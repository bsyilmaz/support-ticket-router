"""The four rungs of the model ladder.

1. TfidfLR          - classical baseline: tf-idf (word 1-2 grams) + logistic regression   (area B)
2. W2VFFNN          - own word2vec (skip-gram, negative sampling) embeddings, averaged, fed to a feedforward network
                      trained with SGD + backpropagation                                  (area A)
3. TransformerClf   - BERT fine-tuned with a classification head (teacher, FP32)          (area C)
4. TransformerClf   - smaller BERT student trained by knowledge distillation from the
   (quantized)        teacher, then dynamic INT8 quantization of all Linear layers         (area D)

Every model exposes predict_proba(list[str]) -> np.ndarray of shape (n, n_classes).
Inputs are already normalized by router.preprocess.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .preprocess import tokenize
from .word2vec import Word2Vec


def dir_size_mb(path: str) -> float:
    if os.path.isfile(path):
        return os.path.getsize(path) / 2**20
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 2**20


# --------------------------------------------------------------------------- #
# 1. tf-idf + logistic regression
# --------------------------------------------------------------------------- #
class TfidfLR:
    name = "tf-idf + LogReg"

    def __init__(self, C: float = 20.0):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.clf = LogisticRegression(C=C, max_iter=2000)

    def fit(self, texts, y):
        self.clf.fit(self.vec.fit_transform(texts), y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))

    def save(self, path):
        joblib.dump({"vec": self.vec, "clf": self.clf}, path)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        d = joblib.load(path)
        obj.vec, obj.clf = d["vec"], d["clf"]
        return obj


# --------------------------------------------------------------------------- #
# 2. word2vec + feedforward network (SGD + backprop)
# --------------------------------------------------------------------------- #
class FFNN(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_out: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


class W2VFFNN:
    name = "word2vec + FFNN"

    def __init__(self, dim: int = 100, hidden: int = 512, seed: int = 0):
        self.dim, self.hidden, self.seed = dim, hidden, seed
        self.kv = None
        self.net = None
        self.mu = self.sd = None

    # averaged embeddings ----------------------------------------------------
    def embed(self, texts):
        X = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            vecs = [self.kv[w] for w in tokenize(t) if w in self.kv]
            if vecs:
                X[i] = np.mean(vecs, axis=0)
        return X

    def fit(self, texts, y, n_classes, epochs=60, lr=0.05, batch=64, w2v_epochs=10, log=print):
        sents = [tokenize(t) for t in texts]
        self.kv = Word2Vec(dim=self.dim, window=5, negative=10, epochs=w2v_epochs,
                           seed=self.seed).fit(sents, log=log)
        X = self.embed(texts)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = torch.tensor((X - self.mu) / self.sd)
        Y = torch.tensor(y)

        torch.manual_seed(self.seed)
        self.net = FFNN(self.dim, self.hidden, n_classes)
        opt = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(X))
            tot = 0.0
            for i in range(0, len(X), batch):
                idx = perm[i:i + batch]
                loss = F.cross_entropy(self.net(X[idx]), Y[idx])
                opt.zero_grad()
                loss.backward()           # backpropagation
                opt.step()                # SGD update
                tot += loss.item() * len(idx)
            sched.step()
            if (ep + 1) % 10 == 0:
                log(f"  [ffnn] epoch {ep+1}/{epochs} loss {tot/len(X):.4f}")
        return self

    @torch.no_grad()
    def predict_proba(self, texts):
        self.net.eval()
        X = torch.tensor((self.embed(texts) - self.mu) / self.sd)
        return F.softmax(self.net(X), -1).numpy()

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        self.kv.save(os.path.join(d, "w2v.npz"))
        torch.save({"state": self.net.state_dict(), "mu": self.mu, "sd": self.sd,
                    "dim": self.dim, "hidden": self.hidden,
                    "n_out": self.net.net[-1].out_features}, os.path.join(d, "ffnn.pt"))

    @classmethod
    def load(cls, d):
        ck = torch.load(os.path.join(d, "ffnn.pt"), weights_only=False)
        obj = cls(ck["dim"], ck["hidden"])
        obj.kv = Word2Vec.load(os.path.join(d, "w2v.npz"))
        obj.mu, obj.sd = ck["mu"], ck["sd"]
        obj.net = FFNN(ck["dim"], ck["hidden"], ck["n_out"])
        obj.net.load_state_dict(ck["state"])
        obj.net.eval()
        return obj


# --------------------------------------------------------------------------- #
# 3./4. Transformer classifiers (fine-tuning, distillation, quantization)
# --------------------------------------------------------------------------- #
class TransformerClf:
    def __init__(self, model, tokenizer, name="BERT", max_len=64, device="cpu"):
        self.model, self.tok, self.name = model, tokenizer, name
        self.max_len, self.device = max_len, device

    def _enc(self, texts):
        return self.tok(texts, padding=True, truncation=True, max_length=self.max_len,
                        return_tensors="pt")

    @torch.no_grad()
    def logits(self, texts, batch=128):
        self.model.eval()
        out = []
        for i in range(0, len(texts), batch):
            enc = {k: v.to(self.device) for k, v in self._enc(texts[i:i + batch]).items()}
            out.append(self.model(**enc).logits.float().cpu())
        return torch.cat(out)

    def predict_proba(self, texts):
        return F.softmax(self.logits(texts), -1).numpy()

    def fit(self, texts, y, epochs=4, lr=5e-5, batch=32, teacher_logits=None,
            T=2.0, alpha=0.5, warmup=0.1, log=print, seed=0):
        """Fine-tune with cross-entropy; if teacher_logits is given, use the
        knowledge-distillation loss  alpha*CE + (1-alpha)*T^2*KL(student_T || teacher_T)."""
        from transformers import get_linear_schedule_with_warmup

        torch.manual_seed(seed)
        self.model.to(self.device).train()
        Y = torch.tensor(y)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * ((len(texts) + batch - 1) // batch)
        sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(len(texts)).tolist()
            tot, t0 = 0.0, time.time()
            for i in range(0, len(perm), batch):
                idx = perm[i:i + batch]
                enc = {k: v.to(self.device) for k, v in self._enc([texts[j] for j in idx]).items()}
                yb = Y[idx].to(self.device)
                with torch.autocast("cuda", enabled=use_amp):
                    logits = self.model(**enc).logits.float()
                    loss = F.cross_entropy(logits, yb)
                    if teacher_logits is not None:
                        tl = teacher_logits[idx].to(self.device)
                        kd = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(tl / T, -1),
                                      reduction="batchmean") * T * T
                        loss = alpha * loss + (1 - alpha) * kd
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item() * len(idx)
            log(f"  [{self.name}] epoch {ep+1}/{epochs} loss {tot/len(texts):.4f} ({time.time()-t0:.0f}s)")
        self.model.eval()
        return self

    def to_cpu(self):
        self.model.to("cpu")
        self.device = "cpu"
        return self

    # FP32 save/load (HF format) ----------------------------------------------
    def save(self, d):
        self.model.save_pretrained(d)
        self.tok.save_pretrained(d)

    @classmethod
    def load(cls, d, name="BERT"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        m = AutoModelForSequenceClassification.from_pretrained(d)
        return cls(m, AutoTokenizer.from_pretrained(d), name=name)

    # INT8 dynamic quantization ------------------------------------------------
    def quantized(self, name=None):
        """Dynamic INT8 quantization: weights of every nn.Linear stored as int8,
        activations quantized on the fly at inference. CPU only."""
        q = torch.ao.quantization.quantize_dynamic(self.model.cpu().eval(), {nn.Linear},
                                                   dtype=torch.qint8)
        return TransformerClf(q, self.tok, name=name or self.name + " INT8")

    def save_quantized(self, d):
        os.makedirs(d, exist_ok=True)
        self.model.config.save_pretrained(d)
        self.tok.save_pretrained(d)
        torch.save(self.model.state_dict(), os.path.join(d, "model_int8.pt"))

    @classmethod
    def load_quantized(cls, d, name="BERT INT8"):
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        cfg = AutoConfig.from_pretrained(d)
        skeleton = AutoModelForSequenceClassification.from_config(cfg).eval()
        q = torch.ao.quantization.quantize_dynamic(skeleton, {nn.Linear}, dtype=torch.qint8)
        q.load_state_dict(torch.load(os.path.join(d, "model_int8.pt"), weights_only=False))
        return cls(q.eval(), AutoTokenizer.from_pretrained(d), name=name)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
