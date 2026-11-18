"""Train and evaluate the whole model ladder on CLINC150, write artifacts/ and results.json.

    python train.py                 # full run (GPU strongly recommended for the BERT steps)
    python train.py --smoke         # quick offline code check: small subset, tiny random-init
                                    # BERTs, no download from Hugging Face (numbers meaningless)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import joblib
import numpy as np
import torch

from router import data as D
from router.evaluate import decide, hardware, latency_ms, metrics, tune_threshold
from router.models import TfidfLR, TransformerClf, W2VFFNN, dir_size_mb
from router.preprocess import SpellCorrector, english_words, normalize

ART = "artifacts"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def add_typos(text: str, rng: random.Random, p: float = 0.35) -> str:
    """Synthetic noise for the robustness ablation: one random edit in some words."""
    out = []
    for w in text.split():
        if len(w) >= 5 and w.isalpha() and rng.random() < p:
            i = rng.randrange(1, len(w) - 1)
            op = rng.choice(["del", "swap", "sub", "dup"])
            if op == "del":
                w = w[:i] + w[i + 1:]
            elif op == "swap":
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif op == "sub":
                w = w[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + w[i + 1:]
            else:
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)


def scratch_tokenizer(texts, out_dir):
    """Offline WordPiece tokenizer (only used by --smoke / --from-scratch)."""
    from tokenizers import BertWordPieceTokenizer
    from transformers import PreTrainedTokenizerFast

    os.makedirs(out_dir, exist_ok=True)
    wp = BertWordPieceTokenizer(lowercase=True)
    wp.train_from_iterator(texts, vocab_size=6000, min_frequency=2)
    from tokenizers.processors import BertProcessing
    wp._tokenizer.post_processor = BertProcessing(("[SEP]", wp.token_to_id("[SEP]")),
                                                  ("[CLS]", wp.token_to_id("[CLS]")))
    tok = PreTrainedTokenizerFast(tokenizer_object=wp._tokenizer, unk_token="[UNK]",
                                  pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
                                  mask_token="[MASK]")
    tok.save_pretrained(out_dir)
    return tok


def build_transformer(name_or_size, n_labels, id2label, tokenizer=None, scratch=False):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig, \
        BertForSequenceClassification

    kw = dict(num_labels=n_labels, id2label=id2label, label2id={v: k for k, v in id2label.items()})
    if scratch:
        L, H = name_or_size
        cfg = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=H, num_hidden_layers=L,
                         num_attention_heads=max(H // 64, 2), intermediate_size=4 * H,
                         max_position_embeddings=128, **kw)
        return BertForSequenceClassification(cfg), tokenizer
    tok = AutoTokenizer.from_pretrained(name_or_size)
    return AutoModelForSequenceClassification.from_pretrained(name_or_size, **kw), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="bert-base-uncased")
    ap.add_argument("--student", default="google/bert_uncased_L-4_H-256_A-4")
    ap.add_argument("--teacher-epochs", type=int, default=4)
    ap.add_argument("--student-epochs", type=int, default=15)
    ap.add_argument("--teacher-lr", type=float, default=4e-5)
    ap.add_argument("--student-lr", type=float, default=1e-4)
    ap.add_argument("--latency-runs", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--from-scratch", action="store_true",
                    help="random-init BERTs + own WordPiece tokenizer (no Hugging Face download)")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    scratch = args.smoke or args.from_scratch

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(ART, exist_ok=True)
    log("device:", device, "|", hardware())

    # ------------------------------------------------------------------ data
    ds = D.load()
    if args.smoke:
        rng = random.Random(0)
        def sub(s, n):
            idx = rng.sample(range(len(s.texts)), n)
            return D.Split([s.texts[i] for i in idx], [s.labels[i] for i in idx])
        ds.train, ds.val, ds.test = sub(ds.train, 3000), sub(ds.val, 600), sub(ds.test, 800)
    l2i = ds.label2id
    oos_id = l2i[D.OOS]
    id2label = {i: l for l, i in l2i.items()}
    team_of = lambda i: ds.team(id2label[int(i)])
    ytr, yva, yte = (np.array([l2i[l] for l in s.labels]) for s in (ds.train, ds.val, ds.test))
    log(f"train {len(ytr)} | val {len(yva)} | test {len(yte)} "
        f"(oos in test: {(yte == oos_id).sum()}) | classes {len(l2i)}")

    # ------------------------------------------------- preprocessing (area B)
    corrector = SpellCorrector([normalize(t) for t in ds.train.texts], known_words=english_words())
    joblib.dump(corrector, os.path.join(ART, "spell_corrector.joblib"))

    def prep(t):
        return corrector.correct(normalize(t))[0]

    Xtr, Xva, Xte = ([prep(t) for t in s.texts] for s in (ds.train, ds.val, ds.test))
    n_corr = sum(1 for a, b in zip(ds.test.texts, Xte) if normalize(a) != b)
    log(f"spelling correction changed {n_corr} of {len(Xte)} test messages")

    results = {"hardware": hardware(), "train_device": device, "smoke": args.smoke,
               "from_scratch": scratch, "n_train": len(ytr), "n_val": len(yva),
               "n_test": len(yte), "n_test_oos": int((yte == oos_id).sum()),
               "n_classes": len(l2i), "models": {}}
    lat_texts = ds.test.texts[: max(args.latency_runs, 100)]
    zoo = {}

    def evaluate(key, model, path, role):
        t0 = time.time()
        pva = model.predict_proba(Xva)
        tau = tune_threshold(pva, yva, oos_id)
        pte = model.predict_proba(Xte)
        pred = decide(pte, oos_id, tau)
        m = metrics(pred, yte, oos_id, team_of)
        lat = latency_ms(lambda b: model.predict_proba([prep(x) for x in b]), lat_texts,
                         n_runs=args.latency_runs)
        results["models"][key] = {"name": model.name, "role": role, "threshold": tau,
                                  "size_mb": dir_size_mb(path), **m, "latency": lat}
        zoo[key] = (model, pte, pred, tau)
        log(f"{model.name:32s} in-scope acc {m['in_scope_accuracy']:.4f} | OOS recall "
            f"{m['oos_recall']:.4f} | size {dir_size_mb(path):.1f} MB | "
            f"{lat['mean_ms']:.2f} ms | tau {tau:.2f} ({time.time()-t0:.0f}s)")

    # ----------------------------------------------------------- 1. baseline
    log("1/4 tf-idf + logistic regression")
    base = TfidfLR().fit(Xtr, ytr)
    p = os.path.join(ART, "tfidf_lr.joblib")
    base.save(p)
    evaluate("baseline", base, p, "Classical baseline")

    # ------------------------------------------------------------ 2. FFNN
    log("2/4 word2vec + feedforward network (SGD)")
    ff = W2VFFNN().fit(Xtr, ytr, len(l2i), epochs=30 if args.smoke else 40,
                      w2v_epochs=3 if args.smoke else 10, log=log)
    p = os.path.join(ART, "w2v_ffnn")
    ff.save(p)
    evaluate("ffnn", ff, p, "Neural network (own training)")

    # ------------------------------------------------------ 3. BERT teacher
    log(f"3/4 Transformer teacher: {'scratch' if scratch else args.teacher}")
    tok = scratch_tokenizer(Xtr, os.path.join(ART, "scratch_tok")) if scratch else None
    tm, ttok = build_transformer((4, 256) if scratch else args.teacher, len(l2i), id2label,
                                 tok, scratch)
    teacher = TransformerClf(tm, ttok, name="BERT-base (FP32)" if not scratch else
                             "BERT L4-H256 scratch (FP32)", device=device)
    teacher.fit(Xtr, ytr, epochs=3 if args.smoke else args.teacher_epochs, lr=
                (5e-4 if scratch else args.teacher_lr), log=log)
    teacher_logits = teacher.logits(Xtr)          # soft targets for distillation
    teacher.to_cpu()
    p = os.path.join(ART, "teacher_fp32")
    teacher.save(p)
    evaluate("teacher_fp32", teacher, p, "Transformer (fine-tuned)")

    # teacher INT8 (extra row: quantization without distillation)
    tq = teacher.quantized(name=teacher.name.replace("FP32", "INT8"))
    p = os.path.join(ART, "teacher_int8")
    tq.save_quantized(p)
    evaluate("teacher_int8", tq, p, "Transformer, INT8 only (extra)")

    # ------------------------------- 4. distilled student -> INT8 quantization
    log(f"4/4 distillation into student: {'scratch' if scratch else args.student}")
    sm, stok = build_transformer((2, 128) if scratch else args.student, len(l2i), id2label,
                                 tok, scratch)
    if scratch:
        stok = ttok
    student = TransformerClf(sm, stok, name="BERT-mini student (FP32)" if not scratch else
                             "BERT L2-H128 student (FP32)", device=device)
    student.fit(Xtr, ytr, epochs=3 if args.smoke else args.student_epochs,
                lr=(1e-3 if scratch else args.student_lr),
                teacher_logits=teacher_logits, T=2.0, alpha=0.5, log=log)
    student.to_cpu()
    p = os.path.join(ART, "student_fp32")
    student.save(p)
    evaluate("student_fp32", student, p, "Distilled student, FP32 (extra)")

    sq = student.quantized(name=student.name.replace("FP32", "INT8"))
    p = os.path.join(ART, "student_int8")
    sq.save_quantized(p)
    sq = TransformerClf.load_quantized(p, name=sq.name)   # check the saved file reloads
    evaluate("quantized", sq, p, "Quantized Transformer (used in app)")

    # --------------------------------------- robustness ablation (typo noise)
    log("ablation: synthetic typos, with vs without edit-distance correction")
    rng = random.Random(1)
    ins = [i for i in range(len(yte)) if yte[i] != oos_id]
    noisy = [add_typos(normalize(ds.test.texts[i]), rng) for i in ins]
    abl = {}
    for key in ("baseline", "quantized"):
        model, _, _, tau = zoo[key]
        r = {}
        for mode, xs in (("no_correction", noisy),
                         ("with_correction", [corrector.correct(x)[0] for x in noisy])):
            pred = decide(model.predict_proba(xs), oos_id, tau)
            r[mode] = float((pred == yte[ins]).mean())
        abl[key] = r
        log(f"  {model.name}: {r}")
    results["typo_ablation"] = {"n": len(ins), "examples": noisy[:5], **abl}

    # ------------------------------------------------------ failure examples
    _, pte, pred, tau = zoo["quantized"]
    fails = [i for i in range(len(yte)) if pred[i] != yte[i]]
    results["failures"] = [{"text": ds.test.texts[i], "true": id2label[int(yte[i])],
                            "pred": id2label[int(pred[i])], "conf": float(pte[i].max())}
                           for i in fails[:25]]
    results["thresholds_note"] = "tau tuned on validation to maximize (in-scope acc + OOS recall)/2"
    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    for path in (args.out, os.path.join(ART, "results.json")):
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
    log("wrote", args.out)


if __name__ == "__main__":
    main()
