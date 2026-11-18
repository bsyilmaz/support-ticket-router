"""Minimal word2vec: skip-gram with negative sampling (Mikolov et al., 2013), in PyTorch.
Trained on the training messages only."""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


class Word2Vec:
    def __init__(self, dim=100, window=5, negative=10, min_count=1, epochs=10, lr=0.01,
                 batch=1024, seed=0):
        self.dim, self.window, self.negative = dim, window, negative
        self.min_count, self.epochs, self.lr, self.batch, self.seed = min_count, epochs, lr, batch, seed
        self.vocab: dict[str, int] = {}
        self.vectors: np.ndarray | None = None

    def fit(self, sentences: list[list[str]], log=print):
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        counts = Counter(w for s in sentences for w in s)
        words = [w for w, c in counts.most_common() if c >= self.min_count]
        self.vocab = {w: i for i, w in enumerate(words)}
        V = len(words)

        # (center, context) pairs with a random window size, as in the original word2vec
        centers, contexts = [], []
        for s in sentences:
            ids = [self.vocab[w] for w in s if w in self.vocab]
            for i, c in enumerate(ids):
                b = rng.integers(1, self.window + 1)
                for j in range(max(0, i - b), min(len(ids), i + b + 1)):
                    if j != i:
                        centers.append(c)
                        contexts.append(ids[j])
        centers, contexts = torch.tensor(centers), torch.tensor(contexts)
        freq = torch.tensor([counts[w] for w in words], dtype=torch.float) ** 0.75
        noise = freq / freq.sum()

        W_in = torch.nn.Embedding(V, self.dim, sparse=True)
        W_out = torch.nn.Embedding(V, self.dim, sparse=True)
        torch.nn.init.uniform_(W_in.weight, -0.5 / self.dim, 0.5 / self.dim)
        torch.nn.init.zeros_(W_out.weight)
        opt = torch.optim.SparseAdam(list(W_in.parameters()) + list(W_out.parameters()), lr=self.lr)
        n = len(centers)
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for k in range(0, n, self.batch):
                idx = perm[k:k + self.batch]
                c, o = centers[idx], contexts[idx]
                neg = torch.multinomial(noise, len(idx) * self.negative, replacement=True)
                neg = neg.view(len(idx), self.negative)
                vc = W_in(c)                                     # (B, d)
                pos = (vc * W_out(o)).sum(-1)                    # (B,)
                negs = torch.bmm(W_out(neg), vc.unsqueeze(-1)).squeeze(-1)   # (B, K)
                loss = -(F.logsigmoid(pos) + F.logsigmoid(-negs).sum(-1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            log(f"  [word2vec] epoch {ep+1}/{self.epochs} loss {tot/n:.4f} ({n:,} pairs, V={V})")
        self.vectors = W_in.weight.detach().numpy().copy()
        return self

    def __contains__(self, w):
        return w in self.vocab

    def __getitem__(self, w):
        return self.vectors[self.vocab[w]]

    def most_similar(self, w, k=5):
        v = self[w]
        M = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9)
        sims = M @ (v / (np.linalg.norm(v) + 1e-9))
        inv = {i: x for x, i in self.vocab.items()}
        return [(inv[i], float(sims[i])) for i in np.argsort(-sims)[1:k + 1]]

    def save(self, path):
        np.savez_compressed(path, vectors=self.vectors, words=np.array(list(self.vocab)))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls(dim=d["vectors"].shape[1])
        obj.vectors = d["vectors"]
        obj.vocab = {w: i for i, w in enumerate(d["words"].tolist())}
        return obj
