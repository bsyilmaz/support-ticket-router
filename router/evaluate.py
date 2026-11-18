"""Shared evaluation: the same held-out CLINC150 test set (4,500 in-scope + 1,000
out-of-scope messages) and the same decision rule for every model."""
from __future__ import annotations

import os
import platform
import time

import numpy as np
import torch


def decide(probs: np.ndarray, oos_id: int, tau: float) -> np.ndarray:
    """argmax, but predict out-of-scope if the top probability is below tau."""
    pred = probs.argmax(1)
    pred[probs.max(1) < tau] = oos_id
    return pred


def metrics(pred: np.ndarray, y: np.ndarray, oos_id: int, team_of=None) -> dict:
    ins, oos = y != oos_id, y == oos_id
    m = {
        "in_scope_accuracy": float((pred[ins] == y[ins]).mean()),
        "oos_recall": float((pred[oos] == oos_id).mean()),
        "overall_accuracy": float((pred == y).mean()),
    }
    tp = ((pred == oos_id) & oos).sum()
    m["oos_precision"] = float(tp / max((pred == oos_id).sum(), 1))
    if team_of is not None:
        m["routing_accuracy"] = float(np.mean([team_of(p) == team_of(t) for p, t in zip(pred, y)]))
    return m


def tune_threshold(probs, y, oos_id, grid=np.linspace(0.0, 0.99, 67)) -> float:
    """Pick tau on the *validation* split, maximizing (in-scope acc + OOS recall)/2."""
    best, best_tau = -1, 0.0
    for tau in grid:
        m = metrics(decide(probs, oos_id, tau), y, oos_id)
        score = (m["in_scope_accuracy"] + m["oos_recall"]) / 2
        if score > best:
            best, best_tau = score, float(tau)
    return best_tau


def latency_ms(predict_fn, texts, n_runs: int = 200, warmup: int = 10) -> dict:
    """Single-message latency (batch size 1), averaged over n_runs different messages."""
    for t in texts[:warmup]:
        predict_fn([t])
    times = []
    for i in range(n_runs):
        t = texts[i % len(texts)]
        t0 = time.perf_counter()
        predict_fn([t])
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "n_runs": n_runs}


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return (f"CPU: {cpu}, {os.cpu_count()} logical cores, torch threads={torch.get_num_threads()}, "
            f"{platform.system()} / Python {platform.python_version()} / torch {torch.__version__}")
