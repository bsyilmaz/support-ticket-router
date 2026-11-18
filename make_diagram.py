"""Draw the architecture diagram (docs/architecture.png) from the components in the code.
Model names / precisions are read from results.json so the figure matches the trained models."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RES = sys.argv[1] if len(sys.argv) > 1 else "results.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/architecture.png"
r = json.load(open(RES))["models"]

C = {"ui": "#DCEBFA", "pre": "#FCEBD2", "model": "#E3F4E1", "main": "#B9E3B4",
     "post": "#EFE1F7", "train": "#EEEEEE", "edge": "#4A4A4A"}

fig, ax = plt.subplots(figsize=(16, 8.6), dpi=150)
ax.set_xlim(0, 160)
ax.set_ylim(0, 86)
ax.axis("off")


def box(x, y, w, h, title, lines=(), color="#fff", bold=False, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=color, ec=C["edge"], lw=2.2 if bold else 1.2))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=fs + 0.5,
            weight="bold")
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.2 - i * 3.3, l, ha="center", va="top", fontsize=fs - 1.3)


def arrow(x1, y1, x2, y2, label=None, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=C["edge"], lw=1.4, linestyle=ls))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, label, ha="center", fontsize=8.5,
                style="italic", color="#333")


# region labels
ax.add_patch(FancyBboxPatch((1, 27), 158, 58, boxstyle="round,pad=0.3", fc="none",
                            ec="#7A9CC6", lw=1.2, ls="--"))
ax.text(3, 83.5, "INFERENCE  —  app.py (Gradio), runs on CPU (laptop / Colab / HF Space)",
        fontsize=10, color="#3B5E8C", weight="bold", va="top")
ax.add_patch(FancyBboxPatch((1, 1), 158, 23, boxstyle="round,pad=0.3", fc="none",
                            ec="#999", lw=1.2, ls="--"))
ax.text(3, 22.5, "TRAINING  —  train.py (offline, GPU recommended)", fontsize=10,
        color="#555", weight="bold", va="top")

# inference path
box(3, 52, 19, 20, "User", ["browser", "types a free-text", "support message"], C["ui"])
box(27, 52, 27, 26, "Preprocessing", ["router/preprocess.py", "1 regex normalization",
                                       "2 regex model-code", "   extraction (DW-6534)",
                                       "3 edit-distance spelling", "   correction (lexicon)"],
    C["pre"])
box(27, 29, 27, 15, "ELIZA rules", ["regex patterns for", "greetings / small talk"], C["pre"])

mx, mw = 60, 42
names = [("baseline", "tf-idf (1-2 grams) + LogReg", C["model"]),
         ("ffnn", "own word2vec (100-d) + FFNN, SGD", C["model"]),
         ("teacher_fp32", None, C["model"]),
         ("quantized", None, C["main"])]
ys = [70.5, 60.5, 50.5, 36.5]
for (k, desc, col), y in zip(names, ys):
    m = r[k]
    title = m["name"]
    if k == "teacher_fp32":
        desc = "fine-tuned, 151-way head"
    if k == "quantized":
        desc = "distilled from the teacher, dynamic quantization"
    h = 12 if k == "quantized" else 9
    prec = "FP64" if k == "baseline" else ("INT8" if k == "quantized" else "FP32")
    box(mx, y, mw, h, title, [f"{desc}  ·  {prec}"] + (["DECISION MODEL (routes the ticket)"] if k == "quantized" else []),
        col, bold=(k == "quantized"), fs=10)

box(108, 31, 25, 47, "Post-processing", ["softmax confidence", "τ threshold (tuned on val)",
                                          "conf < τ or 'oos'", "  → Human Agent",
                                          "intent → team table", "  (router/data.py)",
                                          "small_talk → ELIZA"], C["post"])
box(138, 52, 20, 20, "Gradio UI", ["routed team", "intent + confidence",
                                   "4 models side by", "side + latency"], C["ui"])

arrow(22, 62, 27, 62, "text")
arrow(54, 65, 60, 75)
arrow(54, 65, 60, 65)
arrow(54, 65, 60, 55)
arrow(54, 65, 60, 42.5)
arrow(40.5, 52, 40.5, 44)
arrow(102, 42.5, 108, 52)
for y in (75, 65, 55):
    arrow(102, y, 108, 60, ls=":")
arrow(54, 33, 108, 33, "greeting / small talk → rule reply", ls="--")
arrow(133, 60, 138, 60)
ax.text(81, 29.8, "dotted = comparison only (shown in the UI table)", fontsize=8.5,
        color="#555", ha="center")

# training path
box(3, 3, 20, 15, "CLINC150", ["15,100 train", "3,100 val", "5,500 test"], C["train"])
box(28, 3, 24, 15, "Classical / NN", ["fit tf-idf + LogReg", "train word2vec,",
                                       "FFNN (SGD + backprop)"], C["train"])
box(57, 3, 26, 15, "Teacher", [r["teacher_fp32"]["name"], "fine-tune (AdamW)",
                               "→ soft targets"], C["train"])
box(88, 3, 28, 15, "Distillation + quant.", ["student = α·CE + (1-α)·T²·KL",
                                            "quantize_dynamic(Linear,", "qint8)"], C["train"])
box(121, 3, 36, 15, "artifacts/  (not in git)", ["weights → Hugging Face Hub /",
                                                 "Google Drive; results.json",
                                                 "(metrics, size, latency, τ)"], C["train"])
arrow(23, 10.5, 28, 10.5)
arrow(52, 10.5, 57, 10.5)
arrow(83, 10.5, 88, 10.5)
arrow(116, 10.5, 121, 10.5)
arrow(139, 18, 148, 52)
ax.text(144.5, 34, "weights loaded\nat start-up", fontsize=8.5, style="italic", color="#333")

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
plt.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
