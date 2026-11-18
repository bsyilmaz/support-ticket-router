"""Build the presentation (docs/presentation.pptx + .pdf) from results.json.

    python make_slides.py            # reads presentation_info.json, results.json, docs/architecture.png

Every number on the slides is read from results.json, i.e. from your own train.py run.
"""
import json
import os
import shutil
import subprocess
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

INFO = json.load(open("presentation_info.json"))
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results.json"))
M = R["models"]
OUT = "docs/presentation.pptx"

NAVY, TEAL, GREY, LIGHT = (RGBColor(0x1F, 0x2A, 0x44), RGBColor(0x14, 0x8F, 0x86),
                           RGBColor(0x55, 0x5B, 0x66), RGBColor(0xF3, 0xF5, 0xF8))
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SECTION = {}


def pct(x):
    return f"{100 * x:.1f}%"


def text(slide, x, y, w, h, s, size=18, bold=False, color=NAVY, align=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
        if align:
            p.alignment = align
    return tb


def bullets(slide, x, y, w, h, items, size=17):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        lvl = 1 if it.startswith("  ") else 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("–  " if lvl else "•  ") + it.strip()
        p.level = lvl
        p.font.size = Pt(size - 2 * lvl)
        p.font.color.rgb = GREY if lvl else NAVY
        p.space_after = Pt(6)
    return tb


def new_slide(title, section):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    text(s, 0.5, 0.2, 10, 0.7, title, 28, True, RGBColor(255, 255, 255))
    text(s, 10.3, 0.33, 2.6, 0.5, section.upper(), 13, True, RGBColor(0x8F, 0xD8, 0xD0),
         PP_ALIGN.RIGHT)
    n = len(prs.slides)
    text(s, 12.3, 7.0, 0.8, 0.4, str(n), 11, False, GREY, PP_ALIGN.RIGHT)
    return s


def table(slide, x, y, w, rows, col_w=None, size=12, header_color=TEAL, italic_rows=()):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(0.38 * nr))
    t = shp.table
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = str(v)
            p = c.text_frame.paragraphs[0]
            p.font.size = Pt(size)
            p.font.bold = i == 0
            p.font.italic = i in italic_rows
            if i == 0:
                c.fill.solid()
                c.fill.fore_color.rgb = header_color
                p.font.color.rgb = RGBColor(255, 255, 255)
    return t


# ---------------------------------------------------------------- 1 title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
W = RGBColor(255, 255, 255)
text(s, 0.8, 1.2, 11.5, 1, "Customer Support Ticket Router", 44, True, W)
text(s, 0.8, 2.15, 11.5, 0.8, "SWE015 Homework – Building an Intelligent Application  ·  "
     "Use Case 2", 22, False, RGBColor(0x8F, 0xD8, 0xD0))
text(s, 0.8, 3.5, 11.5, 2.6, [f"Name: {INFO['name']}", f"Student ID: {INFO['student_id']}",
                              "Chosen use case: Use Case 2 – Customer Support Ticket Router",
                              f"Repository: {INFO['repo_url']}", f"Demo video: {INFO['video_url']}"],
     19, False, W)

# ---------------------------------------------------------------- 2 intro
s = new_slide("Introduction – the problem", "Introduction")
bullets(s, 0.6, 1.4, 7.2, 5.6, [
    "A support desk receives thousands of free-text messages a day; each must reach the right "
    "team instantly.",
    "Who it is for: support operations – fewer mis-routed tickets, faster first response.",
    "Use case 2: message → normalize → predict intent + confidence → routed team.",
    "  Out-of-scope or low-confidence messages → human agent",
    "  Greetings / small talk → rule-based (ELIZA-style) reply",
    "  All four models of the ladder are shown side by side",
    "Domain: CLINC150 – 150 intents in 10 domains; each domain = one support team.",
])
bullets(s, 8.1, 1.4, 4.8, 5.6, [
    "What success looks like",
    "  High in-scope intent accuracy",
    "  High out-of-scope recall (unknown requests must not be auto-routed)",
    "  Quantized model fast enough for real time on a CPU (well under 50 ms per message)",
    "  Measured honestly on one shared held-out test set",
], 17)

# ---------------------------------------------------------------- 3 data
s = new_slide("Data and preprocessing", "Development")
table(s, 0.6, 1.4, 6.0, [
    ["Split", "In-scope", "Out-of-scope", "Total"],
    ["Train", f"{R['n_train'] - 100:,}", "100", f"{R['n_train']:,}"],
    ["Validation (τ tuning)", f"{R['n_val'] - 100:,}", "100", f"{R['n_val']:,}"],
    ["Test (held out, all models)", f"{R['n_test'] - R['n_test_oos']:,}",
     f"{R['n_test_oos']:,}", f"{R['n_test']:,}"],
], col_w=[2.7, 1.1, 1.3, 0.9], size=13)
bullets(s, 0.6, 3.3, 6.0, 3.8, [
    "Source: CLINC150 'full' split (Larson et al., EMNLP 2019), github.com/clinc/oos-eval, "
    "CC BY 3.0",
    f"{R['n_classes']} classes = 150 intents + 'oos'; 10 domains → 10 teams",
    "Same splits and same preprocessing for every model",
], 15)
bullets(s, 7.0, 1.4, 5.9, 5.6, [
    "Preprocessing pipeline (router/preprocess.py)",
    "  Unicode NFKC, lower-case, URLs/e-mails removed",
    "  Repeated letters collapsed (\"sooooo\" → \"soo\")",
    "  Regex model-code extraction: \"model DW-6534\" → DW-6534",
    "  Edit-distance spelling correction of OOV words against a lexicon (train vocabulary + "
    "product names), distance ≤ 1 (≤ 2 for long words), ties → most frequent",
    "  ELIZA regex rules checked first for greetings / small talk",
], 16)

# ---------------------------------------------------------------- 4 architecture
s = new_slide("Architecture", "Development")
pic = s.shapes.add_picture("docs/architecture.png", Inches(0.5), Inches(1.2), height=Inches(6.15))
pic.left = int((prs.slide_width - pic.width) / 2)

# ---------------------------------------------------------------- 5 technique map
s = new_slide("Technique map", "Development")
table(s, 0.5, 1.35, 12.3, [
    ["Area", "Technique", "Component in the application", "Code"],
    ["B · Classical NLP", "Regular expressions (normalization, model-code extraction)",
     "Preprocessing of every message", "router/preprocess.py"],
    ["B · Classical NLP", "Minimum edit distance (Levenshtein DP)", "Spelling correction of "
     "misspelled words / product names", "SpellCorrector"],
    ["B · Classical NLP", "ELIZA-style pattern → response rules", "Small-talk replies",
     "eliza_reply()"],
    ["B · Classical NLP", "tf-idf (1–2 grams) + logistic regression", "Model 1: baseline",
     "router/models.py TfidfLR"],
    ["A · Neural networks", "word2vec (skip-gram, own training) + feedforward net, SGD + "
     "backprop, dropout", "Model 2: neural network", "W2VFFNN"],
    ["C · Transformers", "Fine-tuning BERT with a classification head; WordPiece tokenizer; "
     "attention", "Model 3: teacher", "TransformerClf.fit"],
    ["D · Efficient models", "Knowledge distillation (soft targets, T = 2) into BERT-mini",
     "Model 4: student", "fit(teacher_logits=…)"],
    ["D · Efficient models", "Dynamic INT8 quantization (torch quantize_dynamic) + "
     "size/latency benchmark", "Model 4: decision model in the app", "quantized(), evaluate.py"],
], col_w=[2.0, 4.6, 3.4, 2.3], size=12)

# ---------------------------------------------------------------- 6 models
t, st = M["teacher_fp32"], M["quantized"]
s = new_slide("The four models and how they were trained", "Development")
bullets(s, 0.6, 1.35, 6.1, 5.8, [
    f"1  {M['baseline']['name']}",
    "  word uni+bigrams, sublinear tf; L2 logistic regression (C = 20)",
    f"2  {M['ffnn']['name']}",
    "  own skip-gram word2vec (100-d) on training messages; averaged sentence vector",
    "  FFNN 100 → 512 → 512 → 151, ReLU, dropout 0.5; SGD (momentum 0.9, weight decay 1e-3), "
    "backprop, 40 epochs",
    f"3  {t['name']}",
    "  pretrained BERT + linear head, fine-tuned with AdamW, linear warm-up/decay, max 64 "
    "tokens",
], 15)
bullets(s, 7.0, 1.35, 5.9, 5.8, [
    f"4  {st['name']}",
    "  Knowledge distillation: student (4 layers, 256 hidden) learns from teacher soft labels",
    "  loss = 0.5·CE(y) + 0.5·T²·KL(student_T ‖ teacher_T), T = 2",
    "  Quantization: torch.ao.quantization.quantize_dynamic — all nn.Linear weights int8, "
    "activations quantized on the fly; embeddings stay FP32",
    "Decision rule (all models): argmax; if confidence < τ → out-of-scope",
    "  τ tuned on validation only, maximizing (in-scope acc + OOS recall)/2",
], 15)

# ---------------------------------------------------------------- 7 results table
s = new_slide("Results – same held-out test set", "Development")
order = ["baseline", "ffnn", "teacher_fp32", "quantized", "student_fp32", "teacher_int8"]
rows = [["Model", "Role", "In-scope acc.", "OOS recall", "Routing acc.", "Size (MB)",
         "Latency ms (mean / p95)"]]
for k in order:
    m = M[k]
    rows.append([m["name"], m["role"], pct(m["in_scope_accuracy"]), pct(m["oos_recall"]),
                 pct(m["routing_accuracy"]), f"{m['size_mb']:.1f}",
                 f"{m['latency']['mean_ms']:.2f} / {m['latency']['p95_ms']:.2f}"])
table(s, 0.4, 1.3, 12.5, rows, col_w=[2.9, 2.9, 1.2, 1.2, 1.2, 1.0, 2.1], size=12,
      italic_rows=(5, 6))
text(s, 0.4, 4.25, 12.5, 1.2, [
    f"Test set: {R['n_test']:,} messages ({R['n_test_oos']:,} out-of-scope). Latency = one "
    f"message (batch 1) incl. preprocessing, averaged over {M['quantized']['latency']['n_runs']} "
    "runs on CPU. Size = files on disk.",
    f"Hardware: {R['hardware']}",
    "Rows in italics are extra ablations (student before quantization; teacher quantized "
    "without distillation)."], 12, False, GREY)
ab = R["typo_ablation"]
table(s, 0.4, 5.55, 8.0, [
    ["Typo robustness (in-scope test with synthetic typos)", "No correction",
     "With edit-distance correction"],
    [M["baseline"]["name"], pct(ab["baseline"]["no_correction"]),
     pct(ab["baseline"]["with_correction"])],
    [M["quantized"]["name"], pct(ab["quantized"]["no_correction"]),
     pct(ab["quantized"]["with_correction"])],
], col_w=[4.0, 1.7, 2.3], size=12, header_color=NAVY)

# ---------------------------------------------------------------- 8 conclusion
core = ["baseline", "ffnn", "teacher_fp32", "quantized"]
best_acc = max(core, key=lambda k: M[k]["in_scope_accuracy"])
best_oos = max(core, key=lambda k: M[k]["oos_recall"])
fastest = min(core, key=lambda k: M[k]["latency"]["mean_ms"])
b, f, q = M["baseline"], M["ffnn"], M["quantized"]
s = new_slide("Conclusion – what the comparison showed", "Conclusion")
bullets(s, 0.6, 1.35, 12.1, 5.8, [
    f"Highest in-scope accuracy: {M[best_acc]['name']} ({pct(M[best_acc]['in_scope_accuracy'])});"
    f" highest OOS recall: {M[best_oos]['name']} ({pct(M[best_oos]['oos_recall'])}).",
    f"Quantized student vs. FP32 teacher: {t['size_mb']:.0f} MB → {q['size_mb']:.0f} MB "
    f"({t['size_mb'] / q['size_mb']:.1f}× smaller), {t['latency']['mean_ms']:.1f} → "
    f"{q['latency']['mean_ms']:.1f} ms ({t['latency']['mean_ms'] / q['latency']['mean_ms']:.1f}× "
    f"faster), in-scope accuracy {pct(t['in_scope_accuracy'])} → {pct(q['in_scope_accuracy'])}.",
    f"Distillation vs. quantization: student FP32 {pct(M['student_fp32']['in_scope_accuracy'])} → "
    f"INT8 {pct(q['in_scope_accuracy'])} (quantization cost "
    f"{100 * (M['student_fp32']['in_scope_accuracy'] - q['in_scope_accuracy']):+.1f} pp).",
    f"tf-idf baseline is a strong, tiny and fast model ({pct(b['in_scope_accuracy'])}, "
    f"{b['size_mb']:.1f} MB, {b['latency']['mean_ms']:.1f} ms) — word overlap is very informative "
    f"in CLINC150; it is weaker on paraphrases and unseen wording.",
    f"word2vec + FFNN ({pct(f['in_scope_accuracy'])}) loses word order and is limited by "
    "embeddings trained on only ~15k short messages.",
    f"Fastest model: {M[fastest]['name']} ({M[fastest]['latency']['mean_ms']:.2f} ms). "
    + ("All four models meet" if all(M[k]['latency']['mean_ms'] < 50 for k in core) else
       "Models meeting") + " a 50 ms CPU budget"
    + ("." if all(M[k]['latency']['mean_ms'] < 50 for k in core) else
       ": " + ", ".join(M[k]['name'] for k in core if M[k]['latency']['mean_ms'] < 50) + "."),
], 16)

# ---------------------------------------------------------------- 9 limitations
s = new_slide("Limitations, failure examples, next steps", "Conclusion")
fails = R["failures"]
picked = [x for x in fails if x["true"] == "oos"][:2] + [x for x in fails if x["true"] != "oos"][:2]
rows = [["Message (test set)", "True", "Predicted (INT8)", "Conf."]]
for x in picked:
    rows.append([x["text"][:70], x["true"], x["pred"], f"{x['conf']:.2f}"])
table(s, 0.5, 1.3, 12.3, rows, col_w=[6.5, 2.1, 2.7, 1.0], size=12, header_color=NAVY)
bullets(s, 0.6, 3.6, 6.0, 3.6, [
    "Limitations",
    "  CLINC150 is English, short and clean — not real Turkish/English appliance tickets",
    "  Out-of-scope examples in training are few (100); τ is one global threshold",
    "  Confident errors: an OOS message close to a real intent is auto-routed",
    "  Spelling correction can 'fix' a rare correct word into a wrong one",
], 14)
bullets(s, 6.9, 3.6, 6.0, 3.6, [
    "Next steps",
    "  Multilingual student (XLM-R / BERTurk) on MASSIVE incl. Turkish + own tickets",
    "  More / synthetic out-of-scope data, per-team thresholds, calibration",
    "  ONNX Runtime export and static quantization for even lower latency",
    "  Active learning from human-agent decisions",
], 14)

# ---------------------------------------------------------------- 10 AI usage
s = new_slide("AI usage statement", "Conclusion")
bullets(s, 0.6, 1.4, 12.1, 5.6, INFO["ai_usage"], 17)

os.makedirs("docs", exist_ok=True)
prs.save(OUT)
print("wrote", OUT)
soffice = shutil.which("soffice") or shutil.which("libreoffice")
if soffice:
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", "docs", OUT],
                   check=False, capture_output=True)
    print("wrote docs/presentation.pdf" if os.path.exists("docs/presentation.pdf") else
          "PDF conversion failed – export the .pptx to PDF manually")
else:
    print("LibreOffice not found – open docs/presentation.pptx and export it as PDF")
