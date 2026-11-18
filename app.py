"""Customer Support Ticket Router - Gradio application.

    python app.py            # opens http://127.0.0.1:7860

The routing decision is made by the quantized (INT8) distilled BERT student.
The other three models of the ladder are run on the same message and shown side by side.
Model weights are read from ./artifacts (created by train.py) or, if ROUTER_HUB_REPO is set,
downloaded from that Hugging Face Hub model repo.
"""
from __future__ import annotations

import json
import os
import time

import gradio as gr
import joblib
import numpy as np
import torch

from router import data as D
from router.models import TfidfLR, TransformerClf, W2VFFNN
from router.preprocess import eliza_reply, extract_model_codes, normalize, small_talk_reply

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
# Where the trained weights come from: local ./artifacts (after train.py) or, if that folder
# does not exist, the Hugging Face Hub repo below (set it once after upload_artifacts.py).
HUB_REPO = os.environ.get("ROUTER_HUB_REPO", "selimyilm/support-ticket-router")
ART = os.environ.get("ROUTER_ARTIFACTS", "artifacts")
if not os.path.isdir(os.path.join(ART, "student_int8")):
    if HUB_REPO.startswith("<"):
        raise SystemExit("No ./artifacts found. Run `python train.py` first, or set HUB_REPO in "
                         "app.py / the ROUTER_HUB_REPO environment variable.")
    from huggingface_hub import snapshot_download

    print(f"downloading weights from https://huggingface.co/{HUB_REPO} ...")
    ART = snapshot_download(HUB_REPO)

ds = D.load()
ID2LABEL = dict(enumerate(ds.intents))
OOS_ID = ds.label2id[D.OOS]
with open(os.path.join(ART, "results.json")) as f:
    RESULTS = json.load(f)
corrector = joblib.load(os.path.join(ART, "spell_corrector.joblib"))

MODELS = {   # key -> (model, precision label)
    "quantized": (TransformerClf.load_quantized(os.path.join(ART, "student_int8"),
                                                name=RESULTS["models"]["quantized"]["name"]), "INT8"),
    "teacher_fp32": (TransformerClf.load(os.path.join(ART, "teacher_fp32"),
                                         name=RESULTS["models"]["teacher_fp32"]["name"]), "FP32"),
    "ffnn": (W2VFFNN.load(os.path.join(ART, "w2v_ffnn")), "FP32"),
    "baseline": (TfidfLR.load(os.path.join(ART, "tfidf_lr.joblib")), "FP64"),
}
TAU = {k: RESULTS["models"][k]["threshold"] for k in MODELS}
ORDER = ["baseline", "ffnn", "teacher_fp32", "quantized"]


def classify(key, text):
    model, _ = MODELS[key]
    t0 = time.perf_counter()
    probs = model.predict_proba([text])[0]
    ms = (time.perf_counter() - t0) * 1000
    i = int(np.argmax(probs))
    conf = float(probs[i])
    intent = D.OOS if (i == OOS_ID or conf < TAU[key]) else ID2LABEL[i]
    return intent, conf, ID2LABEL[i], ms


def route(message: str):
    if not message or not message.strip():
        return "Please type a message.", "", None
    t0 = time.perf_counter()
    codes = extract_model_codes(message)
    norm = normalize(message)
    text, changes = corrector.correct(norm)
    pre_ms = (time.perf_counter() - t0) * 1000

    rows = []
    for key in ORDER:
        intent, conf, top, ms = classify(key, text)
        team = ds.team(intent)
        rows.append([MODELS[key][0].name, MODELS[key][1], intent, team, f"{conf:.3f}",
                     f"{ms:.1f}"])
    intent, conf, top, ms = classify("quantized", text)

    rule = eliza_reply(message)
    if rule is not None:
        decision = f"### 💬 Small talk → rule-based reply\n> {rule}"
    elif intent == D.OOS:
        why = "out-of-scope" if top == D.OOS else f"low confidence ({conf:.2f} < τ={TAU['quantized']:.2f})"
        decision = f"### 🧑 Routed to: **{D.HUMAN_AGENT}**\nReason: {why}. Best guess was `{top}`."
    elif ds.domain_of_intent[intent] == "small_talk":
        decision = f"### 💬 Small talk → rule-based reply\n> {small_talk_reply(message)}"
    else:
        decision = (f"### ✅ Routed to: **{ds.team(intent)}**\n"
                    f"Intent: `{intent}` · confidence **{conf:.2f}**")
    decision += (f"\n\n<sub>Decision model: {MODELS['quantized'][0].name} · {ms:.1f} ms on CPU</sub>")

    pre = (f"**Normalized:** `{norm}`\n\n"
           f"**After spelling correction:** `{text}`\n\n"
           f"**Corrections (edit distance):** {', '.join(f'{a}→{b}' for a, b in changes) or 'none'}\n\n"
           f"**Model codes (regex):** {', '.join(codes) or 'none'}\n\n"
           f"<sub>preprocessing {pre_ms:.1f} ms</sub>")
    return decision, pre, rows


EXAMPLES = [
    "Can you transfer 200 dollars from my checking to savings?",
    "my credit card was declined at the store, why??",
    "plese tranfer 300 dolars to my savngs acount",
    "My dishwasher won't drain, model DW-6534",
    "Hiii, how are you?",
    "what's the best way to cook a steak medium rare",
    "who won the football game last night",
    "set an alarm for 6am tomorrow",
]

with gr.Blocks(title="Support Ticket Router") as demo:
    gr.Markdown("# 🎫 Customer Support Ticket Router\nType a support message. It is normalized, "
                "spell-corrected and classified into one of 150 intents (10 teams). Low-confidence "
                "or out-of-scope messages go to a **human agent**; small talk gets a rule-based reply.")
    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(label="Customer message", lines=3, placeholder="e.g. my card got declined")
            btn = gr.Button("Route message", variant="primary")
            gr.Examples(EXAMPLES, inp)
        with gr.Column(scale=2):
            out_decision = gr.Markdown()
    out_pre = gr.Markdown()
    out_table = gr.Dataframe(headers=["Model", "Precision", "Intent", "Team", "Confidence",
                                      "Latency (ms)"], label="All four models side by side")
    btn.click(route, inp, [out_decision, out_pre, out_table])
    inp.submit(route, inp, [out_decision, out_pre, out_table])

if __name__ == "__main__":
    demo.launch(share=bool(os.environ.get("GRADIO_SHARE")))
