# Customer Support Ticket Router (SWE015 – Use Case 2)

A support message is **normalized → spell-corrected → classified into one of 150 intents
(10 teams)**. Out-of-scope or low-confidence messages go to a **human agent**, greetings and
small talk get a **rule-based (ELIZA-style) reply**. The routing decision is made by a
**distilled, INT8-quantized BERT**; the other three models of the ladder are shown side by side.

| # | Model | Technique area |
|---|-------|----------------|
| 1 | tf-idf (1–2 grams) + logistic regression | B – classical NLP |
| 2 | own word2vec (skip-gram, negative sampling) + feedforward net, SGD + backprop | A – neural networks |
| 3 | `bert-base-uncased` fine-tuned with a classification head (teacher) | C – Transformers |
| 4 | `google/bert_uncased_L-4_H-256_A-4` distilled from the teacher, then dynamic INT8 quantization | D – efficient models |

Preprocessing (area B): regex normalization, regex model-code extraction (`DW-6534`),
minimum-edit-distance spelling correction, ELIZA rules. The architecture diagram is slide 4 of the
presentation; `python make_diagram.py` regenerates it as `docs/architecture.png`.

Data: [CLINC150](https://github.com/clinc/oos-eval) "full" split (Larson et al., 2019, CC BY 3.0),
downloaded automatically into `data/` on first run. All models use the same held-out test set
(4,500 in-scope + 1,000 out-of-scope messages).

## Setup

Python 3.12 or 3.13 (tested on 3.13; see the tested versions in `requirements.txt`).

```bash
git clone https://github.com/bsyilmaz/support-ticket-router.git
cd support-ticket-router
pip install -r requirements.txt
```

## Run the application (single command)

```bash
python app.py
```

Then open http://127.0.0.1:7860. On the first start the trained weights are downloaded
automatically from the Hugging Face Hub repo
[`selimyilm/support-ticket-router`](https://huggingface.co/selimyilm/support-ticket-router)
(set in `app.py`, `HUB_REPO`). If you trained the models yourself (next section), the local
`artifacts/` folder is used instead. Startup takes ~30 s (BERT-base is loaded for the
side-by-side comparison).

## Train and evaluate all models

```bash
python train.py          # ~15–25 min on a Colab T4 GPU; writes artifacts/ and results.json
```

On a GPU the BERT models train on the GPU, but **all latencies are measured on the CPU**
(batch size 1, averaged over 200 messages), because the application runs on a CPU.
`results.json` has, for every model: in-scope accuracy, out-of-scope recall, routing accuracy,
the tuned threshold, size on disk (MB), latency (mean / p50 / p95 ms) and the hardware.

`python train.py --smoke` runs a 3-minute offline check of the whole pipeline (small subset,
tiny randomly initialized BERTs – the numbers are meaningless).

Google Colab: open `notebooks/colab_train_and_demo.ipynb`
(Runtime → Change runtime type → T4 GPU) and run all cells.

## Slides and diagram

```bash
python make_diagram.py   # docs/architecture.png, uses model names from results.json
python make_slides.py    # docs/presentation.pptx (+ .pdf if LibreOffice is installed)
```

Fill in `presentation_info.json` (name, student ID, links) first. Every number on the slides is
read from `results.json`.

## Weights are not stored in git

`artifacts/` is in `.gitignore`. Publish the weights with

```bash
huggingface-cli login
python upload_artifacts.py selimyilm/support-ticket-router
```

## Repository layout

```
app.py                Gradio application (decision model: INT8 student)
train.py              trains + evaluates the 4-model ladder, writes results.json
router/data.py        CLINC150 loading, intent → domain → team table
router/preprocess.py  regex normalization, model codes, edit distance, ELIZA rules
router/word2vec.py    skip-gram word2vec with negative sampling (PyTorch)
router/models.py      tf-idf+LR, word2vec+FFNN, BERT fine-tuning, distillation, quantization
router/evaluate.py    threshold tuning, metrics, latency benchmark
make_diagram.py       architecture diagram
make_slides.py        presentation generator
upload_artifacts.py   push weights to the Hugging Face Hub
```
