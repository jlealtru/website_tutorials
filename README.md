# Tutorials from my personal site [jesusleal.io](https://jesusleal.io)

This repository contains the full versions of the tutorials published on [my personal website](https://jlealtru.github.io/). Topics: Deep Learning, NLP, and Graph Learning.

The notebooks were originally authored ~2020 against CUDA-only PyTorch. They have since been modernized: every library updated to a current pinned version, code made device-agnostic (CUDA → MPS → CPU), and the environment is now managed by [`uv`](https://docs.astral.sh/uv/).

## Setup

Prereqs: macOS, Linux, or Windows; [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

```bash
# 1. Create the venv and install pinned, hash-verified deps
uv sync --frozen

# 2. Install the spaCy English model (needed by the ETM notebooks)
uv run --with pip python -m spacy download en_core_web_sm

# 3. Launch JupyterLab
uv run jupyter lab
```

`uv sync --frozen` refuses any dependency whose hash doesn't match `uv.lock`, so installs are reproducible across machines.

The hardware choice happens at runtime via `notebooks/_utils.py::pick_device()`:

| Hardware | Result |
|---|---|
| NVIDIA GPU | `cuda` |
| Apple Silicon (M1 / M2 / …) | `mps` |
| Anything else | `cpu` |

On Apple Silicon, `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically so any op that lacks an MPS kernel falls back to CPU for that op only — the rest of training stays on the GPU.

## Where caches and outputs live

All artifacts produced or cached by the notebooks stay inside the repository — no surprise gigabytes under your home directory.

| What | Where |
|---|---|
| HuggingFace models, datasets, hub cache | `data/.hf_cache/` (set via `HF_HOME` in `notebooks/_utils.py`) |
| Trainer checkpoints, training logs | `results/` (TrainingArguments `output_dir='../results'`) |
| nbconvert re-executed copies | `results/_nbruns/` |
| wandb run dirs | `wandb/` (only if you opt in; default `report_to='none'`) |

All of those paths are gitignored. To purge everything: `rm -rf data/.hf_cache results/_nbruns results/checkpoint-* results/runs results/logs wandb`.

## Continuing a long-running training in a new session

Long fine-tunes (Longformer, BigBird, multi-label) take hours on M1. With `save_strategy='epoch'` (set in all transformer notebooks), each completed epoch writes `results/checkpoint-<step>/`. If the training is interrupted:

```bash
# Quick status check
find results -name "checkpoint-*" -type d        # what's been saved
ps -A | grep ipykernel | grep -v grep            # is a kernel still alive?

# Resume RoBERTa+IMDB from the latest checkpoint
uv run python scripts/resume_roberta_imdb.py
```

`scripts/resume_roberta_imdb.py` auto-finds the highest-numbered `checkpoint-N/` under `results/` and continues from there with the same M1-tuned `TrainingArguments` used in the notebook. Equivalent resume scripts can be added for the other transformer notebooks the same way (1 file each, ~120 lines).

## Data prerequisites

Three of the notebooks rely on datasets that are **not bundled with this repository** (size, licensing, etc.). Place each dataset under `data/<name>/` before running its notebook:

| Notebook(s) | Dataset | Where to get it |
|---|---|---|
| `processing_capital_bikeshare_data.ipynb`<br>`node2vec with capitol bikeshare data.ipynb` | Capital Bikeshare trips (yearly zip files) | <https://s3.amazonaws.com/capitalbikeshare-data/index.html> → drop zips into `data/capital_bikes/` |
| `Multi_label_classification_longformer_tutorial.ipynb`<br>`Multi_label_classification_roberta.ipynb` | Jigsaw Toxic Comment Classification (`train.csv`) | Kaggle competition: `jigsaw-toxic-comment-classification-challenge` → place under `data/jigsaw/` |
| `etm_preprocessed_data.ipynb`<br>`etm_spacy_pipeline.ipynb` | Pitchfork album reviews (`pitchfork.csv`) | The Kaggle Pitchfork reviews dataset → place under `data/pitchfork/` |

The IMDB-based notebooks (`RoBERTA with IMDB.ipynb`, `Longformer with IMDB.ipynb`, `BigBird text classification.ipynb`) auto-download IMDB through HuggingFace `datasets` — no manual setup needed.

## Streamlit app (`app.py`)

```bash
uv run streamlit run app.py
```

Requires a local [Ollama](https://ollama.com/) daemon with a Gemma-3 vision model pulled (out of scope for this repo).

## Layout

```
.
├── pyproject.toml          # exact-pinned deps
├── uv.lock                 # hash-verified resolved graph
├── .python-version         # 3.11
├── app.py                  # Streamlit + Ollama OCR demo
├── data/                   # external datasets land here (gitignored)
├── results/                # training outputs / nbconvert reruns (gitignored)
├── notebooks/
│   ├── _utils.py           # pick_device(), set_seed()
│   └── *.ipynb             # the tutorials
└── scripts/                # one-off modernization patchers (run once)
```

## What changed during modernization

- **Packaging**: introduced `pyproject.toml` + `uv.lock`; Python pinned to 3.11.
- **Device**: every notebook routes through `pick_device()` — CUDA → MPS → CPU.
- **Mixed precision**: `fp16=True` → `bf16=True` for MPS compatibility (CPU/CUDA ignore bf16 gracefully).
- **gensim 4.x**: `Word2Vec(size=...)` → `vector_size=...`; `wv.vocab` → `wv.key_to_index`; `wv.index2word` → `wv.index_to_key`.
- **pandas**: `display.max_colwidth=-1` → `None`.
- **HuggingFace Trainer**: `evaluation_strategy=` → `eval_strategy=`; `gradient_checkpointing=False` removed from `from_pretrained()` (use `model.gradient_checkpointing_disable()` instead); `cache_dir='/media/...'` Linux paths removed; `report_to='none'` added (wandb opt-in).
- **node2vec**: replaced unmaintained `stellargraph` with `pecanpy`, which has macOS arm64 wheels and a 1:1 mapping of biased-random-walk parameters.
- **spaCy**: `spacy.prefer_gpu()` wrapped in try/except so it no-ops on hardware without CUDA.
