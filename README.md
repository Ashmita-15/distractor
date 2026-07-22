# Misconception-Aware Retrieval for Distractor Generation

Pedagogically grounded retrieval for multiple-choice distractor generation on
the Eedi *Mining Misconceptions in Mathematics* dataset.

**Hypothesis:** a SentenceTransformer fine-tuned with misconception-aware
triplet supervision retrieves pedagogically more relevant examples (same
student misconception) than generic semantic retrieval.

## Project structure

```
distractor/
├── datasets/                  # Eedi CSVs (not in git — see Data below)
├── outputs/
│   ├── results/               # splits, metrics, retrieval indices
│   ├── embeddings/            # saved corpus/query embeddings
│   ├── triplets/              # training triplets (stage 03 output)
│   ├── models/                # fine-tuned checkpoints
│   └── figures/               # EDA + comparison figures
├── src/                       # library code (config, data, retrieval, eval, training)
├── scripts/
│   ├── 01_prepare_dataset.py  # raw CSVs -> leak-free QDP splits (+ --eda)
│   ├── 02_baseline_retrieval.py  # baseline encode/retrieve/evaluate
│   ├── 03_create_triplets.py  # build + save training triplets
│   ├── 04_train.py            # fine-tune from saved triplets (only)
│   ├── 05_evaluate.py         # evaluate any model; --compare vs baseline
│   ├── train_gpu.py           # minimal Colab/Kaggle training entry point
│   └── evaluate_model.py      # load model -> metrics.json
├── requirements.txt           # full local environment
├── requirements_gpu.txt       # GPU-training-only dependencies (Colab/Kaggle)
└── train_on_colab.ipynb       # Colab notebook: upload triplets, train, download model
```

## Pipeline

Each stage reads only files written by earlier stages, so stages run
independently and on different machines:

```bash
python scripts/01_prepare_dataset.py --eda   # ~1 min
python scripts/02_baseline_retrieval.py      # ~8 min (CPU encoding)
python scripts/03_create_triplets.py         # ~1 min
python scripts/04_train.py                   # slow on CPU -> use Colab (below)
python scripts/05_evaluate.py --compare      # ~8 min + comparison artifacts
```

### GPU training on Colab

CPU fine-tuning takes hours; on a T4 it takes minutes.

1. Run stages 01 and 03 locally.
2. Open `train_on_colab.ipynb` in Colab (GPU runtime), set `REPO_URL`, run all
   cells, upload `outputs/triplets/triplets_package.zip` when prompted.
3. Unzip the downloaded `finetuned_pedagogical.zip` into `outputs/models/`.
4. Run `python scripts/05_evaluate.py --compare` locally.

### Cloud paths

Data/output locations are overridable without code edits (useful on Kaggle,
where competition data mounts read-only):

```bash
DISTRACTOR_DATA_DIR=/kaggle/input/eedi-mining-misconceptions \
DISTRACTOR_OUTPUT_DIR=/kaggle/working/outputs \
python scripts/train_gpu.py
```

## Data

Place the Eedi competition CSVs in `datasets/`:
`train.csv`, `test.csv`, `misconception_mapping.csv`, `sample_submission.csv`.

## Evaluation protocol

- Retrieval unit: **question–distractor pair (QDP)**; each labeled distractor
  is one retrievable item.
- **Question-level** 80/10/10 splits (all QDPs of a question stay together —
  prevents sibling-distractor leakage).
- Corpus = train+val QDPs; queries = test QDPs; exact cosine retrieval.
- Relevance: retrieved QDP shares the query's `MisconceptionId`.
- Metrics: Hit Rate@K, Recall@K, MRR@K, nDCG@K, plus **attainability-aware**
  variants (a query whose misconception never occurs in the corpus is
  unanswerable by any retriever; ~20% of test queries — the attainable
  fraction is the Hit Rate ceiling).
- Significance: McNemar's exact test on paired per-query Hit@10.

## Reproducibility

- Seed 42 everywhere (`src/config.py`); splits, triplet sampling, and t-SNE
  are deterministic.
- All hyperparameters live in `src/config.py`.
- Every evaluation artifact (embeddings, indices, metrics) is persisted, and
  baseline/fine-tuned models are evaluated by the same code path
  (`src/retrieval_experiment.py`).

### Platform note (macOS)

On macOS, faiss-cpu and torch bundle conflicting OpenMP runtimes (segfault).
Retrieval therefore uses an exact NumPy inner-product search on macOS —
numerically identical to `faiss.IndexFlatIP` — and FAISS elsewhere.
