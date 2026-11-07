# Automated Student Essay Scoring System

Predict a holistic Domain 1 score from a student essay and its ASAP essay set. The starter trains one model per prompt and compares a Hierarchical Attention Network (HAN) with a TF-IDF + Ridge baseline on the same essays.

HAN builds sentence vectors with word-level attention and an essay vector with sentence-level attention. Both levels use bidirectional GRUs. A regression head predicts a score normalized to the prompt's rubric range. [Hierarchical Attention Networks for Document Classification](https://aclanthology.org/N16-1174/)

## Project layout

```text
data/README.md                      ASAP-AES download and input format
src/essay_scoring/data.py           TSV loader, rubric ranges, and split
src/essay_scoring/metrics.py        QWK, MSE, and score conversion
src/essay_scoring/models/han.py     Hierarchical encoder and training loop
src/essay_scoring/models/ridge.py   TF-IDF + Ridge baseline
src/essay_scoring/__main__.py       Train, evaluate, and predict commands
tests/test_metrics.py              Scoring examples and edge cases
outputs/                           Local run records and checkpoints
```

## Setup

Use Python 3.12 for the setup below. From the repository root, create and activate a virtual environment. On Windows PowerShell, run the following commands for a CPU setup. [PyTorch installation options](https://pytorch.org/get-started/locally/)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

Download `training_set_rel3.tsv` and put it in `data/raw/`. The commands also accept `--data` for a different path. [Dataset instructions](data/README.md)

## Train and compare

The example uses essay set 1. Pass `--essay-set` with any set from 1 through 8 to train for that prompt.

```powershell
python -m essay_scoring train --essay-set 1 --model ridge --run-dir outputs/set-1/ridge
python -m essay_scoring train --essay-set 1 --model han --epochs 5 --run-dir outputs/set-1/han
```

The loader drops blank essays and keeps one copy of identical text after case and whitespace normalization. Each prompt is split approximately 80/10/10 into training, validation, and test sets. Match `--essay-set`, `--seed` (default 42), and the input TSV to compare the models. Both models build their vocabularies from training essays. Each run saves the dataset checksum, split version, essay IDs, and validation metrics in `run.json`. HAN saves the checkpoint with the lowest validation MSE.

Use a fresh `--run-dir` for each training run. Evaluate saved runs after selecting settings on validation. Load checkpoints from runs you trust, since Ridge uses Python pickle.

```powershell
python -m essay_scoring evaluate --run-dir outputs/set-1/ridge
python -m essay_scoring evaluate --run-dir outputs/set-1/han
```

- **Quadratic weighted kappa (QWK)** uses predictions clipped to the rubric bounds and rounded to the nearest integer, with halves rounded up. It reports `null` when both rating series are constant at the same score.
- **Mean squared error (MSE)** uses the continuous clipped predictions in the original rubric units.

The score bounds follow the published ASAP prompt ranges. [ASAP score ranges, Table 1](https://aclanthology.org/2022.coling-1.240/)

## Score one essay

Save an essay for the trained prompt as UTF-8 text. The command returns the predicted score, the rounded rubric score, and the prompt's score range. HAN splits long sentences into blocks of at most 64 words and keeps the first 48 sentence blocks.

```powershell
python -m essay_scoring predict --run-dir outputs/set-1/han --essay-file path/to/essay.txt
```

Run the metric checks with `python -m unittest discover -s tests`.
