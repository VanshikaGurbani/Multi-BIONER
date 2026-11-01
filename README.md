# BioNER: biomedical named entity recognition, single-task vs multi-task

Fine-tuning BioBERT (and newer biomedical encoders) for entity recognition on five corpora,
evaluated with the standard **entity-level** protocol (seqeval, strict IOBES) over several seeds,
and a **shared-encoder multi-task** model compared honestly against per-corpus single-task models.

| Corpus | Entity types | Train / devel / test sentences |
|:--|:--|:--|
| BC2GM | gene / protein mentions | 12,574 / 2,519 / 5,038 |
| BC4CHEMD | chemicals | 30,682 / 30,639 / 26,364 |
| BC5CDR | chemicals, diseases | 4,560 / 4,581 / 4,797 |
| JNLPBA | protein, DNA, RNA, cell type, cell line | 16,328 / 1,713 / 3,800 (after re-joining cut sentences, see below) |
| NCBI-Disease | diseases | 5,424 / 923 / 940 |

## Results

Numbers in this section are written by `python -m bioner report` from result files that were actually
produced by training runs. Nothing is typed in by hand.

<!-- RESULTS:START -->
Test **entity-level** F1 (%), seqeval strict IOBES, mean ± std over seeds, epoch chosen on the devel split.

| Experiment | Seeds | BC2GM | BC4CHEMD | BC5CDR | JNLPBA | NCBI | Macro avg |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `stl_biobert` | 3 | 83.48 ± 0.14 | 90.25 ± 0.02 | 88.99 ± 0.13 | 75.38 ± 0.20 | 88.79 ± 0.07 | 85.38 ± 0.02 |
| `mtl_proportional` | 3 | 82.90 ± 0.18 | 89.96 ± 0.09 | 89.09 ± 0.20 | 75.48 ± 0.20 | 88.21 ± 0.40 | 85.13 ± 0.06 |
| `mtl_temp2` | 3 | 82.92 ± 0.24 | 89.86 ± 0.05 | 89.35 ± 0.26 | 75.27 ± 0.12 | 88.51 ± 0.22 | 85.18 ± 0.05 |

Details, precision/recall, paired differences and limitations: [docs/RESULTS.md](docs/RESULTS.md).
<!-- RESULTS:END -->

For precision/recall, paired single-task vs multi-task differences, per-entity-type scores and the
list of limitations see [docs/RESULTS.md](docs/RESULTS.md). Error analysis: [docs/ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md).
Model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

### What changed compared with the original thesis notebooks

The first version of this repository (notebooks in [notebooks/legacy](notebooks/legacy)) reported token-level
scores. A code audit ([docs/UPGRADE_PLAN.md](docs/UPGRADE_PLAN.md)) found problems that make those numbers
unsuitable for comparison, so this version re-does the experiments:

* The old "multi-task" notebook trained **five independent BioBERT copies** in a loop; no encoder was shared.
  The new multi-task model has one shared encoder and one tagging head per corpus.
* The old scores were sklearn **token-level** macro-F1 including the `O` class, measured on the **test** set at
  every epoch (and used to pick the "best" model). The new protocol selects the epoch on the **devel** split and
  scores the test split once, with entity-level precision / recall / F1.
* Long sentences were silently truncated at 128 word-pieces; now they are split into windows, so every word is scored.
* The old single-task and multi-task scores counted different things (first word-piece only vs all word-pieces).
* Three seeds per configuration, reported as mean ± std.

Old numbers, kept for history only (token-level macro-F1 incl. `O`, test-set selected, not comparable with entity F1):

| | BC2GM | BC4CHEMD | BC5CDR | JNLPBA | NCBI |
|:--|:-:|:-:|:-:|:-:|:-:|
| legacy single-task | 0.88 | 0.93 | 0.86 | 0.71 | 0.91 |
| legacy "multi-task" (5 separate models) | 0.86 | 0.93 | 0.84 | 0.71 | 0.90 |

## Quick start

```bash
pip install -r requirements.txt && pip install -e . --no-deps   # CPU is enough for tests
python -m bioner prepare-data                                     # ~16 MB, checksum verified
python -m pytest                                                  # fast, CPU only, no downloads
```

Training needs a GPU. Use [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) on Google Colab or Kaggle, or locally:

```bash
python -m bioner train --config configs/stl_biobert.yaml --seeds 13 42 2024
python -m bioner train --config configs/mtl_temp2.yaml
python -m bioner report          # rebuilds docs/RESULTS.md and the table above from results/
```

Experiments (one YAML each in [configs/](configs)):

| Config | What |
|:--|:--|
| `stl_biobert` | one BioBERT v1.1 per corpus (baseline) |
| `mtl_proportional` | shared encoder + head per corpus, corpus drawn in proportion to size |
| `mtl_temp2` | same, temperature sampling `p ∝ n^(1/2)` (counters large-corpus dominance) |
| `mtl_uniform` | same, uniform corpus sampling |
| `mtl_pcgrad` | one batch per corpus per step, conflicting encoder gradients projected (PCGrad) |
| `stl_pubmedbert`, `stl_biolinkbert` | newer biomedical encoders, same recipe as `stl_biobert` |

Optional few-shot Claude baseline (costs money; prints an estimate and stops unless `--yes`):

```bash
python -m bioner llm-baseline --model claude-haiku-5-5 --n 200     # dry run
```

## Repository layout

```
bioner/        data.py (corpus I/O, word-piece alignment) labels.py metrics.py config.py models.py mtl.py
               train.py report.py analysis.py baselines/llm.py cli.py
configs/       one YAML per experiment
tests/         CPU-only, offline, ~7 s (alignment, metrics, config, MTL, end-to-end smoke test on a tiny BERT)
notebooks/     colab_run.ipynb (GPU runner) and legacy/ (the original notebooks)
docs/          UPGRADE_PLAN.md  RESULTS.md (generated)  ERROR_ANALYSIS.md  MODEL_CARD.md
```

## Evaluation protocol

* Metric: seqeval entity-level P/R/F1, `mode="strict"`, IOBES. An entity counts only if type and both boundaries
  are exactly right and the predicted tag run is a well-formed chunk.
* Splits: `train` for fitting, `devel` for choosing the epoch, `test` evaluated once. Same splits for every model.
* Seeds: 13, 42, 2024 (mean ± sample std; n=3 is a rough spread, not a confidence interval).
* Word-piece alignment: each word is tokenized on its own, the first word-piece carries the label, the rest are ignored;
  sentences longer than the window are split on word boundaries (see `tests/test_alignment.py`).
* Same fixed recipe for every model (3 epochs, lr 5e-5, batch 16, 10% warm-up); no per-model tuning.

## Limitations

* JNLPBA as distributed has about 20% of its sentences cut in the middle of an entity. `bioner` re-joins adjacent
  sentences (tags unchanged), so JNLPBA scores are on a re-segmented test set and differ from papers that score the raw files.
* A few test sentences also occur in train (counts in [docs/RESULTS.md](docs/RESULTS.md)).
* Provenance of the exact IOBES files is not recorded in the original repository; they look like the Crichton et al. corpora
  as preprocessed for cross-type multi-task NER (Wang et al.).
* No cross-dataset generalisation test (train on one corpus, test on another) has been run.
* Words in a sentence that needs more than one window are tagged window by window, so an entity that straddles a window
  boundary can be mis-tagged (affects well under 1% of sentences).
* Results are for English abstracts only.
