# Upgrade plan

Goal: turn the M.S. thesis notebooks into a reproducible, tested package whose every reported number is
measured with the field-standard protocol, so that the resume claims about it are defensible.

## 1. Audit of the original code (commit `7371a37`)

| # | Finding | Consequence |
|---|---------|-------------|
| 1 | `Multitask Model/MTL_Model.ipynb` builds a `ModuleDict` of **five separate** `AutoModelForTokenClassification` copies and trains each with its own `Trainer`, one after another. No encoder is shared. | The original "multi-task" model is five single-task models in a loop. A real shared-encoder model has never been run. |
| 2 | STL labels only the first word-piece of each word; MTL labels **every** word-piece (BC2GM test support 193,466 vs 142,875 for STL). | STL vs MTL token tables in the old README are not comparable. |
| 3 | The five STL notebooks are identical code; they train 5 epochs (README says 3). | README hyper-parameters were wrong. |
| 4 | Models are evaluated on the **test** set every epoch; MTL uses `load_best_model_at_end=True` on the test set. `devel.tsv` exists for every corpus but is unused. | Test-set model selection. |
| 5 | `padding="max_length", max_length=128, truncation=True` silently drops words past word-piece 128 (BC4CHEMD STL support 760,336 < 766,033 test tokens). | Truncated words vanish from the score instead of counting as misses. |
| 6 | Reported metric: sklearn token-level report over IOBES tags **including `O`**. seqeval entity F1 is computed in the Trainer log but never reported. | Numbers are not comparable with the literature. |
| 7 | One seed, no variance. | No error bars. |
| 8 | Train data are 16 MB of zips committed to the repo; logic lives in notebooks; NCBI zip path in the MTL notebook does not match the file name. | Not reproducible or testable. |
| 9 | README token counts and a "rule-based baseline" do not match the repo. | Fixed in the new README. |
| 10 | 53 / 190 / 16 / 269 / 1 test sentences (BC2GM / BC4CHEMD / BC5CDR / JNLPBA / NCBI) also occur verbatim in train. | Minor; documented, reported as a caveat. |

Old token-level numbers (macro-F1 0.71 to 0.93) are kept, labelled **token-level, legacy protocol, test-set evaluated**.

## 2. Data

Corpora: BC2GM, BC4CHEMD, BC5CDR, JNLPBA, NCBI-Disease in IOBES format, with `train/devel/test`
(they appear to be the Crichton et al. corpora as preprocessed by Wang et al., *Cross-type biomedical NER with deep
multi-task learning*, Bioinformatics 2019; provenance of the exact files is not recorded in the repo).
`bioner prepare-data` fetches the original zips from the **pinned git history** of this repo (commit `5958784`,
SHA-256 verified), extracts only `train/devel/test.tsv` to `data/<corpus>/` and validates the IOBES tags. `--zip-dir` accepts a local copy. `data/` is git-ignored.

Splits: train for fitting, **devel for model selection**, test touched once at the end. The `merge.tsv` file is not used.

## 3. Package layout

```
bioner/            data.py labels.py metrics.py config.py models.py mtl.py train.py evaluate.py report.py cli.py baselines/
configs/           stl_biobert.yaml mtl_*.yaml ... (one YAML per experiment)
tests/             CPU-only, no network, seconds
notebooks/legacy/  the 6 original notebooks, untouched
notebooks/colab_run.ipynb   thin wrapper: install, prepare-data, run the matrix, zip results
results/           raw per-run JSON (committed only when measured)
docs/              UPGRADE_PLAN.md RESULTS.md (generated) MODEL_CARD.md ERROR_ANALYSIS.md
```

Training loop is plain PyTorch (not `Trainer`): it needs custom multi-dataset batching, loss weights and gradient
surgery, and is not exposed to `Trainer` API churn.

## 4. Alignment (the bug-prone part)

* Each word is tokenized separately, so word to word-piece mapping is explicit. First word-piece carries the label,
  the rest get `-100`. Words are packed into windows of at most `max_len-2` word-pieces so **no word is dropped**; long sentences become several windows and predictions are stitched back per word.
* Decoding returns exactly one tag per original word. Tests cover: multi-piece words, `[CLS]/[SEP]`, over-length sentences, empty/UNK words, label round trip, IOBES span extraction, invalid tag sequences under strict seqeval.

## 5. Evaluation protocol

* seqeval, `mode="strict"`, `scheme=IOBES`: **entity-level** precision / recall / F1 (micro over entities) per corpus, plus per-type scores.
* Seeds 13, 42, 2024 minimum; reported as mean ± sample std (n=3, caveat stated).
* Model selection by devel entity-F1 at epoch end; test metrics computed once for the selected checkpoint.
* `docs/RESULTS.md` and the README table are generated from `results/**/metrics.json`; the generator refuses to emit a cell without a measured run.

## 6. Experiments

| ID | What | Purpose |
|----|------|---------|
| STL-BioBERT | per-corpus fine-tune of `dmis-lab/biobert-base-cased-v1.1` | strong baseline, replaces notebooks |
| MTL-uniform | **real** shared encoder + 5 heads, round-robin/proportional batches (the naive version of what the thesis intended) | shows whether the original idea works |
| MTL-temp | temperature sampling `p_i ∝ n_i^(1/T)`, T in {2, 5} | counters BC4CHEMD dominance |
| MTL-temp+PCGrad | optional | gradient conflict |
| STL-PubMedBERT / BioLinkBERT | newer encoder, same recipe | baseline |
| Claude few-shot | optional, N test sentences, cost and F1 reported | accuracy/cost tradeoff |

If MTL does not beat STL it is reported as such. Compute estimate on a T4 (fp16, max_len 128): one pass of all five
corpora x 3 epochs is roughly 20 to 30 minutes; the full matrix x 3 seeds is on the order of 5 to 8 GPU hours,
split into resumable per-run jobs.

## 7. GPU policy

The development machine has no GPU. Everything under `tests/` runs on CPU with a tiny random BERT and a toy
vocabulary (no downloads). All real training is run by the owner on Colab/Kaggle through `notebooks/colab_run.ipynb`;
result files come back as a zip and are the only source of numbers. No number is estimated or filled in.

## 8. Commit order (branch `upgrade/rigorous-bioner`, PR to `main`)

1. docs + packaging skeleton, legacy notebooks moved, data prepare script, pinned requirements, ruff
2. data/labels/alignment + tests, metrics + tests, config + tests, CI
3. training/eval CLI, results aggregation/report generator
4. real MTL (weighting, temperature sampling, PCGrad)
5. baselines (PubMedBERT/BioLinkBERT, Claude few-shot)
6. error analysis tooling, model card, Colab runner
7. fill in measured results when the owner returns them

## 9. Status (updated as work lands)

* Done and tested on CPU: package, data preparation, alignment, metrics, config, STL/MTL training loop (sampling,
  loss weights, PCGrad), report generator, error analysis, Claude baseline (mock-tested), CI.
* Found while building: JNLPBA sentences cut inside entities (about 20%); fixed by re-joining adjacent sentences.
* **Not done: all GPU training.** No model has been trained in this repository's new code on real data yet, so
  `docs/RESULTS.md` and the README table are intentionally empty until result files from `notebooks/colab_run.ipynb` are added.
