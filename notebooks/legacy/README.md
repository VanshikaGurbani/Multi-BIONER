# Legacy notebooks

The original thesis notebooks, unchanged (moved from `Single Task Model/`, `Multitask Model/` and `nlp outputs/`).
They are kept for provenance only. Known problems (see `docs/UPGRADE_PLAN.md`):

* `multitask/MTL_Model.ipynb` trains five independent BioBERT copies in a loop; no encoder is shared.
* Evaluation and "best model" selection use the test set; no dev set is used.
* Scores are token-level sklearn reports including the `O` class, not entity-level F1.

Use the `bioner` package for anything that is reported.
They expect the zips that used to live in the repo root; run `bioner prepare-data --keep-zips` to recreate them.
