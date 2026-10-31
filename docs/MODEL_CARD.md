# Model card: BioNER taggers

Status: **no model weights are published.** This card describes the models the code trains; metrics are not
repeated here so that they cannot drift from the measured results in [RESULTS.md](RESULTS.md).
Publishing weights to the Hugging Face Hub is optional and will only be done on the owner's explicit say-so.

## Model description

* Encoder: `dmis-lab/biobert-base-cased-v1.1` (also `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
  and `michiyasunaga/BioLinkBERT-base` as baselines), 12 layers, 768 hidden units, about 110M parameters.
* Head: dropout 0.1 and a linear layer over IOBES tags, one head per corpus.
* Single-task: one encoder + head per corpus. Multi-task: one shared encoder, one head per corpus, training steps drawn
  from the five corpora by proportional / temperature / uniform sampling (optionally PCGrad).
* Input: one sentence of pre-tokenized words (the corpora's own tokenization); output: one IOBES tag per word.

## Intended use

Research and teaching on biomedical entity recognition in English PubMed abstracts: gene/protein mentions (BC2GM, JNLPBA),
chemicals (BC4CHEMD, BC5CDR), diseases (BC5CDR, NCBI-Disease), cell types/lines and DNA/RNA (JNLPBA).

## Not intended for

Clinical decision making, patient-record text, other languages, or entity types outside the five corpora. The models
were not evaluated on clinical notes, full text, or social media.

## Training data

Train splits of BC2GM, BC4CHEMD, BC5CDR, JNLPBA and NCBI-Disease in IOBES format (see README for counts and caveats,
including the re-joined JNLPBA sentences). Licences of the underlying corpora are those of their original distributors
(not recorded in the original repository); check them before redistributing derived models.

## Evaluation

Strict entity-level precision/recall/F1 (seqeval) on each corpus's test split, mean ± std over seeds, epoch chosen on
the devel split. See [RESULTS.md](RESULTS.md). Error breakdown: [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md).

## Known weaknesses

* Rare entity types (for example JNLPBA RNA and single-word DNA/cell-line mentions) are expected to score much lower than
  frequent ones; the error analysis reports this per type.
* Boundary errors on long, nested or abbreviated names; type confusions between protein / DNA / RNA in JNLPBA.
* A sentence longer than 126 word-pieces is split into windows; an entity straddling a split can be missed.
* The fixed training recipe was not tuned per model, so the encoder comparison is for one recipe only.

## Ethical considerations

Trained only on public scientific abstracts; contains no personal data by design. Errors in automated entity extraction
can propagate into downstream curation; outputs should be reviewed by a domain expert before use in any decision.
