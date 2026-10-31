"""Aggregate ``results/**/seed*/*.json`` into docs/RESULTS.md and the README results table.

Everything printed comes from a measured result file. A missing (experiment, corpus) cell is
printed as ``not run``; nothing is estimated, copied from papers or filled in.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from bioner.data import CORPORA
from bioner.metrics import format_mean_std, mean_std

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"
NOT_RUN = "not run"

# experiment -> seed -> corpus -> result dict (the per-corpus block of a result file)
Runs = dict[str, dict[int, dict[str, dict]]]


def load_runs(results_dir: str | Path) -> tuple[Runs, dict[str, dict]]:
    """Read every result file. Returns the runs and one metadata dict per experiment."""
    runs: Runs = defaultdict(lambda: defaultdict(dict))
    meta: dict[str, dict] = {}
    for path in sorted(Path(results_dir).glob("*/seed*/*.json")):
        res = json.loads(path.read_text(encoding="utf-8"))
        if "corpora" not in res:
            continue
        exp, seed = res["experiment"], int(res["seed"])
        for corpus, block in res["corpora"].items():
            runs[exp][seed][corpus] = block
        m = meta.setdefault(
            exp,
            {"mode": res["mode"], "model_name": res["model_name"], "config": res["config"]},
        )
        m.setdefault("envs", []).append(res.get("env", {}))
    return {e: dict(s) for e, s in runs.items()}, meta


def _order(runs: Runs, meta: dict[str, dict]) -> list[str]:
    return sorted(runs, key=lambda e: (meta[e]["mode"] != "stl", meta[e]["model_name"], e))


def _values(runs: Runs, exp: str, corpus: str, getter) -> list[float]:
    return [
        getter(by_corpus[corpus])
        for _, by_corpus in sorted(runs[exp].items())
        if corpus in by_corpus
    ]


def _f1(block: dict) -> float:
    return block["test_entity"]["overall"]["f1"]


def _p(block: dict) -> float:
    return block["test_entity"]["overall"]["precision"]


def _r(block: dict) -> float:
    return block["test_entity"]["overall"]["recall"]


def _cell(runs: Runs, exp: str, corpus: str, getter=_f1) -> str:
    vals = _values(runs, exp, corpus, getter)
    return format_mean_std(vals) if vals else NOT_RUN


def _macro_over_corpora(runs: Runs, exp: str, corpora: list[str]) -> str:
    """Mean over corpora per seed, then mean ± std over seeds; needs every corpus in every seed."""
    per_seed = []
    for _, by_corpus in sorted(runs[exp].items()):
        if not all(c in by_corpus for c in corpora):
            continue
        per_seed.append(sum(_f1(by_corpus[c]) for c in corpora) / len(corpora))
    return format_mean_std(per_seed) if per_seed else NOT_RUN


def corpora_present(runs: Runs) -> list[str]:
    seen = {c for by_seed in runs.values() for by_corpus in by_seed.values() for c in by_corpus}
    return [c for c in CORPORA if c in seen]


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|:--" + "|:-:" * (len(header) - 1) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def f1_table(runs: Runs, meta: dict[str, dict]) -> str:
    corpora = corpora_present(runs)
    with_macro = len(corpora) > 1
    rows = []
    for exp in _order(runs, meta):
        row = [f"`{exp}`", str(len(runs[exp]))] + [_cell(runs, exp, c) for c in corpora]
        if with_macro:
            row.append(_macro_over_corpora(runs, exp, corpora))
        rows.append(row)
    header = ["Experiment", "Seeds"] + corpora + (["Macro avg"] if with_macro else [])
    return _md_table(header, rows)


def prf_table(runs: Runs, meta: dict[str, dict]) -> str:
    corpora = corpora_present(runs)
    rows = []
    for exp in _order(runs, meta):
        for label, getter in (("P", _p), ("R", _r)):
            rows.append([f"`{exp}`", label] + [_cell(runs, exp, c, getter) for c in corpora])
    return _md_table(["Experiment", ""] + corpora, rows)


def delta_table(runs: Runs, meta: dict[str, dict], baseline: str | None = None) -> str:
    """Paired (same seed) entity-F1 difference of each other experiment vs the STL baseline."""
    exps = _order(runs, meta)
    if baseline is None:
        baseline = next((e for e in exps if meta[e]["mode"] == "stl"), None)
    if baseline is None or len(exps) < 2:
        return ""
    corpora = corpora_present(runs)
    rows = []
    for exp in exps:
        if exp == baseline or meta[exp]["model_name"] != meta[baseline]["model_name"]:
            continue
        row = [f"`{exp}`"]
        for c in corpora:
            diffs = [
                (
                    runs[exp][s][c]["test_entity"]["overall"]["f1"]
                    - runs[baseline][s][c]["test_entity"]["overall"]["f1"]
                )
                for s in sorted(set(runs[exp]) & set(runs[baseline]))
                if c in runs[exp][s] and c in runs[baseline][s]
            ]
            row.append(_delta_cell(diffs))
        rows.append(row)
    if not rows:
        return ""
    return f"Baseline: `{baseline}`.\n\n" + _md_table(["Experiment"] + corpora, rows)


def _delta_cell(diffs: list[float]) -> str:
    if not diffs:
        return NOT_RUN
    m, s = mean_std(diffs)
    text = f"{m * 100:+.2f}" + (f" ± {s * 100:.2f}" if len(diffs) > 1 else " (n=1)")
    if len(diffs) > 1:
        text += " ▲" if m > s else " ▼" if m < -s else " ≈"
    return text


def token_table(runs: Runs, meta: dict[str, dict]) -> str:
    corpora = corpora_present(runs)
    rows = []
    for exp in _order(runs, meta):
        vals = {c: _values(runs, exp, c, lambda b: b["test_token"]["macro_f1"]) for c in corpora}
        rows.append(
            [f"`{exp}`"]
            + [format_mean_std(vals[c], digits=2) if vals[c] else NOT_RUN for c in corpora]
        )
    return _md_table(["Experiment"] + corpora, rows)


def per_type_table(runs: Runs, exp: str, corpora: list[str]) -> str:
    rows = []
    for c in corpora:
        types: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for _, by_corpus in sorted(runs[exp].items()):
            if c in by_corpus:
                for t, v in by_corpus[c]["test_entity"]["per_type"].items():
                    types[t].append((v["f1"], v["support"]))
        for t in sorted(types):
            f1s = [x[0] for x in types[t]]
            rows.append([c, t, str(types[t][0][1]), format_mean_std(f1s)])
    return _md_table(["Corpus", "Entity type", "Test entities", "F1"], rows) if rows else ""


def legacy_table(legacy_path: str | Path) -> str:
    path = Path(legacy_path)
    if not path.exists():
        return ""
    leg = json.loads(path.read_text(encoding="utf-8"))
    corpora = list(leg["stl_macro_f1"])
    rows = [
        ["legacy STL (token macro-F1)"] + [f"{leg['stl_macro_f1'][c]:.2f}" for c in corpora],
        ['legacy "MTL" (token macro-F1)'] + [f"{leg['mtl_macro_f1'][c]:.2f}" for c in corpora],
    ]
    return _md_table(["Legacy notebooks, token level"] + corpora, rows)


LIMITATIONS = """\
## Limitations

* **Seeds.** Three seeds give a rough spread, not a confidence interval. Differences smaller than
  about one standard deviation (marked `≈`) should be read as ties.
* **Entity-level F1 is strict.** Exact boundaries and type must match; predicted chunks that are not
  well-formed IOBES do not count. This is the usual convention but is harsher than lenient matching.
* **Splits.** Train / devel / test come from the distributed IOBES files; their exact provenance is
  not recorded in the original repository. A few test sentences also occur verbatim in train
  (BC2GM 53, BC4CHEMD 190, BC5CDR 16, JNLPBA 233, NCBI-Disease 1), which slightly favours all models equally.
* **JNLPBA sentence cuts.** In the distributed JNLPBA files about 20% of sentences were cut in the middle
  of an entity, which makes those gold entities unscorable under strict matching. `bioner` re-joins the
  adjacent sentences (tags are untouched; train 2206, devel 219, test 443 joins). JNLPBA numbers are
  therefore on a re-segmented test set and are not directly comparable with papers that score the raw split.
* **One recipe.** All models use the same fixed hyper-parameters (no per-model tuning), so a model that
  needs a different learning rate may be under-represented.
* **Scope.** Results are for these five English corpora and say nothing about clinical text or other
  entity types. No cross-dataset (train on one corpus, test on another) evaluation was run.
* **Legacy numbers** are token-level, include the `O` class, were measured on the test set during
  training, and the legacy "multi-task" model did not share an encoder. They are shown only for history
  and cannot be compared with entity-level F1.
"""


def render_results_md(results_dir: str | Path, legacy_path: str | Path | None = None) -> str:
    runs, meta = load_runs(results_dir)
    out = ["# Results", "", "_Generated by `bioner report`; do not edit by hand._", ""]
    if not runs:
        out += [
            "No measured results are committed yet. Run the experiments with "
            "`notebooks/colab_run.ipynb` and then `bioner report`.",
            "",
            LIMITATIONS,
        ]
        return "\n".join(out)
    n_files = sum(len(by_corpus) for by_seed in runs.values() for by_corpus in by_seed.values())
    out += [
        "## Protocol",
        "",
        "* Metric: seqeval **entity-level** precision / recall / F1, strict IOBES matching, micro-averaged "
        "over entities, on the held-out **test** split.",
        "* Model selection: epoch with the best **devel** F1; the test set is evaluated once.",
        "* Values are mean ± sample std over seeds (n shown in the `Seeds` column), in percent.",
        f"* {len(runs)} experiment(s), {n_files} (seed, corpus) results.",
        "",
        "## Test entity-level F1 (%)",
        "",
        f1_table(runs, meta),
        "",
        "## Test entity-level precision and recall (%)",
        "",
        prf_table(runs, meta),
        "",
    ]
    delta = delta_table(runs, meta)
    if delta:
        out += [
            "## Paired difference to the single-task baseline (F1 points, same seeds)",
            "",
            "`▲` mean difference above one std, `▼` below minus one std, `≈` within noise.",
            "",
            delta,
            "",
        ]
    out += [
        "## Test token-level macro-F1 including `O` (legacy metric, for continuity only)",
        "",
        token_table(runs, meta),
        "",
    ]
    first = _order(runs, meta)[0]
    pt = per_type_table(runs, first, corpora_present(runs))
    if pt:
        out += [f"## Per entity type, `{first}` (F1 %)", "", pt, ""]
    leg = legacy_table(legacy_path) if legacy_path else ""
    if leg:
        out += [
            "## Original thesis notebooks (token level, test-set evaluated, not comparable)",
            "",
            leg,
            "",
        ]
    out += [LIMITATIONS]
    return "\n".join(out)


def render_readme_table(results_dir: str | Path) -> str:
    runs, meta = load_runs(results_dir)
    if not runs:
        return (
            "_No measured results are committed yet. They are produced by `notebooks/colab_run.ipynb` "
            "and written here by `bioner report`._"
        )
    return (
        "Test **entity-level** F1 (%), seqeval strict IOBES, mean ± std over seeds, "
        "epoch chosen on the devel split.\n\n"
        + f1_table(runs, meta)
        + "\n\nDetails, precision/recall, "
        "paired differences and limitations: [docs/RESULTS.md](docs/RESULTS.md)."
    )


def update_readme(readme_path: str | Path, table_md: str) -> bool:
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise ValueError(f"{path} lacks {README_START} / {README_END} markers")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    new = f"{head}{README_START}\n{table_md}\n{README_END}{tail}"
    changed = new != text
    if changed:
        path.write_text(new, encoding="utf-8")
    return changed
