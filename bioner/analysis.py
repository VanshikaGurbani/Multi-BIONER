"""Error analysis from saved test predictions (``predictions_<corpus>.jsonl.gz``).

Every gold entity is put in exactly one bucket:

* ``correct``        - a predicted entity with the same type and boundaries
* ``type_error``     - same boundaries, different type
* ``boundary_error`` - overlaps a predicted entity of the same type but boundaries differ
* ``both_error``     - overlaps a predicted entity of another type with different boundaries
* ``missed``         - no predicted entity overlaps it

Predicted entities that overlap no gold entity are ``spurious``. Only well-formed predicted
chunks count as predicted entities (same rule as strict seqeval scoring).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from bioner.data import read_predictions
from bioner.labels import iobes_spans
from bioner.metrics import token_metrics

BUCKETS = ("correct", "type_error", "boundary_error", "both_error", "missed")


def _overlaps(a: tuple[str, int, int], b: tuple[str, int, int]) -> bool:
    return a[1] <= b[2] and b[1] <= a[2]


def classify_sentence(gold: Sequence[str], pred: Sequence[str]) -> dict:
    g_spans = iobes_spans(gold)
    p_spans = iobes_spans(pred)
    gold_buckets = []
    for g in g_spans:
        over = [p for p in p_spans if _overlaps(g, p)]
        if any(p == g for p in over):
            bucket = "correct"
        elif any(p[1:] == g[1:] for p in over):
            bucket = "type_error"
        elif any(p[0] == g[0] for p in over):
            bucket = "boundary_error"
        elif over:
            bucket = "both_error"
        else:
            bucket = "missed"
        gold_buckets.append((g, bucket, [p for p in over if p[1:] == g[1:] and p[0] != g[0]]))
    spurious = [p for p in p_spans if not any(_overlaps(p, g) for g in g_spans)]
    confusions = []
    for g in g_spans:
        for p in p_spans:
            if p[1:] == g[1:] and p[0] != g[0]:
                confusions.append((g[0], p[0]))
    return {"gold": gold_buckets, "spurious": spurious, "type_confusions": confusions}


def analyse(rows: Iterable[dict]) -> dict:
    """Aggregate buckets by gold type, spurious by predicted type, confusion pairs, span length."""
    by_type: dict[str, Counter] = defaultdict(Counter)
    spurious: Counter = Counter()
    confusions: Counter = Counter()
    by_len: dict[str, Counter] = defaultdict(Counter)
    gold_all, pred_all = [], []
    for row in rows:
        gold_all.append(row["gold"])
        pred_all.append(row["pred"])
        res = classify_sentence(row["gold"], row["pred"])
        for (etype, start, end), bucket, _ in res["gold"]:
            by_type[etype][bucket] += 1
            n = end - start + 1
            by_len["1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4+"][bucket] += 1
        for etype, _, _ in res["spurious"]:
            spurious[etype] += 1
        for g, p in res["type_confusions"]:
            confusions[(g, p)] += 1
    return {
        "by_type": {t: dict(c) for t, c in by_type.items()},
        "spurious": dict(spurious),
        "confusions": {f"{g} -> {p}": n for (g, p), n in confusions.most_common()},
        "by_length": {k: dict(c) for k, c in sorted(by_len.items())},
        "token": token_metrics(gold_all, pred_all),
    }


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}" if total else "-"


def render(corpus: str, result: dict, rare_below: int = 300) -> str:
    out = [f"### {corpus}", ""]
    rows = []
    for t, c in sorted(result["by_type"].items()):
        total = sum(c.get(b, 0) for b in BUCKETS)
        rows.append(
            [t, str(total)]
            + [_pct(c.get(b, 0), total) for b in BUCKETS]
            + [str(result["spurious"].get(t, 0))]
        )
    header = ["Gold type", "Entities", *(b + " %" for b in BUCKETS), "Spurious preds"]
    out += ["| " + " | ".join(header) + " |", "|:--" + "|:-:" * (len(header) - 1) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    if result["confusions"]:
        out += ["", "Type confusions with identical boundaries (gold -> predicted): "]
        out += [f"* {k}: {n}" for k, n in list(result["confusions"].items())[:5]]
    out += ["", "Recall by gold entity length (correct %):", ""]
    out += ["| Words | Entities | Correct % |", "|:--|:-:|:-:|"]
    for k, c in result["by_length"].items():
        total = sum(c.get(b, 0) for b in BUCKETS)
        out.append(f"| {k} | {total} | {_pct(c.get('correct', 0), total)} |")
    rare = [
        (tag, v)
        for tag, v in result["token"]["per_tag"].items()
        if tag.startswith("S-") and 0 < v["support"] < rare_below
    ]
    if rare:
        out += ["", f"Rare single-word tags (< {rare_below} test tokens), token-level:", ""]
        out += ["| Tag | Support | P | R | F1 |", "|:--|:-:|:-:|:-:|:-:|"]
        for tag, v in sorted(rare, key=lambda x: x[1]["support"]):
            out.append(
                f"| {tag} | {v['support']} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} |"
            )
    return "\n".join(out) + "\n"


def render_error_analysis(results_dir: str | Path, experiment: str) -> str:
    """Pool the saved predictions of all seeds of ``experiment`` per corpus."""
    root = Path(results_dir) / experiment
    corpora = sorted(
        {
            p.name[len("predictions_") : -len(".jsonl.gz")]
            for p in root.glob("seed*/predictions_*.jsonl.gz")
        }
    )
    if not corpora:
        return f"# Error analysis\n\nNo saved predictions found for `{experiment}` in `{root}`.\n"
    parts = [
        "# Error analysis",
        "",
        f"Experiment `{experiment}`; test-set predictions pooled over all seeds found "
        f"({len(list(root.glob('seed*')))} seed folders). Generated by `bioner error-analysis`.",
        "",
    ]
    for corpus in corpora:
        rows = []
        for path in sorted(root.glob(f"seed*/predictions_{corpus}.jsonl.gz")):
            rows.extend(read_predictions(path))
        parts.append(render(corpus, analyse(rows)))
    return "\n".join(parts)
