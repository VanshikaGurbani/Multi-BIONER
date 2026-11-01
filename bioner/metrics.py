"""Entity-level (seqeval, strict IOBES) and token-level metrics, and seed aggregation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from seqeval.metrics import classification_report
from seqeval.scheme import IOBES

from bioner.labels import OUTSIDE


def entity_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> dict:
    """Entity-level precision/recall/F1 (micro over entities) with a per-type breakdown.

    An entity counts as correct only if type and both boundaries match exactly, and only if the
    predicted tag run is a well-formed IOBES chunk (seqeval ``mode="strict"``). This is the
    CoNLL / BioCreative style score that the literature reports.
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in number of sentences")
    for g, p in zip(gold, pred, strict=True):
        if len(g) != len(p):
            raise ValueError("gold and pred sentence lengths differ")
    rep = classification_report(
        list(map(list, gold)),
        list(map(list, pred)),
        mode="strict",
        scheme=IOBES,
        output_dict=True,
        zero_division=0,
    )
    per_type = {
        k: _clean(v) for k, v in rep.items() if k not in {"micro avg", "macro avg", "weighted avg"}
    }
    return {
        "level": "entity",
        "overall": _clean(rep["micro avg"]),
        "macro": _clean(rep["macro avg"]),
        "per_type": per_type,
    }


def _clean(d: dict) -> dict:
    return {
        "precision": float(d["precision"]),
        "recall": float(d["recall"]),
        "f1": float(d["f1-score"]),
        "support": int(d["support"]),
    }


def token_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> dict:
    """Per-tag and macro precision/recall/F1 over words, including ``O`` (the legacy protocol).

    Equivalent to sklearn's ``classification_report`` macro average over IOBES tags. Reported
    only so that the old numbers can be compared like-for-like; it is *not* the headline metric.
    """
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    correct = total = 0
    for g_seq, p_seq in zip(gold, pred, strict=True):
        for g, p in zip(g_seq, p_seq, strict=True):
            total += 1
            if g == p:
                tp[g] += 1
                correct += 1
            else:
                fp[p] += 1
                fn[g] += 1
    per_tag = {}
    for tag in sorted(set(tp) | set(fp) | set(fn)):
        pr = tp[tag] / (tp[tag] + fp[tag]) if tp[tag] + fp[tag] else 0.0
        rc = tp[tag] / (tp[tag] + fn[tag]) if tp[tag] + fn[tag] else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per_tag[tag] = {
            "precision": pr,
            "recall": rc,
            "f1": f1,
            "support": tp[tag] + fn[tag],
        }
    gold_tags = [t for t, v in per_tag.items() if v["support"] > 0]
    macro_f1 = _mean([per_tag[t]["f1"] for t in gold_tags])
    return {
        "level": "token",
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "non_o_macro_f1": _mean([per_tag[t]["f1"] for t in gold_tags if t != OUTSIDE]),
        "per_tag": per_tag,
    }


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and *sample* standard deviation (ddof=1); std is 0.0 for a single value."""
    if not values:
        raise ValueError("no values")
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def format_mean_std(values: Sequence[float], scale: float = 100.0, digits: int = 2) -> str:
    m, s = mean_std(values)
    if len(values) == 1:
        return f"{m * scale:.{digits}f} (n=1)"
    return f"{m * scale:.{digits}f} ± {s * scale:.{digits}f}"
