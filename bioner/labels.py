"""IOBES tag utilities: label spaces, validation and span extraction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

OUTSIDE = "O"
IGNORE_INDEX = -100  # label id given to positions that must not contribute to loss/metrics


def split_tag(tag: str) -> tuple[str, str]:
    """Split ``B-GENE`` into ``("B", "GENE")``; ``O`` into ``("O", "")``."""
    if tag == OUTSIDE:
        return OUTSIDE, ""
    prefix, sep, etype = tag.partition("-")
    if not sep or prefix not in {"B", "I", "E", "S"} or not etype:
        raise ValueError(f"not an IOBES tag: {tag!r}")
    return prefix, etype


def entity_types(tags: Iterable[str]) -> list[str]:
    return sorted({split_tag(t)[1] for t in tags if t != OUTSIDE})


def make_label_maps(tags: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Deterministic label space: ``O`` is id 0, the rest sorted alphabetically."""
    unique = sorted({t for t in tags if t != OUTSIDE})
    for t in unique:
        split_tag(t)  # validates
    label_to_id = {OUTSIDE: 0, **{t: i + 1 for i, t in enumerate(unique)}}
    return label_to_id, {i: t for t, i in label_to_id.items()}


def iobes_spans(tags: Sequence[str]) -> list[tuple[str, int, int]]:
    """Well-formed entity spans ``(type, start, end_inclusive)`` of an IOBES sequence.

    Mirrors seqeval's strict IOBES behaviour: a malformed run (e.g. ``B-X`` followed by
    ``O`` or by ``I-Y``) does not produce a span.
    """
    spans: list[tuple[str, int, int]] = []
    start: int | None = None
    etype = ""
    for i, tag in enumerate(tags):
        prefix, t = split_tag(tag)
        if prefix == "S":
            spans.append((t, i, i))
            start = None
        elif prefix == "B":
            start, etype = i, t
        elif prefix == "I":
            if start is None or t != etype:
                start = None
        elif prefix == "E":
            if start is not None and t == etype:
                spans.append((t, start, i))
            start = None
        else:  # O
            start = None
    return spans


def validate_iobes(tags: Sequence[str]) -> list[str]:
    """Return human-readable problems in a gold tag sequence (empty list = valid)."""
    problems: list[str] = []
    open_type: str | None = None
    for i, tag in enumerate(tags):
        prefix, t = split_tag(tag)
        if prefix in {"I", "E"}:
            if open_type != t:
                problems.append(f"position {i}: {tag} without matching B-/I-{t}")
            if prefix == "E":
                open_type = None
        else:
            if open_type is not None:
                problems.append(f"position {i}: entity {open_type} left open before {tag}")
            open_type = t if prefix == "B" else None
    if open_type is not None:
        problems.append(f"sequence ends inside entity {open_type}")
    return problems
