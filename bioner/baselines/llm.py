"""Few-shot Claude baseline for the five NER corpora.

The model reads ``BATCH`` tokenized test sentences per request and returns the entity mentions
as JSON; mentions are mapped back onto the original tokens and scored with exactly the same
strict entity-level metric as the fine-tuned models, on the same sentence subset.

Spending real money: this module never runs unless ``bioner llm-baseline`` is invoked with an
API key in the environment. ``estimate_cost`` gives a token-count based estimate first.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bioner.data import Corpus, Sentence, read_predictions
from bioner.labels import entity_types
from bioner.metrics import entity_metrics

# $ per 1M tokens (input, output), from Anthropic's published first-party price list as cached in
# the claude-api skill on 2026-10-06. Haiku 5.5 prices apply to prompts up to 100K tokens.
# Cache reads cost 0.1x and 5-minute cache writes 1.25x the input price.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

TYPE_HELP = {
    "GENE": "gene or gene-product mention (genes, proteins, RNAs named as genes)",
    "Chemical": "chemical compound or drug name",
    "Disease": "disease, disorder or clinical condition",
    "protein": "protein, protein family or protein complex",
    "DNA": "DNA molecule, gene region, domain or binding site",
    "RNA": "RNA molecule or transcript",
    "cell_type": "cell type (e.g. T lymphocytes)",
    "cell_line": "named cell line (e.g. HeLa)",
}

SYSTEM = """You are a biomedical named entity recognition system.
Task: find every mention of the entity types below in tokenized scientific text.

Entity types:
{types}

Rules:
- Return each mention exactly as it appears in the text (same characters, spaces between tokens as shown).
- Mark the full name of an entity, not a sub-part, and do not include generic words around it.
- Do not output overlapping mentions. Output a mention once per occurrence in reading order.
- If a sentence has no mentions, return an empty list for it.
- Answer with JSON only, no commentary, in this shape:
  {{"results": [{{"id": 1, "entities": [{{"text": "...", "type": "..."}}]}}]}}

Examples:
{examples}"""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0

    def add(self, u) -> None:
        self.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.requests += 1

    def cost_usd(self, model: str) -> float:
        pin, pout = PRICES[model]
        return (
            self.input_tokens * pin
            + self.cache_write_tokens * pin * 1.25
            + self.cache_read_tokens * pin * 0.10
            + self.output_tokens * pout
        ) / 1e6


def _show(words: Sequence[str]) -> str:
    return " ".join(words)


def spans_from_tags(sent: Sentence) -> list[dict]:
    from bioner.labels import iobes_spans

    return [{"text": _show(sent.words[s : e + 1]), "type": t} for t, s, e in iobes_spans(sent.tags)]


def pick_examples(corpus: Corpus, k: int, seed: int, max_words: int = 40) -> list[Sentence]:
    """k short training sentences, ~3/4 with entities, chosen with a fixed seed."""
    rng = random.Random(seed)
    pool = [s for s in corpus.splits["train"] if 5 <= len(s.words) <= max_words]
    with_ent = [s for s in pool if any(t != "O" for t in s.tags)]
    without = [s for s in pool if all(t == "O" for t in s.tags)]
    n_neg = max(1, k // 4) if without else 0
    return rng.sample(with_ent, k - n_neg) + rng.sample(without, n_neg)


def build_system_prompt(corpus: Corpus, examples: Sequence[Sentence]) -> str:
    types = entity_types(t for s in corpus.splits["train"][:2000] for t in s.tags)
    types_txt = "\n".join(f"- {t}: {TYPE_HELP.get(t, t)}" for t in types)
    ex = []
    for i, s in enumerate(examples, 1):
        ex.append(
            f"Text {i}: {_show(s.words)}\n"
            f"Answer: {json.dumps({'entities': spans_from_tags(s)}, ensure_ascii=False)}"
        )
    return SYSTEM.format(types=types_txt, examples="\n\n".join(ex))


def build_user_prompt(batch: Sequence[Sentence]) -> str:
    lines = [f"[{i}] {_show(s.words)}" for i, s in enumerate(batch, 1)]
    return "Extract the entities from each numbered text.\n\n" + "\n".join(lines)


def parse_response(text: str, n: int) -> list[list[dict]]:
    """Per-sentence mention lists; malformed or missing output yields empty lists, never raises."""
    out: list[list[dict]] = [[] for _ in range(n)]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return out
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return out
    for item in data.get("results", []) if isinstance(data, dict) else []:
        try:
            idx = int(item["id"]) - 1
            ents = item["entities"]
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < n and isinstance(ents, list):
            out[idx] = [e for e in ents if isinstance(e, dict) and "text" in e and "type" in e]
    return out


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def mentions_to_tags(words: Sequence[str], mentions: Sequence[dict], types: set[str]) -> list[str]:
    """IOBES tags from mention strings; unknown types, unmatched or overlapping mentions are dropped."""
    tags = ["O"] * len(words)
    cursor = 0
    for m in mentions:
        etype, target = m["type"], _squash(str(m["text"]))
        if etype not in types or not target:
            continue
        span = None
        for lo in (cursor, 0):  # prefer reading order, fall back to any free occurrence
            for i in range(lo, len(words)):
                acc = ""
                for j in range(i, min(len(words), i + 30)):
                    acc += _squash(words[j])
                    if acc == target:
                        if all(tags[k] == "O" for k in range(i, j + 1)):
                            span = (i, j)
                        break
                    if not target.startswith(acc):
                        break
                if span:
                    break
            if span:
                break
        if not span:
            continue
        i, j = span
        if i == j:
            tags[i] = f"S-{etype}"
        else:
            tags[i], tags[j] = f"B-{etype}", f"E-{etype}"
            for k in range(i + 1, j):
                tags[k] = f"I-{etype}"
        cursor = j + 1
    return tags


def sample_indices(n_sentences: int, n: int, seed: int) -> list[int]:
    return sorted(random.Random(seed).sample(range(n_sentences), min(n, n_sentences)))


@dataclass
class LLMRun:
    corpus: str
    model: str
    indices: list[int]
    pred_tags: list[list[str]]
    usage: Usage = field(default_factory=Usage)


def run_corpus(
    client,
    model: str,
    corpus: Corpus,
    n_sentences: int = 200,
    n_shots: int = 12,
    batch: int = 10,
    seed: int = 13,
    max_tokens: int = 4096,
) -> LLMRun:
    """Annotate a seeded random subset of the test split. ``client`` is an ``anthropic.Anthropic``."""
    test = corpus.splits["test"]
    idx = sample_indices(len(test), n_sentences, seed)
    system = build_system_prompt(corpus, pick_examples(corpus, n_shots, seed))
    types = set(entity_types(t for s in corpus.splits["train"][:2000] for t in s.tags))
    run = LLMRun(corpus.name, model, idx, [])
    for i in range(0, len(idx), batch):
        chunk = [test[j] for j in idx[i : i + batch]]
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            cache_control={"type": "ephemeral"},
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": build_user_prompt(chunk)}],
        )
        run.usage.add(resp.usage)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        for sent, ments in zip(chunk, parse_response(text, len(chunk)), strict=True):
            run.pred_tags.append(mentions_to_tags(sent.words, ments, types))
    return run


def estimate_cost(corpus: Corpus, model: str, n_sentences: int, n_shots: int, batch: int) -> dict:
    """Rough pre-flight estimate (4 characters ~ 1 token); the real run reports measured usage."""
    system = build_system_prompt(corpus, pick_examples(corpus, n_shots, 13))
    sample = corpus.splits["test"][:batch]
    in_chars = len(system) + len(build_user_prompt(sample))
    n_req = -(-n_sentences // batch)
    in_tok = in_chars / 4 * n_req
    out_tok = n_sentences * 40
    pin, pout = PRICES[model]
    return {
        "requests": n_req,
        "approx_input_tokens": int(in_tok),
        "approx_output_tokens": int(out_tok),
        "approx_usd_without_caching": round((in_tok * pin + out_tok * pout) / 1e6, 4),
    }


def score_run(run: LLMRun, corpus: Corpus, finetuned_predictions: str | Path | None = None) -> dict:
    test = corpus.splits["test"]
    gold = [list(test[i].tags) for i in run.indices]
    result = {
        "corpus": run.corpus,
        "model": run.model,
        "n_sentences": len(run.indices),
        "sentence_indices": run.indices,
        "llm_entity": entity_metrics(gold, run.pred_tags),
        "usage": run.usage.__dict__,
        "cost_usd": round(run.usage.cost_usd(run.model), 6),
    }
    if finetuned_predictions and Path(finetuned_predictions).exists():
        rows = read_predictions(finetuned_predictions)
        result["finetuned_entity_same_subset"] = entity_metrics(
            gold, [rows[i]["pred"] for i in run.indices]
        )
    return result
