import json
from types import SimpleNamespace

import pytest

from bioner.baselines import llm
from bioner.data import Corpus, Sentence


def S(pairs):
    return Sentence(tuple(w for w, _ in pairs), tuple(t for _, t in pairs))


def test_parse_response_tolerates_noise_and_garbage():
    good = (
        'sure!\n```json\n{"results": [{"id": 2, "entities": '
        '[{"text": "IL-2", "type": "protein"}]}]}\n```'
    )
    assert llm.parse_response(good, 2) == [[], [{"text": "IL-2", "type": "protein"}]]
    assert llm.parse_response("no json", 2) == [[], []]
    assert llm.parse_response('{"results": [{"id": 9, "entities": []}]}', 1) == [[]]
    assert llm.parse_response('{"results": "x"}', 1) == [[]]


def test_mentions_to_tags_multiword_single_and_whitespace_variants():
    words = ["The", "IL", "-", "2", "gene", "and", "p53", "."]
    ments = [{"text": "IL - 2", "type": "protein"}, {"text": "p53", "type": "protein"}]
    tags = llm.mentions_to_tags(words, ments, {"protein"})
    assert tags == ["O", "B-protein", "I-protein", "E-protein", "O", "O", "S-protein", "O"]
    one = llm.mentions_to_tags(words, [{"text": "IL-2", "type": "protein"}], {"protein"})
    assert one[1] == "B-protein"  # written without spaces still matches


def test_mentions_to_tags_ignores_unknown_type_unmatched_and_overlap():
    words = ["a", "b", "c"]
    ments = [
        {"text": "a b", "type": "X"},
        {"text": "b c", "type": "X"},  # overlaps the first
        {"text": "zzz", "type": "X"},  # not present
        {"text": "c", "type": "Nope"},  # unknown type
    ]
    assert llm.mentions_to_tags(words, ments, {"X"}) == ["B-X", "E-X", "O"]


def test_repeated_mention_tags_each_occurrence_in_order():
    words = ["p53", "binds", "p53"]
    ments = [{"text": "p53", "type": "G"}, {"text": "p53", "type": "G"}]
    assert llm.mentions_to_tags(words, ments, {"G"}) == ["S-G", "O", "S-G"]


def test_sample_indices_deterministic_and_bounded():
    assert llm.sample_indices(100, 10, 1) == llm.sample_indices(100, 10, 1)
    assert len(llm.sample_indices(5, 10, 1)) == 5


def test_cost_uses_cache_multipliers():
    u = llm.Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    pin, pout = llm.PRICES["claude-haiku-5-5"]
    assert u.cost_usd("claude-haiku-5-5") == pytest.approx(pin + pin * 1.25 + pin * 0.1 + pout)


class FakeClient:
    def __init__(self, answer):
        self.answer, self.calls = answer, []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kw):
        self.calls.append(kw)
        n = kw["messages"][0]["content"].count("\n[")
        results = [{"id": i + 1, "entities": self.answer} for i in range(n)]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps({"results": results}))],
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=10,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )


def corpus():
    ex = S([("IL", "B-protein"), ("2", "E-protein"), ("acts", "O"), ("here", "O"), ("now", "O")])
    test = [S([("IL", "B-protein"), ("2", "E-protein"), ("x", "O")]) for _ in range(12)]
    return Corpus("JNLPBA", {"train": [ex] * 20, "devel": test[:2], "test": test})


def test_run_and_score_with_fake_client():
    c = corpus()
    fake = FakeClient([{"text": "IL 2", "type": "protein"}])
    run = llm.run_corpus(fake, "claude-haiku-5-5", c, n_sentences=7, n_shots=4, batch=3, seed=1)
    assert len(run.pred_tags) == 7 and run.usage.requests == 3
    assert fake.calls[0]["model"] == "claude-haiku-5-5"
    res = llm.score_run(run, c)
    assert res["llm_entity"]["overall"]["f1"] == 1.0
    assert res["cost_usd"] > 0 and res["n_sentences"] == 7


def test_estimate_cost_is_positive():
    est = llm.estimate_cost(corpus(), "claude-haiku-5-5", 100, 4, 10)
    assert est["requests"] == 10 and est["approx_usd_without_caching"] > 0
