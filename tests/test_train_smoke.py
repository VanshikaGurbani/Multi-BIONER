"""End-to-end smoke tests on CPU with a 1-layer random BERT and a toy vocabulary (no downloads)."""

import json
import random

import pytest
import torch
from transformers import BertConfig, BertModel

from bioner.config import config_from_dict
from bioner.data import read_predictions
from bioner.train import run_experiment
from tests.conftest import VOCAB

WORDS = ["the", "gene", "IL", "p", "kappa", "a", "b", "c"]


def tiny_encoder():
    cfg = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    return BertModel(cfg, add_pooling_layer=False)


def write_corpus(root, name, etype, seed):
    rng = random.Random(seed)
    d = root / name
    d.mkdir(parents=True)
    for split, n in {"train": 24, "devel": 8, "test": 8}.items():
        lines = []
        for _ in range(n):
            ws = [rng.choice(WORDS) for _ in range(rng.randint(3, 9))]
            tags = ["O"] * len(ws)
            for i, w in enumerate(ws):
                if w == "gene":  # a learnable rule
                    tags[i] = f"S-{etype}"
            lines += [f"{w}\t{t}" for w, t in zip(ws, tags, strict=True)] + [""]
        (d / f"{split}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def data_dir(tmp_path):
    root = tmp_path / "data"
    write_corpus(root, "NCBI", "Disease", 1)
    write_corpus(root, "BC5CDR", "Chemical", 2)
    return root


def cfg(**kw):
    base = {
        "name": "smoke",
        "corpora": ["NCBI", "BC5CDR"],
        "seeds": [3],
        "epochs": 2,
        "batch_size": 4,
        "eval_batch_size": 8,
        "max_len": 16,
        "fp16": False,
    }
    return config_from_dict({**base, **kw})


def run(c, data_dir, out_dir, tokenizer, seeds=None):
    return run_experiment(
        c,
        data_dir,
        out_dir,
        torch.device("cpu"),
        seeds=seeds,
        tokenizer=tokenizer,
        encoder_factory=tiny_encoder,
        log=lambda *_: None,
    )


def test_stl_writes_per_corpus_results(data_dir, tmp_path, tokenizer):
    paths = run(cfg(), data_dir, tmp_path / "out", tokenizer)
    assert [p.name for p in paths] == ["NCBI.json", "BC5CDR.json"]
    res = json.loads(paths[0].read_text())
    assert res["seed"] == 3 and res["mode"] == "stl" and set(res["corpora"]) == {"NCBI"}
    c = res["corpora"]["NCBI"]
    assert 0.0 <= c["test_entity"]["overall"]["f1"] <= 1.0
    assert c["test_token"]["level"] == "token" and c["dev_entity"]["level"] == "entity"
    assert len(res["history"]) == 2 and res["best_epoch"] in (1, 2)
    rows = read_predictions(paths[0].parent / "predictions_NCBI.jsonl.gz")
    assert len(rows) == 8 and all(len(r["words"]) == len(r["pred"]) == len(r["gold"]) for r in rows)


def test_stl_learns_a_trivial_rule(data_dir, tmp_path, tokenizer):
    c = cfg(epochs=25, learning_rate=5e-3, corpora=["NCBI"], batch_size=8)
    (path,) = run(c, data_dir, tmp_path / "out", tokenizer)
    f1 = json.loads(path.read_text())["corpora"]["NCBI"]["test_entity"]["overall"]["f1"]
    assert f1 > 0.8


def test_mtl_shares_one_encoder_and_reports_all_corpora(data_dir, tmp_path, tokenizer):
    c = cfg(
        name="mtl_smoke",
        mode="mtl",
        mtl={"sampling": "temperature", "temperature": 2.0, "loss_weights": {"NCBI": 2.0}},
    )
    (path,) = run(c, data_dir, tmp_path / "out", tokenizer)
    res = json.loads(path.read_text())
    assert path.name == "all.json" and set(res["corpora"]) == {"NCBI", "BC5CDR"}


def test_mtl_pcgrad_runs(data_dir, tmp_path, tokenizer):
    c = cfg(name="pc", mode="mtl", mtl={"pcgrad": True})
    (path,) = run(c, data_dir, tmp_path / "out", tokenizer)
    assert json.loads(path.read_text())["config"]["mtl"]["pcgrad"] is True


def test_same_seed_is_reproducible(data_dir, tmp_path, tokenizer):
    a = run(cfg(corpora=["NCBI"]), data_dir, tmp_path / "a", tokenizer)[0]
    b = run(cfg(corpora=["NCBI"]), data_dir, tmp_path / "b", tokenizer)[0]
    ra, rb = json.loads(a.read_text()), json.loads(b.read_text())
    assert ra["history"] == rb["history"]
    assert ra["corpora"]["NCBI"]["test_entity"] == rb["corpora"]["NCBI"]["test_entity"]


def test_finished_runs_are_skipped(data_dir, tmp_path, tokenizer):
    out = tmp_path / "out"
    (p,) = run(cfg(corpora=["NCBI"]), data_dir, out, tokenizer)
    before = p.stat().st_mtime_ns
    run(cfg(corpora=["NCBI"]), data_dir, out, tokenizer)
    assert p.stat().st_mtime_ns == before


def test_unknown_corpus_for_experiment_is_rejected(data_dir, tmp_path, tokenizer):
    with pytest.raises(ValueError, match="not part of experiment"):
        run_experiment(
            cfg(corpora=["NCBI"]),
            data_dir,
            tmp_path / "o",
            torch.device("cpu"),
            corpora=["BC5CDR"],
            tokenizer=tokenizer,
            encoder_factory=tiny_encoder,
        )
