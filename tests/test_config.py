from pathlib import Path

import pytest
import yaml

from bioner.config import config_from_dict, load_config

MTL_PAIR = {"mode": "mtl", "corpora": ["BC2GM", "BC5CDR"]}


def test_defaults_are_valid():
    cfg = config_from_dict({"name": "x"})
    assert cfg.mode == "stl" and cfg.seeds == [13, 42, 2024] and len(cfg.corpora) == 5


def test_yaml_round_trip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "mtl_t",
                "mode": "mtl",
                "corpora": ["BC2GM", "NCBI"],
                "mtl": {"sampling": "temperature", "temperature": 2.0, "loss_weights": {"NCBI": 2}},
            }
        )
    )
    cfg = load_config(p)
    assert cfg.mtl.sampling == "temperature" and cfg.mtl.loss_weights == {"NCBI": 2}
    assert config_from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "raw",
    [
        {"name": ""},
        {"name": "a/b"},
        {"name": "x", "mode": "foo"},
        {"name": "x", "corpora": ["NOPE"]},
        {"name": "x", "corpora": []},
        {"name": "x", "corpora": ["BC2GM", "BC2GM"]},
        {"name": "x", "seeds": [1, 1]},
        {"name": "x", "epochs": 0},
        {"name": "x", "learning_rate": 0},
        {"name": "x", "warmup_ratio": 1.0},
        {"name": "x", "max_len": 2},
        {"name": "x", "mode": "mtl", "corpora": ["BC2GM"]},
        {"name": "x", **MTL_PAIR, "mtl": {"sampling": "bogus"}},
        {"name": "x", **MTL_PAIR, "mtl": {"temperature": 0}},
        {"name": "x", **MTL_PAIR, "mtl": {"loss_weights": {"NCBI": 1.0}}},
        {"name": "x", "mode": "stl", "mtl": {"pcgrad": True}},
        {"name": "x", "typo_key": 1},
        {"name": "x", "mtl": {"typo": 1}},
    ],
)
def test_invalid_configs_are_rejected(raw):
    with pytest.raises(ValueError):
        config_from_dict(raw)


def test_every_shipped_config_loads():
    for p in sorted(Path(__file__).parent.parent.joinpath("configs").glob("*.yaml")):
        load_config(p)
