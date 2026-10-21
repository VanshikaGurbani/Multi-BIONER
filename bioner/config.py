"""Experiment configuration loaded from YAML and validated eagerly."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from bioner.data import CORPORA

MODES = ("stl", "mtl")
SAMPLING = ("proportional", "uniform", "temperature")


@dataclass
class MTLConfig:
    sampling: str = "proportional"  # how the corpus of each training step is chosen
    temperature: float = 1.0  # p_i proportional to n_i ** (1/T); T=1 proportional, large T uniform
    loss_weights: dict[str, float] = field(default_factory=dict)  # per-corpus loss multiplier
    pcgrad: bool = False  # one batch per corpus per step, conflicting encoder grads projected


@dataclass
class ExperimentConfig:
    name: str
    model_name: str = "dmis-lab/biobert-base-cased-v1.1"
    mode: str = "stl"
    corpora: list[str] = field(default_factory=lambda: list(CORPORA))
    seeds: list[int] = field(default_factory=lambda: [13, 42, 2024])
    epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_len: int = 128
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True  # only takes effect on CUDA
    save_checkpoint: bool = False
    mtl: MTLConfig = field(default_factory=MTLConfig)

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not self.name or "/" in self.name or "\\" in self.name:
            errors.append("name must be a non-empty file-name-safe string")
        if self.mode not in MODES:
            errors.append(f"mode must be one of {MODES}, got {self.mode!r}")
        unknown = [c for c in self.corpora if c not in CORPORA]
        if unknown or not self.corpora:
            errors.append(f"corpora must be a non-empty subset of {sorted(CORPORA)}; got {unknown}")
        if len(set(self.corpora)) != len(self.corpora):
            errors.append("corpora contains duplicates")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            errors.append("seeds must be a non-empty list of distinct ints")
        for key in ("epochs", "batch_size", "eval_batch_size", "max_len"):
            if getattr(self, key) < 1:
                errors.append(f"{key} must be >= 1")
        if self.max_len < 3:
            errors.append("max_len must be >= 3")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        if not 0 <= self.warmup_ratio < 1:
            errors.append("warmup_ratio must be in [0, 1)")
        if not 0 <= self.dropout < 1:
            errors.append("dropout must be in [0, 1)")
        if self.mtl.sampling not in SAMPLING:
            errors.append(f"mtl.sampling must be one of {SAMPLING}")
        if self.mtl.temperature <= 0:
            errors.append("mtl.temperature must be > 0")
        extra = [c for c in self.mtl.loss_weights if c not in self.corpora]
        if extra or any(w <= 0 for w in self.mtl.loss_weights.values()):
            errors.append("mtl.loss_weights must be > 0 and only name corpora in `corpora`")
        if self.mode == "stl" and (
            self.mtl.pcgrad or self.mtl.sampling != "proportional" or self.mtl.loss_weights
        ):
            errors.append("mtl.* options only apply with mode: mtl")
        if self.mode == "mtl" and len(self.corpora) < 2:
            errors.append("mode: mtl needs at least two corpora")
        if errors:
            raise ValueError("invalid config:\n  - " + "\n  - ".join(errors))

    def to_dict(self) -> dict:
        return asdict(self)


def config_from_dict(raw: dict) -> ExperimentConfig:
    allowed = {f.name for f in fields(ExperimentConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    raw = dict(raw)
    mtl_raw = raw.pop("mtl", {}) or {}
    mtl_allowed = {f.name for f in fields(MTLConfig)}
    if set(mtl_raw) - mtl_allowed:
        raise ValueError(f"unknown mtl keys: {sorted(set(mtl_raw) - mtl_allowed)}")
    return ExperimentConfig(mtl=MTLConfig(**mtl_raw), **raw)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return config_from_dict(raw)
