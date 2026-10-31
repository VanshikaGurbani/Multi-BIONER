"""Training and evaluation loops (plain PyTorch).

One loop serves both settings: single-task is the degenerate case of one corpus, multi-task draws
the corpus of every step from ``mtl.sampling_probs`` (or, with PCGrad, one batch per corpus).

Protocol: fit on ``train``, select the epoch on ``devel`` entity-F1, evaluate ``test`` exactly once
with the selected weights. The test set is never used for any decision.
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from bioner import mtl
from bioner.batching import collate, endless_batches, n_batches
from bioner.config import ExperimentConfig
from bioner.data import (
    Corpus,
    Window,
    WordPieceEncoder,
    decode_predictions,
    gold_tags,
    load_corpus,
    write_predictions,
)
from bioner.metrics import entity_metrics, token_metrics
from bioner.models import MultiHeadTagger, build_tagger, load_encoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class CorpusData:
    corpus: Corpus
    windows: dict[str, list[Window]] = field(default_factory=dict)


def build_corpus_data(
    names: list[str], data_dir: str | Path, tokenizer, max_len: int
) -> dict[str, CorpusData]:
    encoder = WordPieceEncoder(tokenizer, max_len)
    out = {}
    for name in names:
        corpus = load_corpus(data_dir, name)
        cd = CorpusData(corpus)
        for split, sents in corpus.splits.items():
            cd.windows[split] = encoder.encode_corpus_split(sents, corpus.label_to_id)
        out[name] = cd
    return out


@torch.no_grad()
def predict_split(
    model: MultiHeadTagger,
    name: str,
    cd: CorpusData,
    split: str,
    batch_size: int,
    device: torch.device,
    amp: bool = False,
) -> list[list[str]]:
    """One predicted IOBES tag per word of every sentence in ``split``."""
    model.eval()
    windows = cd.windows[split]
    order = sorted(range(len(windows)), key=lambda i: len(windows[i].input_ids))
    preds: list[list[int] | None] = [None] * len(windows)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = collate([windows[i] for i in idx])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _, logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device), name
            )
        argmax = logits.argmax(-1).cpu().tolist()
        for row, i in zip(argmax, idx, strict=True):
            preds[i] = row[: len(windows[i].input_ids)]
    sentences = cd.corpus.splits[split]
    return decode_predictions(
        windows,
        preds,
        [len(s.words) for s in sentences],
        cd.corpus.id_to_label,  # type: ignore[arg-type]
    )


def evaluate_split(
    model: MultiHeadTagger,
    name: str,
    cd: CorpusData,
    split: str,
    batch_size: int,
    device: torch.device,
    amp: bool = False,
) -> tuple[list[list[str]], dict, dict]:
    pred = predict_split(model, name, cd, split, batch_size, device, amp)
    gold = gold_tags(cd.corpus.splits[split])
    return pred, entity_metrics(gold, pred), token_metrics(gold, pred)


def make_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    decay, plain = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (plain if any(k in n for k in no_decay) else decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
    )


def linear_warmup_decay(opt: torch.optim.Optimizer, total: int, warmup_ratio: float):
    warm = int(total * warmup_ratio)

    def f(step: int) -> float:
        if step < warm:
            return step / max(1, warm)
        return max(0.0, (total - step) / max(1, total - warm))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@dataclass
class TrainOutput:
    model: MultiHeadTagger
    best_epoch: int
    dev: dict[str, dict]  # corpus -> entity metrics at the selected epoch
    history: list[dict]
    seconds: float
    best_state: dict[str, torch.Tensor]


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def train_model(
    cfg: ExperimentConfig,
    names: list[str],
    data: dict[str, CorpusData],
    seed: int,
    device: torch.device,
    encoder_factory: Callable[[], nn.Module] | None = None,
    log: Callable[[str], None] = print,
) -> TrainOutput:
    """Train one tagger over ``names`` (one corpus = single-task); returns best-dev weights."""
    set_seed(seed)
    rng = random.Random(seed)
    encoder = encoder_factory() if encoder_factory else load_encoder(cfg.model_name, cfg.dropout)
    num_labels = {c: len(data[c].corpus.label_to_id) for c in names}
    model = build_tagger(num_labels, dropout=cfg.dropout, encoder=encoder).to(device)

    use_pcgrad = cfg.mtl.pcgrad and len(names) > 1
    amp = cfg.fp16 and device.type == "cuda" and not use_pcgrad
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    weights = {c: float(cfg.mtl.loss_weights.get(c, 1.0)) for c in names}

    sizes = {c: len(data[c].windows["train"]) for c in names}
    steps_per_epoch = sum(n_batches(n, cfg.batch_size) for n in sizes.values())
    if use_pcgrad:
        steps_per_epoch = -(-steps_per_epoch // len(names))  # same number of batches overall
    total = steps_per_epoch * cfg.epochs
    probs = (
        mtl.sampling_probs(sizes, cfg.mtl.sampling, cfg.mtl.temperature)
        if len(names) > 1
        else {names[0]: 1.0}
    )
    schedule = mtl.make_schedule(probs, total, rng)
    streams = {c: endless_batches(data[c].windows["train"], cfg.batch_size, rng) for c in names}
    log(
        f"[{cfg.name} seed={seed}] corpora={names} steps/epoch={steps_per_epoch} total={total} "
        f"sampling={ {c: round(p, 3) for c, p in probs.items()} } amp={amp} pcgrad={use_pcgrad}"
    )

    opt = make_optimizer(model, cfg)
    sched = linear_warmup_decay(opt, total, cfg.warmup_ratio)
    shared = model.shared_parameters()

    def batch_loss(corpus: str) -> torch.Tensor:
        batch = {k: v.to(device) for k, v in collate(next(streams[corpus])).items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss, _ = model(batch["input_ids"], batch["attention_mask"], corpus, batch["labels"])
        return loss * weights[corpus]

    best_f1, best_epoch, best_state, best_dev = -1.0, 0, {}, {}
    history: list[dict] = []
    started = time.time()
    running, seen = 0.0, 0
    for step in range(total):
        model.train()
        if use_pcgrad:
            opt.zero_grad(set_to_none=True)
            grads = []
            for c in names:
                loss = batch_loss(c)
                loss.backward()
                running += loss.item()
                grads.append(mtl.flat_grad(shared))
                for p in shared:
                    p.grad = None
            mtl.assign_flat_grad(shared, mtl.pcgrad(grads, rng))
            seen += len(names)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
        else:
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(schedule[step])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            seen += 1
        sched.step()

        if (step + 1) % steps_per_epoch == 0:
            epoch = (step + 1) // steps_per_epoch
            dev_metrics = {
                c: evaluate_split(model, c, data[c], "devel", cfg.eval_batch_size, device, amp)[1]
                for c in names
            }
            f1s = {c: m["overall"]["f1"] for c, m in dev_metrics.items()}
            mean_f1 = sum(f1s.values()) / len(f1s)
            history.append(
                {"epoch": epoch, "train_loss": running / max(1, seen), "dev_entity_f1": f1s}
            )
            avg = running / max(1, seen)
            log(f"[{cfg.name} seed={seed}] epoch {epoch} loss={avg:.4f} dev={f1s}")
            running, seen = 0.0, 0
            if mean_f1 > best_f1:
                best_f1, best_epoch, best_dev = mean_f1, epoch, dev_metrics
                best_state = _cpu_state(model)

    model.load_state_dict(best_state)
    return TrainOutput(model, best_epoch, best_dev, history, time.time() - started, best_state)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None


def _env_info(device: torch.device) -> dict:
    import transformers

    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
        "git_commit": _git_commit(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def evaluate_test_and_save(
    cfg: ExperimentConfig,
    names: list[str],
    data: dict[str, CorpusData],
    out: TrainOutput,
    seed: int,
    device: torch.device,
    result_path: Path,
) -> dict:
    amp = cfg.fp16 and device.type == "cuda" and not (cfg.mtl.pcgrad and len(names) > 1)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    corpora_res = {}
    for c in names:
        pred, ent, tok = evaluate_split(
            out.model, c, data[c], "test", cfg.eval_batch_size, device, amp
        )
        write_predictions(
            result_path.parent / f"predictions_{c}.jsonl.gz", data[c].corpus.splits["test"], pred
        )
        corpora_res[c] = {
            "dev_entity": out.dev[c],
            "test_entity": ent,
            "test_token": tok,
            "n_train_sentences": len(data[c].corpus.splits["train"]),
        }
    result = {
        "experiment": cfg.name,
        "mode": cfg.mode,
        "model_name": cfg.model_name,
        "seed": seed,
        "best_epoch": out.best_epoch,
        "train_seconds": round(out.seconds, 1),
        "history": out.history,
        "corpora": corpora_res,
        "config": cfg.to_dict(),
        "env": _env_info(device),
    }
    _write_json(result_path, result)
    if cfg.save_checkpoint:
        ckpt = result_path.parent / "checkpoint"
        ckpt.mkdir(exist_ok=True)
        torch.save(out.best_state, ckpt / f"{result_path.stem}.pt")
    return result


def run_experiment(
    cfg: ExperimentConfig,
    data_dir: str | Path,
    out_dir: str | Path,
    device: torch.device,
    seeds: list[int] | None = None,
    corpora: list[str] | None = None,
    tokenizer=None,
    encoder_factory: Callable[[], nn.Module] | None = None,
    skip_existing: bool = True,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Run ``cfg`` for each seed. Finished (seed, corpus) results are skipped, so a run that was
    interrupted on Colab/Kaggle resumes where it stopped. Returns the result files."""
    seeds = seeds or cfg.seeds
    names = corpora or cfg.corpora
    unknown = [c for c in names if c not in cfg.corpora]
    if unknown:
        raise ValueError(f"corpora {unknown} are not part of experiment {cfg.name!r}")
    root = Path(out_dir) / cfg.name
    _write_json(root / "config.json", cfg.to_dict())

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    data = build_corpus_data(names, data_dir, tokenizer, cfg.max_len)

    written = []
    for seed in seeds:
        groups = [[c] for c in names] if cfg.mode == "stl" else [names]
        for group in groups:
            stem = group[0] if cfg.mode == "stl" else "all"
            path = root / f"seed{seed}" / f"{stem}.json"
            if skip_existing and path.exists():
                log(f"skip {path} (exists)")
                written.append(path)
                continue
            out = train_model(cfg, group, data, seed, device, encoder_factory, log)
            evaluate_test_and_save(cfg, group, data, out, seed, device, path)
            log(f"wrote {path}")
            written.append(path)
            del out
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return written
