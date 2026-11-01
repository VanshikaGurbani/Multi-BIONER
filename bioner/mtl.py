"""Multi-task training ingredients: corpus sampling, loss weighting and PCGrad.

Why this exists: with proportional sampling BC4CHEMD (about 31k training sentences) supplies ~40% of
all steps while NCBI-Disease (5k) supplies ~8%, so the shared encoder mostly learns chemicals.
Temperature sampling ``p_i ∝ n_i^(1/T)`` flattens that; ``T=1`` is proportional, ``T→∞`` uniform.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch


def sampling_probs(
    sizes: dict[str, int], sampling: str = "proportional", temperature: float = 1.0
) -> dict[str, float]:
    if not sizes or any(n <= 0 for n in sizes.values()):
        raise ValueError("every corpus needs a positive size")
    if sampling == "proportional":
        exp = 1.0
    elif sampling == "uniform":
        exp = 0.0
    elif sampling == "temperature":
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        exp = 1.0 / temperature
    else:
        raise ValueError(f"unknown sampling {sampling!r}")
    raw = {c: n**exp for c, n in sizes.items()}
    z = sum(raw.values())
    return {c: v / z for c, v in raw.items()}


def make_schedule(probs: dict[str, float], n_steps: int, rng: random.Random) -> list[str]:
    """Corpus used at each training step, drawn i.i.d. from ``probs``."""
    names = sorted(probs)
    return rng.choices(names, weights=[probs[n] for n in names], k=n_steps)


def pcgrad(grads: Sequence[torch.Tensor], rng: random.Random) -> torch.Tensor:
    """Project conflicting task gradients (Yu et al., 2020) and return their **mean**.

    For each task gradient ``g_i`` and every other ``g_j`` (random order): if ``g_i·g_j < 0``,
    remove from ``g_i`` its component along ``g_j``. The mean (not sum) keeps the update on the
    same scale as a single-task step.
    """
    if not grads:
        raise ValueError("no gradients")
    projected = []
    for i, g in enumerate(grads):
        gi = g.clone()
        others = [j for j in range(len(grads)) if j != i]
        rng.shuffle(others)
        for j in others:
            gj = grads[j]
            dot = torch.dot(gi, gj)
            if dot < 0:
                denom = torch.dot(gj, gj)
                if denom > 0:
                    gi = gi - dot / denom * gj
        projected.append(gi)
    return torch.stack(projected).mean(dim=0)


def flat_grad(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat(
        [(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params]
    )


def assign_flat_grad(params: Sequence[torch.nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = flat[offset : offset + n].view_as(p).clone()
        offset += n
