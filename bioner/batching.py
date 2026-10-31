"""Turn alignment windows into padded tensors."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

import torch

from bioner.data import Window
from bioner.labels import IGNORE_INDEX


def collate(windows: Sequence[Window], pad_id: int = 0) -> dict[str, torch.Tensor]:
    n = max(len(w.input_ids) for w in windows)
    ids = torch.full((len(windows), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(windows), n), dtype=torch.long)
    labels = torch.full((len(windows), n), IGNORE_INDEX, dtype=torch.long)
    for i, w in enumerate(windows):
        k = len(w.input_ids)
        ids[i, :k] = torch.tensor(w.input_ids)
        mask[i, :k] = 1
        labels[i, :k] = torch.tensor(w.labels)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def shuffled_batches(
    windows: Sequence[Window], batch_size: int, rng: random.Random
) -> Iterator[list[Window]]:
    order = list(range(len(windows)))
    rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        yield [windows[j] for j in order[i : i + batch_size]]


def n_batches(n_items: int, batch_size: int) -> int:
    return -(-n_items // batch_size)


def endless_batches(
    windows: Sequence[Window], batch_size: int, rng: random.Random
) -> Iterator[list[Window]]:
    """Batches forever, reshuffling after each full pass (a partial last batch is kept)."""
    while True:
        yield from shuffled_batches(windows, batch_size, rng)
