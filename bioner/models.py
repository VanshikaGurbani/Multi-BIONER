"""One shared transformer encoder with a linear tagging head per corpus.

Single-task models are the same class with a single head, so STL and MTL differ only in what is
shared and how batches are drawn. (The legacy "multi-task" notebook instead instantiated one full
BioBERT per corpus and shared nothing.)
"""

from __future__ import annotations

import torch
from torch import nn

from bioner.labels import IGNORE_INDEX


class MultiHeadTagger(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        num_labels: dict[str, int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({c: nn.Linear(hidden_size, n) for c, n in num_labels.items()})
        self.num_labels = dict(num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        corpus: str,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.heads[corpus](self.dropout(hidden))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return loss, logits

    def shared_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def head_parameters(self, corpus: str) -> list[nn.Parameter]:
        return list(self.heads[corpus].parameters())


def load_encoder(model_name: str, dropout: float = 0.1) -> nn.Module:
    from transformers import AutoModel

    try:
        return AutoModel.from_pretrained(
            model_name, add_pooling_layer=False, hidden_dropout_prob=dropout
        )
    except TypeError:  # architectures without a pooler / these kwargs
        return AutoModel.from_pretrained(model_name)


def build_tagger(
    num_labels: dict[str, int],
    model_name: str | None = None,
    dropout: float = 0.1,
    encoder: nn.Module | None = None,
) -> MultiHeadTagger:
    if encoder is None:
        if model_name is None:
            raise ValueError("need model_name or an encoder")
        encoder = load_encoder(model_name, dropout)
    return MultiHeadTagger(encoder, encoder.config.hidden_size, num_labels, dropout)
