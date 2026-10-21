"""Shared fixtures. Everything here is offline: a toy WordPiece vocabulary, no model downloads."""

import pytest
from transformers import BertTokenizerFast

VOCAB = (
    ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    + ["the", "gene", "IL", "-", "2", "kappa", "##B", "p", "##53", "a", "b", "c", ".", ","]
    + ["Inter", "##leu", "##kin"]
)


@pytest.fixture(scope="session")
def vocab_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    path.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def tokenizer(vocab_file):
    return BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False)
