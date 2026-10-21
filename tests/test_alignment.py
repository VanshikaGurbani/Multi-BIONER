"""Word-piece <-> word alignment: the part of the pipeline most prone to silent label bugs."""

import random

import pytest
from transformers import BertTokenizer

from bioner.data import WordPieceEncoder, decode_predictions
from bioner.labels import IGNORE_INDEX

CLS, SEP = 2, 3  # ids in tests/conftest.py VOCAB


def enc(tokenizer, max_len=128):
    return WordPieceEncoder(tokenizer, max_len=max_len)


def test_single_piece_words_all_labelled(tokenizer):
    (w,) = enc(tokenizer).encode(["the", "gene"], [0, 1])
    assert w.input_ids == [CLS, 5, 6, SEP]
    assert w.labels == [IGNORE_INDEX, 0, 1, IGNORE_INDEX]
    assert w.word_index == [None, 0, 1, None]


def test_only_first_piece_of_a_multi_piece_word_gets_the_label(tokenizer):
    (w,) = enc(tokenizer).encode(["Interleukin", "the"], [3, 0])
    assert len(w.input_ids) == 1 + 3 + 1 + 1  # CLS, 3 pieces, "the", SEP
    assert w.labels == [IGNORE_INDEX, 3, IGNORE_INDEX, IGNORE_INDEX, 0, IGNORE_INDEX]
    assert w.word_index == [None, 0, None, None, 1, None]


def test_punctuation_inside_a_word_stays_with_that_word(tokenizer):
    # "IL-2" is split by BERT's basic tokenizer into IL, -, 2: one label, on "IL".
    (w,) = enc(tokenizer).encode(["IL-2"], [4])
    assert w.input_ids[1:-1] == [7, 8, 9]
    assert [x for x in w.labels if x != IGNORE_INDEX] == [4]
    assert w.word_index == [None, 0, None, None, None]


def test_special_tokens_are_ignored(tokenizer):
    (w,) = enc(tokenizer).encode(["the"], [0])
    assert w.labels[0] == IGNORE_INDEX and w.labels[-1] == IGNORE_INDEX


def test_unknown_word_becomes_unk_not_dropped(tokenizer):
    (w,) = enc(tokenizer).encode(["zzzz", "the"], [1, 0])
    assert w.input_ids[1] == tokenizer.unk_token_id
    assert [x for x in w.labels if x != IGNORE_INDEX] == [1, 0]


def test_word_that_tokenizes_to_nothing_is_kept_as_unk(tokenizer):
    # U+200B is stripped by BERT's basic tokenizer, so tokenize() returns [].
    assert tokenizer.tokenize("​") == []
    (w,) = enc(tokenizer).encode(["​", "the"], [2, 0])
    assert [x for x in w.labels if x != IGNORE_INDEX] == [2, 0]
    assert w.input_ids[1] == tokenizer.unk_token_id


def test_empty_sentence_yields_one_window(tokenizer):
    (w,) = enc(tokenizer).encode([], [])
    assert w.input_ids == [CLS, SEP] and w.word_index == [None, None]


def test_long_sentence_is_windowed_without_losing_words(tokenizer):
    words = ["the"] * 15
    wins = enc(tokenizer, max_len=8).encode(words, list(range(15)))
    assert len(wins) > 1
    assert all(len(w.input_ids) <= 8 for w in wins)
    assert all(w.input_ids[0] == CLS and w.input_ids[-1] == SEP for w in wins)
    seen = [i for w in wins for i in w.word_index if i is not None]
    assert seen == list(range(15))  # every word exactly once, in order
    labels = [x for w in wins for x in w.labels if x != IGNORE_INDEX]
    assert labels == list(range(15))


def test_a_word_is_never_split_across_windows(tokenizer):
    # budget = max_len - 2 = 4. "Interleukin" has 3 pieces and must start a new window.
    wins = enc(tokenizer, max_len=6).encode(["the", "the", "Interleukin"], [0, 1, 2])
    assert len(wins) == 2
    assert wins[0].word_index == [None, 0, 1, None]
    assert wins[1].word_index == [None, 2, None, None, None]
    assert wins[1].labels == [IGNORE_INDEX, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]


def test_overlong_single_word_is_truncated_but_keeps_its_label(tokenizer):
    (w,) = enc(tokenizer, max_len=4).encode(["Interleukin"], [3])  # budget 2 < 3 pieces
    assert len(w.input_ids) == 4
    assert [x for x in w.labels if x != IGNORE_INDEX] == [3]


def test_labels_must_match_words(tokenizer):
    with pytest.raises(ValueError):
        enc(tokenizer).encode(["the", "gene"], [0])


def test_max_len_too_small_is_rejected(tokenizer):
    with pytest.raises(ValueError):
        WordPieceEncoder(tokenizer, max_len=2)


def test_inference_windows_carry_no_labels(tokenizer):
    (w,) = enc(tokenizer).encode(["the", "gene"], None)
    assert set(w.labels) == {IGNORE_INDEX}
    assert w.word_index == [None, 0, 1, None]


def test_decode_round_trip_restores_one_tag_per_word(tokenizer):
    id_to_label = {0: "O", 1: "B-X", 2: "E-X", 3: "S-X"}
    sents = [["the", "Interleukin", "gene"], [], ["p53", "kappaB", "the", "IL-2", "a", "b", "c"]]
    gold = [[0, 1, 2], [], [3, 0, 0, 1, 2, 0, 3]]
    e = enc(tokenizer, max_len=6)
    windows, lengths = [], []
    for i, (words, labs) in enumerate(zip(sents, gold, strict=True)):
        windows += e.encode(words, labs, i)
        lengths.append(len(words))
    # a "perfect model": predicts the gold id on first pieces, garbage on the rest
    preds = [[x if x != IGNORE_INDEX else 3 for x in w.labels] for w in windows]
    decoded = decode_predictions(windows, preds, lengths, id_to_label)
    assert decoded == [[id_to_label[i] for i in g] for g in gold]


def test_decode_ignores_predictions_on_continuation_pieces(tokenizer):
    e = enc(tokenizer)
    (w,) = e.encode(["Interleukin"], [0])
    preds = [[3, 1, 3, 3, 3]]  # 1 on the first piece (position 1), 3 elsewhere
    assert decode_predictions([w], preds, [1], {0: "O", 1: "B-X", 3: "S-X"}) == [["B-X"]]


def test_decode_rejects_length_mismatch(tokenizer):
    (w,) = enc(tokenizer).encode(["the"], [0])
    with pytest.raises(ValueError):
        decode_predictions([w], [[0]], [1], {0: "O"})


def test_slow_and_fast_tokenizers_align_identically(tokenizer, vocab_file):
    slow = BertTokenizer(vocab_file=str(vocab_file), do_lower_case=False)
    words = ["the", "Interleukin", "IL-2", "zzzz", "p53", "​", "kappaB"]
    labels = list(range(len(words)))
    a = WordPieceEncoder(tokenizer, 16).encode(words, labels)
    b = WordPieceEncoder(slow, 16).encode(words, labels)
    assert [(w.input_ids, w.labels, w.word_index) for w in a] == [
        (w.input_ids, w.labels, w.word_index) for w in b
    ]


def test_random_sentences_keep_every_word_exactly_once(tokenizer):
    rng = random.Random(0)
    pool = ["the", "gene", "IL-2", "Interleukin", "p53", "kappaB", "zzzz", "a", ",", "."]
    for _ in range(200):
        n = rng.randint(0, 25)
        words = [rng.choice(pool) for _ in range(n)]
        labels = [rng.randint(0, 5) for _ in range(n)]
        max_len = rng.randint(7, 20)
        wins = WordPieceEncoder(tokenizer, max_len).encode(words, labels)
        assert all(len(w.input_ids) <= max_len for w in wins)
        assert all(len(w.input_ids) == len(w.labels) == len(w.word_index) for w in wins)
        assert [i for w in wins for i in w.word_index if i is not None] == list(range(n))
        assert [x for w in wins for x in w.labels if x != IGNORE_INDEX] == labels
