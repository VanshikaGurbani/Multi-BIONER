import hashlib
import zipfile

import pytest

from bioner import data
from bioner.data import (
    Corpus,
    Sentence,
    corpus_stats,
    prepare_corpus,
    read_conll,
    read_predictions,
    write_predictions,
)

SAMPLE = "the\tO\nIL-2\tS-protein\n\nkappaB\tB-protein\ngene\tE-protein\n"


def test_read_conll(tmp_path):
    p = tmp_path / "a.tsv"
    p.write_text(SAMPLE, encoding="utf-8")
    assert read_conll(p) == [
        Sentence(("the", "IL-2"), ("O", "S-protein")),
        Sentence(("kappaB", "gene"), ("B-protein", "E-protein")),
    ]


def test_read_conll_without_trailing_blank_line_and_with_extra_blanks(tmp_path):
    p = tmp_path / "a.tsv"
    p.write_text("a\tO\n\n\n\nb\tO", encoding="utf-8")
    assert len(read_conll(p)) == 2


def test_read_conll_rejects_malformed_line(tmp_path):
    p = tmp_path / "a.tsv"
    p.write_text("only_one_column\n", encoding="utf-8")
    with pytest.raises(ValueError, match="a.tsv:1"):
        read_conll(p)


def test_corpus_label_space_and_stats():
    train = [Sentence(("a", "b"), ("B-X", "E-X")), Sentence(("c",), ("O",))]
    test = [Sentence(("a", "b"), ("B-X", "E-X")), Sentence(("d",), ("S-Y",))]
    c = Corpus("T", {"train": train, "test": test})
    assert c.label_to_id["O"] == 0 and set(c.label_to_id) == {"O", "B-X", "E-X", "S-Y"}
    st = corpus_stats(c)
    assert st["splits"]["train"]["entities"] == 1
    assert st["test_sentences_also_in_train"] == 1
    assert st["splits"]["test"]["malformed_sentences"] == 0


def _fake_zip(tmp_path, monkeypatch, name="NCBI", good_sha=True):
    info = data.CORPORA[name]
    path = tmp_path / "zips" / info["zip"]
    path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for s in ("train", "devel", "test", "merge"):
            zf.writestr(f"{info['dir']}/{s}.tsv", f"w\tO\n# {s}\n")
    if good_sha:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        monkeypatch.setitem(data.CORPORA, name, {**info, "sha256": sha})
    return path


def test_prepare_corpus_extracts_three_splits_and_checks_sha(tmp_path, monkeypatch):
    zp = _fake_zip(tmp_path, monkeypatch)
    out = prepare_corpus("NCBI", tmp_path / "data", zip_dir=zp.parent)
    assert sorted(p.name for p in out.iterdir()) == ["devel.tsv", "test.tsv", "train.tsv"]


def test_prepare_corpus_rejects_wrong_checksum(tmp_path, monkeypatch):
    zp = _fake_zip(tmp_path, monkeypatch, good_sha=False)
    with pytest.raises(ValueError, match="sha256"):
        prepare_corpus("NCBI", tmp_path / "data", zip_dir=zp.parent)


def test_prepare_corpus_downloads_via_injected_fetch(tmp_path, monkeypatch):
    zp = _fake_zip(tmp_path, monkeypatch)
    calls = []

    def fetch(url, dest):
        calls.append(url)
        dest.write_bytes(zp.read_bytes())

    prepare_corpus("NCBI", tmp_path / "data", fetch=fetch)
    assert calls and data.LEGACY_COMMIT in calls[0]
    prepare_corpus("NCBI", tmp_path / "data", fetch=fetch)  # already present: no second download
    assert len(calls) == 1


def test_prediction_dump_round_trip(tmp_path):
    sents = [Sentence(("a", "b"), ("O", "S-X"))]
    write_predictions(tmp_path / "p.jsonl.gz", sents, [["O", "O"]])
    (row,) = read_predictions(tmp_path / "p.jsonl.gz")
    assert row == {"words": ["a", "b"], "gold": ["O", "S-X"], "pred": ["O", "O"]}
