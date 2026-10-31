"""Corpus I/O and word-piece <-> word alignment.

The alignment contract (tested in ``tests/test_alignment.py``):

* every word is tokenized on its own, so the word -> word-piece mapping is explicit;
* the **first** word-piece of a word carries the word's label, the remaining pieces get
  ``IGNORE_INDEX``; ``[CLS]``/``[SEP]`` also get ``IGNORE_INDEX``;
* a sentence that does not fit into ``max_len`` is split into several windows on word
  boundaries, so no word is ever dropped (the legacy notebooks silently truncated);
* decoding yields exactly one tag per original word.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bioner.labels import IGNORE_INDEX, OUTSIDE, make_label_maps, split_tag, validate_iobes

SPLITS = ("train", "devel", "test")

# The original dataset zips were committed in this repository; they remain available at the
# pinned commit below. SHA-256 sums guarantee the bytes are the ones the thesis used.
LEGACY_COMMIT = "5958784cabcfa9fa1e11bb5d980d4f64c2d5a50b"
LEGACY_URL = "https://raw.githubusercontent.com/VanshikaGurbani/Multi-BIONER/{commit}/{zip}"

CORPORA: dict[str, dict[str, str]] = {
    "BC2GM": {
        "zip": "BC2GM-IOBES.zip",
        "dir": "BC2GM-IOBES",
        "sha256": "28b978ab3d1f29c81de663b7578f7e950e8d209402771395ff1c4d07ebe88a78",
    },
    "BC4CHEMD": {
        "zip": "BC4CHEMD-IOBES.zip",
        "dir": "BC4CHEMD-IOBES",
        "sha256": "3c37283616a0d6cf962e968401fe4aac6568bc7cff75a687040961dc53fc2364",
    },
    "BC5CDR": {
        "zip": "BC5CDR-IOBES.zip",
        "dir": "BC5CDR-IOBES",
        "sha256": "c76a709aebb3812e876bf797aa24ccf9332dd20e0da37a51f8e907cb99e736eb",
    },
    "JNLPBA": {
        "zip": "JNLPBA-IOBES.zip",
        "dir": "JNLPBA-IOBES",
        "sha256": "a19b17fead38f4c0ef9d58658783bd3018aa9d8111280e5d978e8623477abef9",
    },
    "NCBI": {
        "zip": "NCBI-disease-IOBES.zip",
        "dir": "NCBI-disease-IOBES",
        "sha256": "740c9bab3ea136bb9509705d97d4eb26d61aa7cfc4410143ab7ffe7e9e86af51",
    },
}


# --------------------------------------------------------------------------- corpus I/O


@dataclass(frozen=True)
class Sentence:
    words: tuple[str, ...]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.words) != len(self.tags):
            raise ValueError("words and tags differ in length")


def read_conll(path: str | Path) -> list[Sentence]:
    """Read a two-column ``token<TAB>tag`` file; blank lines separate sentences."""
    sentences: list[Sentence] = []
    words: list[str] = []
    tags: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line.strip():
                if words:
                    sentences.append(Sentence(tuple(words), tuple(tags)))
                    words, tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: expected 'token<TAB>tag', got {line!r}")
            words.append(parts[0])
            tags.append(parts[1])
    if words:
        sentences.append(Sentence(tuple(words), tuple(tags)))
    return sentences


@dataclass
class Corpus:
    name: str
    splits: dict[str, list[Sentence]]
    label_to_id: dict[str, int] = field(init=False)
    id_to_label: dict[int, str] = field(init=False)
    joined_sentences: dict[str, int] = field(default_factory=dict)  # split -> repaired cuts

    def __post_init__(self) -> None:
        tags = (t for s in self.splits.values() for sent in s for t in sent.tags)
        self.label_to_id, self.id_to_label = make_label_maps(tags)


def merge_split_entities(sentences: list[Sentence]) -> tuple[list[Sentence], int]:
    """Re-join sentences that the corpus splitter cut in the middle of an entity.

    If sentence *i* ends inside an entity (last tag ``B-X``/``I-X``) and sentence *i+1* continues it
    (first tag ``I-X``/``E-X``, same type), they are concatenated. Only segmentation changes, never
    a tag. JNLPBA as distributed has ~20% of its sentences cut like this, which makes those gold
    entities unscorable (strict seqeval drops malformed gold chunks). Returns the new list and the
    number of joins.
    """
    out: list[Sentence] = []
    joins = 0
    for sent in sentences:
        if out and sent.tags and out[-1].tags:
            pre_last, type_last = split_tag(out[-1].tags[-1])
            pre_first, type_first = split_tag(sent.tags[0])
            if pre_last in {"B", "I"} and pre_first in {"I", "E"} and type_last == type_first:
                prev = out.pop()
                out.append(Sentence(prev.words + sent.words, prev.tags + sent.tags))
                joins += 1
                continue
        out.append(sent)
    return out, joins


def load_corpus(
    data_dir: str | Path, name: str, splits: Iterable[str] = SPLITS, repair: bool = True
) -> Corpus:
    """Load ``train/devel/test`` of a prepared corpus.

    ``repair`` re-joins sentences cut inside an entity (see ``merge_split_entities``); it is a
    no-op for corpora whose gold is already well formed.
    """
    if name not in CORPORA:
        raise KeyError(f"unknown corpus {name!r}; known: {sorted(CORPORA)}")
    root = Path(data_dir) / name
    loaded = {}
    joins = {}
    for split in splits:
        path = root / f"{split}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `bioner prepare-data` first")
        sents = read_conll(path)
        joins[split] = 0
        if repair:
            sents, joins[split] = merge_split_entities(sents)
        loaded[split] = sents
    corpus = Corpus(name, loaded)
    corpus.joined_sentences = joins
    return corpus


def corpus_stats(corpus: Corpus) -> dict:
    """Sentence/token/entity counts per split, malformed-gold count and train/test overlap."""
    stats: dict = {
        "labels": len(corpus.label_to_id),
        "sentences_rejoined": dict(corpus.joined_sentences),
        "splits": {},
    }
    for split, sents in corpus.splits.items():
        n_ent = sum(t.startswith(("B-", "S-")) for s in sents for t in s.tags)
        bad = sum(bool(validate_iobes(s.tags)) for s in sents)
        stats["splits"][split] = {
            "sentences": len(sents),
            "tokens": sum(len(s.words) for s in sents),
            "entities": n_ent,
            "malformed_sentences": bad,
            "max_words": max((len(s.words) for s in sents), default=0),
        }
    if "train" in corpus.splits and "test" in corpus.splits:
        train = {s.words for s in corpus.splits["train"]}
        stats["test_sentences_also_in_train"] = sum(s.words in train for s in corpus.splits["test"])
    return stats


# ------------------------------------------------------------------------ data download


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_corpus(
    name: str,
    data_dir: str | Path,
    zip_dir: str | Path | None = None,
    keep_zip: bool = False,
    fetch: Callable[[str, Path], None] | None = None,
) -> Path:
    """Place ``train/devel/test.tsv`` of one corpus under ``data_dir/name``.

    The zip comes from ``zip_dir`` if given, otherwise it is downloaded from the pinned legacy
    commit. The SHA-256 is always verified. ``merge.tsv`` (train+devel) is deliberately skipped.
    """
    info = CORPORA[name]
    data_dir = Path(data_dir)
    out = data_dir / name
    if all((out / f"{s}.tsv").exists() for s in SPLITS):
        return out

    if zip_dir is not None:
        zip_path = Path(zip_dir) / info["zip"]
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
    else:
        zip_path = data_dir / "_zips" / info["zip"]
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists():
            url = LEGACY_URL.format(commit=LEGACY_COMMIT, zip=info["zip"])
            (fetch or _download)(url, zip_path)

    digest = _sha256(zip_path)
    if digest != info["sha256"]:
        raise ValueError(f"{zip_path}: sha256 {digest} != expected {info['sha256']}")

    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for split in SPLITS:
            member = f"{info['dir']}/{split}.tsv"
            with zf.open(member) as src, open(out / f"{split}.tsv", "wb") as dst:
                shutil.copyfileobj(src, dst)
    if zip_dir is None and not keep_zip:
        zip_path.unlink()
    return out


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:  # noqa: S310
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)


# -------------------------------------------------------------------- word-piece alignment


@dataclass
class Window:
    """One model input: a contiguous run of words of a sentence."""

    sentence_index: int
    input_ids: list[int]
    labels: list[int]  # IGNORE_INDEX except on the first piece of each word
    word_index: list[int | None]  # word index (in the sentence) on first pieces, else None


class WordPieceEncoder:
    """Tokenizes words one by one (cached) and packs them into windows of ``<= max_len``."""

    def __init__(self, tokenizer, max_len: int = 128):
        if max_len < 3:
            raise ValueError("max_len must leave room for [CLS], one piece and [SEP]")
        self.tok = tokenizer
        self.max_len = max_len
        self._cache: dict[str, list[int]] = {}

    def pieces(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is None:
            toks = self.tok.tokenize(word)
            if not toks:  # e.g. a lone control character: keep the word alive as [UNK]
                toks = [self.tok.unk_token]
            cached = self.tok.convert_tokens_to_ids(toks)
            self._cache[word] = cached
        return cached

    def encode(
        self,
        words: Sequence[str],
        label_ids: Sequence[int] | None = None,
        sentence_index: int = 0,
    ) -> list[Window]:
        if label_ids is not None and len(label_ids) != len(words):
            raise ValueError("label_ids and words differ in length")
        budget = self.max_len - 2
        windows: list[Window] = []
        ids: list[int] = []
        labs: list[int] = []
        widx: list[int | None] = []

        def flush() -> None:
            if not ids:
                return
            windows.append(
                Window(
                    sentence_index,
                    [self.tok.cls_token_id, *ids, self.tok.sep_token_id],
                    [IGNORE_INDEX, *labs, IGNORE_INDEX],
                    [None, *widx, None],
                )
            )
            ids.clear()
            labs.clear()
            widx.clear()

        for i, word in enumerate(words):
            p = self.pieces(word)[:budget]  # a single absurdly long word is cut, never dropped
            if len(ids) + len(p) > budget:
                flush()
            ids.extend(p)
            first = IGNORE_INDEX if label_ids is None else label_ids[i]
            labs.extend([first] + [IGNORE_INDEX] * (len(p) - 1))
            widx.extend([i] + [None] * (len(p) - 1))
        flush()
        if not windows:  # empty sentence: still emit one window so indices stay aligned
            windows.append(
                Window(
                    sentence_index,
                    [self.tok.cls_token_id, self.tok.sep_token_id],
                    [IGNORE_INDEX, IGNORE_INDEX],
                    [None, None],
                )
            )
        return windows

    def encode_corpus_split(
        self, sentences: Sequence[Sentence], label_to_id: dict[str, int]
    ) -> list[Window]:
        out: list[Window] = []
        for i, s in enumerate(sentences):
            out.extend(self.encode(s.words, [label_to_id[t] for t in s.tags], i))
        return out


def decode_predictions(
    windows: Sequence[Window],
    window_preds: Sequence[Sequence[int]],
    sentence_lengths: Sequence[int],
    id_to_label: dict[int, str],
) -> list[list[str]]:
    """Map per-position predictions back to one tag per word, per sentence.

    ``window_preds[k]`` holds the argmax ids for window ``k`` (same length as its input_ids,
    padding already stripped). Words that were never seen default to ``O``.
    """
    out = [[OUTSIDE] * n for n in sentence_lengths]
    for win, preds in zip(windows, window_preds, strict=True):
        if len(preds) != len(win.input_ids):
            raise ValueError("prediction length does not match window length")
        for pos, widx in enumerate(win.word_index):
            if widx is not None:
                out[win.sentence_index][widx] = id_to_label[int(preds[pos])]
    return out


def gold_tags(sentences: Sequence[Sentence]) -> list[list[str]]:
    return [list(s.tags) for s in sentences]


# ---------------------------------------------------------------------- prediction dumps


def write_predictions(
    path: str | Path, sentences: Sequence[Sentence], pred_tags: Sequence[Sequence[str]]
) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for s, p in zip(sentences, pred_tags, strict=True):
            fh.write(json.dumps({"words": s.words, "gold": s.tags, "pred": p}) + "\n")


def read_predictions(path: str | Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]
