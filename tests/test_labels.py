import pytest

from bioner.labels import entity_types, iobes_spans, make_label_maps, split_tag, validate_iobes


def test_split_tag():
    assert split_tag("O") == ("O", "")
    assert split_tag("B-cell_line") == ("B", "cell_line")
    for bad in ["B", "X-GENE", "B-", "I_GENE", ""]:
        with pytest.raises(ValueError):
            split_tag(bad)


def test_label_maps_are_deterministic_with_o_first():
    l2i, i2l = make_label_maps(["S-B", "O", "B-A", "O", "E-A"])
    assert l2i == {"O": 0, "B-A": 1, "E-A": 2, "S-B": 3}
    assert all(i2l[i] == t for t, i in l2i.items())
    assert make_label_maps(["E-A", "B-A", "S-B"])[0] == {"O": 0, "B-A": 1, "E-A": 2, "S-B": 3}


def test_entity_types():
    assert entity_types(["O", "B-DNA", "E-DNA", "S-RNA"]) == ["DNA", "RNA"]


def test_spans_basic():
    tags = ["O", "B-X", "I-X", "E-X", "S-Y", "O", "B-Y", "E-Y"]
    assert iobes_spans(tags) == [("X", 1, 3), ("Y", 4, 4), ("Y", 6, 7)]


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["B-X", "O"], []),  # unterminated
        (["B-X", "I-Y", "E-Y"], []),  # type switch inside a chunk
        (["I-X", "E-X"], []),  # no opener
        (["E-X"], []),
        (["B-X", "B-X", "E-X"], [("X", 1, 2)]),  # restart: only the second chunk is well formed
    ],
)
def test_malformed_runs_are_not_spans(tags, expected):
    assert iobes_spans(tags) == expected


def test_validate_iobes():
    assert validate_iobes(["O", "B-X", "I-X", "E-X", "S-Y"]) == []
    assert validate_iobes(["B-X", "O"])
    assert validate_iobes(["I-X"])
    assert validate_iobes(["B-X"])
    assert validate_iobes(["B-X", "E-Y"])
