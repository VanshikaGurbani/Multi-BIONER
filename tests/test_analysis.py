from bioner.analysis import analyse, classify_sentence, render_error_analysis
from bioner.data import Sentence, write_predictions


def bucket(gold, pred):
    return [b for _, b, _ in classify_sentence(gold, pred)["gold"]]


def test_each_bucket():
    assert bucket(["S-X"], ["S-X"]) == ["correct"]
    assert bucket(["S-X"], ["S-Y"]) == ["type_error"]
    assert bucket(["B-X", "E-X", "O"], ["B-X", "I-X", "E-X"]) == ["boundary_error"]
    assert bucket(["B-X", "E-X", "O"], ["O", "S-Y", "O"]) == ["both_error"]
    assert bucket(["S-X", "O"], ["O", "S-X"]) == ["missed"]


def test_spurious_and_confusions():
    res = classify_sentence(["S-X", "O", "O"], ["S-Y", "O", "S-Y"])
    assert res["spurious"] == [("Y", 2, 2)]
    assert res["type_confusions"] == [("X", "Y")]


def test_malformed_predictions_are_not_entities():
    assert bucket(["S-X"], ["B-X"]) == ["missed"]


def test_analyse_aggregates():
    rows = [
        {"gold": ["S-X", "O"], "pred": ["S-X", "O"]},
        {"gold": ["S-X", "O"], "pred": ["O", "O"]},
        {"gold": ["B-Y", "E-Y"], "pred": ["B-Y", "E-Y"]},
    ]
    r = analyse(rows)
    assert r["by_type"]["X"] == {"correct": 1, "missed": 1}
    assert r["by_type"]["Y"] == {"correct": 1}
    assert r["by_length"]["1"] == {"correct": 1, "missed": 1}


def test_render_reads_saved_predictions(tmp_path):
    d = tmp_path / "exp" / "seed1"
    d.mkdir(parents=True)
    write_predictions(
        d / "predictions_NCBI.jsonl.gz",
        [Sentence(("a", "b"), ("S-Disease", "O"))],
        [["S-Disease", "O"]],
    )
    md = render_error_analysis(tmp_path, "exp")
    assert "### NCBI" in md and "Disease" in md
    assert "No saved predictions" in render_error_analysis(tmp_path, "nope")
