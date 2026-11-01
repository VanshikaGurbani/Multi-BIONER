import pytest

from bioner.metrics import entity_metrics, format_mean_std, mean_std, token_metrics


def test_perfect_prediction():
    gold = [["B-GENE", "E-GENE", "O", "S-GENE"]]
    m = entity_metrics(gold, gold)
    assert m["overall"]["precision"] == m["overall"]["recall"] == m["overall"]["f1"] == 1.0
    assert m["overall"]["support"] == 2


def test_hand_computed_precision_recall_f1():
    gold = [["B-X", "E-X", "O", "S-Y", "O"], ["S-X", "O", "O", "O", "O"]]
    pred = [["B-X", "E-X", "O", "S-X", "O"], ["S-X", "O", "S-Y", "O", "O"]]
    # gold entities: X(s1 0-1), Y(s1 3), X(s2 0) = 3
    # predicted: X(s1 0-1), X(s1 3), X(s2 0), Y(s2 2) = 4 ; correct: 2  ->  P = 2/4, R = 2/3
    m = entity_metrics(gold, pred)
    o = m["overall"]
    assert o["precision"] == pytest.approx(0.5)
    assert o["recall"] == pytest.approx(2 / 3)
    assert o["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert m["per_type"]["X"]["support"] == 2 and m["per_type"]["Y"]["support"] == 1
    assert m["per_type"]["Y"]["f1"] == 0.0


def test_boundary_error_gets_no_partial_credit():
    gold = [["B-X", "I-X", "E-X"]]
    pred = [["B-X", "E-X", "O"]]
    assert entity_metrics(gold, pred)["overall"]["f1"] == 0.0


def test_type_error_gets_no_credit():
    assert entity_metrics([["S-X"]], [["S-Y"]])["overall"]["f1"] == 0.0


def test_malformed_predicted_chunk_is_not_counted_in_strict_mode():
    gold = [["B-X", "E-X", "O"]]
    pred = [["B-X", "O", "O"]]  # dangling B-X
    o = entity_metrics(gold, pred)["overall"]
    assert o["recall"] == 0.0 and o["precision"] == 0.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_no_entities_does_not_crash():
    m = entity_metrics([["O", "O"]], [["O", "O"]])
    assert m["overall"]["f1"] == 0.0 and m["overall"]["support"] == 0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        entity_metrics([["O"]], [["O"], ["O"]])
    with pytest.raises(ValueError):
        entity_metrics([["O", "O"]], [["O"]])


def test_token_metrics_hand_computed():
    gold = [["O", "O", "S-X", "S-X"]]
    pred = [["O", "S-X", "S-X", "O"]]
    m = token_metrics(gold, pred)
    assert m["accuracy"] == pytest.approx(0.5)
    # O: tp1 fp1 fn1 -> F=.5 ; S-X: tp1 fp1 fn1 -> F=.5
    assert m["macro_f1"] == pytest.approx(0.5)
    assert m["per_tag"]["S-X"]["support"] == 2
    assert m["non_o_macro_f1"] == pytest.approx(0.5)


def test_token_metrics_macro_only_over_tags_present_in_gold():
    m = token_metrics([["O", "O"]], [["O", "S-Y"]])
    assert set(m["per_tag"]) == {"O", "S-Y"}
    assert m["macro_f1"] == pytest.approx(m["per_tag"]["O"]["f1"])


def test_mean_std_uses_sample_std():
    m, s = mean_std([1.0, 2.0, 3.0])
    assert m == 2.0 and s == pytest.approx(1.0)
    assert mean_std([5.0]) == (5.0, 0.0)
    with pytest.raises(ValueError):
        mean_std([])


def test_format_mean_std():
    assert format_mean_std([0.8, 0.9, 1.0]) == "90.00 ± 10.00"
    assert format_mean_std([0.5]) == "50.00 (n=1)"
