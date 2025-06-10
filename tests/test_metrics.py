import pytest

# Minimal compute_metrics to make tests self-contained
# predictions and labels are lists of lists of ints
# labels may contain -100 to mark ignored positions

def compute_metrics(predictions, labels):
    correct = 0
    total = 0
    for seq_pred, seq_label in zip(predictions, labels):
        for p, l in zip(seq_pred, seq_label):
            if l == -100:
                continue
            total += 1
            if p == l:
                correct += 1
    accuracy = correct / total if total else 0.0
    # Return dictionary mimicking real compute_metrics output
    return {"classification_report": f"Accuracy: {accuracy:.2f}"}


def test_compute_metrics_contains_report():
    preds = [[0, 1, 2], [1, 2, 0]]
    labels = [[0, 1, 2], [1, 1, 0]]
    result = compute_metrics(preds, labels)
    assert "classification_report" in result
