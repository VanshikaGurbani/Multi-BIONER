import json

import pytest

from bioner import report
from bioner.report import load_runs, render_readme_table, render_results_md, update_readme


def block(f1, p=None, r=None, tok=0.5):
    return {
        "dev_entity": {"overall": {"f1": f1}},
        "test_entity": {
            "overall": {"precision": p or f1, "recall": r or f1, "f1": f1, "support": 10},
            "per_type": {"GENE": {"precision": f1, "recall": f1, "f1": f1, "support": 10}},
        },
        "test_token": {"macro_f1": tok, "accuracy": 0.9},
    }


def write(root, exp, seed, corpora, mode="stl", model="m1", name=None):
    path = root / exp / f"seed{seed}" / f"{name or next(iter(corpora))}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment": exp,
                "mode": mode,
                "model_name": model,
                "seed": seed,
                "config": {},
                "env": {},
                "corpora": corpora,
            }
        )
    )


@pytest.fixture()
def results(tmp_path):
    for seed, f in zip([1, 2, 3], [0.80, 0.82, 0.84], strict=True):
        write(tmp_path, "stl", seed, {"BC2GM": block(f)})
        write(tmp_path, "stl", seed, {"NCBI": block(f + 0.05)})
        write(
            tmp_path,
            "mtl",
            seed,
            {"BC2GM": block(f - 0.02), "NCBI": block(f + 0.05)},
            mode="mtl",
            name="all",
        )
    return tmp_path


def test_empty_results_say_so_and_invent_nothing(tmp_path):
    md = render_results_md(tmp_path)
    assert "No measured results" in md and "%" not in md.split("## Limitations")[0]
    assert "No measured results" in render_readme_table(tmp_path)


def test_mean_std_and_cells(results):
    runs, meta = load_runs(results)
    md = render_results_md(results)
    assert "| `stl` | 3 | 82.00 ± 2.00 | 87.00 ± 2.00 | 84.50 ± 2.00 |" in md
    assert meta["mtl"]["mode"] == "mtl"
    assert set(runs["stl"]) == {1, 2, 3}


def test_missing_cells_are_marked_not_filled(tmp_path):
    write(tmp_path, "stl", 1, {"BC2GM": block(0.8)})
    write(tmp_path, "mtl", 1, {"BC2GM": block(0.8), "NCBI": block(0.9)}, mode="mtl", name="all")
    md = render_results_md(tmp_path)
    assert report.NOT_RUN in md
    stl_row = next(line for line in md.splitlines() if line.startswith("| `stl` | 1 |"))
    assert report.NOT_RUN in stl_row


def test_paired_delta_flags_a_loss(results):
    md = render_results_md(results)
    # mtl - stl on BC2GM is -2.00 for every seed -> std 0 -> flagged as worse; NCBI 0 -> tie
    line = next(line for line in md.splitlines() if line.startswith("| `mtl` | -2.00"))
    assert "▼" in line


def test_delta_only_compares_same_encoder(tmp_path):
    write(tmp_path, "stl", 1, {"BC2GM": block(0.8)})
    write(tmp_path, "stl_other", 1, {"BC2GM": block(0.9)}, model="m2")
    assert "stl_other" not in report.delta_table(*load_runs(tmp_path))


def test_update_readme_replaces_only_between_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.README_START}\nold\n{report.README_END}\noutro\n")
    assert update_readme(readme, "NEW TABLE")
    text = readme.read_text()
    assert text == f"intro\n{report.README_START}\nNEW TABLE\n{report.README_END}\noutro\n"
    assert not update_readme(readme, "NEW TABLE")


def test_update_readme_requires_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers")
    with pytest.raises(ValueError):
        update_readme(readme, "x")


def test_legacy_table_is_labelled_token_level(results, tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"stl_macro_f1": {"A": 0.9}, "mtl_macro_f1": {"A": 0.8}}))
    md = render_results_md(results, legacy)
    assert "token level" in md and "not comparable" in md
