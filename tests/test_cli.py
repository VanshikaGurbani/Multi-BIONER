import pytest

from bioner.cli import build_parser, main


def test_parser_knows_all_commands():
    p = build_parser()
    for cmd in (
        ["prepare-data"],
        ["train", "--config", "c.yaml"],
        ["eval", "--run-dir", "r", "--corpus", "NCBI"],
        ["report"],
    ):
        assert p.parse_args(cmd).func


def test_unknown_corpus_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--config", "c.yaml", "--corpora", "NOPE"])


def test_report_with_no_results_writes_an_honest_placeholder(tmp_path, capsys):
    readme = tmp_path / "README.md"
    readme.write_text("<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n")
    out = tmp_path / "RESULTS.md"
    rc = main(
        [
            "report",
            "--results-dir",
            str(tmp_path / "empty"),
            "--out",
            str(out),
            "--readme",
            str(readme),
            "--legacy",
            str(tmp_path / "none.json"),
        ]
    )
    assert rc == 0
    assert "No measured results" in out.read_text()
    assert "No measured results" in readme.read_text() and "old" not in readme.read_text()
