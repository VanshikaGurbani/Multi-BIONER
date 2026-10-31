"""Command line: ``python -m bioner <prepare-data|train|eval|report> ...``"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bioner.data import CORPORA, corpus_stats, load_corpus, prepare_corpus


def _cmd_prepare(args: argparse.Namespace) -> int:
    for name in args.corpora or CORPORA:
        out = prepare_corpus(name, args.data_dir, args.zip_dir, args.keep_zips)
        stats = corpus_stats(load_corpus(args.data_dir, name))
        print(f"{name}: {out}")
        for split, s in stats["splits"].items():
            print(
                f"  {split:6} sentences={s['sentences']:6} tokens={s['tokens']:8} "
                f"entities={s['entities']:6} malformed={s['malformed_sentences']}"
            )
        if any(stats["sentences_rejoined"].values()):
            print(f"  sentence cuts inside entities re-joined: {stats['sentences_rejoined']}")
        print(f"  test sentences also in train: {stats.get('test_sentences_also_in_train')}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from bioner.config import load_config
    from bioner.train import get_device, run_experiment

    cfg = load_config(args.config)
    device = get_device(args.device)
    print(f"experiment={cfg.name} device={device}")
    paths = run_experiment(
        cfg,
        args.data_dir,
        args.out_dir,
        device,
        seeds=args.seeds,
        corpora=args.corpora,
        skip_existing=not args.no_skip,
    )
    print("\n".join(f"result: {p}" for p in paths))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from bioner.train import evaluate_checkpoint, get_device

    res = evaluate_checkpoint(
        args.run_dir,
        args.stem or args.corpus,
        args.corpus,
        args.split,
        args.data_dir,
        get_device(args.device),
    )
    o = res["entity"]["overall"]
    print(json.dumps(res["entity"]["per_type"], indent=2))
    print(f"entity P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")
    print(f"(token macro-F1 incl. O, legacy metric: {res['token']['macro_f1']:.4f})")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from bioner.report import render_readme_table, render_results_md, update_readme

    legacy = Path(args.legacy)
    md = render_results_md(args.results_dir, legacy if legacy.exists() else None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    if args.readme:
        changed = update_readme(args.readme, render_readme_table(args.results_dir))
        print(f"{'updated' if changed else 'unchanged'} {args.readme}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bioner", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare-data", help="download, verify and extract the corpora")
    s.add_argument("--data-dir", default="data")
    s.add_argument("--corpora", nargs="+", choices=sorted(CORPORA))
    s.add_argument("--zip-dir", help="use zips from this folder instead of downloading")
    s.add_argument("--keep-zips", action="store_true", help="keep downloaded zips in data/_zips")
    s.set_defaults(func=_cmd_prepare)

    s = sub.add_parser("train", help="train (and test-evaluate) an experiment from a YAML config")
    s.add_argument("--config", required=True)
    s.add_argument("--data-dir", default="data")
    s.add_argument("--out-dir", default="results")
    s.add_argument("--seeds", nargs="+", type=int, help="override the config's seeds")
    s.add_argument("--corpora", nargs="+", choices=sorted(CORPORA), help="subset of the config's")
    s.add_argument("--device", default="auto", help="auto | cpu | cuda")
    s.add_argument("--no-skip", action="store_true", help="redo results that already exist")
    s.set_defaults(func=_cmd_train)

    s = sub.add_parser("eval", help="re-score a saved checkpoint")
    s.add_argument("--run-dir", required=True, help="results/<experiment>/seed<N>")
    s.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    s.add_argument("--stem", help="checkpoint name; default = corpus (use 'all' for MTL)")
    s.add_argument("--split", default="devel", choices=["devel", "test"])
    s.add_argument("--data-dir", default="data")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=_cmd_eval)

    s = sub.add_parser("report", help="build docs/RESULTS.md and the README table from results/")
    s.add_argument("--results-dir", default="results")
    s.add_argument("--out", default="docs/RESULTS.md")
    s.add_argument("--readme", default="README.md", help="'' to skip")
    s.add_argument("--legacy", default="docs/legacy_results.json")
    s.set_defaults(func=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
