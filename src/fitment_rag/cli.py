"""fitment-rag CLI -- for sweeps and CI. Day-to-day work happens in notebooks/.

    fitment-rag doctor
    fitment-rag build-evalset --config configs/phase1_smoke.yaml -n 200
    fitment-rag run     --config configs/phase1_smoke.yaml
    fitment-rag sweep   --configs "configs/phase1/emb/*.yaml"
    fitment-rag compare --sort recall@5

Phase 1 is retrieval only -- there is no generation step. See context/02-plan.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from .config import RunConfig
from .report import leaderboard


def cmd_doctor(_args) -> int:
    print(f"{'python':22}{sys.version.split()[0]}")
    missing = []
    for mod in ("torch", "sentence_transformers", "faiss", "rank_bm25"):
        try:
            __import__(mod)
            print(f"{mod:22}ok")
        except ImportError:
            print(f"{mod:22}MISSING")
            missing.append(mod)

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{'device':22}{device}")
    except ImportError:
        pass

    if missing:
        print('\n-> poetry install --extras "nb dev"')
    return 1 if missing else 0


def cmd_build_evalset(args) -> int:
    from .data.amazon import load_documents
    from .evalset.build import build_eval_set

    cfg = RunConfig.load(args.config)
    out = Path(args.out or cfg.eval.eval_set)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[2] / out

    items = build_eval_set(load_documents(cfg.data), out, n_questions=args.n, seed=args.seed)
    print(f"wrote {len(items)} questions -> {out}")
    return 0


def cmd_run(args) -> int:
    from .pipeline import run

    cfg = RunConfig.load(args.config)
    if args.limit:
        cfg.eval.limit = args.limit
    art = run(cfg)
    print(json.dumps(
        {k: v for k, v in art.metrics.items() if k.startswith(("recall@", "mrr", "ndcg@", "hit@"))},
        indent=2,
    ))
    return 0


def cmd_sweep(args) -> int:
    from .pipeline import run

    paths = sorted(glob.glob(args.configs))
    if not paths:
        print(f"no configs matched: {args.configs}")
        return 1

    for i, p in enumerate(paths, 1):
        cfg = RunConfig.load(p)
        print(f"\n===== [{i}/{len(paths)}] {cfg.name} =====")
        try:
            run(cfg)
        except Exception as exc:
            print(f"FAILED: {exc}")

    print()
    return cmd_compare(argparse.Namespace(sort=args.sort, csv=None))


def cmd_compare(args) -> int:
    df = leaderboard(sort=args.sort)
    if df.empty:
        print("no results yet -- run `fitment-rag run --config configs/phase1_smoke.yaml`")
        return 0
    print(df.to_markdown(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fitment-rag", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor").set_defaults(func=cmd_doctor)

    p = sub.add_parser("build-evalset")
    p.add_argument("--config", required=True)
    p.add_argument("--out")
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=17)
    p.set_defaults(func=cmd_build_evalset)

    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep")
    p.add_argument("--configs", required=True)
    p.add_argument("--sort", default="recall@5")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("compare")
    p.add_argument("--sort", default="recall@5")
    p.add_argument("--csv")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
