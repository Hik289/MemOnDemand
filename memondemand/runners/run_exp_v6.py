"""Compatibility entry point for V6 invocations using the current retrieval pipeline."""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from memondemand.runners import run_stream_v5 as _base
from memondemand.runners.prompts import load_prompt


ANSWER_SYSTEM_V6 = load_prompt("answer_v2.txt")
ANSWER_USER_V6 = _base.ANSWER_USER_TMPL


def format_dual_context(ctx_items: List[Tuple[str, str, str]]) -> str:
    return _base._format_dual_context(ctx_items)


def parse_cited_v6(answer_text: str, candidate_ids: List[str]) -> List[str]:
    candidates = set(candidate_ids)
    result: List[str] = []
    for match in re.findall(r"CITED\s*:\s*([^\n]*)", answer_text or ""):
        for token in match.split(","):
            token = token.strip()
            if token.startswith("[EVID|") and token.endswith("]"):
                token = token[6:-1].strip()
            if token in candidates and token not in result:
                result.append(token)
    return result


def run_query_v6(
    ctx: _base.RunnerCtx,
    qid: str,
    query: str,
    query_idx: int,
    max_nav_steps: int = 3,
) -> _base.QueryRecord:
    current = replace(ctx, max_steps=max_nav_steps)
    return _base.run_llm_nav(
        current, qid, query, query_idx,
        use_promotion=current.promotion is not None,
    )


def _environment(name: str) -> Optional[str]:
    return os.environ.get(f"MEMONDEMAND_{name}") or os.environ.get(f"MEMLOOP_{name}")


def _hierarchy_path(tier: str, parser: argparse.ArgumentParser) -> str:
    root = Path(_environment("REPO_ROOT") or Path(__file__).resolve().parents[2])
    tier_name = tier if tier.startswith("erag_") else f"erag_{tier}"
    directory = root / "results/v5_emb_large_llm_distill" / tier_name / "hierarchy"
    for filename in (
        "hierarchy_L1L2_REBUILT_gpt55_nocap.json",
        "hierarchy_v4l1kf_merged_l1l2gpt54_l1l2distillgpt54.json",
        "hierarchy_v4l1kf.json",
        "hierarchy.json",
    ):
        candidate = directory / filename
        if candidate.is_file():
            return str(candidate)
    parser.error(
        f"No hierarchy found for {tier!r}; pass --hierarchy or set "
        "MEMONDEMAND_HIERARCHY to an existing hierarchy."
    )
    raise AssertionError("argparse.error must exit")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="erag_10M")
    parser.add_argument("--hierarchy", default=_environment("HIERARCHY"))
    parser.add_argument("--queries", default=_environment("QUERIES"))
    parser.add_argument("--out_dir", "--out", dest="out_dir", required=True)
    parser.add_argument("--max_queries", "--n_smoke", dest="max_queries", type=int, default=50)
    parser.add_argument("--method", default="V5", choices=sorted(_base.VALID_METHODS))
    parser.add_argument("--resume", action="store_true")
    args, remaining = parser.parse_known_args(argv)

    hierarchy = args.hierarchy or _hierarchy_path(args.tier, parser)
    root = Path(_environment("REPO_ROOT") or Path(__file__).resolve().parents[2])
    queries = args.queries or str(root / "manifests/erag_queries.parquet")
    forwarded = [
        "--method", args.method,
        "--hierarchy", str(hierarchy),
        "--queries", str(queries),
        "--out", args.out_dir,
        "--n_smoke", str(args.max_queries),
    ]
    if args.resume:
        forwarded.append("--resume")
    forwarded.extend(remaining)

    original_argv = sys.argv[:]
    try:
        sys.argv = ["memondemand run-v6", *forwarded]
        result = _base.main()
        return int(result) if result is not None else 0
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
