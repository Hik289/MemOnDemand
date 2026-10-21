"""Convert saved runner answers to the official evaluator's JSONL schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def convert_answers(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": row.get("query_id") or row.get("question_id"),
            "answer": row.get("answer_text") or row.get("answer") or "",
            "document_ids": row.get("cited_evidence_ids") or row.get("document_ids") or [],
        }
        for row in rows
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path, help="saved runner answers.jsonl")
    parser.add_argument("output", type=Path, help="official-format output JSONL")
    args = parser.parse_args(argv)
    if args.answers.resolve() == args.output.resolve():
        parser.error("input and output paths must differ")
    with args.answers.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    converted = convert_answers(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Converted {len(converted)} answers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
