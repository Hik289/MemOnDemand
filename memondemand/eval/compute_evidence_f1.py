"""Score saved citations against evaluator-only expected document identifiers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compute_metrics(
    answers: list[dict[str, Any]], questions: list[dict[str, Any]]
) -> dict[str, int | float]:
    gold = {row["question_id"]: set(row.get("expected_doc_ids") or []) for row in questions}
    precisions, recalls, f1s = [], [], []
    skipped = 0
    for answer in answers:
        qid = answer.get("query_id") or answer.get("question_id")
        if qid not in gold:
            continue
        expected = gold[qid]
        if not expected:
            skipped += 1
            continue
        cited = set(answer.get("cited_evidence_ids") or answer.get("document_ids") or [])
        matched = len(cited & expected)
        precision = matched / len(cited) if cited else 0.0
        recall = matched / len(expected)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    size = len(f1s)
    return {
        "n_questions_scored": size,
        "n_skipped_no_gold_docs": skipped,
        "mean_precision": round(sum(precisions) / size, 4) if size else 0.0,
        "mean_recall": round(sum(recalls) / size, 4) if size else 0.0,
        "mean_f1": round(sum(f1s) / size, 4) if size else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path, help="saved runner or official-format answers JSONL")
    parser.add_argument("questions", type=Path, help="official questions JSONL with expected_doc_ids")
    args = parser.parse_args(argv)
    print(json.dumps(compute_metrics(load_jsonl(args.answers), load_jsonl(args.questions)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
