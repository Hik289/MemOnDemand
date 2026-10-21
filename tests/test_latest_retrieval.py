from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from memondemand.methods.dual_node import DualNode
from memondemand.methods import promotion_controller as promotion
from memondemand.runners import run_stream_v5 as runner
from memondemand.runners.prompts import load_prompt
from memondemand.runners.runtime_config import configure_runtime, resolve_prompt


class OfflineEmbedder:
    dim = 2
    backend = "offline-test"

    def encode(self, texts, **kwargs):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def node(name, text, level="L0", children=()):
    return DualNode(
        node_id=name, level=level, distilled_text=text, detailed_text=text,
        distilled_tokens=5, detailed_tokens=10, source_evidence_ids=[name],
        extra={"children_ids": list(children)},
    )


@pytest.fixture
def runtime(monkeypatch):
    settings = {
        "L0_RETRIEVAL": "bm25", "SKIP_L0_EMBED": "1", "CLUSTER_FILTER": "1",
        "BM25_WIDE_K": "50", "ANSWER_MODE": "detailed_full", "CONTEXT_FORMAT": "dual", "ANSWER_PROMPT": "v2",
        "PROMOTE_RELEVANCE_FLOOR": "0", "MAX_NAV_STEPS": "3",
        "DENSE_RERANK": "0", "RRF": "0", "TE3_RRF": "0", "QUERY_EXPAND": "0",
        "V6_DUAL_MEM": "0", "V7_ENTITY": "0", "V61": "0",
    }
    for suffix, value in settings.items():
        monkeypatch.setenv("MEMONDEMAND_" + suffix, value)
    for suffix in ("NAV_SYSTEM_OVERRIDE_FILE", "NAV_USER_OVERRIDE_FILE", "ANSWER_SYSTEM_OVERRIDE_FILE"):
        for prefix in ("MEMONDEMAND_", "MEMLOOP_"):
            monkeypatch.delenv(prefix + suffix, raising=False)
    monkeypatch.setattr(runner, "_BM25_CACHE", {"index": None, "doc_ids": None})


def test_release_environment_aliases_keep_explicit_public_settings(monkeypatch):
    monkeypatch.delenv("MEMONDEMAND_PROMOTE_RELEVANCE_FLOOR", raising=False)
    monkeypatch.setenv("MEMLOOP_PROMOTE_RELEVANCE_FLOOR", "0.7")
    monkeypatch.setenv("MEMONDEMAND_CLUSTER_FILTER", "0")
    configure_runtime()
    import os
    assert os.environ["MEMONDEMAND_PROMOTE_RELEVANCE_FLOOR"] == "0.7"
    assert os.environ["MEMONDEMAND_CLUSTER_FILTER"] == "0"


@pytest.mark.parametrize("filename,content", [
    ("prompt.txt", "Ground answers in evidence."),
    ("release_prompt.py", "Ground answers in evidence."),
    ("module.py", 'ANSWER_SYSTEM = "Ground answers in evidence."'),
])
def test_prompt_override_accepts_release_text_and_existing_modules(monkeypatch, tmp_path, filename, content):
    path = tmp_path / filename
    path.write_text(content)
    monkeypatch.setenv("MEMONDEMAND_ANSWER_SYSTEM_OVERRIDE_FILE", str(path))
    assert resolve_prompt("MEMONDEMAND_ANSWER_SYSTEM_OVERRIDE_FILE", "ANSWER_SYSTEM", "fallback") == "Ground answers in evidence."


@pytest.mark.parametrize("retrieval", ["bm25", "bm25_promo"])
def test_cluster_rerank_keeps_global_fallback_candidates(runtime, monkeypatch, retrieval):
    monkeypatch.setenv("MEMONDEMAND_L0_RETRIEVAL", retrieval)
    hierarchy = {
        "good": node("good", "precision safety threshold"),
        "bad": node("bad", "safety safety safety safety threshold"),
        "other": node("other", "employee travel policy"),
        "extra": node("extra", "expense report approval"),
        "spare": node("spare", "customer service handbook"),
        "topic": node("topic", "precision", "L1", ["good"]),
    }
    ctx = SimpleNamespace(hierarchy=hierarchy, parent_child={"topic": ["good"]})
    global_hits = runner._bm25_top_k(ctx, "safety", 4)
    assert global_hits[0][0] == "bad"
    hits = runner.final_l0_retrieval(ctx, np.array([1, 0]), ["topic"], 4, "safety")
    assert hits[0][0] == "good"
    assert {name for name, _ in hits} == {name for name, _ in global_hits}


@pytest.mark.parametrize("method,retrieval,rerank", [
    ("B_flat", "bm25", "0"), ("B_fixed_hier", "bm25", "0"),
    ("B_dynamic_hier", "bm25", "0"), ("V5", "dense", "0"),
    ("V5", "bm25", "1"),
])
def test_bm25_defaults_do_not_disable_dense_retrieval_paths(runtime, monkeypatch, method, retrieval, rerank):
    monkeypatch.setenv("MEMONDEMAND_L0_RETRIEVAL", retrieval)
    monkeypatch.setenv("MEMONDEMAND_DENSE_RERANK", rerank)
    ctx = SimpleNamespace(method=method, hierarchy={"doc": node("doc", "source")},
                          by_level={"L0": ["doc"]}, indexes={}, embedder=OfflineEmbedder())
    assert runner.get_index(ctx, "L0").search(np.array([1, 0]), top_k=1)[0][0] == "doc"


@pytest.mark.parametrize("mode", ["deterministic", "llm", "failed_llm"])
def test_promotion_floor_applies_to_every_decision_path(runtime, monkeypatch, mode):
    monkeypatch.setenv("MEMONDEMAND_PROMOTE_RELEVANCE_FLOOR", "0.9")

    class Embeddings:
        def encode(self, texts, **kwargs):
            vectors = {"query": [1, 0], "high": [1, 0], "medium": [0.6, 0.8]}
            return np.asarray([vectors[text] for text in texts], dtype=np.float32)

    def api(*args, **kwargs):
        if mode == "failed_llm":
            raise RuntimeError("offline failure")
        return {"text": json.dumps({"decisions": [
            {"node_id": name, "decision": "PROMOTE"} for name in ("high", "medium")
        ]}), "usage": {}}

    monkeypatch.setattr(promotion, "api_call", api)
    controller = promotion.PromotionController(
        {name: node(name, name) for name in ("high", "medium")}, Embeddings(),
        deterministic_only=mode == "deterministic",
    )
    decisions = controller.decide("query", ["high", "medium"], 0)
    assert {item.node_id: item.decision for item in decisions} == {
        "high": "PROMOTE", "medium": "KEEP_LIGHT",
    }


@pytest.mark.parametrize("legacy_flag", [None, "V6_DUAL_MEM", "V7_ENTITY", "V61"])
def test_reference_pipeline_navigates_answers_logs_and_resumes_offline(runtime, monkeypatch, tmp_path, legacy_flag):
    if legacy_flag:
        monkeypatch.setenv("MEMONDEMAND_" + legacy_flag, "1")
    nodes = [
        node("root", "precision safety", "L2", ["topic"]),
        node("topic", "precision safety", "L1", ["good"]),
        node("good", "The precision safety threshold is 12."),
        node("bad", "safety safety safety unrelated reference"),
        node("other", "employee travel policy"),
        node("extra", "expense report approval"),
    ]
    hierarchy = tmp_path / "hierarchy.json"
    hierarchy.write_text("".join(json.dumps(n.to_dict()) + "\n" for n in nodes))
    queries = tmp_path / "queries.parquet"
    pd.DataFrame([{"query_id": "q1", "query_text": "What is the safety threshold?"}]).to_parquet(queries)
    calls = []
    navigation = iter([
        {"action": "DESCEND", "chosen_node_ids": ["root"]},
        {"action": "ANSWER", "chosen_node_ids": ["topic"]},
    ])

    def api(alias, messages, **kwargs):
        system, user = messages[0]["content"], messages[1]["content"]
        calls.append(system)
        if system == promotion.PROMOTION_SYSTEM_PROMPT:
            text = json.dumps({"decisions": [
                {"node_id": name, "decision": "PROMOTE"}
                for name in ("good", "bad", "other", "extra")
            ]})
        elif system.startswith("You navigate"):
            assert "CONCEPTUAL" in system and "CONCEPTUAL" in user
            assert "{max_steps}" not in system
            text = json.dumps(next(navigation))
        else:
            assert system == load_prompt("answer_v2")
            assert "[EVID|good] [FULL]" in user
            assert "[CTX|topic]" in user
            text = "The precision safety threshold is 12.\nCITED: good"
        return {"text": text, "usage": {"input_tokens": 11, "output_tokens": 13}}

    monkeypatch.setattr(runner, "Embedder", OfflineEmbedder)
    monkeypatch.setattr(runner, "api_call", api)
    monkeypatch.setattr(promotion, "api_call", api)
    out = tmp_path / "run"
    arguments = ["memondemand run", "--method", "V5", "--hierarchy", str(hierarchy),
                 "--queries", str(queries), "--out", str(out), "--promotion_budget", "1"]
    monkeypatch.setattr(sys, "argv", arguments)
    assert runner.main() == 0
    records = [json.loads(line) for line in (out / "answers.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["final_action"] == "ANSWER"
    assert records[0]["cited_evidence_ids"] == ["good"]
    assert records[0]["n_navigation_steps"] == 2
    assert records[0]["n_promote_events"] == 4
    assert records[0]["n_demote_events"] == 3
    events = [json.loads(line) for line in (out / "state_log.ndjson").read_text().splitlines()]
    assert sum(e["event"] == "PROMOTE" for e in events) == 4
    assert sum(e["event"] == "DEMOTE" for e in events) == 3
    assert json.loads((out / "token_ledger.json").read_text())["n_records"] == 4
    monkeypatch.setattr(sys, "argv", arguments + ["--resume"])
    assert runner.main() == 0
    assert len(calls) == 4
    assert len((out / "answers.jsonl").read_text().splitlines()) == 1
