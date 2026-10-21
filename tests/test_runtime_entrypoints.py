from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from memondemand.runners import run_exp_v6
from memondemand.runners import run_stream_v5 as runner
from memondemand.runners.prompts import load_prompt


EXPANDED = "memondemand.pipelines.enterprise_rag_1_14b.answer_generation"


@pytest.mark.parametrize("name", ["run_stream_v5", "run_stream_v5_force_answer_patched"])
def test_expanded_module_patches_reach_shared_runner(monkeypatch, name):
    monkeypatch.setenv("MEMONDEMAND_FORCE_ANSWER", "0")
    expanded = importlib.import_module(f"{EXPANDED}.{name}")
    assert expanded is runner
    monkeypatch.setattr(expanded, "ANSWER_SYSTEM", "caller-supplied prompt")
    assert runner.ANSWER_SYSTEM == "caller-supplied prompt"


@pytest.mark.parametrize("name", ["run_stream_v5_truncate_patch", "run_empty14_force_answer_wrapper"])
def test_compatibility_wrappers_do_not_replace_shared_context_policy(monkeypatch, name):
    monkeypatch.setenv("MEMONDEMAND_FORCE_ANSWER", "0")
    original = runner._node_text_for_answer
    wrapper = importlib.import_module(f"{EXPANDED}.{name}")
    assert wrapper.R is runner
    assert runner._node_text_for_answer is original


def test_documented_answer_override_uses_current_prompt():
    override = importlib.import_module(f"{EXPANDED}.improved_answer_prompt")
    assert override.ANSWER_SYSTEM == load_prompt("answer_v2.txt")


@pytest.mark.parametrize("prefix", ["MEMONDEMAND", "MEMLOOP"])
def test_run_v6_delegates_legacy_arguments_to_current_runner(monkeypatch, tmp_path, prefix):
    for name in ("MEMONDEMAND_HIERARCHY", "MEMLOOP_HIERARCHY", "MEMONDEMAND_QUERIES", "MEMLOOP_QUERIES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(f"{prefix}_HIERARCHY", str(tmp_path / "hierarchy.json"))
    monkeypatch.setenv(f"{prefix}_QUERIES", str(tmp_path / "queries.parquet"))
    captured = []

    def fake_main():
        captured.extend(sys.argv[1:])
        return 7

    monkeypatch.setattr(runner, "main", fake_main)
    before = sys.argv[:]
    assert run_exp_v6.main([
        "--tier", "erag_FULL", "--out_dir", str(tmp_path / "answers"),
        "--max_queries", "11", "--resume", "--top_k_distilled", "23",
    ]) == 7
    assert captured[captured.index("--hierarchy") + 1] == str(tmp_path / "hierarchy.json")
    assert captured[captured.index("--queries") + 1] == str(tmp_path / "queries.parquet")
    assert captured[captured.index("--out") + 1] == str(tmp_path / "answers")
    assert captured[captured.index("--n_smoke") + 1] == "11"
    assert captured[captured.index("--top_k_distilled") + 1] == "23"
    assert "--resume" in captured
    assert sys.argv == before


def test_run_v6_accepts_explicit_paths_and_discovers_existing_tier(monkeypatch, tmp_path):
    for prefix in ("MEMONDEMAND", "MEMLOOP"):
        monkeypatch.delenv(f"{prefix}_HIERARCHY", raising=False)
    monkeypatch.setenv("MEMONDEMAND_REPO_ROOT", str(tmp_path))
    hierarchy = tmp_path / "results/v5_emb_large_llm_distill/erag_10M/hierarchy/hierarchy.json"
    hierarchy.parent.mkdir(parents=True)
    hierarchy.write_text("")
    calls = []
    monkeypatch.setattr(runner, "main", lambda: calls.append(sys.argv[:]) or 0)
    assert run_exp_v6.main(["--out_dir", str(tmp_path / "out")]) == 0
    assert calls[-1][calls[-1].index("--hierarchy") + 1] == str(hierarchy)
    explicit = tmp_path / "chosen.json"
    assert run_exp_v6.main([
        "--out_dir", str(tmp_path / "out"), "--hierarchy", str(explicit),
        "--queries", "chosen.parquet",
    ]) == 0
    assert calls[-1][calls[-1].index("--hierarchy") + 1] == str(explicit)
    assert calls[-1][calls[-1].index("--queries") + 1] == "chosen.parquet"


def test_run_v6_missing_hierarchy_explains_required_input(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MEMONDEMAND_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("MEMONDEMAND_HIERARCHY", raising=False)
    monkeypatch.delenv("MEMLOOP_HIERARCHY", raising=False)
    with pytest.raises(SystemExit) as exc:
        run_exp_v6.main(["--out_dir", str(tmp_path / "out")])
    assert exc.value.code == 2
    assert "--hierarchy" in capsys.readouterr().err


def test_run_query_v6_uses_shared_navigation_without_mutating_context(monkeypatch):
    @dataclass
    class Context:
        max_steps: int
        promotion: object

    original = Context(max_steps=12, promotion=object())
    captured = {}
    result = object()

    def fake_navigation(ctx, qid, query, query_idx, *, use_promotion):
        captured.update(ctx=ctx, qid=qid, query=query, query_idx=query_idx, promotion=use_promotion)
        return result

    monkeypatch.setattr(runner, "run_llm_nav", fake_navigation)
    assert run_exp_v6.run_query_v6(original, "q1", "question", 4, max_nav_steps=3) is result
    assert captured["ctx"].max_steps == 3
    assert captured["promotion"] is True
    assert (captured["qid"], captured["query_idx"]) == ("q1", 4)
    assert original.max_steps == 12
    assert run_exp_v6.parse_cited_v6("Answer\nCITED: [EVID|doc-1],doc-1,unknown", ["doc-1"]) == ["doc-1"]


@pytest.mark.parametrize("module", [
    "memondemand.runners.run_stream_v5",
    "memondemand.runners.run_exp_v6",
    f"{EXPANDED}.run_stream_v5",
    f"{EXPANDED}.run_stream_v5_force_answer_patched",
    f"{EXPANDED}.run_stream_v5_truncate_patch",
    f"{EXPANDED}.run_empty14_force_answer_wrapper",
])
def test_runtime_help_requires_no_model_or_api(module):
    program = """
import runpy
import sys
from memondemand.runners import run_stream_v5 as runner

def fail(*args, **kwargs):
    raise AssertionError("Help attempted to initialize a model or call an API")

runner.Embedder = fail
runner.api_call = fail
sys.argv = [sys.argv[1], "--help"]
runpy.run_module(sys.argv[0], run_name="__main__")
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, module],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--hierarchy" in completed.stdout
