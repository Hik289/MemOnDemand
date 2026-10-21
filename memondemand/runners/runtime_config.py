"""Portable runtime settings for the latest semantic-navigation pipeline."""
from __future__ import annotations

import ast
import os
import runpy
from pathlib import Path


REFERENCE_DEFAULTS = {
    "L0_RETRIEVAL": "bm25",
    "MAX_NAV_STEPS": "3",
    "SKIP_L0_EMBED": "1",
    "ANSWER_MODE": "detailed_full",
    "ANSWER_PROMPT": "v2",
    "CONTEXT_FORMAT": "dual",
    "CLUSTER_FILTER": "1",
    "BM25_WIDE_K": "50",
    "PROMOTE_RELEVANCE_FLOOR": "0.0",
}


def configure_runtime() -> None:
    """Keep explicit settings; accept the release's historical env prefix."""
    for key, value in list(os.environ.items()):
        if key.startswith("MEMLOOP_"):
            os.environ.setdefault("MEMONDEMAND_" + key[len("MEMLOOP_"):], value)
    for suffix, value in REFERENCE_DEFAULTS.items():
        os.environ.setdefault("MEMONDEMAND_" + suffix, value)


def resolve_prompt(variable: str, constant: str, default: str) -> str:
    """Read a text prompt or a legacy Python module exporting the constant."""
    filename = os.environ.get(variable, "").strip()
    if not filename:
        return default
    path = Path(filename).expanduser()
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError:
            pass  # The original release stored plain-text prompts as .py files.
        else:
            text = runpy.run_path(str(path)).get(constant)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Prompt file {path} must provide nonempty {constant} text")
    return text
