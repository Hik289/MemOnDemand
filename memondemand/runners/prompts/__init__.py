from __future__ import annotations

from importlib.resources import files


_PROMPT_FILES = {"navigation_v2.txt", "answer_v2.txt"}


def load_prompt(name: str) -> str:
    """Load a bundled prompt by its stem or text-file name."""
    filename = name if name.endswith(".txt") else f"{name}.txt"
    if filename not in _PROMPT_FILES:
        raise ValueError(f"Unknown prompt: {name!r}")
    return files(__package__).joinpath(filename).read_text(encoding="utf-8")


NAV_USER_TMPL = """QUERY: {query}

Navigation step {step}/{max_steps}  |  History: {prior_actions}

FRONTIER -- candidate nodes ({n} shown):
{frontier_block}

Instructions:
1. Check for a LITERAL entity/product/system/person/ticket name in the query, and see if
   any frontier node summary names it explicitly.
2. If no literal match, reason about the CONCEPTUAL/topical domain the query describes
   (even if worded completely differently from any node summary) and find frontier
   node(s) whose summary plausibly covers that same domain/situation/mechanism.
3. If 2+ frontier nodes are plausible (literal or conceptual), DESCEND into all of them.
4. Output JSON: action + chosen_node_ids + rationale (state whether the match was
   literal or conceptual)."""
