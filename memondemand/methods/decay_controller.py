"""V4 Decay Controller.

Implements the LIGHT/PROMOTED state-decay rules described in README2.md §3.4::

    utility_t   = detail_use_count / max(1, promotion_count)
    recency_t   = exp(-(current_query_idx - last_used_query_idx) / tau)
    keep_score  = utility_t * recency_t

    if promoted_node_count > promotion_budget:
        demote lowest keep_score nodes (count = overflow)
    if current_query_idx - last_used_query_idx >= decay_window:
        demote node

§11 fairness:
- This controller reads ONLY method-observable signals: per-node counters
  that record what the *runner* did with the node. No gold answer, gold
  evidence, or future-query information is used. The current_query_idx is
  the index of the query that JUST finished — the controller never peeks
  forward.

Checkpoint semantics:
- The DecayController itself is stateless w.r.t. node identity (all node
  state lives on the DualNode objects in `hierarchy_dict`). `save_state` /
  `load_state` therefore only persist the controller's hyper-params + a
  small bookkeeping counter so it can be paired with a hierarchy snapshot.
"""
from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from memondemand.methods.dual_node import (
    DualNode,
    NODE_STATE_LIGHT,
    NODE_STATE_PROMOTED,
)
from memondemand.methods.state_log import (
    EVENT_DEMOTE,
    EVENT_KEEP_PROMOTED,
    StateLog,
    StateLogEntry,
)


# ---------------------------------------------------------------------------
# Demotion-reason constants (for state-log rationale tagging)
# ---------------------------------------------------------------------------

REASON_BUDGET_OVERFLOW = "budget_overflow"
REASON_DECAY_WINDOW = "decay_window"
REASON_KEEP = "keep"


@dataclass
class DecayResult:
    demoted_ids: List[str]
    kept_ids: List[str]
    reasons: Dict[str, str] = field(default_factory=dict)
    keep_scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class DecayController:
    """Drives §3.4 decay cycle once per query."""

    def __init__(
        self,
        *,
        promotion_budget: int = 200,
        decay_window: int = 50,
        tau: float = 30.0,
    ):
        if promotion_budget <= 0:
            raise ValueError("promotion_budget must be > 0")
        if decay_window <= 0:
            raise ValueError("decay_window must be > 0")
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.promotion_budget = int(promotion_budget)
        self.decay_window = int(decay_window)
        self.tau = float(tau)
        # Bookkeeping: number of decay cycles executed (purely informational).
        self.n_cycles = 0

    # ----- core math -----

    def compute_keep_score(self, node: DualNode, current_query_idx: int) -> float:
        if node.state != NODE_STATE_PROMOTED:
            return 0.0
        utility = node.detail_use_count / max(1, node.promotion_count)
        # If the node has never been used at all, anchor recency to the last
        # promotion event so freshly promoted nodes get a brief grace period.
        anchor = node.last_used_query_idx
        if anchor < 0:
            anchor = node.last_promoted_query_idx
        if anchor < 0:
            anchor = current_query_idx  # safety
        gap = max(0, current_query_idx - anchor)
        recency = math.exp(-gap / self.tau)
        return utility * recency

    # ----- selection -----

    def select_for_demotion(
        self,
        hierarchy_dict: Dict[str, DualNode],
        current_query_idx: int,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, float]]:
        """Return (demote_ids, reasons, keep_scores).

        Rules (README2.md §3.4):
          1. If a promoted node has not been *used* for `decay_window` queries
             (counting from its last_used_query_idx, or last_promoted_query_idx
             when the node has never been marked used), demote it.
          2. After applying (1), if promoted-node count still exceeds the
             budget, demote the lowest keep_score nodes until under budget.
        """
        promoted: List[DualNode] = [
            n for n in hierarchy_dict.values() if n.state == NODE_STATE_PROMOTED
        ]
        keep_scores: Dict[str, float] = {
            n.node_id: self.compute_keep_score(n, current_query_idx) for n in promoted
        }
        reasons: Dict[str, str] = {}
        demote_set: Dict[str, None] = {}  # ordered set

        # Rule (1): decay-window expiry.
        for n in promoted:
            anchor = n.last_used_query_idx
            if anchor < 0:
                anchor = n.last_promoted_query_idx
            gap = current_query_idx - anchor
            if gap >= self.decay_window:
                demote_set[n.node_id] = None
                reasons[n.node_id] = REASON_DECAY_WINDOW

        # Rule (2): budget overflow on the remaining promoted set.
        remaining = [n for n in promoted if n.node_id not in demote_set]
        overflow = len(remaining) - self.promotion_budget
        if overflow > 0:
            # Sort by keep_score ASCENDING, tie-break by node_id for determinism.
            ranked = sorted(
                remaining,
                key=lambda x: (keep_scores[x.node_id], x.node_id),
            )
            for n in ranked[:overflow]:
                demote_set[n.node_id] = None
                reasons.setdefault(n.node_id, REASON_BUDGET_OVERFLOW)

        return list(demote_set.keys()), reasons, keep_scores

    # ----- mutation -----

    def apply_demotions(
        self,
        hierarchy_dict: Dict[str, DualNode],
        demote_ids: List[str],
        current_query_idx: int,
        *,
        query_id: str = "",
        state_log: Optional[StateLog] = None,
        keep_scores: Optional[Dict[str, float]] = None,
        reasons: Optional[Dict[str, str]] = None,
    ) -> None:
        keep_scores = keep_scores or {}
        reasons = reasons or {}
        for nid in demote_ids:
            n = hierarchy_dict.get(nid)
            if n is None or n.state != NODE_STATE_PROMOTED:
                continue
            n.state = NODE_STATE_LIGHT
            # last_promoted_query_idx stays — that's history.
            if state_log is not None:
                state_log.append(StateLogEntry(
                    query_idx=current_query_idx,
                    query_id=query_id,
                    node_id=nid,
                    event=EVENT_DEMOTE,
                    score_components={
                        "keep_score": keep_scores.get(nid, 0.0),
                        "detail_use_count": n.detail_use_count,
                        "promotion_count": n.promotion_count,
                        "last_used_query_idx": n.last_used_query_idx,
                        "last_promoted_query_idx": n.last_promoted_query_idx,
                    },
                    rationale=reasons.get(nid, "demote"),
                ))

    def log_kept_promoted(
        self,
        hierarchy_dict: Dict[str, DualNode],
        demote_ids: List[str],
        current_query_idx: int,
        *,
        query_id: str = "",
        state_log: Optional[StateLog] = None,
        keep_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        if state_log is None:
            return
        keep_scores = keep_scores or {}
        dset = set(demote_ids)
        for n in hierarchy_dict.values():
            if n.state == NODE_STATE_PROMOTED and n.node_id not in dset:
                state_log.append(StateLogEntry(
                    query_idx=current_query_idx,
                    query_id=query_id,
                    node_id=n.node_id,
                    event=EVENT_KEEP_PROMOTED,
                    score_components={"keep_score": keep_scores.get(n.node_id, 0.0)},
                    rationale=REASON_KEEP,
                ))

    def run_cycle(
        self,
        hierarchy_dict: Dict[str, DualNode],
        current_query_idx: int,
        *,
        query_id: str = "",
        state_log: Optional[StateLog] = None,
        log_kept: bool = False,
    ) -> DecayResult:
        demote_ids, reasons, keep_scores = self.select_for_demotion(
            hierarchy_dict, current_query_idx,
        )
        self.apply_demotions(
            hierarchy_dict, demote_ids, current_query_idx,
            query_id=query_id, state_log=state_log,
            keep_scores=keep_scores, reasons=reasons,
        )
        if log_kept:
            self.log_kept_promoted(
                hierarchy_dict, demote_ids, current_query_idx,
                query_id=query_id, state_log=state_log,
                keep_scores=keep_scores,
            )
        kept_ids = [
            n.node_id for n in hierarchy_dict.values()
            if n.state == NODE_STATE_PROMOTED
        ]
        self.n_cycles += 1
        return DecayResult(
            demoted_ids=demote_ids,
            kept_ids=kept_ids,
            reasons=reasons,
            keep_scores=keep_scores,
        )

    # ----- checkpoint -----

    def save_state(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "promotion_budget": self.promotion_budget,
                "decay_window": self.decay_window,
                "tau": self.tau,
                "n_cycles": self.n_cycles,
            }, f)

    def load_state(self, path: str) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.promotion_budget = int(d["promotion_budget"])
        self.decay_window = int(d["decay_window"])
        self.tau = float(d["tau"])
        self.n_cycles = int(d.get("n_cycles", 0))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> int:
    # Build a tiny hierarchy: 5 PROMOTED nodes.
    nodes = {}
    for i in range(5):
        n = DualNode(
            node_id=f"n{i}", level="L1", tenant_id="t",
            distilled_text="d", detailed_text="dd", distilled_tokens=1, detailed_tokens=4,
            source_evidence_ids=["ev"], state=NODE_STATE_PROMOTED,
            promotion_count=1, last_promoted_query_idx=0,
            last_used_query_idx=i,  # n4 just used; n0 oldest
            detail_use_count=(1 if i in (3, 4) else 0),
        )
        nodes[n.node_id] = n
    # Case A: budget=3 → should demote 2 lowest keep_score nodes.
    dc = DecayController(promotion_budget=3, decay_window=999, tau=30.0)
    res = dc.run_cycle(nodes, current_query_idx=5)
    assert len(res.demoted_ids) == 2, res.demoted_ids
    # n0,n1,n2 have utility 0 (use=0) → keep_score = 0; tie-broken by node_id.
    assert "n0" in res.demoted_ids and "n1" in res.demoted_ids, res.demoted_ids
    print(f"  budget overflow demoted: {res.demoted_ids}")

    # Reset
    for n in nodes.values():
        n.state = NODE_STATE_PROMOTED
    # Case B: decay_window=2 — anyone whose last_used_query_idx <= 100-2 should demote.
    dc2 = DecayController(promotion_budget=999, decay_window=2, tau=30.0)
    res2 = dc2.run_cycle(nodes, current_query_idx=100)
    # All 5 nodes have last_used_query_idx in 0..4 → all gaps >= 96 >= 2 → all demoted.
    assert len(res2.demoted_ids) == 5, res2.demoted_ids
    print(f"  decay window demoted: {res2.demoted_ids}")

    # Case C: checkpoint round-trip
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "dc.pkl")
        dc.save_state(p)
        dc3 = DecayController(promotion_budget=999, decay_window=1, tau=1.0)
        dc3.load_state(p)
        assert dc3.promotion_budget == 3
        assert dc3.decay_window == 999
        assert dc3.tau == 30.0
    print("[PASS] decay_controller self-test")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
