"""V4 Promotion Controller — query-time state machine (README2.md §3.3).

Each query the controller is given a set of candidate nodes from the
distilled index. For every candidate it computes::

    promotion_score =
        w1 * relevance
      + w2 * uncertainty
      + w3 * expected_information_gain
      + w4 * historical_detail_utility
      - w5 * detail_token_cost

then asks `gpt_5_4` to choose PROMOTE / KEEP_LIGHT / STOP_INSUFFICIENT for
each candidate. Decisions are then materialized onto the DualNode state
(LIGHT → PROMOTED) and recorded in the state log.

================================================================
§11 FAIRNESS GUARANTEES (mandatory)
================================================================
1.  `historical_detail_utility` reads ONLY past per-node counters
    (`detail_use_count`, `promotion_count`). Those counters are mutated
    by `mark_detail_used` AFTER the runner has already generated the
    answer for the current query. Therefore — by construction — the
    counters seen at query t reflect only events from queries 0..t-1.
    `score_promotion` asserts `node.last_used_query_idx < query_idx`,
    so a programming mistake that mutates state too early will raise.
2.  No gold evidence, gold answer, or evaluator score is ever read by
    this controller. `decide` consumes ONLY the query text and the
    candidate node's distilled_text + score signals.
3.  Decisions are deterministic given (seed, query order, candidate
    ordering, signals): we use `temperature=0.0` for the LLM call,
    and we sort candidates by node_id for the prompt body so the
    promotion-decision tokens are reproducible across runs.

================================================================
Checkpointing
================================================================
`save_state` / `load_state` persist:
- the cumulative number of promotion decisions made (a counter),
- the controller's hyper-parameters,
- the embed-cache (a `Dict[query_id, np.ndarray]`) so a resumed run does
  not re-pay embedding cost for queries that were already partially
  processed.

The DualNode-level state (which nodes are PROMOTED, all counters) lives
on the hierarchy objects themselves and is saved by the runner alongside
the hierarchy snapshot.
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from memondemand.core.api_adapter import call as api_call, APIError  # noqa: E402
from memondemand.methods.dual_node import (  # noqa: E402
    DualNode,
    NODE_STATE_LIGHT,
    NODE_STATE_PROMOTED,
)
from memondemand.methods.token_ledger import (  # noqa: E402
    PHASE_PROMOTION_DECISION,
    TokenLedger,
)
from memondemand.methods.state_log import (  # noqa: E402
    EVENT_KEEP_LIGHT,
    EVENT_PROMOTE,
    EVENT_STOP_INSUFFICIENT,
    EVENT_MARK_USED,
    StateLog,
    StateLogEntry,
)

logger = logging.getLogger("v4.promotion")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

DECISION_PROMOTE = "PROMOTE"
DECISION_KEEP_LIGHT = "KEEP_LIGHT"
DECISION_STOP_INSUFFICIENT = "STOP_INSUFFICIENT"
VALID_DECISIONS = {DECISION_PROMOTE, DECISION_KEEP_LIGHT, DECISION_STOP_INSUFFICIENT}


@dataclass
class PromotionSignals:
    relevance: float
    uncertainty: float
    expected_information_gain: float
    historical_detail_utility: float
    detail_token_cost: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevance": round(self.relevance, 6),
            "uncertainty": round(self.uncertainty, 6),
            "expected_information_gain": round(self.expected_information_gain, 6),
            "historical_detail_utility": round(self.historical_detail_utility, 6),
            "detail_token_cost": int(self.detail_token_cost),
        }


@dataclass
class PromotionDecision:
    query_id: str
    node_id: str
    decision: str
    promotion_score: float
    signals: PromotionSignals
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        if self.decision not in VALID_DECISIONS:
            raise ValueError(f"unknown decision {self.decision!r}")
        return {
            "query_id": self.query_id,
            "node_id": self.node_id,
            "decision": self.decision,
            "promotion_score": round(self.promotion_score, 6),
            "signals": self.signals.to_dict(),
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Default LLM prompts
# ---------------------------------------------------------------------------

PROMOTION_SYSTEM_PROMPT = """You are a Memory promotion controller for a retrieval system.

For each candidate Memory node you must decide ONE label:
- PROMOTE: load the node's detailed representation into the LLM context (worth the token cost)
- KEEP_LIGHT: the distilled representation already suffices for this query
- STOP_INSUFFICIENT: the candidate is unrelated; do not load detail and do not search deeper here

You are given numeric SIGNALS for each candidate:
- relevance: cosine similarity to the query (higher = more on-topic)
- uncertainty: how confused the distilled retrieval is about the candidate (higher = need more detail)
- expected_information_gain: normalized detailed length the candidate would add
- historical_detail_utility: ratio of past PROMOTIONS of this node that actually got cited
- detail_token_cost: raw token count of the detailed representation
- promotion_score: weighted sum of the signals

Rules:
- Prefer PROMOTE when relevance is high AND (uncertainty is moderate or historical_detail_utility is high).
- Prefer KEEP_LIGHT when relevance is moderate but the distilled text seems sufficient.
- Use STOP_INSUFFICIENT only when the candidate is clearly off-topic.
- Output ONLY a JSON object — no commentary, no markdown fences, no preamble.

Output schema (STRICT):
{"decisions": [
  {"node_id": "<id>", "decision": "PROMOTE|KEEP_LIGHT|STOP_INSUFFICIENT", "reason": "<<= 25 words>"}
]}
"""


PROMOTION_USER_TEMPLATE = """Query: {query}

Candidates (sorted by node_id for stable ordering):
{candidates_block}

Return JSON only."""


def _format_candidate_for_prompt(
    node: DualNode, signals: PromotionSignals, promotion_score: float,
) -> str:
    distilled_preview = (node.distilled_text or "").strip().replace("\n", " ")
    if len(distilled_preview) > 280:
        distilled_preview = distilled_preview[:280] + "..."
    return (
        f"- node_id={node.node_id} level={node.level}\n"
        f"  distilled: {distilled_preview}\n"
        f"  signals: relevance={signals.relevance:.3f} "
        f"uncertainty={signals.uncertainty:.3f} "
        f"info_gain={signals.expected_information_gain:.3f} "
        f"hist_util={signals.historical_detail_utility:.3f} "
        f"detail_tokens={signals.detail_token_cost} "
        f"promotion_score={promotion_score:.3f}"
    )


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class PromotionController:
    """Implements the README2.md §3.3 promotion state machine."""

    def __init__(
        self,
        hierarchy_dict: Dict[str, DualNode],
        embedder: Any,
        *,
        w1: float = 1.0,
        w2: float = 0.5,
        w3: float = 0.5,
        w4: float = 1.0,
        w5: float = 0.01,
        promotion_budget: int = 200,
        decay_window: int = 50,
        tau: float = 30.0,
        alias_high: str = "general",
        alias_low: str = "general",
        max_candidates_per_decision: int = 10,
        max_detailed_tokens_at_level: Optional[Dict[str, int]] = None,
        deterministic_only: bool = False,
    ):
        self.hierarchy = hierarchy_dict
        self.embedder = embedder
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.w4 = float(w4)
        self.w5 = float(w5)
        self.promotion_budget = int(promotion_budget)
        self.decay_window = int(decay_window)
        self.tau = float(tau)
        self.alias_high = alias_high
        self.alias_low = alias_low
        self.max_candidates_per_decision = int(max_candidates_per_decision)
        self.deterministic_only = bool(deterministic_only)

        # Pre-compute normalization constants per level (used by info_gain).
        if max_detailed_tokens_at_level is None:
            agg: Dict[str, int] = {}
            for n in hierarchy_dict.values():
                agg[n.level] = max(agg.get(n.level, 1), int(n.detailed_tokens or 1))
            self.max_detailed_tokens_at_level = agg
        else:
            self.max_detailed_tokens_at_level = dict(max_detailed_tokens_at_level)

        # Cache of distilled-text embeddings, keyed by node_id.
        self._node_embed_cache: Dict[str, np.ndarray] = {}
        # Cache of query embeddings, keyed by query_id.
        self._query_embed_cache: Dict[str, np.ndarray] = {}
        # Counter — how many decisions have been issued (informational).
        self.n_decisions_issued = 0

    # ----- embedding helpers -----

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.embedder.encode(
            list(texts), batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def get_query_embedding(self, query_id: str, query_text: str) -> np.ndarray:
        v = self._query_embed_cache.get(query_id)
        if v is None:
            v = self._embed([query_text])[0]
            self._query_embed_cache[query_id] = v
        return v

    def get_node_embedding(self, node: DualNode) -> np.ndarray:
        v = self._node_embed_cache.get(node.node_id)
        if v is None:
            v = self._embed([node.distilled_text or node.node_id])[0]
            self._node_embed_cache[node.node_id] = v
        return v

    def warmup_node_embeddings(self, node_ids: Sequence[str]) -> None:
        """Batch-embed nodes that aren't yet in the cache. Cheap warm-up."""
        missing_nodes = [
            self.hierarchy[nid] for nid in node_ids
            if nid in self.hierarchy and nid not in self._node_embed_cache
        ]
        if not missing_nodes:
            return
        texts = [n.distilled_text or n.node_id for n in missing_nodes]
        vecs = self._embed(texts)
        for n, v in zip(missing_nodes, vecs):
            self._node_embed_cache[n.node_id] = v

    # ----- scoring -----

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    def score_promotion(
        self,
        query: str,
        candidate_node_id: str,
        query_idx: int,
        *,
        query_id: Optional[str] = None,
        peer_similarities: Optional[Sequence[float]] = None,
    ) -> Tuple[float, PromotionSignals]:
        """Compute promotion_score for one candidate.

        Parameters
        ----------
        query : str
            Raw query text.
        candidate_node_id : str
            Hierarchy node id.
        query_idx : int
            The index of the current query in the global query stream.
        query_id : str, optional
            Stable id for embedding cache; defaults to a hash of `query`.
        peer_similarities : Sequence[float], optional
            Distilled-index top-k similarity scores at this level. Used to
            estimate the uncertainty signal as `1.0 - max(peer_similarities)`.
            If omitted we fall back to `1.0 - relevance` (single-candidate proxy).

        Fairness guard rail: this method MUST NOT see future queries' state.
        We assert `node.last_used_query_idx < query_idx` (i.e. any utility
        signal we read was recorded before the current query).
        """
        node = self.hierarchy[candidate_node_id]
        # § 11 fairness: utility counters at this point reflect strictly past
        # queries because the runner is supposed to call mark_detail_used()
        # only after answering the current query.
        assert node.last_used_query_idx < query_idx, (
            f"§11 fairness violation: node {node.node_id} last_used_query_idx="
            f"{node.last_used_query_idx} >= query_idx={query_idx}. "
            "Did mark_detail_used() run too early?"
        )

        qid = query_id or f"_inline_{hash(query) & 0xffffffff:08x}"
        q_vec = self.get_query_embedding(qid, query)
        n_vec = self.get_node_embedding(node)
        relevance = max(0.0, self._cosine(q_vec, n_vec))

        if peer_similarities:
            max_peer = max(peer_similarities)
            uncertainty = max(0.0, min(1.0, 1.0 - max_peer))
        else:
            uncertainty = max(0.0, min(1.0, 1.0 - relevance))

        max_lvl_tok = max(1, int(self.max_detailed_tokens_at_level.get(node.level, 1)))
        info_gain = min(1.0, float(node.detailed_tokens or 0) / max_lvl_tok)

        # historical_detail_utility — past-only by construction (see assertion above).
        hist_util = node.detail_use_count / max(1, node.promotion_count)

        detail_token_cost = int(node.detailed_tokens or 0)

        score = (
            self.w1 * relevance
            + self.w2 * uncertainty
            + self.w3 * info_gain
            + self.w4 * hist_util
            - self.w5 * detail_token_cost
        )

        signals = PromotionSignals(
            relevance=relevance,
            uncertainty=uncertainty,
            expected_information_gain=info_gain,
            historical_detail_utility=hist_util,
            detail_token_cost=detail_token_cost,
        )
        return float(score), signals

    # ----- LLM decision -----

    @staticmethod
    def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON extraction from an LLM reply."""
        if not text:
            return None
        # Strip code fences
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?", "", t).strip()
            if t.endswith("```"):
                t = t[:-3].strip()
        # Try direct parse
        try:
            return json.loads(t)
        except Exception:
            pass
        # Find first { ... } block
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

    def _deterministic_fallback(
        self,
        scored: List[Tuple[DualNode, float, PromotionSignals]],
    ) -> List[PromotionDecision]:
        """Cheap, deterministic decision rule used as a fallback when the
        LLM is unavailable or `deterministic_only=True`.

        Rule (intentionally simple, deterministic, signal-driven; does NOT
        depend on detail_token_cost because raw token counts can dominate
        the score for high-level nodes):

        - PROMOTE if relevance >= 0.30 OR historical_detail_utility >= 0.5
        - STOP_INSUFFICIENT if relevance < 0.10 AND uncertainty < 0.3
        - KEEP_LIGHT otherwise
        """
        out: List[PromotionDecision] = []
        for n, score, sig in scored:
            if sig.relevance >= 0.30 or sig.historical_detail_utility >= 0.5:
                d = DECISION_PROMOTE
                reason = "deterministic: relevance>=0.30 or hist_util>=0.5"
            elif sig.relevance < 0.10 and sig.uncertainty < 0.30:
                d = DECISION_STOP_INSUFFICIENT
                reason = "deterministic: very low relevance and low uncertainty"
            else:
                d = DECISION_KEEP_LIGHT
                reason = "deterministic: middling signal"
            out.append(PromotionDecision(
                query_id="", node_id=n.node_id, decision=d,
                promotion_score=score, signals=sig, reason=reason,
            ))
        return out

    def decide(
        self,
        query: str,
        candidate_node_ids: Sequence[str],
        query_idx: int,
        *,
        query_id: str = "",
        alias: Optional[str] = None,
        ledger: Optional[TokenLedger] = None,
        peer_similarities: Optional[Dict[str, float]] = None,
    ) -> List[PromotionDecision]:
        """Score candidates + ask the LLM to pick a decision per candidate."""
        if not candidate_node_ids:
            return []
        # Stable ordering for reproducibility (independent of caller order).
        ordered_ids = sorted(set(candidate_node_ids))[: self.max_candidates_per_decision]
        nodes = [self.hierarchy[i] for i in ordered_ids if i in self.hierarchy]
        peer_sims_list: Optional[List[float]] = None
        if peer_similarities is not None:
            peer_sims_list = list(peer_similarities.values())

        scored: List[Tuple[DualNode, float, PromotionSignals]] = []
        for n in nodes:
            score, signals = self.score_promotion(
                query, n.node_id, query_idx,
                query_id=query_id, peer_similarities=peer_sims_list,
            )
            scored.append((n, score, signals))

        if self.deterministic_only:
            decs = self._deterministic_fallback(scored)
            for d in decs:
                d.query_id = query_id
            self._apply_relevance_floor(decs)
            self.n_decisions_issued += len(decs)
            return decs

        # Build prompt and call LLM.
        candidates_block = "\n".join(
            _format_candidate_for_prompt(n, s, sc) for (n, sc, s) in scored
        )
        user_prompt = PROMOTION_USER_TEMPLATE.format(
            query=query, candidates_block=candidates_block,
        )
        alias = alias or self.alias_high

        decisions_by_node: Dict[str, Tuple[str, str]] = {}
        t0 = time.time()
        try:
            resp = api_call(
                alias,
                [
                    {"role": "system", "content": PROMOTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=900,
                temperature=0.0,
                timeout=120.0,
                max_retries=3,
            )
            wall = time.time() - t0
            in_t = int(resp.get("usage", {}).get("input_tokens", 0))
            out_t = int(resp.get("usage", {}).get("output_tokens", 0))
            if ledger is not None:
                ledger.record(
                    phase=PHASE_PROMOTION_DECISION, model_alias=alias,
                    input_tokens=in_t, output_tokens=out_t,
                    wall_seconds=wall, query_id=query_id,
                    n_candidates=len(scored),
                )
            payload = self._extract_json_payload(resp.get("text", ""))
            if payload and isinstance(payload, dict):
                for item in payload.get("decisions", []) or []:
                    nid = str(item.get("node_id", "")).strip()
                    dec = str(item.get("decision", "")).strip().upper()
                    reason = str(item.get("reason", ""))[:300]
                    if nid and dec in VALID_DECISIONS:
                        decisions_by_node[nid] = (dec, reason)
        except (APIError, Exception) as exc:  # noqa: BLE001
            logger.warning(
                f"promotion LLM call failed ({type(exc).__name__}: {str(exc)[:120]}); "
                "falling back to deterministic rule"
            )

        out: List[PromotionDecision] = []
        for (n, score, sig) in scored:
            dec, reason = decisions_by_node.get(
                n.node_id,
                self._deterministic_decision_for(n, score, sig),
            )
            out.append(PromotionDecision(
                query_id=query_id, node_id=n.node_id, decision=dec,
                promotion_score=score, signals=sig, reason=reason,
            ))
        self._apply_relevance_floor(out)
        self.n_decisions_issued += len(out)
        return out

    @staticmethod
    def _apply_relevance_floor(decisions: Sequence[PromotionDecision]) -> None:
        """Apply the same relevance gate to LLM and deterministic decisions."""
        try:
            floor = float(os.environ.get("MEMONDEMAND_PROMOTE_RELEVANCE_FLOOR", "0.0"))
        except (TypeError, ValueError):
            floor = 0.0
        for decision in decisions:
            if (floor > 0.0 and decision.decision == DECISION_PROMOTE
                    and decision.signals.relevance < floor):
                decision.decision = DECISION_KEEP_LIGHT
                decision.reason = (
                    f"floor_override: relevance {decision.signals.relevance:.3f} < {floor:.3f}"
                )

    @staticmethod
    def _deterministic_decision_for(
        n: DualNode, score: float, sig: PromotionSignals,
    ) -> Tuple[str, str]:
        if sig.relevance >= 0.30 or sig.historical_detail_utility >= 0.5:
            return DECISION_PROMOTE, "fallback: relevance>=0.30 or hist_util>=0.5"
        if sig.relevance < 0.10 and sig.uncertainty < 0.30:
            return DECISION_STOP_INSUFFICIENT, "fallback: very low relevance"
        return DECISION_KEEP_LIGHT, "fallback: middling signal"

    # ----- mutation -----

    def apply_decisions(
        self,
        decisions: Sequence[PromotionDecision],
        query_idx: int,
        *,
        state_log: Optional[StateLog] = None,
    ) -> Dict[str, int]:
        """Mutate node state per decision; log to state_log if provided."""
        counts = {DECISION_PROMOTE: 0, DECISION_KEEP_LIGHT: 0, DECISION_STOP_INSUFFICIENT: 0}
        for d in decisions:
            n = self.hierarchy.get(d.node_id)
            if n is None:
                continue
            if d.decision == DECISION_PROMOTE:
                # LIGHT -> PROMOTED (idempotent if already promoted)
                if n.state != NODE_STATE_PROMOTED:
                    n.state = NODE_STATE_PROMOTED
                n.last_promoted_query_idx = query_idx
                n.promotion_count += 1
                n.promotion_score = d.promotion_score
                event = EVENT_PROMOTE
            elif d.decision == DECISION_KEEP_LIGHT:
                event = EVENT_KEEP_LIGHT
            else:
                event = EVENT_STOP_INSUFFICIENT
            counts[d.decision] += 1
            if state_log is not None:
                state_log.append(StateLogEntry(
                    query_idx=query_idx,
                    query_id=d.query_id,
                    node_id=d.node_id,
                    event=event,
                    score_components=d.signals.to_dict() | {"promotion_score": round(d.promotion_score, 6)},
                    rationale=d.reason,
                ))
        return counts

    def mark_detail_used(
        self,
        node_id: str,
        query_idx: int,
        *,
        query_id: str = "",
        state_log: Optional[StateLog] = None,
    ) -> None:
        n = self.hierarchy.get(node_id)
        if n is None:
            return
        n.detail_use_count += 1
        n.last_used_query_idx = query_idx
        if state_log is not None:
            state_log.append(StateLogEntry(
                query_idx=query_idx,
                query_id=query_id,
                node_id=node_id,
                event=EVENT_MARK_USED,
                score_components={
                    "detail_use_count": n.detail_use_count,
                    "promotion_count": n.promotion_count,
                },
                rationale="answer cited detailed evidence",
            ))

    # ----- checkpoint -----

    def save_state(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "weights": (self.w1, self.w2, self.w3, self.w4, self.w5),
                "promotion_budget": self.promotion_budget,
                "decay_window": self.decay_window,
                "tau": self.tau,
                "alias_high": self.alias_high,
                "alias_low": self.alias_low,
                "max_candidates_per_decision": self.max_candidates_per_decision,
                "max_detailed_tokens_at_level": self.max_detailed_tokens_at_level,
                "n_decisions_issued": self.n_decisions_issued,
                "node_embed_cache": self._node_embed_cache,
                "query_embed_cache": self._query_embed_cache,
                "deterministic_only": self.deterministic_only,
            }, f)

    def load_state(self, path: str) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.w1, self.w2, self.w3, self.w4, self.w5 = d["weights"]
        self.promotion_budget = int(d["promotion_budget"])
        self.decay_window = int(d["decay_window"])
        self.tau = float(d["tau"])
        self.alias_high = d.get("alias_high", self.alias_high)
        self.alias_low = d.get("alias_low", self.alias_low)
        self.max_candidates_per_decision = int(d.get(
            "max_candidates_per_decision", self.max_candidates_per_decision))
        self.max_detailed_tokens_at_level = dict(d.get(
            "max_detailed_tokens_at_level", self.max_detailed_tokens_at_level))
        self.n_decisions_issued = int(d.get("n_decisions_issued", 0))
        self._node_embed_cache = dict(d.get("node_embed_cache", {}))
        self._query_embed_cache = dict(d.get("query_embed_cache", {}))
        self.deterministic_only = bool(d.get("deterministic_only", False))


# ---------------------------------------------------------------------------
# Self-test (deterministic-only path — no LLM, no embedding model required if
# we plug in a tiny stub)
# ---------------------------------------------------------------------------


def _self_test() -> int:
    class _StubEmbedder:
        """Cheap deterministic embedder for self-test."""
        def encode(self, texts, **_):
            out = []
            for t in texts:
                v = np.zeros(8, dtype=np.float32)
                for ch in (t or ""):
                    v[ord(ch) % 8] += 1.0
                # normalize
                norm = float(np.linalg.norm(v)) or 1.0
                out.append(v / norm)
            return np.asarray(out, dtype=np.float32)

    nodes = {}
    for i in range(3):
        nodes[f"n{i}"] = DualNode(
            node_id=f"n{i}", level="L2", tenant_id="t",
            distilled_text=f"distilled summary about topic {i}",
            detailed_text=f"detailed body for topic {i} " * 20,
            distilled_tokens=10, detailed_tokens=200 + i,
            source_evidence_ids=["ev"], state=NODE_STATE_LIGHT,
        )
    pc = PromotionController(
        nodes, _StubEmbedder(), deterministic_only=True,
        promotion_budget=10, decay_window=5, tau=10.0,
    )
    decs = pc.decide(
        "tell me about topic 1",
        [n.node_id for n in nodes.values()],
        query_idx=0, query_id="q0",
    )
    assert len(decs) == 3
    pc.apply_decisions(decs, query_idx=0)
    # Test mark_detail_used and the fairness assertion
    promoted = [d for d in decs if d.decision == DECISION_PROMOTE]
    if promoted:
        pc.mark_detail_used(promoted[0].node_id, query_idx=0)
    # next query — score_promotion must succeed (utility from past)
    score, sig = pc.score_promotion(
        "tell me about topic 1", "n1", query_idx=1, query_id="q1",
    )
    print(f"  query2 score for n1: {score:.3f} signals={sig.to_dict()}")

    # Checkpoint test
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "pc.pkl")
        pc.save_state(p)
        pc2 = PromotionController(
            nodes, _StubEmbedder(), deterministic_only=True,
            promotion_budget=99, decay_window=99, tau=99.0,
        )
        pc2.load_state(p)
        assert pc2.promotion_budget == 10
        assert pc2.n_decisions_issued == pc.n_decisions_issued
    print("[PASS] promotion_controller self-test")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
