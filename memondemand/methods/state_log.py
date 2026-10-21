"""V4 Promotion State Log.

Newline-delimited JSON event log used by the promotion + decay controllers
and consumed by the §9.2 metrics evaluator.

Per README2.md §3.3 each event entry has the shape::

    {
      "query_idx": int,
      "query_id": str,
      "node_id": str,
      "event": "PROMOTE|DEMOTE|KEEP_PROMOTED|MARK_USED|STOP_INSUFFICIENT|KEEP_LIGHT",
      "score_components": {...},   # optional, depends on event
      "rationale": str,            # short text
      "timestamp": float
    }

The log is append-only on disk. We deliberately keep this very small so it
stays compatible with checkpoint resume — recovery just truncates the log
to the last checkpoint's recorded byte offset (the controller saves that
offset along with the rest of its state).

This module performs NO LLM call and NO embedding work; it is pure I/O.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Event-name constants
# ---------------------------------------------------------------------------

EVENT_PROMOTE = "PROMOTE"
EVENT_KEEP_LIGHT = "KEEP_LIGHT"
EVENT_STOP_INSUFFICIENT = "STOP_INSUFFICIENT"
EVENT_DEMOTE = "DEMOTE"
EVENT_KEEP_PROMOTED = "KEEP_PROMOTED"
EVENT_MARK_USED = "MARK_USED"
EVENT_DETAIL_USED = EVENT_MARK_USED

VALID_EVENTS = {
    EVENT_PROMOTE, EVENT_KEEP_LIGHT, EVENT_STOP_INSUFFICIENT,
    EVENT_DEMOTE, EVENT_KEEP_PROMOTED, EVENT_MARK_USED,
}


@dataclass
class StateLogEntry:
    query_idx: int
    query_id: str
    node_id: str
    event: str
    score_components: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        if self.event not in VALID_EVENTS:
            raise ValueError(f"unknown event {self.event!r}")
        return {
            "query_idx": int(self.query_idx),
            "query_id": str(self.query_id),
            "node_id": str(self.node_id),
            "event": self.event,
            "score_components": self.score_components,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
        }


class StateLog:
    """Append-only NDJSON state-log writer with in-memory mirror.

    The mirror lets controllers + tests inspect the most recent events
    without re-reading from disk.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Open in append mode so a checkpoint-resume can continue cleanly.
        self._f = open(path, "a", encoding="utf-8")
        self._mirror: List[Dict[str, Any]] = []

    def append(self, entry: StateLogEntry) -> None:
        d = entry.to_dict()
        self._f.write(json.dumps(d, ensure_ascii=False) + "\n")
        self._f.flush()
        self._mirror.append(d)

    def record(
        self,
        event: str,
        node_id: str,
        query_idx: int,
        query_id: str = "",
        score: float = 0.0,
        reason: str = "",
        **extra: Any,
    ) -> None:
        """Accept the legacy convenience API and write the full event schema."""
        components = dict(extra.pop("score_components", {}) or {})
        components.setdefault("promotion_score", score)
        components.update(extra)
        self.append(StateLogEntry(
            query_idx=query_idx,
            query_id=query_id,
            node_id=node_id,
            event=EVENT_MARK_USED if event == "DETAIL_USED" else event,
            score_components=components,
            rationale=reason,
        ))

    def append_many(self, entries: Iterable[StateLogEntry]) -> None:
        for e in entries:
            self.append(e)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    # ----- queries -----

    @property
    def entries(self) -> List[Dict[str, Any]]:
        return list(self._mirror)

    def count_event(self, event: str) -> int:
        return sum(1 for e in self._mirror if e["event"] == event)

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ev in VALID_EVENTS:
            out[ev] = self.count_event(ev)
        out["total"] = len(self._mirror)
        return out

    # ----- I/O for resume -----

    def tell(self) -> int:
        return self._f.tell()

    @staticmethod
    def read_all(path: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not os.path.exists(path):
            return out
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    out.append(json.loads(ln))
        return out

    @staticmethod
    def truncate_to(path: str, byte_offset: int) -> None:
        """Truncate log file to a byte offset (for checkpoint resume)."""
        if not os.path.exists(path):
            return
        with open(path, "rb+") as f:
            f.truncate(byte_offset)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "log.ndjson")
        log = StateLog(p)
        log.append(StateLogEntry(0, "q0", "n0", EVENT_PROMOTE,
                                 {"score": 1.2}, "high relevance"))
        log.append(StateLogEntry(0, "q0", "n1", EVENT_KEEP_LIGHT,
                                 {"score": 0.1}, "low signal"))
        log.append(StateLogEntry(1, "q1", "n0", EVENT_MARK_USED,
                                 {}, "cited"))
        log.append(StateLogEntry(2, "q2", "n0", EVENT_DEMOTE,
                                 {"keep_score": 0.0}, "budget overflow"))
        s = log.summary()
        assert s["PROMOTE"] == 1, s
        assert s["KEEP_LIGHT"] == 1, s
        assert s["MARK_USED"] == 1, s
        assert s["DEMOTE"] == 1, s
        assert s["total"] == 4, s
        log.close()
        roundtrip = StateLog.read_all(p)
        assert len(roundtrip) == 4
    print("[PASS] state_log self-test")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
