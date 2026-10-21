"""Explicit best-effort answering with the shared retrieval implementation."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MEMONDEMAND_FORCE_ANSWER", "1")

from memondemand.runners import run_stream_v5 as _runner


if __name__ == "__main__":
    raise SystemExit(_runner.main())

sys.modules[__name__] = _runner
