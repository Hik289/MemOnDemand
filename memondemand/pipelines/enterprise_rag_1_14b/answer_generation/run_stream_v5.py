"""Keep the Expanded runner import path on the shared retrieval implementation."""
from __future__ import annotations

import sys

from memondemand.runners import run_stream_v5 as _runner


if __name__ == "__main__":
    raise SystemExit(_runner.main())

# Callers historically patch module attributes; they must reach function globals.
sys.modules[__name__] = _runner
