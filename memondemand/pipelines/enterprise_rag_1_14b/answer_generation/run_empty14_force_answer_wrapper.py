"""Compatibility entry point for explicit best-effort verification reruns."""
from __future__ import annotations

import os

os.environ.setdefault("MEMONDEMAND_FORCE_ANSWER", "1")

from memondemand.runners import run_stream_v5 as R


if __name__ == "__main__":
    raise SystemExit(R.main())
