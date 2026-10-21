"""Compatibility entry point; the shared runner handles truncation and overrides."""
from __future__ import annotations

from memondemand.runners import run_stream_v5 as R


if __name__ == "__main__":
    raise SystemExit(R.main())
