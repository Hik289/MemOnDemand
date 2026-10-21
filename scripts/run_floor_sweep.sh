#!/usr/bin/env bash
# Explicitly running this script calls the configured models for each floor.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_floor_sweep.sh [options] [-- runner-options]
  --hierarchy PATH    Existing hierarchy (or HIERARCHY)
  --queries PATH      Query parquet (or QUERIES)
  --output-root PATH  Per-floor output root (or OUTPUT_ROOT)
  --floors "0.1 0.3"  Space/comma-separated floors (or FLOORS)
  --help              Show this message without running models

PYTHON_BIN selects Python. Defaults sweep 0.1 through 0.9.
Running the sweep calls configured model APIs; it does not build memory or judge answers.
USAGE
}

HIERARCHY=${HIERARCHY:-}
QUERIES=${QUERIES:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-}
FLOORS=${FLOORS:-"0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"}
PYTHON_BIN=${PYTHON_BIN:-python3}
EXTRA_ARGS=()
while (( $# )); do
  case "$1" in
    --hierarchy|--queries|--output-root|--floors)
      if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --hierarchy) HIERARCHY=$2 ;;
        --queries) QUERIES=$2 ;;
        --output-root) OUTPUT_ROOT=$2 ;;
        --floors) FLOORS=$2 ;;
      esac
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$HIERARCHY" || -z "$QUERIES" || -z "$OUTPUT_ROOT" ]]; then
  echo "Set hierarchy, queries, and output-root before running a sweep." >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$HIERARCHY" || ! -f "$QUERIES" ]]; then
  echo "Hierarchy and query files must exist." >&2
  exit 2
fi
read -r -a FLOOR_VALUES <<< "${FLOORS//,/ }"
if (( ${#FLOOR_VALUES[@]} == 0 )); then echo "No floors specified." >&2; exit 2; fi
for floor in "${FLOOR_VALUES[@]}"; do
  if [[ ! "$floor" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
    echo "Invalid floor: $floor (expected a number from 0 to 1)" >&2
    exit 2
  fi
done
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
for floor in "${FLOOR_VALUES[@]}"; do
  echo "Running relevance floor $floor"
  MEMONDEMAND_PROMOTE_RELEVANCE_FLOOR="$floor" "$PYTHON_BIN" -m memondemand run \
    --method V5 \
    --hierarchy "$HIERARCHY" \
    --queries "$QUERIES" \
    --seed 20260702 \
    --top_k_distilled 50 \
    --promotion_budget 20 \
    --out "$OUTPUT_ROOT/floor_$floor" \
    --resume "${EXTRA_ARGS[@]}"
done
