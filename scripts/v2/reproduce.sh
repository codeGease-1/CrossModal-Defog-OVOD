#!/usr/bin/env bash
# Run from repository root in the same PyTorch/V100 environment.
set -eu
out="${1:-experiments/v2_20261101}"
python scripts/v2/test_v2.py
for stage in prepare calibrate diagnose; do
  python scripts/v2/run.py --stage "$stage" --output "$out"
done
# Optional helper scripts use this experiment's canonical directory.
# For another output, invoke only the stage CLI or adapt helpers in a new run.
python scripts/v2/run.py --stage train --output "$out"
python scripts/v2/run.py --stage freeze --output "$out"
for stage in development test confirmation external; do
  python scripts/v2/run.py --stage "$stage" --output "$out"
done
