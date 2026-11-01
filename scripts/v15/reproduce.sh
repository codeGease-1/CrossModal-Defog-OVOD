#!/usr/bin/env bash
# Run from the repository root. An existing output is resumed only if its
# frozen configuration and source hashes match. Never overwrite old runs.
set -eu
out="${1:-experiments/v15_reproduction}"
archive="${2:-/mnt/RSHaze+.zip}"
mkdir -p "$out"
python scripts/v15/audit_archive.py --archive "$archive" --output "$out/audit" > "$out/archive_audit.log"
python scripts/v15/test_v15.py > "$out/unit_tests.log" 2>&1
for stage in prepare smoke fit development freeze test external; do
  python scripts/v15/run.py --stage "$stage" --archive "$archive" --output "$out" > "$out/$stage.log" 2>&1
done
python scripts/v15/verified_spectral_test.py --archive "$archive" --output "$out" > "$out/verified_spectral.log" 2>&1
python scripts/v15/summarize.py --output "$out" > "$out/secondary.log" 2>&1
python scripts/v15/finalize.py --output "$out"
