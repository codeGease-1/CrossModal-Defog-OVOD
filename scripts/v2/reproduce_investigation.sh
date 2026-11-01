#!/usr/bin/env bash
# Auxiliary scripts use the fixed output path in configs/v2_investigation.json.
# Use a separate repository copy for a full fresh reproduction; retain this run.
set -eu
python scripts/v2/test_student_gate.py
python scripts/v2/investigate.py --stage prepare
python scripts/v2/investigate_real.py  # register the secondary endpoint before evaluation
python scripts/v2/rgb_control.py --register
python scripts/v2/spatial_gate_controls.py --register
for stage in preflight diagnose; do
  python scripts/v2/investigate.py --stage "$stage"
done
python scripts/v2/resume_compat.py --target investigation --stage train
python scripts/v2/resume_compat.py --target rgb
python scripts/v2/investigate.py --stage freeze
python scripts/v2/investigate.py --stage evaluate
python scripts/v2/investigate_real.py
python scripts/v2/evaluate_rgb_control.py
python scripts/v2/analyze_investigation.py
python scripts/v2/summarize_rgb_comparison.py
python scripts/v2/spatial_gate_controls.py
python scripts/v2/verify_student_resume.py
python scripts/v2/inference_smoke.py
