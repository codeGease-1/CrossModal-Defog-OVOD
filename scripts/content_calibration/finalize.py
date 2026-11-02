"""Package review evidence without modifying historical experiments or Git."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v15.run import read, write, digest


def main():
    out = ROOT/'experiments/content_calibration_20261102'
    if (out/'delivery_manifest.json').exists():
        raise SystemExit('Delivery already finalized; preserve it')
    before = read(out/'handover_snapshot.json')
    exact, appended, failures = [], [], []
    with zipfile.ZipFile(out/'preexisting_changes.zip') as backup:
        for entry in before['files']:
            p = ROOT/entry['path']
            data = p.read_bytes()
            if hashlib.sha256(data).hexdigest() == entry['sha256']:
                exact.append(entry['path'])
            elif entry['path'] == '.gitignore' and data.startswith(backup.read('.gitignore')):
                appended.append(entry['path'])
            else:
                failures.append(entry['path'])
    if failures:
        raise RuntimeError(f'Preexisting files changed unexpectedly: {failures}')
    write(out/'preservation_check.json', {'exactly_unchanged': exact, 'original_prefix_preserved_appended_only': appended,
          'unexpected_changes': failures, 'cloud': 'Read-only SSH audit only; no sync or cloud mutation',
          'original_backup_sha256': digest(out/'preexisting_changes.zip')})
    run = out/'cpu_run'
    protocol = read(run/'protocol.json')
    assert all(digest(ROOT/p) == h for p, h in protocol['source_sha256'].items()), 'Recorded computational sources changed'
    old = read(ROOT/'experiments/v2_investigation_20261101/budget.json')
    stages = old['stages']
    student = sum(e['seconds'] for e in stages if e['stage'].startswith('train_student_'))
    rgb = sum(e['seconds'] for e in stages if e['stage'].startswith('rgb_') and '_epoch' in e['stage'])
    estimate = 2*student + rgb
    ledger = read(out/'historical_evidence.json')
    write(out/'budget_forecast.json', {
          'historical_gpu_seconds_verified_local_and_cloud': ledger['historical_seconds'],
          'arithmetic_remaining_seconds': ledger['arithmetic_remaining_seconds'],
          'new_authorized_gpu_seconds': 0, 'actual_gpu_seconds': 0,
          'current_CPU_main_loop_seconds': read(run/'completion.json')['elapsed_this_invocation_seconds'],
          'current_CPU_additional_diagnostics_seconds': read(out/'diagnostics/results.json')['cpu_elapsed_seconds'],
          'decision': 'Stop expansion now: joint errors criterion failed; no new GPU request recommended until qualification improves',
          'conditional_future_CPU_plan': {'originals_train_development': [48, 16], 'image_size': 256,
              'versions_per_original': 32, 'cases': 2048, 'estimated_minutes_range': [8, 15],
              'basis': 'Area and case-count extrapolation from observed CPU stages and measured 256 full pipeline; NOT a measured promise',
              'uncompressed_GiB_one_9channel_features_plus_target': 2048*256*256*10*4/2**30,
              'uncompressed_GiB_shared_RGB_raw_three_candidate_sets_target': 2048*256*256*16*4/2**30,
              'approved_or_executed': False},
          'conditional_future_GPU_plan': {'prerequisites': ['candidate/common-error gate passed', 'qualified scene grouping/target/domain',
                     'frozen method and untouched new holdout', 'application noninferiority margin chosen if claiming near-but-cheaper'],
              'major_neural_variants': ['B new-protocol global candidates + dynamic', 'D same frozen C candidates + dynamic',
                                         'same-input direct regression with same pairs and auxiliary supervision opportunity'],
              'variants_per_round_max': 3, 'seeds': [42, 43, 44], 'epochs': 18,
              'old_student_three_seed_seconds': student, 'old_RGB_three_seed_seconds': rgb,
              'training_estimate_seconds': estimate, 'overhead_reserve_minutes': [15, 25],
              'total_minutes_range': [estimate/60+15, estimate/60+25],
              'model_timing_on_new_GPU_unmeasured': True, 'authorized_or_started': False,
              'controls': 'Changed raw-cue/calibration protocol requires B and direct reference retraining; old scores are not fair controls'},
          'no_resource_purchase_or_change': True, 'not_a_cloud_bill': True})
    rgb_raw = read(ROOT/'experiments/v2_investigation_20261101/rgb_control/nominal/summary.json')
    ledger['second_v2_RGB_raw_summary'] = rgb_raw
    write(out/'historical_evidence.json', ledger)
    write(out/'comparison_matrix.json', {
          'historical_original_v2_fixed_and_dynamic': 'Preserved evidence; reference only under changed protocol',
          'new_global_vs_content_fixed': 'Actually fit/evaluated on identical small train/dev source protocol',
          'content_vs_content_physical': 'Same binning capacity, source pairs, support, shrinkage and independent candidate supervision',
          'C_vs_D': 'Same frozen candidates proven; D accuracy training pending, do not reuse old dynamic scores',
          'same_input_direct_regression': '9-channel interface and trunk/cost measured only; no new effect evidence',
          'physical_interpretation': 'Exact synthetic supervision collapses this term to signal-dependent weighted H loss',
          'protocol_wording_clarification': 'protocol.json says same knots: the identical quantities are x-quantile BIN EDGES and bin-count upper bound; '
              'weighted centroids and PAVA merges differ after fitting. No exact fitted-knot identity is claimed.',
          'diagnostic_pooling': 'Use diagnostics/results.json boundary_pooling for global conditional amplitude and max; '
              'cpu_run summary contains means of per-image corresponding quantities',
          'stop_decision': read(out/'diagnostics/results.json')['gate']})
    source = sorted(set([*ROOT.glob('src/**/*.py'), *ROOT.glob('benchmarks/haze_density/**/*.py'),
                         *ROOT.glob('scripts/content_calibration/*.py'), *ROOT.glob('scripts/v2/*.py'),
                         ROOT/'scripts/v15/run.py', ROOT/'configs/content_calibration.json',
                         ROOT/'configs/v15_calibration.json', ROOT/'requirements.txt']))
    source_manifest = {str(p.relative_to(ROOT)).replace('\\', '/'): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in source}
    write(out/'source_manifest.json', {'git_head': before['head'], 'files': source_manifest})
    with zipfile.ZipFile(out/'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in source:
            archive.write(p, str(p.relative_to(ROOT)))
    report = ROOT/'docs/三先验内容校正独立实验报告_20261102.md'
    files = {str(p.relative_to(out)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in out.rglob('*') if p.is_file() and p.name != 'delivery_manifest.json'}
    write(out/'delivery_manifest.json', {'status': 'Implementation complete; efficacy gate failed; stop expansion',
          'report': {'path': str(report.relative_to(ROOT)), 'sha256': digest(report)},
          'files': files, 'source_snapshot_sha256': digest(out/'source_snapshot.zip'),
          'git_commit_or_push': False, 'GPU_seconds': 0, 'cloud_sync': False})
    print(f'Finalized {len(files)} artifacts, {len(source)} source/config files. Existing files preserved.')


if __name__ == '__main__':
    main()
