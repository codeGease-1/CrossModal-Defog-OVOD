"""Archive actual V1 audit artifacts and save compact, versioned evidence.

Does not commit, tag, train, publish, or change the acceptance verdict.
"""
import argparse
import ast
import base64
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.haze_density.common import read_json, write_json, sha256_file, fresh_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    run, evidence = args.run.resolve(), fresh_directory(args.evidence).resolve()
    history = ROOT / 'experiments/haze_benchmark'
    assert read_json(history / 'full_results/results.json')['summary'] == read_json(run / 'historical_rerun/results.json')['summary']
    assert all(c['returncode'] == 0 for c in read_json(run / 'checks.json'))
    assert [m['status'] for m in read_json(run / 'results/results.json')['methods']] == ['PASS', 'PASS', 'NOT RUN']
    write_json(run / 'historical_rerun_comparison.json', {'status': 'PASS', 'summaries_identical': True,
        'historical_sha256': sha256_file(history / 'full_results/results.json'),
        'rerun_sha256': sha256_file(run / 'historical_rerun/results.json')})
    versions = {p: importlib.metadata.version(p) for p in
                ('torch', 'torchvision', 'numpy', 'scipy', 'Pillow', 'PyYAML', 'matplotlib')}
    write_json(run / 'runtime_versions.json', {'python': sys.version, 'packages': versions,
        'note': 'Observed versions, not a portable resolver lock; full pip inventory kept separately.'})
    # Validate the exact self-contained code package, detached from the repo.
    notebook = read_json(ROOT / 'docs/haze_density_benchmark.ipynb')
    setup = ''.join(next(c['source'] for c in notebook['cells'] if c['cell_type'] == 'code'))
    node = next(n for n in ast.parse(setup).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'payload' for t in n.targets))
    payload = base64.b64decode(ast.literal_eval(node.value.args[0]))
    assert hashlib.sha256(payload).hexdigest() == notebook['metadata']['bundle_sha256']
    (run / 'source_snapshot.zip').write_bytes(payload)
    with tempfile.TemporaryDirectory(prefix='v1_sealed_bundle_') as temporary:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(temporary)
        origin = read_json(Path(temporary) / 'bundle_origin.json')
        for name, digest in origin['files'].items():
            assert sha256_file(Path(temporary) / name) == digest
        command = [sys.executable, '-B', '-X', 'utf8', '-m', 'unittest',
                   'scripts.test.test_haze_benchmark', 'scripts.test.test_stage1_bugfixes',
                   'scripts.test.test_review_regressions', 'scripts.test.test_v1_acceptance', '-v']
        # Notebook integration tests require the notebook files, deliberately
        # excluded from the embedded source bundle; tested in the real repo.
        result = subprocess.run(command, cwd=temporary, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (run / 'bundle_tests.log').write_bytes(result.stdout)
        write_json(run / 'bundle_validation.json', {'returncode': result.returncode,
            'command': command, 'bundle_sha256': hashlib.sha256(payload).hexdigest(),
            'file_count': len(origin['files']), 'expected_tests': 71,
            'scope': 'Detached source package, CPU; NOT a cloud GPU execution'})
        if result.returncode:
            raise RuntimeError('Detached bundle checks failed; see bundle_tests.log')
    copies = ['historical_audit.json', 'historical_rerun_comparison.json', 'normalization_audit.json',
              'weight_audit.json', 'gpu_access.json', 'config.json', 'environment.json', 'runtime_versions.json',
              'checks.json', 'unit_tests_clean.log', 'smoke.log', 'static_check.log',
              'bundle_tests.log', 'bundle_validation.json', 'screened_rois.png']
    for name in copies:
        shutil.copyfile(run / name, evidence / name)
    manifest = read_json(run / 'scenes_portable.json')
    for scene in manifest['scenes']:
        scene['path'] = '../../../experiments/haze_benchmark/source_samples/' + Path(scene['path']).name
    write_json(evidence / 'scenes.json', manifest)
    for folder in ('results', 'historical_rerun', 'performance_cuda'):
        target = evidence / folder
        target.mkdir()
        for path in (run / folder).glob('*'):
            if path.suffix in ('.csv', '.json'):
                shutil.copyfile(path, target / path.name)
    shutil.copyfile(history / 'performance_cpu/performance.json', evidence / 'historical_cpu_performance.json')
    shutil.copyfile(run / 'results/figures/curve_00.png', evidence / 'ranking_example.png')
    shutil.copyfile(run / 'results/figures/comparison_00.png', evidence / 'spatial_example.png')
    # Full local artifact bundle includes input sources, old-weight identity,
    # independent truths and predictions; nothing is uploaded by this script.
    paths = list(run.rglob('*')) + list((history / 'full_data').rglob('*'))
    paths += list((history / 'source_samples').glob('*.png'))
    paths += list((history / 'performance_cpu').glob('*'))
    paths += [ROOT / 'experiments/haze_density/checkpoints/best.pth']
    paths = sorted({p for p in paths if p.is_file()})
    archive_path = run.with_name(run.name + '_artifacts.zip')
    if archive_path.exists():
        raise FileExistsError(archive_path)
    artifacts = {p.relative_to(ROOT).as_posix(): {'sha256': sha256_file(p), 'bytes': p.stat().st_size} for p in paths}
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    code = [p for folder in ('src', 'benchmarks', 'scripts', 'configs')
            for p in (ROOT / folder).rglob('*') if p.suffix in ('.py', '.yaml', '.json') and '__pycache__' not in p.parts]
    write_json(evidence / 'freeze_manifest.json', {'verdict': 'NO-GO: full formal baseline acceptance',
        'snapshot_kind': 'audited limited-evidence research snapshot, not an accepted complete baseline',
        'audit_tag': 'v1-audited-20261022',
        'git_identity_note': 'Tag points to the final commit containing this manifest; logged HEAD is the experiment starting point. This avoids a self-referential commit hash.',
        'starting_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'seed': 42, 'prior_version': 'spatial_minmax_valid_boundary_v3',
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(code)},
        'raw_archive': {'path': archive_path.relative_to(ROOT).as_posix(), 'sha256': sha256_file(archive_path),
                        'bytes': archive_path.stat().st_size, 'uploaded': False},
        'artifact_files': artifacts,
        'evidence_files': {p.relative_to(evidence).as_posix(): sha256_file(p) for p in sorted(evidence.rglob('*')) if p.is_file()},
        'weight_status': 'Only old unversioned epoch=5; current V1 student NOT RUN',
        'gpu_status': 'NOT RUN', 'geographic_groups': 1,
        'protocol': 'Historical 128-case rerun plus 256-case rotated spatial sensitivity; all test, no calibration or training'})
    print('Sealed:', evidence, archive_path, archive_path.stat().st_size)


if __name__ == '__main__':
    main()
