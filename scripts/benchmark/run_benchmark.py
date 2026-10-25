"""Run the independent pilot on a GPU server, locally or in Jupyter, without training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--fetch-small-sample', action='store_true',
                        help='Explicitly allow the bounded public sample download (24 MiB hard cap)')
    args = parser.parse_args()
    output = args.output or ROOT / 'experiments' / ('benchmark_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stages = []

    def run(name, command):
        print('\nRUN:', name, flush=True)
        with (output / (name + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-B', '-X', 'utf8', *command], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
        print((output / (name + '.log')).read_text(encoding='utf-8'), flush=True)
        stages.append({'stage': name, 'status': 'PASS' if result.returncode == 0 else 'NOT RUN' if result.returncode == 2 else 'FAIL',
                       'exit_code': result.returncode})
        (output / 'run_status.json').write_text(json.dumps(stages, indent=2), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name}: see {output / (name + ".log")}')

    run('unit_tests', ['-m', 'unittest', 'scripts.test.test_haze_benchmark',
                      'scripts.test.test_stage1_bugfixes', 'scripts.test.test_review_regressions',
                      'scripts.test.test_v1_acceptance'])
    spec = json.loads((ROOT / 'configs/haze_benchmark_pilot_scenes.json').read_text(encoding='utf-8'))
    sources = output / 'source_samples'
    sources.mkdir()
    roots = [args.data_root.resolve()] if args.data_root else [ROOT / 'experiments/haze_benchmark/source_samples',
        ROOT / 'datasets']
    # Bounded path candidates; never recursively traverse an entire mounted disk.
    for scene in spec['scenes']:
        for root in filter(None, roots):
            candidates = [root / scene['path'], root / scene['archive_member'],
                          root / 'RSHaze+' / 'RSHaze_G/test/cleanpng' / Path(scene['archive_member']).name,
                          root / 'RSHaze_G/test/cleanpng' / Path(scene['archive_member']).name,
                          root / Path(scene['archive_member']).name]
            match = next((p for p in candidates if p.is_file() and
                          hashlib.sha256(p.read_bytes()).hexdigest() == scene['source_sha256']), None)
            if match:
                shutil.copyfile(match, sources / scene['path'])
                break
    if not all((sources / s['path']).exists() for s in spec['scenes']):
        if not args.fetch_small_sample:
            stages.append({'stage': 'data', 'status': 'NOT RUN', 'reason': 'Exact pilot sources missing; provide --data-root or explicitly use --fetch-small-sample'})
            (output / 'run_status.json').write_text(json.dumps(stages, indent=2), encoding='utf-8')
            print('NOT RUN: pilot images missing. Results:', output)
            return 2
        downloaded = output / 'downloaded_samples'
        run('fetch_samples', ['scripts/data/fetch_benchmark_samples.py', '--output', str(downloaded), '--count', '8'])
        for scene in spec['scenes']:
            shutil.copyfile(downloaded / scene['path'], sources / scene['path'])
    manifest = sources / 'pilot_scenes.json'
    manifest.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = ['--checkpoint', str(args.checkpoint.resolve())] if args.checkpoint else []
    # Do not guess checkpoint lineage from its filename. Missing weights remain NOT EVALUATED.
    cli = ['-m', 'benchmarks.haze_density']
    for profile in ('minimal', 'full'):
        data, results = output / (profile + '_data'), output / (profile + '_results')
        run(profile + '_generate', cli + ['generate', '--scenes', str(manifest), '--output', str(data), '--profile', profile])
        run(profile + '_evaluate', cli + ['evaluate', '--dataset', str(data), '--output', str(results), '--device', device] + checkpoint)
        run(profile + '_report', cli + ['report', '--output', str(results), '--max-figures', '32'])
    run('performance', cli + ['performance', '--output', str(output / 'performance')] + checkpoint)
    print('Completed. Read method-level statuses; PASS does not mean the algorithm is accurate. Results:', output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
