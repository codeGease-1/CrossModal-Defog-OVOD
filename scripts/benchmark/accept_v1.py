"""Reproducible V1 acceptance audit; never trains or overwrites a run.

Run with the project environment: python scripts/benchmark/accept_v1.py --output NEW_DIR
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
import yaml

from benchmarks.haze_density.common import (fresh_directory, read_json, write_json,
    sha256_file, seed_runtime, environment)
from benchmarks.haze_density.synthetic import generate, synthesize
from benchmarks.haze_density.common import NotAvailable


def independent_audit(output):
    """Recompute from arrays using SciPy/direct comparisons, not project metrics."""
    from scipy.stats import spearmanr
    base = ROOT / 'experiments/haze_benchmark'
    data = read_json(base / 'full_data/dataset.json')
    old = read_json(base / 'full_results/results.json')
    groups, spatial, controls = defaultdict(list), defaultdict(list), defaultdict(list)
    template_scores = []
    rng = np.random.default_rng(42)
    maximum_scipy_difference = maximum_mixture_error = maximum_saved_mean_error = 0.
    checked_cases, checked_predictions = set(), 0
    for row in old['images']:
        case = base / 'full_data' / row['path']
        assert sha256_file(case) == row['sha256']
        predfile = base / 'full_results' / row['prediction_path']
        assert sha256_file(predfile) == row['prediction_sha256']
        p = np.load(predfile, allow_pickle=False).astype(np.float64)
        with np.load(case, allow_pickle=False) as z:
            t = z['transmission'].astype(np.float64)
            if row['case_id'] not in checked_cases:
                _, expected, _ = synthesize(z['original_rgb'], z['transmission'], z['airlight'], row['brightness'], row['domain'])
                maximum_mixture_error = max(maximum_mixture_error, float(np.abs(expected-z['hazy_rgb']).max()))
                checked_cases.add(row['case_id'])
        maximum_saved_mean_error = max(maximum_saved_mean_error, abs(float(p.mean())-row['response_mean']))
        checked_predictions += 1
        if row['pattern'] == 'uniform':
            groups[(row['method'], row['scene_id'], tuple(row['airlight']), row['brightness'])].append((1-row['uniform_t'], float(p.mean())))
        else:
            # Independent fixed 4x4 pooling, data here are exactly 256x256.
            strength = (1-t).reshape(4, 64, 4, 64).mean(axis=(1, 3)).ravel()
            response = p.reshape(4, 64, 4, 64).mean(axis=(1, 3)).ravel()
            rho = float(spearmanr(strength, response).statistic)
            template_scores.append(float(spearmanr(strength, strength).statistic))
            maximum_scipy_difference = max(maximum_scipy_difference, abs(rho-row['region_spearman']))
            spatial[row['method']].append(rho)
            permuted = [float(spearmanr(strength, rng.permutation(response)).statistic) for _ in range(100)]
            controls[row['method']].append(float(np.mean(permuted)))
    result = {'status': 'PASS', 'checked_cases': len(checked_cases), 'checked_predictions': checked_predictions,
              'source_result_sha256': sha256_file(base / 'full_results/results.json'),
              'max_synthesis_reconstruction_error': maximum_mixture_error,
              'max_saved_response_mean_error': maximum_saved_mean_error,
              'max_region_spearman_difference_vs_scipy': maximum_scipy_difference,
              'methods': [], 'spatial_controls': {
                  'constant_prediction_correlation': None,
                  'constant_prediction_reason': 'undefined variance, not zero/perfect',
                  'known_template_spearman': float(np.mean(template_scores)),
                  'known_template_reason': 'same spatial map per group/pattern; a template oracle ignoring RGB scores 1. This is a protocol limitation, not an image method.',
                  'permutations_per_image': 100, 'seed': 42},
              'splits': sorted({c['split'] for c in data['cases']}),
              'geographic_groups': sorted({c['group_id'] for c in data['cases']})}
    for method in ('dcp', 'prior'):
        correct = pairs = 0
        rhos = []
        for key, rows in groups.items():
            if key[0] != method:
                continue
            x, y = np.asarray(sorted(rows)).T
            for i in range(len(x)):
                for j in range(i+1, len(x)):
                    pairs += 1
                    correct += int((y[j]-y[i]) * np.sign(x[j]-x[i]) > 1e-6)
            rhos.append(float(spearmanr(x, y).statistic))
        result['methods'].append({'method': method, 'correct_pairs': correct, 'comparable_pairs': pairs,
                                  'pairwise_accuracy': correct/pairs, 'spearman': float(np.mean(rhos)),
                                  'region_spearman': float(np.mean(spatial[method])),
                                  'shuffled_region_spearman_mean': float(np.mean(controls[method]))})
    assert maximum_scipy_difference < 1e-12 and maximum_mixture_error == 0
    write_json(output / 'historical_audit.json', result)
    return result


def prepare_sources(output):
    sources = ROOT / 'experiments/haze_benchmark/source_samples'
    spec = read_json(ROOT / 'configs/haze_benchmark_pilot_scenes.json')
    for scene in spec['scenes']:
        scene['path'] = str(sources / scene['path'])
    available = {s['scene_id']: s for s in read_json(sources / 'scenes.json')['scenes']}
    additions = [
        ('G_test_1013', 'urban', [0, 0, 240, 240]),
        ('G_test_1058', 'vegetation', [210, 0, 370, 160]),
        ('G_test_1058', 'water', [350, 340, 510, 500]),
        ('G_test_1141', 'bare_soil', [220, 180, 480, 440]),
    ]
    for source_id, cover, roi in additions:
        scene = {**available[source_id], 'scene_id': source_id + '_' + cover,
                 'group_id': spec['scenes'][0]['group_id'], 'land_cover': cover, 'roi': roi,
                 'path': str(sources / available[source_id]['path']),
                 'label_source': 'assistant visual screening of source RGB before new inference, 2026-10-22; NOT human ground truth',
                 'label_notes': 'Small mixed-cover diagnostic ROI; no verified geography, all sources conservatively ONE test group.'}
        spec['scenes'].append(scene)
    spec['sampling_note'] = 'Eight diagnostic ROIs (two per cover), seven source images, ONE geographic group; labels provisional, no automated classification.'
    write_json(output / 'scenes.json', spec)
    # Preserve a repository-portable manifest as well as resolved execution paths.
    portable = json.loads(json.dumps(spec))
    for s in portable['scenes']:
        s['path'] = '../../experiments/haze_benchmark/source_samples/' + Path(s['path']).name
    write_json(output / 'scenes_portable.json', portable)
    sheet = Image.new('RGB', (4*256, 2*286), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, s in enumerate(spec['scenes']):
        with Image.open(s['path']) as im:
            im = im.crop(tuple(s['roi'])).resize((256, 256))
            x, y = index % 4 * 256, index // 4 * 286
            sheet.paste(im, (x, y))
            draw.text((x+3, y+257), s['scene_id'] + ' / ' + s['land_cover'], fill='black')
    sheet.save(output / 'screened_rois.png')
    return output / 'scenes.json'


def rotate_spatial_cases(directory):
    """Sensitivity protocol: rotate existing truth before rendering RGB.

    No prediction or result is consulted. Uniform cases are preserved. Each
    case chooses a seeded quarter-turn, reducing shared positional templates.
    This still is not new geography or realistic atmospheric validation.
    """
    data = read_json(directory / 'dataset.json')
    for c in data['cases']:
        if c['pattern'] == 'uniform':
            continue
        k = int(hashlib.sha256(('42:rotation:' + c['case_id']).encode()).hexdigest()[:8], 16) % 4
        path = directory / c['path']
        with np.load(path, allow_pickle=False) as z:
            original, t, a = z['original_rgb'], np.rot90(z['transmission'], k).copy(), z['airlight']
        clean, hazy, clipping = synthesize(original, t, a, c['brightness'], c['domain'])
        np.savez_compressed(path, original_rgb=original, clear_rgb=clean, hazy_rgb=hazy, transmission=t, airlight=a)
        c['pattern_parameters']['quarter_turns'] = k
        c['sha256'] = sha256_file(path)
    data['protocol'] = 'supplementary seeded per-case quarter-turn spatial sensitivity; not the historical fixed-layout protocol'
    write_json(directory / 'dataset.json', data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--audit-history-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.audit_history_only:
        independent_audit(args.output.resolve())
        return
    # SciPy and this Windows torch wheel ship incompatible OpenMP runtimes.
    # Independent statistics run in a separate process; no unsafe DLL bypass.
    import torch
    from benchmarks.haze_density.evaluate import evaluate
    from benchmarks.haze_density.baselines import build_method
    from src.models.haze_density.physical_prior import min_max_normalize
    output = args.output.resolve() if args.run_prepared else fresh_directory(args.output).resolve()
    config = yaml.safe_load((ROOT / 'configs/haze_benchmark.yaml').read_text())
    seed_runtime(42, 1)
    if not args.run_prepared:
        subprocess.run([sys.executable, '-B', '-X', 'utf8', __file__, '--output', str(output), '--audit-history-only'], check=True)
        manifest = prepare_sources(output)
        checkpoint = ROOT / 'experiments/haze_density/checkpoints/best.pth'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        config['checkpoint'] = str(checkpoint)
        try:
            build_method('student', 'cpu', config)
            status, reason = 'AVAILABLE', 'Verify training scene isolation separately'
        except NotAvailable as error:
            status, reason = 'NOT RUN', str(error)
        write_json(output / 'weight_audit.json', {'status': status, 'reason': reason,
            'path': str(checkpoint), 'sha256': sha256_file(checkpoint), 'epoch': saved.get('epoch'),
            'config': saved.get('config'), 'training_scene_isolation': 'UNVERIFIED'})
        x = torch.linspace(.1, .8, 256).reshape(1, 1, 16, 16)
        shifted = .4 * x + .5
        write_json(output / 'normalization_audit.json', {
            'input_means': [float(x.mean()), float(shifted.mean())],
            'normalized_means': [float(min_max_normalize(x).mean()), float(min_max_normalize(shifted).mean())],
            'maximum_normalized_difference': float((min_max_normalize(x)-min_max_normalize(shifted)).abs().max()),
            'interpretation': 'Per-image min-max removes positive affine scale/offset (up to epsilon and rounding). No ablation or causal attribution of full-prior failure.'})
        write_json(output / 'environment.json', environment(42, config))
        write_json(output / 'config.json', config)
        if args.prepare_only:
            return
    else:
        manifest = output / 'scenes.json'
        config = read_json(output / 'config.json')
    generate(manifest, output / 'data', config, profile='full')
    rotate_spatial_cases(output / 'data')
    evaluate(output / 'data', output / 'results', config, 'cpu')
    print('Results:', output)


if __name__ == '__main__':
    main()
