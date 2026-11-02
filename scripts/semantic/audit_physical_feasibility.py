"""CPU-only equation audit; clean images/A/t are offline diagnostic inputs.

No estimator is trained or evaluated here. This does not establish a deployable
baseline estimator, natural-haze accuracy, or a new physical law.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from benchmarks.haze_density.synthetic import (
    linear_to_srgb, srgb_to_linear, synthesize, transmission_map)
from scripts.v15.run import original_scenes, procedural
from src.models.haze_density.calibrated_prior import raw_cues


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cues(rgb):
    image = torch.from_numpy(np.asarray(rgb).transpose(2, 0, 1).copy()).double()[None]
    value = raw_cues(image, window_size=15)[0].numpy()
    value[1:] *= -1  # positive raw contrast and color difference
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/physical_feasibility_20261102'))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists() or (out/'plan.json').exists():
        raise SystemExit('Existing audit: use a new output directory; preserve evidence.')
    cfg = json.loads((ROOT/'configs/v15_calibration.json').read_text(encoding='utf-8'))
    scenes = original_scenes(cfg, 'development')[:4]
    patterns = [('uniform', 1.), ('uniform', .6), ('uniform', .4),
                ('gradient', None), ('blob', None), ('regions', None)]
    variants = ['correct_linear_known_A', 'wrong_display_equations',
                'linear_with_perturbed_A']
    plan = {
        'purpose': 'Offline equation/conditioning feasibility, not model efficacy',
        'split': 'previously exposed development only', 'scenes': scenes,
        'image_size': cfg['image_size'], 'window': 15,
        'patterns': patterns, 'airlights': cfg['airlights'], 'brightness': 1.,
        'variants': variants, 'A_perturbation': [.02, -.02, .01],
        'conditioning_thresholds': [.0001, .001, .01, .05],
        'reported_inverse_threshold': .01,
        'baseline_perturbation': .005,
        'boundary': 'within seven pixels of regions quadrant seam, diagnostic only',
        'truth_policy': 'J,A,t used only for offline mechanism/oracle diagnostics',
        'gpu_used': False, 'training_steps': 0,
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
            Path(__file__).resolve(), ROOT/'benchmarks/haze_density/synthetic.py',
            ROOT/'src/models/haze_density/calibrated_prior.py',
            ROOT/'src/models/haze_density/physical_prior.py',
            ROOT/'src/models/haze_density/guided_filter.py',
            ROOT/'scripts/v15/run.py', ROOT/'configs/v15_calibration.json']}}
    write(out/'plan.json', plan)
    started = time.perf_counter()
    torch.set_num_threads(4)
    records, manifest = [], []
    size = cfg['image_size']
    yy, xx = np.mgrid[:size, :size]
    seam = (np.abs(xx-(size-1)/2) <= 7) | (np.abs(yy-(size-1)/2) <= 7)
    thresholds = plan['conditioning_thresholds']
    for scene in scenes:
        original, _unused_procedural_name = procedural(scene['seed'], size)
        for ai, a_list in enumerate(cfg['airlights']):
            a = np.asarray(a_list, dtype=np.float32)
            for pattern, level in patterns:
                t, _ = transmission_map(pattern, size, scene['seed'], level)
                clean, hazy, _ = synthesize(original, t, a, 1., 'linear_srgb')
                case = f'{scene["scene_id"]}_a{ai}_{pattern}_{level}'
                manifest.append({'case_id': case, 'group_id': scene['group_id'],
                    'clean_sha256': hashlib.sha256(clean.tobytes()).hexdigest(),
                    'hazy_sha256': hashlib.sha256(hazy.tobytes()).hexdigest(),
                    'target_sha256': hashlib.sha256(t.tobytes()).hexdigest()})
                for variant in variants:
                    if variant == 'wrong_display_equations':
                        j, image = clean.astype(float), hazy.astype(float)
                        assumed_a = linear_to_srgb(a).astype(float)
                    else:
                        j, image = srgb_to_linear(clean).astype(float), srgb_to_linear(hazy).astype(float)
                        assumed_a = a.astype(float)
                        if variant == 'linear_with_perturbed_A':
                            assumed_a = np.clip(assumed_a + [.02, -.02, .01], .01, 1.)
                    # No per-image stretching or clipping of A-normalized RGB.
                    baseline, observed = cues(j/assumed_a), cues(image/assumed_a)
                    expected = baseline*t[None]
                    expected[0] += 1-t
                    for k, name in enumerate(['dark', 'contrast', 'color_difference']):
                        b, q = baseline[k], observed[k]
                        denominator = 1-b if k == 0 else b
                        numerator = 1-q if k == 0 else q
                        valid = np.abs(denominator) >= .01
                        residual = np.abs(q-expected[k])
                        row = dict(case_id=case, scene_id=scene['scene_id'], pattern=pattern,
                                   uniform_t=level, airlight_index=ai, variant=variant, prior=name,
                                   response_mae=float(residual.mean()),
                                   response_max_error=float(residual.max()),
                                   inverse_coverage=float(valid.mean()))
                        for threshold in thresholds:
                            row[f'coverage_at_{threshold}'] = float((np.abs(denominator) >= threshold).mean())
                        if valid.any():
                            t_hat = numerator[valid]/denominator[valid]
                            row['offline_inverse_H_mae'] = float(np.abs(np.clip(t_hat, 0, 1)-t[valid]).mean())
                            row['offline_inverse_unclipped_H_mae'] = float(np.abs(t_hat-t[valid]).mean())
                            perturb_errors = []
                            for delta in [-.005, .005]:
                                changed = denominator-delta if k == 0 else denominator+delta
                                take = valid & (np.abs(changed) >= .0001)
                                if take.any():
                                    estimate = numerator[take]/changed[take]
                                    perturb_errors.append(float(np.abs(np.clip(estimate, 0, 1)-t[take]).mean()))
                            row['baseline_perturb_H_mae_mean'] = float(np.mean(perturb_errors)) if perturb_errors else None
                        else:
                            row['offline_inverse_H_mae'] = None
                            row['offline_inverse_unclipped_H_mae'] = None
                            row['baseline_perturb_H_mae_mean'] = None
                        row['regions_seam_response_mae'] = float(residual[seam].mean()) if pattern == 'regions' else None
                        row['regions_interior_response_mae'] = float(residual[~seam].mean()) if pattern == 'regions' else None
                        records.append(row)
    keys = ['response_mae', 'inverse_coverage', 'offline_inverse_H_mae',
            'offline_inverse_unclipped_H_mae', 'baseline_perturb_H_mae_mean',
            'regions_seam_response_mae', 'regions_interior_response_mae'] + [f'coverage_at_{x}' for x in thresholds]
    summary = {}
    for variant in variants:
        summary[variant] = {}
        for pattern in ['all', 'uniform', 'gradient', 'blob', 'regions']:
            summary[variant][pattern] = {}
            for name in ['dark', 'contrast', 'color_difference']:
                rr = [r for r in records if r['variant'] == variant and r['prior'] == name
                      and (pattern == 'all' or r['pattern'] == pattern)]
                values = {key: [r[key] for r in rr if r[key] is not None] for key in keys}
                summary[variant][pattern][name] = {key: float(np.mean(v)) if v else None for key, v in values.items()}
    # Mathematical sanity checks; only the explicitly applicable cases.
    for row in records:
        if row['variant'] == 'correct_linear_known_A' and (
                row['pattern'] == 'uniform' or row['prior'] == 'color_difference'):
            if row['response_max_error'] > 1e-5:
                raise AssertionError(f'Applicable response equation failed: {row}')
    identical_rgb = .8
    ambiguity = [{'A': 1., 't': t_value, 'J': (identical_rgb-(1-t_value))/t_value,
                  'I': identical_rgb, 'H': 1-t_value} for t_value in [.5, .8]]
    result = {'status': 'offline mechanism audit completed, no trained estimator',
              'source_cases': len(manifest), 'diagnostic_rows': len(records),
              'aggregation': 'equal mean over cases; inverse errors conditional on denominator>=0.01; coverage always reported',
              'summary': summary, 'nonidentifiability_example': ambiguity,
              'cpu_elapsed_seconds': time.perf_counter()-started,
              'gpu_used': False, 'training_steps': 0,
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'numpy': np.__version__, 'device': 'cpu', 'threads': 4}}
    write(out/'results.json', result)
    write(out/'manifest.json', manifest)
    with (out/'per_case.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    write(out/'artifact_sha256.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k: v for k, v in result.items() if k != 'summary'}, ensure_ascii=True))


if __name__ == '__main__':
    main()
