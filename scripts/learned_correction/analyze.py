"""Offline scene-grouped comparisons; no model/input changes or test tuning."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read(p): return json.loads(p.read_text(encoding='utf-8'))


def write(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def aggregate(rows, method, metric):
    groups = {}
    for r in rows:
        if r['method'] != method or r[metric] in ['', None]: continue
        groups.setdefault(r['group_id'], []).append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in groups.items()}


def compare(rows, cfg, candidate_mode='corrector16', reference_mode='rgb32'):
    seeds = cfg['seeds']; metrics = ['pixel_mae', 'region_spearman', 'clear_response']
    grouped = {metric: [] for metric in metrics}
    per_seed = {}
    for seed in seeds:
        names = [f'{candidate_mode}_seed{seed}', f'{reference_mode}_seed{seed}']
        per_seed[str(seed)] = {}
        for metric in metrics:
            maps = [aggregate(rows, name, metric) for name in names]
            keys = sorted(set(maps[0]) & set(maps[1]))
            if not keys: raise ValueError(f'Missing comparable results for {names}, {metric}')
            a, b = [np.array([mapping[k] for k in keys]) for mapping in maps]
            grouped[metric].append((a, b))
            per_seed[str(seed)][metric] = {'candidate': float(a.mean()), 'reference': float(b.mean()),
                'relative_increase': float(a.mean()/b.mean()-1) if b.mean() else None}
    result = {'per_seed': per_seed, 'bootstrap_scope': 'Average fixed three training seeds per scene; resample original scenes. Does not characterize all training-seed uncertainty.'}
    rng = np.random.default_rng(20261102)
    for metric, arrays in grouped.items():
        a = np.mean([v[0] for v in arrays], axis=0); b = np.mean([v[1] for v in arrays], axis=0)
        ix = rng.integers(0, len(a), (10000, len(a)))
        if metric == 'pixel_mae':
            observed = a.mean()/b.mean()-1; draws = a[ix].mean(1)/b[ix].mean(1)-1
            margin = cfg['relative_MAE_margin']
        elif metric == 'region_spearman':
            observed = b.mean()-a.mean(); draws = b[ix].mean(1)-a[ix].mean(1); margin = cfg['spatial_margin']
        else:
            observed = a.mean()-b.mean(); draws = a[ix].mean(1)-b[ix].mean(1); margin = cfg['clear_response_margin']
        interval = np.quantile(draws, [.025, .975]).tolist()
        result[metric] = {'candidate_mean': float(a.mean()), 'reference_mean': float(b.mean()),
                          'difference_or_relative_increase': float(observed), 'scene_bootstrap_95': interval,
                          'margin': margin, 'point_within_margin': bool(observed <= margin),
                          'upper_95_within_margin': bool(interval[1] <= margin), 'original_scenes': len(a)}
    result['all_point_margins_passed'] = all(result[k]['point_within_margin'] for k in metrics)
    result['all_upper_95_margins_passed'] = all(result[k]['upper_95_within_margin'] for k in metrics)
    return result


def ranking(rows, method):
    rr = [r for r in rows if r['method'] == method and r['pattern'] == 'uniform']
    correct = count = same_correct = same_count = 0
    for i, a in enumerate(rr):
        for b in rr[i+1:]:
            ya, yb = float(a['truth_mean']), float(b['truth_mean'])
            if abs(ya-yb) < 1e-6: continue
            good = (float(a['prediction_mean'])-float(b['prediction_mean']))*np.sign(ya-yb) > 1e-6
            if a['group_id'] != b['group_id']: correct += good; count += 1
            elif a['airlight_index'] == b['airlight_index'] and a['brightness_index'] == b['brightness_index']:
                same_correct += good; same_count += 1
    return {'same_scene_rank_accuracy': float(same_correct/same_count) if same_count else None,
            'cross_scene_nuisance_rank_accuracy': float(correct/count) if count else None,
            'same_pairs': same_count, 'cross_pairs': count}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--split', choices=['development', 'confirmation'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(exist_ok=False, parents=True)
    cfg = read(args.run/'protocol.json')['config']; suffix = '_'.join(map(str, cfg['seeds']))
    output = {}
    for condition in ['nominal', 'unseen_A', 'display_rgb']:
        folder = args.run/args.split/condition
        path = folder/f'per_image_{suffix}.csv'
        if not path.exists(): continue
        with path.open(encoding='utf-8', newline='') as f: rows = list(csv.DictReader(f))
        diagnostics = read(folder/f'candidate_diagnostics_{suffix}.json')
        candidate_summary = {}
        for mode in ['original', *[f'corrector16_seed{s}' for s in cfg['seeds']]]:
            dd = [d for d in diagnostics if d['method'] == mode]
            candidate_summary[mode] = {k: np.mean([d[k] for d in dd], axis=0).tolist() for k in
                ['candidate_mae', 'candidate_bias', 'joint_high', 'joint_low', 'outside_fraction', 'oracle_mae', 'oracle_mse']}
            candidate_summary[mode]['outside_global_max'] = max(d['outside_max'] for d in dd)
            candidate_summary[mode]['outside_conditional_amplitude'] = sum(d['oracle_mae'] for d in dd)/sum(d['outside_fraction'] for d in dd)
        small = 'direct_same16' if 'direct_same16' in cfg['modes'] else 'rgb_context16'
        output[condition] = {'vs_RGB32': compare(rows, cfg), 'vs_same_capacity_same_information_direct': compare(rows, cfg, reference_mode=small),
            'small_direct_vs_RGB32': compare(rows, cfg, candidate_mode=small),
            'candidate_diagnostics': candidate_summary,
            'rankings': {name: ranking(rows, name) for name in sorted(set(r['method'] for r in rows))}}
    write(args.output/'comparison.json', output)
    for condition, result in output.items():
        a = result['vs_RGB32']
        print(condition, 'C vs RGB32', a['pixel_mae'], 'all upper margins:', a['all_upper_95_margins_passed'])


if __name__ == '__main__': main()
