"""Correctness against independently generated t, never against pseudo-labels."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import torch
from .baselines import build_method
from .common import read_json, write_json, sha256_file, fresh_directory, environment, NotAvailable
from .metrics import ranking_metrics, spatial_metrics, response_summary, confound_regions


def macro(rows, key):
    by_scene = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None:
            by_scene[row['group_id']].append(value)
    values = [float(np.mean(values)) for values in by_scene.values()]
    return {'value': float(np.mean(values)) if values else None,
            'n_valid_scene_groups': len(values), 'status': 'PASS' if values else 'N/A'}


def summarize(image_rows, rankings):
    summaries = []
    methods = sorted({row['method'] for row in image_rows})
    covers = sorted({row['land_cover'] for row in image_rows})
    for method in methods:
        for cover in ['ALL'] + covers:
            images = [r for r in image_rows if r['method'] == method and (cover == 'ALL' or r['land_cover'] == cover)]
            ranks = [r for r in rankings if r['method'] == method and (cover == 'ALL' or r['land_cover'] == cover)]
            if not images:
                continue
            clear = [r for r in images if r['pattern'] == 'uniform' and r['uniform_t'] == 1]
            summaries.append({'method': method, 'land_cover': cover,
                              'n_scene_groups': len({r['group_id'] for r in images}),
                              'n_cases': len(images),
                              **{k: macro(ranks, k) for k in ('pairwise_accuracy', 'tie_rate', 'spearman', 'monotonicity_violation_rate')},
                              'clear_response': macro(clear, 'response_mean'),
                              'region_spearman': macro(images, 'region_spearman'),
                              'dense_minus_light_response': macro(images, 'dense_minus_light_response')})
    return summaries


def evaluate(dataset_directory, output, config, device='cpu', split='test', methods=None):
    dataset_directory = Path(dataset_directory)
    data = read_json(dataset_directory / 'dataset.json')
    output = fresh_directory(output)
    cases = [r for r in data['cases'] if r['split'] == split]
    if not cases:
        raise NotAvailable(f'No cases in split {split}')
    # Verify saved truth and provenance before invoking any predictor.
    for case in cases:
        if sha256_file(dataset_directory / case['path']) != case['sha256']:
            raise ValueError(f'Case checksum mismatch: {case["case_id"]}')
    image_rows, method_results = [], []
    for method in methods or config['methods']:
        try:
            model, info = build_method(method, device, config)
        except NotAvailable as error:
            method_results.append({'method': method, 'status': 'NOT RUN',
                                   'accuracy_status': 'NOT EVALUATED', 'reason': str(error)})
            continue
        method_rows = []
        try:
            with torch.inference_mode():
                for index, case in enumerate(cases):
                    with np.load(dataset_directory / case['path'], allow_pickle=False) as arrays:
                        rgb, t, clear = arrays['hazy_rgb'], arrays['transmission'], arrays['clear_rgb']
                    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                        raise ValueError('Invalid saved RGB input')
                    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
                    prediction = model(tensor)
                    if tuple(prediction.shape) != (1, 1, *t.shape):
                        raise ValueError(f'Wrong prediction shape: {tuple(prediction.shape)}')
                    if not torch.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
                        raise ValueError('Prediction outside finite [0,1] contract')
                    p = prediction[0, 0].cpu().numpy()
                    target = output / 'predictions' / method / (case['case_id'] + '.npy')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.save(target, p, allow_pickle=False)
                    spatial = spatial_metrics(t, p, config['region_grid'])
                    method_rows.append({**case, 'method': method,
                                        'prediction_path': str(target.relative_to(output)).replace('\\', '/'),
                                        'prediction_sha256': sha256_file(target),
                                        **response_summary(p, config['response_reference_threshold']),
                                        'spatial_status': spatial['status'],
                                        'region_spearman': spatial['region_spearman'],
                                        'region_pairwise_accuracy': spatial['region_pairwise_accuracy'],
                                        'dense_minus_light_response': spatial['dense_minus_light_response'],
                                        'confound_regions': confound_regions(clear, p) if case['uniform_t'] == 1 else None})
                    if index % 20 == 0:
                        print(f'{method}: {index + 1}/{len(cases)} cases', flush=True)
            image_rows.extend(method_rows)
            method_results.append({**info, 'status': 'PASS', 'accuracy_judgment': 'DESCRIPTIVE: no arbitrary pass threshold',
                                   'evaluated_cases': len(method_rows)})
        except (ValueError, RuntimeError) as error:
            method_results.append({**info, 'status': 'FAIL', 'reason': str(error)})
        finally:
            del model
    groups = defaultdict(list)
    for row in image_rows:
        if row['pattern'] == 'uniform':
            groups[(row['method'], row['scene_id'], json.dumps(row['airlight']), row['brightness'])].append(row)
    rankings = []
    for (method, scene_id, airlight, brightness), rows in groups.items():
        ordered = sorted(rows, key=lambda r: 1 - r['uniform_t'])
        scores = [r['response_mean'] for r in ordered]
        strengths = [1 - r['uniform_t'] for r in ordered]
        rankings.append({'method': method, 'scene_id': scene_id, 'group_id': rows[0]['group_id'],
                         'land_cover': rows[0]['land_cover'], 'airlight': json.loads(airlight), 'brightness': brightness,
                         'strengths': strengths, 'scores': scores,
                         'strong_minus_clear_response': scores[-1] - scores[0],
                         **ranking_metrics(strengths, scores, config['tie_tolerance'])})
    stratified = []
    for key in sorted({(r['method'], r['land_cover'], json.dumps(r['airlight']), r['brightness']) for r in rankings}):
        method, cover, a, b = key
        rows = [r for r in rankings if (r['method'], r['land_cover'], json.dumps(r['airlight']), r['brightness']) == key]
        stratified.append({'method': method, 'land_cover': cover, 'airlight': json.loads(a), 'brightness': b,
                           'pairwise_accuracy': macro(rows, 'pairwise_accuracy'), 'spearman': macro(rows, 'spearman')})
    overall = 'FAIL' if any(r['status'] == 'FAIL' for r in method_results) else 'PASS' if image_rows else 'NOT RUN'
    spatial_strata = []
    for key in sorted({(r['method'], r['land_cover'], r['pattern']) for r in image_rows if r['pattern'] != 'uniform'}):
        method, cover, pattern = key
        rows = [r for r in image_rows if (r['method'], r['land_cover'], r['pattern']) == key]
        spatial_strata.append({'method': method, 'land_cover': cover, 'pattern': pattern,
                               **{k: macro(rows, k) for k in ('region_spearman', 'region_pairwise_accuracy', 'dense_minus_light_response')}})
    result = {'status': overall, 'data_kind': data['data_kind'], 'split': split,
              'dataset_directory': str(dataset_directory.resolve()),
              'dataset_manifest_sha256': sha256_file(dataset_directory / 'dataset.json'),
              'config': config, 'environment': environment(config['seed'], config),
              'methods': method_results, 'images': image_rows, 'rankings': rankings,
              'summary': summarize(image_rows, rankings), 'condition_strata': stratified,
              'spatial_strata': spatial_strata,
              'coverage': {cover: {'n_scene_groups': len({c['group_id'] for c in cases if c['land_cover'] == cover}),
                                  'status': 'PASS' if any(c['land_cover'] == cover for c in cases) else 'NOT RUN'}
                           for cover in ('urban', 'vegetation', 'water', 'bare_soil')},
              'physical_mae': {'status': 'N/A', 'reason': 'relative output; no independently fitted calibration'},
              'failure_candidates': sorted(rankings, key=lambda r: r['pairwise_accuracy'] if r['pairwise_accuracy'] is not None else 2)[:8],
              'spatial_failure_candidates': sorted([r for r in image_rows if r['region_spearman'] is not None],
                                                   key=lambda r: r['region_spearman'])[:8]}
    write_json(output / 'results.json', result)
    return result
