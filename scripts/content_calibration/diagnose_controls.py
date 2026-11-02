"""Additional bounded CPU diagnostics; frozen estimators, no model search."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.content_calibration.run_cpu import models, records
from scripts.v15.run import read, write, tensor, digest, csv_write, procedural
from src.models.haze_density.content_calibration import response_objective, FixedCandidateFusion
from src.models.haze_density.semantic_prior import candidate_diagnostics
from benchmarks.haze_density.metrics import spatial_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'experiments/content_calibration_20261102/cpu_run')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/content_calibration_20261102/diagnostics')
    args = parser.parse_args()
    run, out = args.run, args.output
    out.mkdir(exist_ok=True, parents=True)
    if (out/'results.json').exists():
        raise SystemExit('Existing diagnostic preserved; choose a new directory')
    plan = read(run/'protocol.json'); cfg = plan['config']; base = plan['base']
    torch.set_num_threads(cfg['torch_threads'])
    data = dict(np.load(run/'sampled_training.npz', allow_pickle=False))
    z, w = response_objective(data['target'], data['baseline'], data['observed'], data['valid'], cfg['physical_strength'])
    equivalence = {'max_soft_target_minus_H': float(np.abs(z-data['target'][:, None]).max()),
                   'effective_weights_min_max': [float(w.min()), float(w.max())],
                   'effective_weight_mean_D_C_K': w.mean(0).tolist(),
                   'identity': 'If q=b+(s-b)H exactly, H_MSE+lambda*response_MSE = (1+lambda*valid*(s-b)^2)*H_MSE.',
                   'implication': 'Known J/A and independent H make this soft term signal-dependent sample weighting; '
                                  'no extra physical information is established. Weight-matched control required for future attribution.'}
    ms = models(run); fits = read(run/'fixed_weights.json')
    scenes = [s for s in plan['scenes'] if s['split'] == 'development']
    conditions = {'same_protocol': dict(base), 'display_domain': dict(base, domain='display_rgb'),
                  'unseen_A': dict(base, airlights=[[.55, .65, .75]])}
    rows, proxies, conditions_manifest = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for condition, config in conditions.items():
            config = dict(config, brightness=[1.], uniform_t=[1., .6], spatial_patterns=['gradient', 'regions'])
            for _, meta, rgb, y in records(config, scenes):
                image = tensor(rgb, 'cpu'); raw, context = ms['content'].extract(image)
                rng = np.random.default_rng(int(cfg['seed']) + int(meta['seed']))
                perm = torch.from_numpy(rng.permutation(y.size))
                shuffled = context.flatten(2)[:, :, perm].reshape_as(context)
                for name, model in ms.items():
                    for transform, ctx in [('aligned', context), ('shuffled_context', shuffled)]:
                        c = model.correct(raw, ctx)[0]
                        p = FixedCandidateFusion(fits[name]['weights'])(c)[0, 0].numpy()
                        d = candidate_diagnostics(c[0].numpy(), y)
                        rows.append({'condition': condition, 'method': name, 'context': transform,
                                     'case_id': meta['case_id'], 'group_id': meta['group_id'],
                                     'pixel_mae': float(np.abs(p-y).mean()),
                                     'region_spearman': spatial_metrics(1-y, p)['region_spearman'],
                                     'clear_response': float(p.mean()) if meta['uniform_t'] == 1 else None,
                                     **{k: d[k] for k in ['offline_oracle_mae', 'offline_oracle_mse', 'joint_high_fraction', 'joint_low_fraction']}})
                conditions_manifest.append({'condition': condition, **meta, 'domain': config['domain'],
                                            'A': config['airlights'][meta['airlight_index']]})
        # Complete pipeline timings including fixed multiplication/sum, not just table lookup.
        rgb, _ = procedural(scenes[0]['seed'], 256); image = tensor(rgb, 'cpu')
        timing = {}
        for name, model in ms.items():
            fusion = FixedCandidateFusion(fits[name]['weights'])
            for _ in range(2): fusion(model(image)[1])
            times = []
            for _ in range(7):
                start = time.perf_counter(); fusion(model(image)[1]); times.append(1000*(time.perf_counter()-start))
            timing[name] = {'full_fixed_pipeline_p50_ms': float(np.median(times)),
                            'full_fixed_pipeline_p95_ms': float(np.quantile(times, .95))}
    summary = {}
    for condition in conditions:
        summary[condition] = {}
        for name in ms:
            summary[condition][name] = {}
            for transform in ['aligned', 'shuffled_context']:
                rr = [r for r in rows if r['condition'] == condition and r['method'] == name and r['context'] == transform]
                summary[condition][name][transform] = {k: float(np.mean([r[k] for r in rr if r[k] is not None]))
                    for k in ['pixel_mae', 'region_spearman', 'clear_response', 'offline_oracle_mae', 'offline_oracle_mse', 'joint_high_fraction', 'joint_low_fraction']}
    # Correct pooling definitions: max of image maxima; conditional amplitude
    # weighted by outside-pixel fractions (all images same size).
    boundary = {}
    diagnostics = read(run/'candidate_diagnostics.json')
    for name in ms:
        dd = [d for d in diagnostics if d['method'] == name]
        total_fraction = sum(d['outside_envelope_fraction'] for d in dd)
        boundary[name] = {'outside_amplitude_mean_given_outside': sum(d['outside_envelope_fraction']*d['outside_envelope_conditional_mean'] for d in dd)/total_fraction,
                          'outside_amplitude_max': max(d['outside_envelope_max'] for d in dd)}
    baseline = read(run/'development_metrics.json')
    # Conservative predeclared user criterion: joint error and lower bound must
    # improve, rather than declaring success from envelope coverage alone.
    a, b = baseline['global'], baseline['content']
    gate = {'content_reduces_each_candidate_MAE': bool(all(np.asarray(b['prior_mae']) < a['prior_mae'])),
            'content_reduces_both_oracle_bounds': b['offline_oracle_mae'] < a['offline_oracle_mae'] and b['offline_oracle_mse'] < a['offline_oracle_mse'],
            'content_reduces_total_joint_error_fraction': b['joint_high_fraction']+b['joint_low_fraction'] < a['joint_high_fraction']+a['joint_low_fraction'],
            'fixed_fusion_uses_multiple_priors': any(sum(v > 1e-5 for v in fits[m]['weights']) > 1 for m in ms),
            'decision': 'Stop expansion/full training: joint-error criterion failed; three-prior necessity and physical attribution unproven. '
                        'Keep implementation and failure evidence. No dynamic/regression accuracy claim.'}
    csv_write(out/'per_case.csv', rows)
    write(out/'manifest.json', conditions_manifest)
    write(out/'results.json', {'frozen_run_sha256': digest(run/'artifact_sha256.json'),
          'source_sha256': {str(Path(__file__).relative_to(ROOT)).replace('\\', '/'): digest(__file__)},
          'physical_weight_equivalence': equivalence, 'boundary_pooling': boundary,
          'frozen_context_and_stress_diagnostics': summary, 'full_pipeline_timing_256': timing,
          'timing': 'CPU 4 threads, batch1, tensor-to-H including cues, context, tables and fixed fusion; warmup2 repeat7; no disk',
          'gate': gate, 'gpu_seconds': 0, 'new_confirmation': False,
          'cpu_elapsed_seconds': time.perf_counter()-started})
    print(json.dumps({'physical_equivalence': equivalence, 'gate': gate, 'stress': summary}, indent=2))


if __name__ == '__main__':
    main()
