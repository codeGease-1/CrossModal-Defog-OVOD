"""Bounded CPU-only feasibility loop. Never opens a confirmation split.

Stages are restartable at completed artifact boundaries with --resume. Training
samples/physics metadata live separately from frozen RGB-only candidate caches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
import torch
from benchmarks.haze_density.synthetic import synthesize, srgb_to_linear
from benchmarks.haze_density.metrics import spatial_metrics, ranking_metrics
from scripts.v15.run import read, write, digest, procedural, original_scenes, cases, tensor, csv_write
from scripts.v2.global_optimum import solve_simplex
from src.models.haze_density.calibrated_prior import raw_cues
from src.models.haze_density.content_calibration import (
    content_features, fit_content_calibration, ContentCalibratedCandidates, FixedCandidateFusion, ContentDirectReference)
from src.models.haze_density.semantic_prior import candidate_diagnostics
from src.models.haze_density.student_weight_gate import StudentWeightGate

METHODS = ['global', 'content', 'content_physical']


def source_files():
    return [Path(__file__).resolve(), ROOT/'configs/content_calibration.json',
            ROOT/'scripts/content_calibration/test_contracts.py',
            ROOT/'src/models/haze_density/content_calibration.py',
            *[ROOT/f'src/models/haze_density/{n}.py' for n in
              ['calibrated_prior', 'physical_prior', 'guided_filter', 'semantic_prior', 'dynamic_prior', 'student_weight_gate']],
            ROOT/'scripts/v15/run.py', ROOT/'scripts/v2/global_optimum.py',
            ROOT/'benchmarks/haze_density/synthetic.py', ROOT/'benchmarks/haze_density/metrics.py',
            ROOT/'configs/v15_calibration.json']


def stamp(out, stage, **extra):
    write(out/'recovery_state.json', {'stage': stage, 'device': 'cpu',
          'gpu_seconds': 0, 'optimizer': None, 'rng': 'deterministic per-case numpy seed; no neural training', **extra})


def formation_terms(clear, rgb, target, meta, base):
    """TRAIN/offline only, on the actual brightness-adjusted clean source."""
    a = np.asarray(base['airlights'][meta['airlight_index']], dtype=np.float32)
    clean, _, _ = synthesize(clear, 1-target, a, meta['brightness'], base['domain'])
    j, image = srgb_to_linear(clean), srgb_to_linear(rgb)
    b = raw_cues(tensor(j/a, 'cpu'), base['prior']['window_size'])[0].numpy()
    q = raw_cues(tensor(image/a, 'cpu'), base['prior']['window_size'])[0].numpy()
    b[1:] *= -1
    q[1:] *= -1
    valid = np.ones_like(b, dtype=bool)
    # Conservative TRAIN-only eligibility: d/c only on known uniform variants.
    # No true-t-derived mask or baseline is ever stored in the inference cache.
    valid[:2] = meta['pattern'] == 'uniform'
    return b, q, valid


def records(base, scenes):
    for scene in scenes:
        original, _unused_design_name = procedural(scene['seed'], base['image_size'])
        for meta, rgb, y in cases(original, scene, base):
            yield original, meta, rgb, y


def prepare(out, cfg, resume):
    sources = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in source_files()}
    if (out/'protocol.json').exists():
        previous = read(out/'protocol.json')
        if not resume:
            raise SystemExit('Existing run: use --resume or a fresh output; no overwrite')
        if previous['config'] != cfg or previous['source_sha256'] != sources:
            raise SystemExit('Config/source changed: preserve this run and choose a new output')
        return previous['base'], previous['scenes']
    base = read(ROOT/'configs/v15_calibration.json')
    base.update({k: cfg[k] for k in ['image_size', 'uniform_t', 'spatial_patterns', 'airlights', 'brightness', 'domain']})
    base['prior']['window_size'] = cfg['cue_window']
    if cfg['domain'] != 'linear_srgb' or cfg['new_gpu_authorized_seconds'] != 0 or cfg['device'] != 'cpu':
        raise ValueError('This runner only supports the declared CPU linear-sRGB fixture')
    scenes = [s for split, n in [('train', cfg['train_scenes']), ('development', cfg['development_scenes'])]
              for s in original_scenes(base, split)[:n]]
    groups, hashes = set(), set()
    for s in scenes:
        original, _ = procedural(s['seed'], cfg['image_size'])
        h = hashlib.sha256(original.tobytes()).hexdigest()
        assert s['group_id'] not in groups and h not in hashes
        groups.add(s['group_id']); hashes.add(h)
        s['original_sha256'] = h
    protocol = {'config': cfg, 'base': base, 'scenes': scenes, 'source_sha256': sources,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'command': subprocess.list2cmdline(sys.argv),
                'environment': {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
                                'numpy': np.__version__, 'platform': platform.platform(), 'device': 'cpu',
                                'threads': cfg['torch_threads'], 'cuda_used': False},
                'data_policy': 'Exposed procedural train/dev originals only; same original all versions same group. '
                               'Design names are not semantic labels. No natural-haze truth or new confirmation.',
                'physics': 'Training-only actual J after brightness and clipping; positive constant known A, scalar t. '
                           'Raw positive d,c,k of J/A and I/A in linear domain, no clipping/power/guided filtering. '
                           'd/c uniform cases only, k all cases. No inference physics applicability gate.',
                'comparability': 'All three fits share samples/knots/support/inputs; physical penalty differs only in objective. '
                                 'Historical v2 scores/weights are not fair controls for this changed raw-cue protocol.'}
    write(out/'protocol.json', protocol)
    (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    return base, scenes


def sample_train(out, cfg, base, scenes):
    path = out/'sampled_training.npz'
    if path.exists():
        return dict(np.load(path, allow_pickle=False))
    arrays = {k: [] for k in ['raw', 'target', 'context', 'groups', 'baseline', 'observed', 'valid']}
    manifest, reconstruction, ideal_residual, clip_fractions = [], [], [], []
    started = time.perf_counter()
    for original, meta, rgb, y in records(base, [s for s in scenes if s['split'] == 'train']):
        image = tensor(rgb, 'cpu')
        raw = raw_cues(image, cfg['cue_window'])[0].numpy().reshape(3, -1).T
        context = content_features(image, cfg['content_window'])[0].numpy().reshape(2, -1).T
        b, q, valid = formation_terms(original, rgb, y, meta, base)
        expected = b*(1-y)[None]
        expected[0] += y
        ideal_residual.append(float(np.abs(q-expected)[valid].max()))
        a = np.asarray(base['airlights'][meta['airlight_index']], np.float32)
        j = np.clip(srgb_to_linear(original)*meta['brightness'], 0, 1)
        reconstruction.append(float(np.abs(srgb_to_linear(rgb)-(j*(1-y)[..., None]+a*y[..., None])).max()))
        clip_fractions.append(float((srgb_to_linear(original)*meta['brightness'] > 1).mean()))
        rng = np.random.default_rng(cfg['seed'] + int(hashlib.sha256(meta['case_id'].encode()).hexdigest()[:8], 16))
        idx = rng.choice(y.size, min(cfg['pixels_per_case'], y.size), replace=False)
        values = {'raw': raw[idx], 'context': context[idx], 'target': y.ravel()[idx],
                  'groups': np.repeat(meta['group_id'], len(idx)),
                  'baseline': b.reshape(3, -1).T[idx], 'observed': q.reshape(3, -1).T[idx],
                  'valid': valid.reshape(3, -1).T[idx]}
        for k, v in values.items():
            arrays[k].append(v)
        manifest.append({**meta, 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                         'target_sha256': hashlib.sha256(y.tobytes()).hexdigest(), 'sample_index_sha256': hashlib.sha256(idx.tobytes()).hexdigest()})
    data = {k: np.concatenate(v) for k, v in arrays.items()}
    assert max(ideal_residual) < 2e-5 and max(reconstruction) < 2e-6
    np.savez_compressed(path, **data)
    write(out/'train_manifest.json', manifest)
    write(out/'data_audit.json', {'original_groups_disjoint': True, 'decoded_original_duplicates': 0,
          'training_cases': len(manifest), 'sampled_pixels': len(data['target']),
          'target': 'known synthetic scalar H=1-t; independent of old teacher',
          'linear_formation_max_abs_error': max(reconstruction),
          'eligible_forward_response_max_abs_error': max(ideal_residual),
          'brightness_clipped_fraction_max': max(clip_fractions),
          'context_sample_quantiles': np.quantile(data['context'], [0, .01, .5, .99, 1], axis=0).tolist(),
          'physics_eligible_fraction_D_C_K': data['valid'].mean(0).tolist(),
          'seconds': time.perf_counter()-started,
          'RSHaze': 'Existing G reconstruction failure and S channelwise target restrictions retained; no new RSHaze labels fitted',
          'coverage': 'Small artificial texture/geometric family only. No validated roof/water/vegetation semantic coverage',
          'semantics': 'No segmenter used; prior semantic failures remain evidence, no semantic robustness claim'})
    stamp(out, 'training_samples_complete')
    return data


def fit(out, cfg, data):
    if (out/'calibration_content_physical.json').exists():
        return
    args = {k: cfg[k] for k in ['bins', 'min_pixels', 'min_scenes', 'shrinkage', 'max_alpha', 'max_delta', 'cue_window', 'content_window']}
    started = time.perf_counter()
    physical = {'verified_protocol': 'linear_srgb_scalar_t_constant_positive_A',
                **{k: data[k] for k in ['baseline', 'observed', 'valid']}}
    for name, strength in [('content', 0.), ('content_physical', cfg['physical_strength'])]:
        calibration = fit_content_calibration(data['raw'], data['target'], data['context'], data['groups'],
                         split='train', physical_strength=strength, physical=physical, **args)
        write(out/f'calibration_{name}.json', calibration)
    write(out/'fit_cost.json', {'cpu_seconds': time.perf_counter()-started, 'gpu_seconds': 0,
          'objective': 'Per-candidate sampled pixel H MSE; optional raw forward-response MSE. No fused-loss calibration.'})
    stamp(out, 'calibration_complete')


def models(out):
    plain = read(out/'calibration_content.json')
    physical = read(out/'calibration_content_physical.json')
    return {'global': ContentCalibratedCandidates(plain, conditional=False),
            'content': ContentCalibratedCandidates(plain),
            'content_physical': ContentCalibratedCandidates(physical)}


def fixed_weights(out, cfg, base, scenes, ms):
    if (out/'fixed_weights.json').exists():
        return read(out/'fixed_weights.json')
    stats = {m: [np.zeros((3, 3)), np.zeros(3), np.zeros((3, 3)), np.zeros(3), np.zeros((3, 3))] for m in METHODS}
    n = nc = 0
    started = time.perf_counter()
    for _, meta, rgb, y in records(base, [s for s in scenes if s['split'] == 'train']):
        raw, context = ms['content'].extract(tensor(rgb, 'cpu'))
        clear = meta['pattern'] == 'uniform' and meta['uniform_t'] == 1.
        for name, model in ms.items():
            p = model.correct(raw, context)[0][0].numpy().reshape(3, -1).astype(float)
            q, r, qg, rg, qc = stats[name]
            q += p@p.T/p.shape[1]; r += p@y.ravel()/p.shape[1]
            mean = p.mean(1); qg += np.outer(mean, mean); rg += mean*y.mean()
            if clear:
                qc += p@p.T/p.shape[1]
        n += 1; nc += int(clear)
    result = {}
    for name, (q, r, qg, rg, qc) in stats.items():
        Q = q/n + cfg['global_loss_weight']*qg/n + cfg['clear_loss_weight']*qc/nc
        R = r/n + cfg['global_loss_weight']*rg/n
        result[name] = {'weights': solve_simplex(Q, R).tolist(), 'Q': Q.tolist(), 'r': R.tolist(),
                        'training_cases': n, 'clear_cases': nc}
    result['_cost'] = {'cpu_seconds': time.perf_counter()-started, 'fitted_on': 'train only; seven simplex faces'}
    write(out/'fixed_weights.json', result)
    stamp(out, 'fixed_fusion_complete')
    return result


def evaluate(out, cfg, base, scenes, ms, fits):
    if (out/'development_metrics.json').exists():
        return read(out/'development_metrics.json')
    rows, diagnostics, probes, manifest, cache_manifest = [], [], [], [], []
    candidate_dir = out/'frozen_candidates'; candidate_dir.mkdir(exist_ok=True)
    sample_image = None
    for original, meta, rgb, y in records(base, [s for s in scenes if s['split'] == 'development']):
        image = tensor(rgb, 'cpu')
        raw, context = ms['content'].extract(image)
        b, q, valid = formation_terms(original, rgb, y, meta, base)
        # Evaluation-only clear-content proxies; never inference inputs.
        gray = original @ np.array([.299, .587, .114])
        dy, dx = np.gradient(gray)
        masks = {'bright_proxy': gray > .8, 'low_texture_proxy': np.hypot(dx, dy) < .01}
        for name, model in ms.items():
            maps, used = model.correct(raw, context)
            c = maps[0].numpy()
            p = FixedCandidateFusion(fits[name]['weights'])(maps)[0, 0].numpy()
            rho = spatial_metrics(1-y, p)['region_spearman']
            rows.append({**{k: meta[k] for k in ['case_id', 'scene_id', 'group_id', 'airlight_index', 'brightness_index', 'pattern', 'uniform_t']},
                         'method': name, 'truth_mean': float(y.mean()), 'prediction_mean': float(p.mean()),
                         'pixel_mae': float(np.abs(p-y).mean()), 'global_abs_error': float(abs(p.mean()-y.mean())),
                         'region_spearman': rho, 'clear_response': float(p.mean()) if meta['uniform_t'] == 1. else None})
            diag = candidate_diagnostics(c, y, cfg['bias_tolerance'])
            lo, hi = c.min(0), c.max(0)
            outside = np.maximum(lo-y, 0) + np.maximum(y-hi, 0)
            diag.update({'method': name, 'case_id': meta['case_id'],
                         'outside_envelope_conditional_mean': float(outside[outside > 0].mean()) if (outside > 0).any() else 0.,
                         'outside_envelope_max': float(outside.max()),
                         'fallback_fraction_D_C_K': (1-used.float().mean((0, 2, 3))).tolist()})
            expected = b*(1-c); expected[0] += c[0]
            diag['offline_forward_response_mae_D_C_K'] = [float(np.abs(q[k]-expected[k])[valid[k]].mean()) if valid[k].any() else None for k in range(3)]
            diagnostics.append(diag)
            for region, mask in masks.items():
                if mask.any():
                    probes.append({'case_id': meta['case_id'], 'method': name, 'proxy': region, 'pixels': int(mask.sum()),
                                   'fused_mae': float(np.abs(p-y)[mask].mean()),
                                   **{f'candidate_{k}_mae': float(np.abs(c[k]-y)[mask].mean()) for k in range(3)}})
            if sample_image is None:
                np.savez_compressed(candidate_dir/f'{name}.npz', candidates=c, rgb=rgb)
                cache_manifest.append({'method': name, 'case_id': meta['case_id'], 'sha256': digest(candidate_dir/f'{name}.npz'),
                                       'calibration_sha256': digest(out/('calibration_content_physical.json' if name == 'content_physical' else 'calibration_content.json')),
                                       'keys': ['candidates', 'rgb'], 'truth_or_clean_or_A_or_scene_input': False})
        if sample_image is None:
            sample_image = image
            Image.fromarray(np.round(rgb*255).astype('uint8')).save(out/'input_rgb.png')
        if meta['pattern'] == 'regions' and meta['airlight_index'] == 0 and meta['brightness_index'] == 1:
            render(out/f'evidence_{meta["scene_id"]}.png', rgb, y, raw, context, ms, fits)
        manifest.append({**meta, 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'target_sha256': hashlib.sha256(y.tobytes()).hexdigest()})
    metrics = {}
    for name in METHODS:
        rr = [r for r in rows if r['method'] == name]
        dd = [d for d in diagnostics if d['method'] == name]
        metrics[name] = {k: float(np.mean([r[k] for r in rr if r[k] is not None])) for k in
                         ['pixel_mae', 'global_abs_error', 'region_spearman', 'clear_response']}
        for key in ['prior_mae', 'prior_bias', 'joint_high_fraction', 'joint_low_fraction', 'outside_envelope_fraction',
                    'offline_oracle_mae', 'offline_oracle_mse', 'outside_envelope_conditional_mean', 'outside_envelope_max',
                    'fallback_fraction_D_C_K']:
            metrics[name][key] = np.mean([d[key] for d in dd], axis=0).tolist()
        # All different-scene/nuisance uniform pairs with unequal truth; ties count as errors.
        uniform = [r for r in rr if r['pattern'] == 'uniform']
        a, n, same_a, same_n = 0, 0, 0, 0
        for i, r in enumerate(uniform):
            for s in uniform[i+1:]:
                if abs(r['truth_mean']-s['truth_mean']) < 1e-6:
                    continue
                correct = (r['prediction_mean']-s['prediction_mean'])*np.sign(r['truth_mean']-s['truth_mean']) > 1e-6
                if r['scene_id'] != s['scene_id']:
                    a += correct; n += 1
                elif r['airlight_index'] == s['airlight_index'] and r['brightness_index'] == s['brightness_index']:
                    same_a += correct; same_n += 1
        metrics[name].update(cross_scene_rank=float(a/n), same_scene_rank=float(same_a/same_n), cases=len(rr))
    csv_write(out/'development_per_image.csv', rows)
    csv_write(out/'content_proxy_errors.csv', probes)
    write(out/'candidate_diagnostics.json', diagnostics)
    write(out/'development_manifest.json', manifest)
    write(out/'frozen_candidate_manifest.json', cache_manifest)
    write(out/'development_metrics.json', metrics)
    stamp(out, 'development_diagnostics_complete')
    return metrics


def render(path, rgb, y, raw, context, ms, fits):
    maps = [('H target', y)]
    for name, model in ms.items():
        c = model.correct(raw, context)[0]
        maps.extend([(name+' '+str(k), c[0, k].numpy()) for k in range(3)])
        maps.append((name+' fused', FixedCandidateFusion(fits[name]['weights'])(c)[0, 0].numpy()))
    size = y.shape[0]; width = size*5
    canvas = Image.new('RGB', (width, (size+24)*3+30), 'white')
    draw = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray((rgb*255).astype('uint8')), (0, 24)); draw.text((2, 3), 'hazy RGB', fill='black')
    for i, (label, value) in enumerate(maps, 1):
        xx, yy = (i % 5)*size, (i//5)*(size+24)
        v = np.clip(value, 0, 1)
        color = np.stack([v, .25 + .5*(1-np.abs(2*v-1)), 1-v], -1)
        canvas.paste(Image.fromarray((color*255).astype('uint8')), (xx, yy+24))
        draw.text((xx+2, yy+3), label, fill='black')
    draw.text((2, canvas.height-20), 'All H maps: fixed 0 (blue) to 1 (red); no per-image stretching', fill='black')
    canvas.save(path)


def cost_and_contract(out, cfg, ms):
    if (out/'cost_and_contract.json').exists():
        return
    image = tensor(np.asarray(Image.open(out/'input_rgb.png'), np.float32)/255, 'cpu')
    from scripts.v2.rgb_control import RGBControl
    torch.manual_seed(cfg['seed'])
    gate, regressor, matched = StudentWeightGate().eval(), RGBControl().eval(), ContentDirectReference().eval()
    costs = {}
    # Model cost is measured only; random network outputs are not effect results.
    for size in [128, 256]:
        x = torch.nn.functional.interpolate(image, size=(size, size), mode='bilinear', align_corners=False)
        for name, fn in [('raw_prior', lambda: raw_cues(x, cfg['cue_window'])),
                         ('content_features', lambda: content_features(x, cfg['content_window'])),
                         ('global_candidates', lambda: ms['global'](x)),
                         ('content_candidates', lambda: ms['content'](x)),
                         ('content_plus_dynamic', lambda: gate(*ms['content'](x))),
                         ('same_features_regression', lambda: matched(*ms['content'](x))),
                         ('RGB_regression', lambda: regressor(x))]:
            for _ in range(2): fn()
            times = []
            for _ in range(5):
                started = time.perf_counter(); fn(); times.append((time.perf_counter()-started)*1000)
            costs[f'{name}_{size}'] = {'p50_ms': float(np.median(times)), 'p95_ms': float(np.quantile(times, .95))}
    f, c = ms['content'](image)
    p, w = gate(f, c)
    assert torch.allclose(p, (w*c).sum(1, keepdim=True))
    assert torch.allclose(w.sum(1), torch.ones_like(w[:, 0]), atol=1e-6)
    repeat = ms['content'](image)[1]
    assert torch.equal(c, repeat)
    # Per-candidate prior removal probe: hold context fixed and replace each
    # own cue by TRAIN median; no re-fitting and no cross-candidate information.
    data = np.load(out/'sampled_training.npz')
    raw, context = ms['content'].extract(image)
    replaced = raw.clone()
    for k in range(3): replaced[:, k] = float(np.median(data['raw'][:, k]))
    without = ms['content'].correct(replaced, context)[0]
    table_scalars = lambda cal: sum(len(t['x'])+len(t['y']) for t in cal['global']) + sum(
        sum(len(t['x'])+len(t['y']) for t in e['tables']) for e in cal['cells'].values())
    write(out/'cost_and_contract.json', {'timings': costs, 'warmups': 2, 'repeats': 5, 'threads': cfg['torch_threads'],
          'timing_scope': 'CPU tensor-to-output, batch1, no disk reads. candidate calls include content + prior + tables; '
                          'dynamic includes all those costs. Fixed fusion adds three multiplies and sum, separately negligible but not timed here.',
          'parameters': {'calibrator_neural_trainable': 0, 'calibrator_neural_total': 0,
                         'content_fitted_table_xy_scalars': table_scalars(ms['content'].calibration),
                         'physical_fitted_table_xy_scalars': table_scalars(ms['content_physical'].calibration),
                         'fixed_fusion_scalars': 3, 'dynamic_trainable_and_total': sum(p.numel() for p in gate.parameters()),
                         'RGB_trainable_and_total': sum(p.numel() for p in regressor.parameters()), 'semantic_model': 0},
          'same_features_reference_parameters': sum(p.numel() for p in matched.parameters()),
          'same_frozen_candidates_for_C_D': True, 'dynamic_fusion_identity_passed': True,
          'RGB_only_inference_saved_png_passed': True,
          'prior_replaced_mean_candidate_change': (without-c).abs().mean((0, 2, 3)).tolist(),
          'prior_probe_scope': 'one exposed RGB input, not an ablation efficacy result; retrained prior-removal control remains pending',
          'GPU_memory': None, 'GPU_memory_reason': 'No CUDA execution authorized or performed',
          'CPU_max_RSS': None, 'CPU_RSS_reason': 'Per-model peak not instrumented; do not misattribute process aggregate',
          'precompute_bytes': sum(p.stat().st_size for p in out.rglob('*.npz')),
          'models_trained': 'nonparametric train-only table fits + fixed-weight solve; gate and regressor cost/contract only'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/content_calibration.json')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/content_calibration_20261102/cpu_run')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = read(args.config); out = args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg['torch_threads'])
    started = time.perf_counter()
    base, scenes = prepare(out, cfg, args.resume)
    if (out/'completion.json').exists():
        print('Completed run preserved; source/config verified. No recomputation.'); return
    with torch.inference_mode():
        data = sample_train(out, cfg, base, scenes)
        fit(out, cfg, data)
        ms = models(out)
        fits = fixed_weights(out, cfg, base, scenes, ms)
        metrics = evaluate(out, cfg, base, scenes, ms, fits)
        cost_and_contract(out, cfg, ms)
    write(out/'completion.json', {'status': 'CPU implementation feasibility only; no efficacy claim',
          'elapsed_this_invocation_seconds': time.perf_counter()-started, 'gpu_seconds': 0,
          'new_confirmation_accessed': False, 'neural_training_steps': 0,
          'physical_constraint': 'Paired nonparametric weighted forward-response objective implemented; same table capacity, no hard inversion',
          'future_dynamic_and_regression_effects': 'NOT TRAINED OR EVALUATED FOR ACCURACY'})
    stamp(out, 'complete')
    write(out/'artifact_sha256.json', {str(p.relative_to(out)).replace('\\', '/'): digest(p)
          for p in out.rglob('*') if p.is_file() and p.name not in ['artifact_sha256.json']})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
