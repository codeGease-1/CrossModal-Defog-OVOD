"""Frozen-model cost, correction ablations and portable weight export."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(exist_ok=False, parents=True)
    sys.path.insert(0, str(args.source))
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from scripts.v15.run import read, write, digest, procedural, cases, tensor, csv_write
    from scripts.learned_correction.run import model
    from src.models.haze_density.dynamic_prior import FrozenCandidates
    from src.models.haze_density.dynamic_prior import single_scores
    from src.models.haze_density.calibrated_prior import apply_monotone
    from src.models.haze_density.learned_correction import observed_context
    from benchmarks.haze_density.metrics import spatial_metrics
    torch.set_num_threads(1)
    protocol = read(args.run/'protocol.json'); cfg = protocol['config']; base = protocol['base']
    (args.output/'calibration.json').write_bytes(Path(protocol['calibration_path']).read_bytes())
    ext = FrozenCandidates(read(protocol['calibration_path'])).cuda().eval()
    fits = read(args.run/'fusion_weights.json')
    selected = {mode: min(cfg['seeds'], key=lambda s: read(args.run/f'{mode}_seed{s}'/'complete.json')['best_development_loss']) for mode in cfg['modes']}
    write(args.output/'selected_development_seeds.json', selected)
    started = time.monotonic()
    costs, exported, ablations, proxy_rows = [], [], [], []
    def measure(fn):
        for _ in range(8): fn()
        torch.cuda.synchronize(); times = []
        for _ in range(40):
            torch.cuda.synchronize(); start = time.perf_counter(); fn(); torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        return {'p50_ms': float(np.median(times)), 'p95_ms': float(np.quantile(times, .95))}
    with torch.no_grad():
        for mode in cfg['modes']:
            seed = selected[mode]; name = f'{mode}_seed{seed}'
            net = model(mode, cfg).cuda().eval()
            checkpoint = args.run/name/'best.pth'
            net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
            state = {'model': {k: v.cpu() for k, v in net.state_dict().items()}, 'mode': mode, 'config': cfg,
                     'calibration_sha256': protocol['calibration_sha256'],
                     'fusion_weights': fits[name]['weights'] if mode.startswith('corrector') else None,
                     'checkpoint_sha256': digest(checkpoint), 'selected_by': 'development only'}
            weight_path = args.output/(mode+'_inference.pth'); torch.save(state, weight_path)
            exported.append({'mode': mode, 'path': weight_path.name, 'sha256': digest(weight_path), 'bytes': weight_path.stat().st_size})
            for size in [256, 512]:
                rgb, _ = procedural(base['seed'], size); image = tensor(rgb, 'cuda')
                def pipeline():
                    if mode == 'rgb32': f = image
                    else:
                        f, _ = ext(image); f = torch.cat([f, observed_context(image)], 1)
                    with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.get('amp', False)):
                        pred, _ = net(f)
                    if mode.startswith('corrector'):
                        return (pred*pred.new_tensor(fits[name]['weights'])[None, :, None, None]).sum(1, keepdim=True)
                    return pred
                for _ in range(8): pipeline()
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(40):
                    torch.cuda.synchronize(); start = time.perf_counter(); pipeline(); torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)*1000)
                costs.append({'mode': mode, 'size': size, 'parameters_trainable': sum(v.numel() for v in net.parameters()),
                    'parameters_inference_neural': sum(v.numel() for v in net.parameters()),
                    'p50_ms': float(np.median(times)), 'p95_ms': float(np.quantile(times, .95)),
                    'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'includes': 'RGB tensor to H, priors/calibration/context if required + model + fixed fusion; no disk read',
                    'precision': 'same AMP policy for all neural models, float32 prior extraction', 'warmup': 8, 'repeats': 40})
                # Separate synchronized measurements. Components need not sum to
                # the complete pipeline: cached inputs/allocation differ.
                def neural(f):
                    with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.get('amp', False)):
                        return net(f)[0]
                components = {}
                if mode != 'rgb32':
                    features, _ = ext(image)
                    features = torch.cat([features, observed_context(image)], 1)
                    _, scores = single_scores(image, ext.calibration)
                    components['raw_cues_fixed_scaling_power_guided'] = measure(lambda: single_scores(image, ext.calibration))
                    components['train_fitted_candidate_tables'] = measure(lambda: torch.cat([
                        apply_monotone(scores[:, k:k+1], ext.calibration['candidate_tables'][k]) for k in range(3)], 1))
                    components['observed_content_features'] = measure(lambda: observed_context(image))
                else: features = image
                components['network_and_bounded_correction_if_used'] = measure(lambda: neural(features))
                if mode.startswith('corrector'):
                    cached_prediction = neural(features)
                    weights = cached_prediction.new_tensor(fits[name]['weights'])[None, :, None, None]
                    components['fixed_fusion'] = measure(lambda: (cached_prediction*weights).sum(1, keepdim=True))
                    del cached_prediction, weights
                costs[-1]['components_cached_inputs'] = components
                costs[-1]['semantic_cost'] = 'No semantic model or semantic precompute'
                del features
                if mode != 'rgb32': del scores
                del image; torch.cuda.empty_cache()
            if mode.startswith('corrector'):
                gain_stats, shift_stats = [], []
                for scene in [s for s in protocol['scenes'] if s['split'] == 'development']:
                    clear, _ = procedural(scene['seed'], base['image_size'])
                    for meta, rgb, y in cases(clear, scene, base):
                        image = tensor(rgb, 'cuda'); f, raw_p = ext(image)
                        f = torch.cat([f, observed_context(image)], 1)
                        with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.get('amp', False)):
                            corrected, auxiliary = net(f)
                        g, b = auxiliary['gain'], auxiliary['offset']
                        gray = clear @ np.array([.299, .587, .114]); dy, dx = np.gradient(gray)
                        proxies = {'bright_proxy': gray > .8, 'low_texture_proxy': np.hypot(dx, dy) < .01}
                        for label, values in [('original', raw_p), ('corrected', corrected)]:
                            c_np = values[0].float().cpu().numpy()
                            for region, mask in proxies.items():
                                if mask.any():
                                    proxy_rows.append({'case_id': meta['case_id'], 'maps': label, 'proxy': region,
                                        'pixels': int(mask.sum()),
                                        **{f'candidate_{k}_mae': float(np.abs(c_np[k]-y)[mask].mean()) for k in range(3)},
                                        **{f'candidate_{k}_bias': float((c_np[k]-y)[mask].mean()) for k in range(3)}})
                        rng = np.random.default_rng(cfg['seed']+scene['seed'])
                        ix = torch.from_numpy(rng.permutation(y.size)).cuda()
                        candidates = {'learned': corrected,
                            'image_mean_coefficients': net.transform(raw_p, g.mean((2, 3), keepdim=True), b.mean((2, 3), keepdim=True)),
                            'shuffled_coefficients': net.transform(raw_p, g.flatten(2)[:, :, ix].reshape_as(g), b.flatten(2)[:, :, ix].reshape_as(b)),
                            'mean_candidate_maps': net.transform(raw_p.mean((2, 3), keepdim=True).expand_as(raw_p), g, b)}
                        for variation, c in candidates.items():
                            h = (c*c.new_tensor(fits[name]['weights'])[None, :, None, None]).sum(1)[0].cpu().numpy()
                            ablations.append({'case_id': meta['case_id'], 'group_id': meta['group_id'], 'variation': variation,
                                'pixel_mae': float(np.abs(h-y).mean()), 'region_spearman': spatial_metrics(1-y, h)['region_spearman']})
                        gain_stats.append(g.float().mean((0, 2, 3)).cpu().numpy())
                        shift_stats.append(b.float().mean((0, 2, 3)).cpu().numpy())
                        if scene['scene_id'] == 'development_000' and meta['pattern'] == 'regions' and meta['brightness_index'] == 1 and meta['airlight_index'] == 0:
                            Image.fromarray((rgb*255).astype('uint8')).save(args.output/'input_rgb.png')
                            values = [('target', y), *[(f'original_{k}', raw_p[0, k].cpu().numpy()) for k in range(3)],
                                      *[(f'corrected_{k}', corrected[0, k].cpu().numpy()) for k in range(3)]]
                            canvas = Image.new('RGB', (256*4, 280*2+22), 'white'); draw = ImageDraw.Draw(canvas)
                            for i, (label, a) in enumerate(values):
                                v = np.clip(a, 0, 1); color = np.stack([v, .25+.5*(1-np.abs(2*v-1)), 1-v], -1)
                                x, yy = (i%4)*256, (i//4)*280
                                canvas.paste(Image.fromarray((color*255).astype('uint8')), (x, yy+22)); draw.text((x+2, yy+2), label, fill='black')
                            draw.text((2, 560), 'All H maps fixed 0 blue to 1 red; no per-image scaling', fill='black')
                            canvas.save(args.output/'candidate_comparison.png')
                summary = {variation: {'pixel_mae': float(np.mean([r['pixel_mae'] for r in ablations if r['variation'] == variation])),
                                      'region_spearman': float(np.mean([r['region_spearman'] for r in ablations if r['variation'] == variation and r['region_spearman'] is not None]))}
                           for variation in sorted(set(r['variation'] for r in ablations))}
                write(args.output/'correction_ablations.json', {'summary': summary,
                    'mean_gain_D_C_K': np.mean(gain_stats, axis=0).tolist(), 'mean_offset_D_C_K': np.mean(shift_stats, axis=0).tolist(),
                    'scope': 'Frozen development inference interventions only; prior replacement alone cannot prove trained-model necessity. Same-input direct control supplies the main comparison.'})
                del image, f, raw_p, corrected, auxiliary, g, b, ix, candidates, c, values
            del net; gc.collect(); torch.cuda.empty_cache()
    csv_write(args.output/'ablation_per_case.csv', ablations)
    csv_write(args.output/'content_proxy_errors.csv', proxy_rows)
    write(args.output/'full_inference_cost.json', costs)
    write(args.output/'exported_weights.json', exported)
    write(args.output/'source.json', {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'run_protocol_sha256': digest(args.run/'protocol.json'), 'selected_by': 'development, no confirmation tuning',
        'elapsed_wall_seconds': time.monotonic()-started,
        'GPU_memory_scope': 'PyTorch allocated/reserved; framework/driver memory outside allocator not included'})
    print(json.dumps({'selected_seeds': selected, 'costs': costs}, indent=2))


if __name__ == '__main__': main()
