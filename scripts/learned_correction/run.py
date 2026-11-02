"""Isolated V100 training/evaluation; finite 3-model matrix, no old writes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.v15.run import read, write, digest, original_scenes, procedural, cases, tensor, csv_write
from scripts.v2.global_optimum import solve_simplex
from src.models.haze_density.dynamic_prior import FrozenCandidates
from src.models.haze_density.learned_correction import RestrictedCorrector, RGBReference, observed_context
from benchmarks.haze_density.metrics import spatial_metrics


def model(mode, cfg):
    if mode.startswith('corrector'):
        return RestrictedCorrector(int(mode.removeprefix('corrector')), cfg['rho'], cfg['shift'])
    if mode == 'direct_same16': return RGBReference(16, with_priors=True)
    return RGBReference(16 if mode == 'rgb_context16' else 32, mode == 'rgb_context16')


def task_loss(pred, target, cfg):
    out = (pred-target).square().mean() + cfg['global_loss_weight']*(pred.mean((2, 3))-target.mean((2, 3))).square().mean()
    clear = target.mean((1, 2, 3)) < 1e-7
    if clear.any(): out = out + cfg['clear_loss_extra_weight']*pred[clear].square().mean()
    return out


def loss_for(mode, pred, target, cfg):
    if not mode.startswith('corrector'): return task_loss(pred, target, cfg)
    weights = cfg['candidate_loss_weights']
    return sum(w*task_loss(pred[:, k:k+1], target, cfg) for k, w in enumerate(weights))/sum(weights)


def atomic_torch_save(value, path):
    temp = path.with_suffix('.pending.pth'); torch.save(value, temp); temp.replace(path)


def charge(out, stage, start):
    p = out/'budget.json'
    b = read(p) if p.exists() else {'new_authorization': 'Use existing V100 for training; prior 3600s ceiling superseded for this task', 'stages': [], 'gpu_stage_wall_seconds': 0.}
    elapsed = time.monotonic()-start
    b['stages'].append({'stage': stage, 'seconds': elapsed}); b['gpu_stage_wall_seconds'] += elapsed
    write(p, b)


def setup(out, cfg, calibration_path):
    out.mkdir(parents=True, exist_ok=True)
    sources = [*ROOT.glob('src/**/*.py'), *ROOT.glob('benchmarks/haze_density/*.py'),
               *ROOT.glob('scripts/learned_correction/*.py'), ROOT/'scripts/v15/run.py', ROOT/'scripts/v2/global_optimum.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    if (out/'protocol.json').exists():
        p = read(out/'protocol.json')
        assert p['config'] == cfg and p['source_sha256'] == hashes, 'Frozen source/config mismatch; new directory required'
        assert p['calibration_sha256'] == digest(calibration_path)
        return p
    base = read(ROOT/'configs/v15_calibration.json')
    scenes = [s for split in ['train', 'development'] for s in original_scenes(base, split)]
    for s in scenes:
        rgb, _ = procedural(s['seed'], base['image_size'])
        s['pixel_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
    assert len(set(s['group_id'] for s in scenes)) == len(scenes)
    assert len(set(s['pixel_sha256'] for s in scenes)) == len(scenes)
    p = {'config': cfg, 'base': base, 'scenes': scenes, 'source_sha256': hashes,
         'calibration_path': str(calibration_path), 'calibration_sha256': digest(calibration_path),
         'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                         'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda},
         'command': sys.argv, 'data': '48/16 original scenes, 32 versions each; synthetic scalar H=1-t, not natural atmospheric truth',
         'inputs': 'RGB and deterministic RGB-derived priors/context only; no J,t,A,scene identity or oracle in forward',
         'noninferiority': 'User selected relative MAE margin .05 before any new confirmation; rho drop <=.01, clear increase <=.005',
         'confirmation_state': 'No new confirmation generated before freeze',
         'selection': cfg['selection']}
    write(out/'protocol.json', p)
    (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    return p


def cache(p, ext, split):
    xs, ys, metas = [], [], []
    with torch.no_grad():
        for s in [s for s in p['scenes'] if s['split'] == split]:
            clear, _ = procedural(s['seed'], p['base']['image_size'])
            for meta, rgb, y in cases(clear, s, p['base']):
                image = tensor(rgb, 'cuda'); features, _ = ext(image)
                xs.append(torch.cat([features, observed_context(image)], 1).cpu())
                ys.append(torch.from_numpy(y.copy())[None, None]); metas.append(meta)
            print('CACHE', split, s['scene_id'], flush=True)
    return torch.cat(xs), torch.cat(ys), metas


def validation(m, mode, x, y, cfg):
    m.eval(); total = []; maes = []
    with torch.no_grad():
        for i in range(0, len(x), cfg['batch_size']):
            target = y[i:i+cfg['batch_size']].cuda()
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.get('amp', False)):
                pred, _ = m(x[i:i+cfg['batch_size']].cuda())
            total.append(float(loss_for(mode, pred, target, cfg)))
            maes.append((pred-target).abs().mean((0, 2, 3)).cpu().numpy())
    return float(np.mean(total)), np.mean(maes, axis=0).tolist()


def resume_smoke(out, cfg):
    if (out/'resume_smoke.json').exists(): return
    start = time.monotonic(); torch.manual_seed(842)
    x = torch.rand(2, 11, 32, 32, device='cuda'); y = torch.rand(2, 1, 32, 32, device='cuda')
    m = RestrictedCorrector(8).cuda(); opt = torch.optim.Adam(m.parameters(), lr=.001)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.get('amp', False))
    def step(net, optimizer, scale):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.get('amp', False)):
            pred, _ = net(x)
            loss = (pred-y).square().mean()
        scale.scale(loss).backward(); scale.step(optimizer); scale.update()
    step(m, opt, scaler)
    state = {'model': m.state_dict(), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
    atomic_torch_save(state, out/'resume_smoke.pth')
    step(m, opt, scaler)
    saved = torch.load(out/'resume_smoke.pth', map_location='cpu', weights_only=False)
    other = RestrictedCorrector(8).cuda(); other.load_state_dict(saved['model'])
    op = torch.optim.Adam(other.parameters(), lr=.001); op.load_state_dict(saved['optimizer'])
    other_scaler = torch.amp.GradScaler('cuda', enabled=cfg.get('amp', False)); other_scaler.load_state_dict(saved['scaler'])
    torch.set_rng_state(saved['cpu_rng'].cpu()); torch.cuda.set_rng_state_all([a.cpu() for a in saved['cuda_rng']])
    step(other, op, other_scaler)
    delta = max(float((m.state_dict()[k]-other.state_dict()[k]).abs().max()) for k in m.state_dict())
    assert delta == 0, delta
    write(out/'resume_smoke.json', {'max_parameter_difference': delta, 'device': 'cuda', 'amp': cfg.get('amp', False), 'actual_save_load_and_optimizer_resume': True})
    charge(out, 'resume_smoke', start)


def train(out, cfg, p, ext, only_seed=None):
    resume_smoke(out, cfg)
    if all((out/f'{mode}_seed{seed}'/'complete.json').exists() for mode in cfg['modes'] for seed in cfg['seeds']): return
    start = time.monotonic(); x, y, meta = cache(p, ext, 'train'); vx, vy, vm = cache(p, ext, 'development')
    charge(out, 'RGB_derived_precompute_and_target_cache', start)
    write(out/'train_manifest.json', meta); write(out/'development_manifest.json', vm)
    print('CACHE RAM bytes', x.numel()*x.element_size()+y.numel()*4+vx.numel()*4+vy.numel()*4, flush=True)
    for seed in ([only_seed] if only_seed is not None else cfg['seeds']):
        for mode in cfg['modes']:
            d = out/f'{mode}_seed{seed}'; d.mkdir(exist_ok=True)
            if (d/'complete.json').exists(): continue
            torch.manual_seed(seed); rng = torch.Generator().manual_seed(seed)
            m = model(mode, cfg).cuda(); opt = torch.optim.Adam(m.parameters(), lr=cfg['learning_rate'])
            scaler = torch.amp.GradScaler('cuda', enabled=cfg.get('amp', False))
            epoch0, best, hist, step = 0, float('inf'), [], 0
            if (d/'latest.pth').exists():
                s = torch.load(d/'latest.pth', map_location='cpu', weights_only=False)
                m.load_state_dict(s['model']); opt.load_state_dict(s['optimizer'])
                if 'scaler' in s: scaler.load_state_dict(s['scaler'])
                epoch0, best, hist, step = s['epoch'], s['best'], s['history'], s['step']
                rng.set_state(s['shuffle_rng'].cpu()); torch.set_rng_state(s['torch_rng'].cpu())
                torch.cuda.set_rng_state_all([a.cpu() for a in s['cuda_rng']])
            for epoch in range(epoch0, cfg['epochs']):
                start = time.monotonic(); m.train(); losses = []; order = torch.randperm(len(x), generator=rng)
                torch.cuda.reset_peak_memory_stats()
                for j in range(0, len(x), cfg['batch_size']):
                    ix = order[j:j+cfg['batch_size']]; opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.get('amp', False)):
                        pred, _ = m(x[ix].cuda()); loss = loss_for(mode, pred, y[ix].cuda(), cfg)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss; preserve checkpoint')
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 5.)
                    scaler.step(opt); scaler.update(); losses.append(float(loss.detach())); step += 1
                val, mae = validation(m, mode, vx, vy, cfg)
                improved = val < best; best = min(best, val)
                hist.append({'epoch': epoch+1, 'step': step, 'train_loss': float(np.mean(losses)),
                             'development_loss': val, 'development_mae_per_output': mae,
                             'seconds': time.monotonic()-start, 'train_peak_allocated_bytes': torch.cuda.max_memory_allocated()})
                state = {'model': m.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch+1, 'step': step,
                         'best': best, 'history': hist, 'shuffle_rng': rng.get_state(), 'torch_rng': torch.get_rng_state(),
                         'cuda_rng': torch.cuda.get_rng_state_all(), 'mode': mode, 'seed': seed,
                         'scaler': scaler.state_dict(),
                         'protocol_sha256': digest(out/'protocol.json')}
                atomic_torch_save(state, d/'latest.pth')
                if improved: atomic_torch_save(state, d/'best.pth')
                write(d/'history.json', hist); charge(out, f'{mode}_seed{seed}_epoch{epoch+1}', start)
                print('TRAIN', mode, seed, epoch+1, 'dev', val, 'MAE', mae, 'seconds', hist[-1]['seconds'], flush=True)
            write(d/'complete.json', {'mode': mode, 'seed': seed, 'best_development_loss': best,
                  'parameters': sum(q.numel() for q in m.parameters()), 'steps': step})
            del m, opt; torch.cuda.empty_cache()
    print('TRAINING BATCH COMPLETE', flush=True)


def load_models(out, cfg, seeds):
    ms = {}
    for seed in seeds:
        for mode in cfg['modes']:
            d = out/f'{mode}_seed{seed}'
            assert (d/'complete.json').exists(), d
            m = model(mode, cfg).cuda()
            m.load_state_dict(torch.load(d/'best.pth', map_location='cpu', weights_only=False)['model'])
            ms[f'{mode}_seed{seed}'] = m.eval()
    return ms


def fit_fusion(out, cfg, p, ext, ms):
    path = out/'fusion_weights.json'
    previous = read(path) if path.exists() else {}
    todo = {name: m for name, m in ms.items() if name.startswith('corrector') and name not in previous}
    if not todo: return previous
    start = time.monotonic()
    stats = {name: [np.zeros((3, 3)), np.zeros(3), np.zeros((3, 3)), np.zeros(3), np.zeros((3, 3))] for name in todo}
    stats['original'] = [np.zeros((3, 3)), np.zeros(3), np.zeros((3, 3)), np.zeros(3), np.zeros((3, 3))]
    x, y, metas = cache(p, ext, 'train'); nc = sum(m['uniform_t'] == 1. for m in metas); n = len(x)
    with torch.no_grad():
        for i in range(0, n, cfg['batch_size']):
            f = x[i:i+cfg['batch_size']].cuda(); yy = y[i:i+cfg['batch_size']].cuda()
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.get('amp', False)):
                predictions = {name: m(f)[0] for name, m in todo.items()}
            predictions['original'] = f[:, 3:6]
            for name, pred in predictions.items():
                for k in range(len(pred)):
                    c = pred[k].reshape(3, -1).double(); target = yy[k, 0].flatten().double()
                    q = (c@c.T/c.shape[1]).cpu().numpy(); r = (c@target/c.shape[1]).cpu().numpy()
                    mean = c.mean(1).cpu().numpy(); ym = float(target.mean())
                    Q, R, Qg, Rg, Qc = stats[name]
                    Q += q; R += r; Qg += np.outer(mean, mean); Rg += mean*ym
                    if metas[i+k]['uniform_t'] == 1.: Qc += q
    for name, (Q, R, Qg, Rg, Qc) in stats.items():
        matrix = Q/n+cfg['global_loss_weight']*Qg/n+cfg['clear_loss_extra_weight']*Qc/nc
        rr = R/n+cfg['global_loss_weight']*Rg/n
        previous[name] = {'weights': solve_simplex(matrix, rr).tolist(), 'Q': matrix.tolist(), 'r': rr.tolist(),
                          'fitted_on': 'train only; correction frozen', 'candidate_checkpoint_sha256': digest(out/name/'best.pth') if name != 'original' else p['calibration_sha256']}
    write(path, previous); charge(out, 'frozen_candidate_train_only_fixed_fusion', start)
    return previous


def diagnostic(c, y):
    error = c-y[None]; lo, hi = c.min(0), c.max(0)
    distance = np.maximum(lo-y, 0)+np.maximum(y-hi, 0)
    return {'candidate_mae': np.abs(error).mean((1, 2)).tolist(), 'candidate_bias': error.mean((1, 2)).tolist(),
            'joint_high': float((lo > y+.03).mean()), 'joint_low': float((hi < y-.03).mean()),
            'outside_fraction': float((distance > 0).mean()), 'outside_max': float(distance.max()),
            'oracle_mae': float(distance.mean()), 'oracle_mse': float((distance**2).mean())}


def summarize(rows):
    result = {}
    for name in sorted(set(r['method'] for r in rows)):
        rr = [r for r in rows if r['method'] == name]
        result[name] = {k: float(np.mean([r[k] for r in rr if r[k] is not None])) for k in
                        ['pixel_mae', 'pixel_mse', 'global_mae', 'region_spearman', 'clear_response']}
        result[name]['cases'] = len(rr)
    return result


def evaluate(out, cfg, p, ext, seeds, confirmation=False):
    start = time.monotonic(); ms = load_models(out, cfg, seeds)
    fits = fit_fusion(out, cfg, p, ext, ms)
    if confirmation:
        frozen = read(out/'frozen.json')
        for name in ms: assert digest(out/name/'best.pth') == frozen['checkpoints'][name]
        scenes = [{'scene_id': f'newholdout_{i:03d}', 'group_id': f'procedural_original_{p["base"]["seed"]+cfg["confirmation_seed_offset"]+i}',
                   'seed': p['base']['seed']+cfg['confirmation_seed_offset']+i, 'split': 'confirmation'} for i in range(cfg['confirmation_scene_count'])]
        assert not ({s['group_id'] for s in scenes} & {s['group_id'] for s in p['scenes']})
        conditions = ['nominal', 'unseen_A', 'display_rgb']
    else:
        scenes = [s for s in p['scenes'] if s['split'] == 'development']; conditions = ['nominal']
    with torch.no_grad():
        for condition in conditions:
            d = out/('confirmation' if confirmation else 'development')/condition
            d.mkdir(parents=True, exist_ok=True)
            suffix = '_'.join(map(str, seeds))
            if (d/f'summary_{suffix}.json').exists(): continue
            base = copy.deepcopy(p['base'])
            if condition == 'unseen_A': base['airlights'] = cfg['unseen_airlights']
            if condition == 'display_rgb': base['domain'] = 'display_rgb'
            rows, candidates, manifest = [], [], []
            for scene in scenes:
                clear, _ = procedural(scene['seed'], base['image_size'])
                for meta, rgb, y in cases(clear, scene, base):
                    image = tensor(rgb, 'cuda'); f, original = ext(image)
                    f = torch.cat([f, observed_context(image)], 1)
                    predictions = {'original_fixed': (original*original.new_tensor(fits['original']['weights'])[None, :, None, None]).sum(1, keepdim=True)}
                    for name, net in ms.items():
                        with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.get('amp', False)):
                            pred, aux = net(f)
                        if name.startswith('corrector'):
                            c = pred[0].cpu().numpy()
                            candidates.append({'method': name, 'case_id': meta['case_id'], **diagnostic(c, y)})
                            weights = pred.new_tensor(fits[name]['weights'])[None, :, None, None]
                            predictions[name] = (pred*weights).sum(1, keepdim=True)
                        else: predictions[name] = pred
                    candidates.append({'method': 'original', 'case_id': meta['case_id'], **diagnostic(original[0].cpu().numpy(), y)})
                    for name, pred in predictions.items():
                        a = pred[0, 0].cpu().numpy()
                        rows.append({'method': name, 'case_id': meta['case_id'], 'scene_id': meta['scene_id'], 'group_id': meta['group_id'],
                                     'pattern': meta['pattern'], 'airlight_index': meta['airlight_index'], 'brightness_index': meta['brightness_index'],
                                     'truth_mean': float(y.mean()), 'prediction_mean': float(a.mean()),
                                     'pixel_mae': float(np.abs(a-y).mean()), 'pixel_mse': float(((a-y)**2).mean()),
                                     'global_mae': float(abs(a.mean()-y.mean())),
                                     'region_spearman': spatial_metrics(1-y, a)['region_spearman'],
                                     'clear_response': float(a.mean()) if meta['uniform_t'] == 1. else None})
                    manifest.append({**meta, 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                                     'target_sha256': hashlib.sha256(y.tobytes()).hexdigest()})
                print('EVAL', condition, scene['scene_id'], flush=True)
            csv_write(d/f'per_image_{suffix}.csv', rows)
            write(d/f'candidate_diagnostics_{suffix}.json', candidates)
            write(d/f'manifest_{suffix}.json', manifest)
            summary = summarize(rows); write(d/f'summary_{suffix}.json', summary)
            print('SUMMARY', condition, json.dumps(summary), flush=True)
    charge(out, 'confirmation_evaluation' if confirmation else 'development_evaluation', start)


def freeze(out, cfg):
    if (out/'frozen.json').exists(): return
    checkpoint = {}
    for mode in cfg['modes']:
        for seed in cfg['seeds']:
            d = out/f'{mode}_seed{seed}'; assert (d/'complete.json').exists(), d
            checkpoint[f'{mode}_seed{seed}'] = digest(d/'best.pth')
    assert (out/'fusion_weights.json').exists()
    write(out/'frozen.json', {'checkpoints': checkpoint, 'config': cfg,
          'fusion_weights_sha256': digest(out/'fusion_weights.json'),
          'all_seeds_reported': True, 'selection': 'development only', 'new_confirmation_seen': False})
    print('FROZEN before confirmation', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/learned_correction.json')
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['train', 'development', 'freeze', 'confirmation'], required=True)
    parser.add_argument('--only-seed', type=int)
    args = parser.parse_args(); cfg = read(args.config)
    torch.set_num_threads(1); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    p = setup(args.output, cfg, args.calibration)
    ext = FrozenCandidates(read(args.calibration)).cuda().eval()
    if args.stage == 'train': train(args.output, cfg, p, ext, args.only_seed)
    elif args.stage == 'freeze': freeze(args.output, cfg)
    else: evaluate(args.output, cfg, p, ext, [args.only_seed] if args.only_seed is not None else cfg['seeds'], args.stage == 'confirmation')


if __name__ == '__main__': main()
