"""Known original-S members: exposed secondary endpoint, never model selection."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(exist_ok=False, parents=True); sys.path.insert(0, str(args.source))
    import numpy as np
    import torch
    from PIL import Image
    from scripts.v15.run import read, write, digest, tensor, csv_write
    from scripts.learned_correction.run import load_models, diagnostic
    from src.models.haze_density.dynamic_prior import FrozenCandidates
    from src.models.haze_density.learned_correction import observed_context
    from benchmarks.haze_density.metrics import spatial_metrics
    torch.set_num_threads(1)
    assert (args.run/'frozen.json').exists(), 'Freeze first'
    protocol = read(args.run/'protocol.json'); cfg = protocol['config']; ms = load_models(args.run, cfg, cfg['seeds'])
    fits = read(args.run/'fusion_weights.json'); ext = FrozenCandidates(read(protocol['calibration_path'])).cuda().eval()
    old = args.project/'experiments/v2_investigation_20261101/original_S'
    scenes = read(old/'manifest.json'); checks = {a['scene_id']: a for a in read(old/'label_checks.json')}
    started = time.monotonic()
    rows, bounds, inputs = [], [], []
    with torch.no_grad(), zipfile.ZipFile('/mnt/RSHaze+.zip') as z:
        for s in scenes:
            assert checks[s['scene_id']]['status'] == 'PASS', 'Reuse existing eligibility audit, no reinterpretation'
            n = s['archive_member']; clean_data = z.read(n)
            assert hashlib.sha256(clean_data).hexdigest() == s['file_sha256']
            h_data = z.read(n.replace('/cleanpng/', '/synhazypng/'))
            t_data = z.read(n.replace('/cleanpng/', '/transpng/'))
            h = Image.open(io.BytesIO(h_data)).convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
            rgb = np.asarray(h, np.float32)/255
            t = np.asarray(Image.open(io.BytesIO(t_data)), np.float32)/255
            y = np.asarray(Image.fromarray((1-t.mean(2)).astype(np.float32)).resize((256, 256), Image.Resampling.BILINEAR))
            image = tensor(rgb, 'cuda'); f, c = ext(image); f = torch.cat([f, observed_context(image)], 1)
            preds = {'original_fixed': (c*c.new_tensor(fits['original']['weights'])[None, :, None, None]).sum(1, keepdim=True)}
            bounds.append({'method': 'original', 'case_id': s['scene_id'], **diagnostic(c[0].cpu().numpy(), y)})
            for name, m in ms.items():
                with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.get('amp', False)): pred, _ = m(f)
                if name.startswith('corrector'):
                    bounds.append({'method': name, 'case_id': s['scene_id'], **diagnostic(pred[0].cpu().numpy(), y)})
                    pred = (pred*pred.new_tensor(fits[name]['weights'])[None, :, None, None]).sum(1, keepdim=True)
                preds[name] = pred
            for name, pred in preds.items():
                a = pred[0, 0].float().cpu().numpy()
                rows.append({'method': name, 'scene_id': s['scene_id'], 'group_id': s['group_id'],
                    'pixel_mae': float(np.abs(a-y).mean()), 'global_mae': float(abs(a.mean()-y.mean())),
                    'region_spearman': spatial_metrics(1-y, a)['region_spearman']})
            inputs.append({**s, 'hazy_sha256': hashlib.sha256(h_data).hexdigest(), 'channel_transmission_sha256': hashlib.sha256(t_data).hexdigest()})
    result = {}
    for name in sorted(set(r['method'] for r in rows)):
        rr = [r for r in rows if r['method'] == name]
        result[name] = {k: float(np.mean([r[k] for r in rr if r[k] is not None])) for k in ['pixel_mae', 'global_mae', 'region_spearman']}
    csv_write(args.output/'per_image.csv', rows); write(args.output/'candidate_diagnostics.json', bounds)
    write(args.output/'manifest.json', inputs)
    write(args.output/'summary.json', {'metrics': result, 'endpoint': 'H_eff=1-mean_RGB(t_c), spectral secondary endpoint',
        'geographic_groups': 1, 'confidence_interval': None, 'role': 'Already exposed development failure diagnostic; no model tuning',
        'not_natural_haze_truth': True, 'old_label_audit_sha256': digest(old/'label_checks.json'),
        'elapsed_wall_seconds': time.monotonic()-started,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
