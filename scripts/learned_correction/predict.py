"""Portable CPU/CUDA RGB-file inference from exported frozen learned weights."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import json
import time
import numpy as np
from PIL import Image
import torch
from src.models.haze_density.learned_correction import RestrictedCorrector, RGBReference, observed_context
from src.models.haze_density.dynamic_prior import FrozenCandidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=Path, required=True); p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args(); args.output.mkdir(exist_ok=False, parents=True); torch.set_num_threads(4)
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    assert hashlib.sha256(args.calibration.read_bytes()).hexdigest() == state['calibration_sha256']
    mode, cfg = state['mode'], state['config']
    if mode.startswith('corrector'): net = RestrictedCorrector(16, cfg['rho'], cfg['shift'])
    else: net = RGBReference(16, with_priors=True) if mode == 'direct_same16' else RGBReference(32)
    net.load_state_dict(state['model']); net.to(args.device).eval()
    ext = FrozenCandidates(json.loads(args.calibration.read_text(encoding='utf-8'))).to(args.device).eval()
    with Image.open(args.image) as im:
        if im.mode != 'RGB': raise ValueError('Verify RGB encoding before inference')
        rgb = torch.from_numpy((np.asarray(im, np.float32)/255).transpose(2, 0, 1).copy())[None].to(args.device)
    with torch.no_grad():
        start = time.perf_counter()
        if mode == 'rgb32': f = rgb
        else:
            f, _ = ext(rgb); f = torch.cat([f, observed_context(rgb)], 1)
        with torch.autocast(args.device, dtype=torch.float16, enabled=args.device == 'cuda' and cfg.get('amp', False)):
            out, _ = net(f)
        candidates = out if mode.startswith('corrector') else None
        if candidates is not None: out = (out*out.new_tensor(state['fusion_weights'])[None, :, None, None]).sum(1, keepdim=True)
        h = out[0, 0].float().cpu().numpy(); seconds = time.perf_counter()-start
    np.savez_compressed(args.output/'prediction.npz', H=h, **({'candidates': candidates[0].cpu().numpy()} if candidates is not None else {}))
    Image.fromarray(np.round(np.clip(h, 0, 1)*255).astype('uint8')).save(args.output/'H_fixed_0_1.png')
    (args.output/'inference.json').write_text(json.dumps({'mode': mode, 'device': args.device, 'H_mean': float(h.mean()),
        'input': 'hazy RGB only', 'first_call_seconds': seconds, 'fixed_scale': [0, 1], 'warning': 'Synthetic-protocol effective attenuation, not measured natural fog concentration'}, indent=2), encoding='utf-8')
    print('Saved', args.output)


if __name__ == '__main__': main()
