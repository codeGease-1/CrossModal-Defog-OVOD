"""Standalone saved-RGB inference; no truth, A, clean image or scene ID API."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read, write, tensor, digest
from src.models.haze_density.content_calibration import ContentCalibratedCandidates, FixedCandidateFusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--variant', choices=['global', 'content', 'content_physical'], default='content')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cal_path = args.run/('calibration_content_physical.json' if args.variant == 'content_physical' else 'calibration_content.json')
    model = ContentCalibratedCandidates(read(cal_path), conditional=args.variant != 'global')
    fusion = FixedCandidateFusion(read(args.run/'fixed_weights.json')[args.variant]['weights'])
    with Image.open(args.image) as im:
        if im.mode != 'RGB': raise ValueError('Require already verified display RGB file; no silent sensor/bit-depth conversion')
        image = tensor(np.asarray(im, np.float32)/255, 'cpu')
    with torch.inference_mode():
        start = time.perf_counter(); _, c = model(image); h = fusion(c)
        seconds = time.perf_counter()-start
    a = h[0, 0].numpy()
    np.savez_compressed(args.output/'prediction.npz', H=a, candidates=c[0].numpy())
    Image.fromarray(np.round(a*255).astype('uint8')).save(args.output/'H_fixed_0_1.png')
    write(args.output/'inference.json', {'input_sha256': digest(args.image), 'calibration_sha256': digest(cal_path),
          'variant': args.variant, 'input': 'hazy display RGB only', 'domain_warning': 'unknown sensor domain may be out of calibration',
          'H_mean': float(a.mean()), 'seconds_first_call_no_warmup': seconds,
          'color_scale': 'uint8=round(255*H), fixed [0,1], no image stretching',
          'source_sha256': digest(__file__), 'gpu_seconds': 0})
    print('Saved RGB-only inference:', args.output)


if __name__ == '__main__':
    main()
