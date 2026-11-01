"""V1.5 train-only calibration; the legacy prior is deliberately unchanged.

Fixed percentile ranges put differently scaled cues on a dimensionless common
range. A frozen monotone scalar map then calibrates the fused response to 1-t.
This is supervised calibration, not a new neural density estimator.
"""
import numpy as np
import torch

from .physical_prior import dark_channel, min_max_normalize, generate_s_final, _local_mean
from .guided_filter import guided_filter


def raw_cues(image, window_size=15):
    """Return D, -absolute local contrast, -color difference (high = more haze)."""
    gray = (image * image.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    contrast = (gray - _local_mean(gray, window_size)).abs()
    color = (image[:, :1]-image[:, 1:2]).abs() + (image[:, :1]-image[:, 2:3]).abs()
    return torch.cat([dark_channel(image, window_size), -contrast, -color], 1)


def normalized_cues(raw):
    # Inverting AFTER minmax reproduces the legacy constant-map convention.
    return torch.cat([min_max_normalize(raw[:, :1]),
                      1-min_max_normalize(-raw[:, 1:2]),
                      1-min_max_normalize(-raw[:, 2:3])], 1)


def fixed_cues(raw, limits):
    low = raw.new_tensor([p[0] for p in limits])[None, :, None, None]
    high = raw.new_tensor([p[1] for p in limits])[None, :, None, None]
    return ((raw-low)/(high-low).clamp_min(1e-8)).clamp(0, 1)


def fuse(image, cues, radius=15, eps=.01):
    weights = image.new_tensor([.5, .3, .2])[None, :, None, None]
    score = (cues*weights).sum(1, keepdim=True).clamp(0, 1).pow(1.5)
    gray = (image * image.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    return guided_filter(gray, score, radius=radius, eps=eps)


def fit_monotone(x, y, bins=128):
    """Deterministic binned PAVA, weighted by equal-pixel sample counts."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if x.shape != y.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Calibration requires matching finite nonempty arrays')
    order = np.argsort(x, kind='stable')
    # Equal x values must never receive inconsistent fitted values.
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins+1)))
    bucket = np.searchsorted(edges[1:-1], x[order], side='right')
    blocks = []
    for idx in np.unique(bucket):
        take = order[bucket == idx]
        blocks.append([len(take), float(x[take].sum()), float(y[take].sum())])
        while len(blocks) >= 2 and blocks[-2][2]/blocks[-2][0] > blocks[-1][2]/blocks[-1][0]:
            b = blocks.pop(); a = blocks.pop()
            blocks.append([a[k]+b[k] for k in range(3)])
    return {'x': [b[1]/b[0] for b in blocks],
            'y': [float(np.clip(b[2]/b[0],0,1)) for b in blocks],
            'n_pixels': int(x.size), 'bins': bins}


def apply_monotone(score, table):
    x, y = score.new_tensor(table['x']), score.new_tensor(table['y'])
    if x.numel() == 1:
        return torch.full_like(score, y.item())
    index = torch.searchsorted(x, score.contiguous()).clamp(1, len(x)-1)
    frac = ((score-x[index-1])/(x[index]-x[index-1]).clamp_min(1e-12)).clamp(0, 1)
    return y[index-1] + frac*(y[index]-y[index-1])


class CalibratedPrior(torch.nn.Module):
    """Load JSON parameters fitted exclusively on the recorded training scenes."""
    def __init__(self, calibration, variant='fixed'):
        super().__init__()
        if variant not in ('normalized', 'fixed'):
            raise ValueError('variant must be normalized or fixed')
        self.calibration = calibration
        self.variant = variant

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or not torch.isfinite(image).all():
            raise ValueError('Expected finite B,3,H,W RGB')
        if image.min() < 0 or image.max() > 1:
            raise ValueError('RGB must be in [0,1]')
        cfg = self.calibration['prior']
        if self.variant == 'normalized':
            score = generate_s_final(image, **cfg)
        else:
            raw = raw_cues(image, cfg['window_size'])
            score = fuse(image, fixed_cues(raw, self.calibration['limits']), cfg['guided_radius'], cfg['guided_eps'])
        return apply_monotone(score, self.calibration['tables'][self.variant])
