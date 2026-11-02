"""Independent, low-capacity correction BEFORE fusion (research only).

Each H candidate is a monotone function of its OWN oriented raw cue. Two
observed-RGB context statistics select a shrinkage table, never fusion weights.
There is no learned RGB head, clean-baseline predictor, or hard inversion.
"""
import itertools
import numpy as np
import torch
from torch import nn

from .calibrated_prior import raw_cues, apply_monotone
from .physical_prior import _local_mean


def content_features(image, window=31):
    """B,2,H,W: local luminance mean; mean absolute luminance deviation.

    Both are from hazy display RGB, not semantic classes or clear reflectance.
    Haze changes these descriptors too; content invariance is NOT assumed.
    """
    if (image.ndim != 4 or image.shape[1] != 3 or not torch.isfinite(image).all()
            or image.min() < 0 or image.max() > 1):
        raise ValueError('Expected finite display RGB B,3,H,W in [0,1]')
    gray = (image * image.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    mean = _local_mean(gray, window)
    texture = _local_mean((gray - mean).abs(), window)
    return torch.cat([mean, texture], 1)


def weighted_monotone(x, y, weights, bins=32):
    """Binned weighted PAVA, same capacity with or without response penalty.

    Equal-frequency x bins are independent of y and the physical coefficient.
    Bounds are applied to fitted block means, not to physical observations.
    """
    x, y, w = [np.asarray(a, dtype=np.float64).ravel() for a in (x, y, weights)]
    if (x.shape != y.shape or x.shape != w.shape or not len(x) or bins < 2
            or not all(np.isfinite(a).all() for a in (x, y, w)) or (w <= 0).any()):
        raise ValueError('Expected matching finite arrays and positive weights')
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    bucket = np.searchsorted(edges[1:-1], x, side='right')
    blocks = []
    for i in np.unique(bucket):
        take = bucket == i
        blocks.append([w[take].sum(), (w[take]*x[take]).sum(), (w[take]*y[take]).sum()])
        while len(blocks) > 1 and blocks[-2][2]/blocks[-2][0] > blocks[-1][2]/blocks[-1][0]:
            b, a = blocks.pop(), blocks.pop()
            blocks.append([a[j] + b[j] for j in range(3)])
    return {'x': [float(b[1]/b[0]) for b in blocks],
            'y': [float(np.clip(b[2]/b[0], 0, 1)) for b in blocks], 'bins': bins}


def response_objective(target, baseline, observed, valid, strength):
    """TRAIN ONLY: square-complete H loss + lambda*forward-response loss.

    F(P,b) = b + (s-b)P, s=(1,0,0), on POSITIVE raw d,c,k of J/A,I/A.
    No division by b or (1-b). Effective denominator is >=1 even at zero signal.
    The caller must verify linear domain, positive constant A, scalar t, and
    local constancy for d/c. Nothing here is an inference feature or mask.
    """
    y = np.asarray(target, float)
    b, q, v = np.asarray(baseline, float), np.asarray(observed, float), np.asarray(valid)
    if (y.ndim != 1 or b.shape != (len(y), 3) or q.shape != b.shape
            or v.shape != b.shape or v.dtype != np.bool_ or not np.isfinite(strength)
            or strength < 0 or not all(np.isfinite(a).all() for a in (y, b, q))):
        raise ValueError('Invalid forward-response training arrays')
    a = np.array([1., 0., 0.]) - b
    w = 1 + strength * v * a*a
    z = (y[:, None] + strength * v * a*(q-b)) / w
    return z, w


def fit_content_calibration(raw, target, context, scene_ids, *, split,
                            bins=32, min_pixels=512, min_scenes=4,
                            shrinkage=4096, max_alpha=.5, max_delta=.1,
                            cue_window=15, content_window=31,
                            physical_strength=0., physical=None):
    """Fit only sampled TRAIN pixels. No scene ID is consumed at inference."""
    if split != 'train':
        raise ValueError('Calibration may only fit train originals')
    x, y, f, groups = map(np.asarray, (raw, target, context, scene_ids))
    if (x.ndim != 2 or x.shape[1] != 3 or y.shape != (len(x),)
            or f.shape != (len(x), 2) or groups.shape != y.shape or not len(x)
            or not all(np.isfinite(a).all() for a in (x, y, f))
            or (y < 0).any() or (y > 1).any()):
        raise ValueError('Expected N,3 cues / N H / N,2 context / N original IDs')
    if (min_pixels < 1 or min_scenes < 1 or shrinkage < 0 or not 0 <= max_alpha <= 1
            or not 0 <= max_delta <= 1 or not np.isfinite(physical_strength) or physical_strength < 0):
        raise ValueError('Invalid support/shrinkage configuration')
    z, w = np.repeat(y[:, None], 3, axis=1), np.ones_like(x, dtype=float)
    if physical_strength:
        if physical is None or physical.get('verified_protocol') != 'linear_srgb_scalar_t_constant_positive_A':
            raise ValueError('Physical term needs verified TRAIN formation metadata')
        z, w = response_objective(y, physical['baseline'], physical['observed'],
                                  physical['valid'], physical_strength)
    tables = [weighted_monotone(x[:, k], z[:, k], w[:, k], bins) for k in range(3)]
    cuts = [np.unique(np.quantile(f[:, 0], [1/3, 2/3])).tolist(),
            np.unique(np.quantile(f[:, 1], [.5])).tolist()]
    labels = [np.searchsorted(cuts[j], f[:, j], side='right') for j in range(2)]
    result = {'version': 'raw_content_monotone_v1', 'target': 'H=1-t',
              'global': tables, 'cells': {}, 'support': {}, 'cuts': cuts,
              'context_bounds': [[float(f[:, j].min()), float(f[:, j].max())] for j in range(2)],
              'raw_limits': [[float(x[:, k].min()), float(x[:, k].max())] for k in range(3)],
              'train_scene_ids': sorted(set(groups.tolist())),
              'cue_window': cue_window, 'content_window': content_window,
              'max_delta': max_delta, 'physical_strength': physical_strength,
              'context': ['hazy_RGB_local_luminance_mean', 'hazy_RGB_local_absolute_texture'],
              'physical_policy': 'training supervision only; no clean/A/t or applicability mask at inference'}
    for i, j in itertools.product(range(len(cuts[0])+1), range(len(cuts[1])+1)):
        mask = (labels[0] == i) & (labels[1] == j)
        n, ng = int(mask.sum()), len(set(groups[mask].tolist()))
        active = n >= min_pixels and ng >= min_scenes
        key = f'{i}_{j}'
        result['support'][key] = {'pixels': n, 'originals': ng, 'active': active}
        if active:
            result['cells'][key] = {
                'alpha': min(max_alpha, n/(n+shrinkage)),
                'raw_bounds': [[float(x[mask, k].min()), float(x[mask, k].max())] for k in range(3)],
                'tables': [weighted_monotone(x[mask, k], z[mask, k], w[mask, k], bins) for k in range(3)]}
    return result


class ContentCalibratedCandidates(nn.Module):
    """Frozen correction only. forward(rgb) has no target or fusion argument."""
    def __init__(self, calibration, conditional=True):
        super().__init__()
        self.calibration = calibration
        self.conditional = conditional

    def extract(self, image):
        context = content_features(image, self.calibration['content_window'])
        return raw_cues(image, self.calibration['cue_window']), context

    def correct(self, raw, context):
        if (raw.ndim != 4 or raw.shape[1] != 3 or context.shape != (raw.shape[0], 2, *raw.shape[2:])
                or not torch.isfinite(raw).all() or not torch.isfinite(context).all()):
            raise ValueError('Invalid raw/context tensors')
        cfg = self.calibration
        global_maps = torch.cat([apply_monotone(raw[:, k:k+1], t) for k, t in enumerate(cfg['global'])], 1)
        out = global_maps.clone()
        used = torch.zeros_like(raw, dtype=torch.bool)
        if self.conditional:
            labels, supported = [], torch.ones_like(context[:, :1], dtype=torch.bool)
            for j in range(2):
                labels.append(torch.bucketize(context[:, j:j+1].contiguous(), context.new_tensor(cfg['cuts'][j]), right=True))
                low, high = cfg['context_bounds'][j]
                supported &= (context[:, j:j+1] >= low) & (context[:, j:j+1] <= high)
            for key, entry in cfg['cells'].items():
                i, j = map(int, key.split('_'))
                cell = supported & (labels[0] == i) & (labels[1] == j)
                for k, table in enumerate(entry['tables']):
                    low, high = entry['raw_bounds'][k]
                    mask = cell[:, 0] & (raw[:, k] >= low) & (raw[:, k] <= high)
                    if mask.any():
                        base = global_maps[:, k][mask]
                        delta = (apply_monotone(raw[:, k][mask], table)-base).clamp(-cfg['max_delta'], cfg['max_delta'])
                        out[:, k][mask] = base + entry['alpha']*delta
                        used[:, k][mask] = True
        return out, used

    def forward(self, image):
        raw, context = self.extract(image)
        candidates, _ = self.correct(raw, context)
        low = raw.new_tensor([p[0] for p in self.calibration['raw_limits']])[None, :, None, None]
        high = raw.new_tensor([p[1] for p in self.calibration['raw_limits']])[None, :, None, None]
        features = torch.cat([image, candidates, (raw-low)/(high-low).clamp_min(1e-8)], 1)
        return features, candidates


class FixedCandidateFusion(nn.Module):
    """Three global weights only; can share the same frozen candidates with D."""
    def __init__(self, weights):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32)
        if w.shape != (3,) or not torch.isfinite(w).all() or (w < 0).any() or not torch.isclose(w.sum(), w.new_tensor(1.)):
            raise ValueError('Expected simplex weights')
        self.register_buffer('weights', w[None, :, None, None])

    def forward(self, candidates):
        return (candidates*self.weights).sum(1, keepdim=True)


class ContentDirectReference(nn.Module):
    """Control ONLY: identical 9 inference channels and contextual trunk as D.

    This unrestricted reference is never a correction path or default output.
    Physical auxiliary supervision can be applied to its H output in a future
    authorized comparison using the same TRAIN J/A pairs and eligible masks.
    """
    def __init__(self, base_channels=32):
        super().__init__()
        from .student_weight_gate import StudentWeightGate
        self.network = StudentWeightGate(base_channels)
        self.network.head = nn.Sequential(nn.Conv2d(base_channels//2+9, 16, 3, padding=1),
                                          nn.ReLU(), nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, features, candidates=None):
        return self.network.logits(features), None
