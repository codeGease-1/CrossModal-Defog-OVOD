"""Train-only semantic conditional calibration, separate from fusion.

Semantics are derived from hazy RGB. Each prior has its own global/conditional
monotone map. No semantic value changes the three fusion weights here.
"""
import numpy as np
import torch
from torch import nn
from .calibrated_prior import fit_monotone, apply_monotone
from .student_weight_gate import StudentWeightGate
from .dynamic_prior import single_scores


def fit_calibration(scores, target, probabilities, scene_ids, *, split,
                    confidence=.6, min_pixels=1024, min_scenes=4,
                    shrinkage=4096, bins=64):
    """Fit sampled training pixels only. scene_ids refer to original scenes."""
    if split != 'train':
        raise ValueError('Calibration may only fit training scenes')
    x, y, p = map(np.asarray, (scores, target, probabilities))
    groups = np.asarray(scene_ids)
    if (x.ndim != 2 or x.shape[1] != 3 or y.shape != (len(x),)
            or p.ndim != 2 or p.shape[0] != len(x) or groups.shape != y.shape
            or not all(np.isfinite(a).all() for a in (x, y, p))
            or not np.allclose(p.sum(1), 1, atol=1e-5) or (p < 0).any()
            or (y < 0).any() or (y > 1).any()):
        raise ValueError('Expected finite N,3 scores / N target / N,K probabilities / N scene IDs')
    if not 0 <= confidence <= 1 or min_pixels < 1 or min_scenes < 1 or shrinkage < 0:
        raise ValueError('Invalid support or confidence thresholds')
    result = {'global': [fit_monotone(x[:, i], y, bins) for i in range(3)],
              'classes': {}, 'support': {}, 'confidence': confidence,
              'train_scene_ids': sorted(set(groups.tolist())),
              'target': 'H=1-t', 'semantic_input': 'frozen predictions from hazy RGB only'}
    labels, conf = p.argmax(1), p.max(1)
    for k in range(p.shape[1]):
        mask = (labels == k) & (conf >= confidence)
        n, g = int(mask.sum()), len(set(groups[mask].tolist()))
        # Group zero aggregates all unrequested ADE categories. It has no
        # definite land-cover meaning and must fall back to the global map.
        active = k != 0 and n >= min_pixels and g >= min_scenes
        result['support'][str(k)] = {'pixels': n, 'scenes': g, 'active': active,
                                    'eligible': k != 0}
        if active:
            result['classes'][str(k)] = {
                'alpha': n / (n + shrinkage),
                'tables': [fit_monotone(x[mask, i], y[mask], bins) for i in range(3)]}
    return result


class SemanticCalibration(nn.Module):
    """Frozen per-prior estimator. Inference API has no reference or target."""
    def __init__(self, calibration):
        super().__init__()
        self.calibration = calibration

    def forward(self, scores, probabilities, conditional=True):
        if (scores.ndim != 4 or scores.shape[1] != 3 or probabilities.ndim != 4
                or probabilities.shape[0] != scores.shape[0]
                or probabilities.shape[2:] != scores.shape[2:]
                or not torch.isfinite(scores).all() or not torch.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not torch.allclose(probabilities.sum(1), torch.ones_like(scores[:, 0]), atol=1e-5)):
            raise ValueError('Expected B,3,H,W scores and normalized B,K,H,W semantics')
        out = torch.cat([apply_monotone(scores[:, i:i+1], table)
                         for i, table in enumerate(self.calibration['global'])], 1)
        if conditional:
            conf, label = probabilities.max(1, keepdim=True)
            for k, entry in self.calibration['classes'].items():
                mask = (label == int(k)) & (conf >= self.calibration['confidence'])
                mapped = torch.cat([apply_monotone(scores[:, i:i+1], table)
                                    for i, table in enumerate(entry['tables'])], 1)
                out = torch.where(mask, out * (1-entry['alpha']) + mapped * entry['alpha'], out)
        return out


def candidate_diagnostics(candidates, target, tolerance=.03):
    """Offline scalar convex-envelope bound; NEVER an inference prediction."""
    c, y = np.asarray(candidates, dtype=float), np.asarray(target, dtype=float)
    if c.ndim != 3 or c.shape[0] != 3 or y.shape != c.shape[1:]:
        raise ValueError('Expected 3,H,W candidates and H,W target')
    if not np.isfinite(c).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite candidate or target')
    error = c-y[None]
    lo, hi = c.min(0), c.max(0)
    distance = np.maximum(lo-y, 0) + np.maximum(y-hi, 0)
    return {'prior_mae': np.abs(error).mean((1, 2)).tolist(),
            'prior_bias': error.mean((1, 2)).tolist(),
            'joint_high_fraction': float((lo > y+tolerance).mean()),
            'joint_low_fraction': float((hi < y-tolerance).mean()),
            'outside_envelope_fraction': float(((y < lo) | (y > hi)).mean()),
            'offline_oracle_mae': float(distance.mean()),
            'offline_oracle_mse': float((distance**2).mean())}


class FrozenSemanticCandidates(nn.Module):
    """One candidate extractor shared by C and D; all calibration is frozen."""
    def __init__(self, prior_calibration, semantic_calibration):
        super().__init__()
        self.prior_calibration = prior_calibration
        self.correction = SemanticCalibration(semantic_calibration)

    def forward(self, image, probabilities):
        raw, scores = single_scores(image, self.prior_calibration)
        candidates = self.correction(scores, probabilities)
        limits = self.prior_calibration['limits']
        lo = image.new_tensor([a[0] for a in limits])[None,:,None,None]
        hi = image.new_tensor([a[1] for a in limits])[None,:,None,None]
        features = torch.cat([image, candidates, (raw-lo)/(hi-lo).clamp_min(1e-8)], 1)
        return features, candidates


class SemanticWeightGate(StudentWeightGate):
    """D: same C candidates, three softmax weights, no residual output."""
    def __init__(self, semantic_channels=4, base_channels=32, temperature=2.):
        super().__init__(base_channels, temperature)
        self.head = nn.Sequential(nn.Conv2d(base_channels//2+9+semantic_channels, 16, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(16, 3, 1))
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.copy_(temperature*torch.tensor([.5, .3, .2]).log())


class SemanticDirectRegressor(StudentWeightGate):
    """Information-matched RGB + identical derived semantics reference."""
    def __init__(self, semantic_channels=4, base_channels=32):
        super().__init__(base_channels)
        self.head = nn.Sequential(nn.Conv2d(base_channels//2+3+semantic_channels, 16, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, rgb_semantics, candidates=None):
        return self.logits(rgb_semantics), None
