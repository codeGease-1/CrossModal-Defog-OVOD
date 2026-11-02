"""Restricted per-candidate calibration, distinct from RGB regression controls."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .student_weight_gate import StudentWeightGate
from .physical_prior import _local_mean


def observed_context(rgb):
    gray = (rgb*rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    mean = _local_mean(gray, 31)
    return torch.cat([mean, _local_mean((gray-mean).abs(), 31)], 1)


class ContextTrunk(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        source = StudentWeightGate(base)
        self.encoder, self.multiscale, self.fusion, self.up = source.encoder, source.multiscale, source.fusion, source.up

    def forward(self, rgb):
        z = self.encoder(rgb)
        a, b, c, _ = self.multiscale(z, return_separate=True)
        z = self.up(self.fusion(a, b, c))
        return F.interpolate(z, rgb.shape[-2:], mode='bilinear', align_corners=False) if z.shape[-2:] != rgb.shape[-2:] else z


class RestrictedCorrector(nn.Module):
    """RGB/context predicts bounded gain/offset, each acting on ONE own prior.

    P'_k = (1-rho)P_k + rho*sigmoid(a_k*logit(P_k)+b_k).
    0.5<=a<=2, |b|<=3, rho=.75. No free H output/residual or fusion weights.
    Explicit retained prior path prevents complete removal of cue dependence.
    This is statistical calibration, NOT a physical inversion or identifiable
    separation of haze and content. Necessity needs replacement ablations.
    """
    def __init__(self, base=16, rho=.75, shift=3.):
        super().__init__()
        self.trunk = ContextTrunk(base)
        self.head = nn.Sequential(nn.Conv2d(base//2+5, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 6, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.rho, self.shift = rho, shift

    def coefficients(self, rgb, context):
        z = self.head(torch.cat([self.trunk(rgb), rgb, context], 1))
        return torch.exp(math.log(2)*torch.tanh(z[:, :3])), self.shift*torch.tanh(z[:, 3:])

    def forward(self, features):
        rgb, candidates, context = features[:, :3], features[:, 3:6], features[:, 9:11]
        gain, offset = self.coefficients(rgb, context)
        corrected = self.transform(candidates, gain, offset)
        return corrected, {'gain': gain, 'offset': offset}

    def transform(self, candidates, gain, offset):
        safe = candidates.clamp(1e-4, 1-1e-4)
        mapped = torch.sigmoid(gain*torch.logit(safe)+offset)
        return (1-self.rho)*candidates + self.rho*mapped


class RGBReference(nn.Module):
    """Same trunk family; all inputs are RGB-derived, no hidden supervision."""
    def __init__(self, base=32, with_context=False, with_priors=False):
        super().__init__()
        self.trunk = ContextTrunk(base)
        self.with_context = with_context
        self.with_priors = with_priors
        self.head = nn.Sequential(nn.Conv2d(base//2+3+(8 if with_priors else (2 if with_context else 0)), 16, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, features):
        rgb = features[:, :3]
        inputs = [self.trunk(rgb), rgb]
        if self.with_priors: inputs.append(features[:, 3:11])
        elif self.with_context: inputs.append(features[:, 9:11])
        return self.head(torch.cat(inputs, 1)), None
