"""Frozen candidate correction followed by an independently trained convex gate."""
import math
import torch
from torch import nn
from .learned_correction import RestrictedCorrector, observed_context
from .dynamic_prior import FrozenCandidates


class MatchedFusion(nn.Module):
    """Both arms share frozen correction-trained features; only candidate maps differ.

    forward_features takes the same 11 deterministic RGB-derived channels as the
    original experiment. It never receives J, t, A, identity, or an oracle.
    Raw arm retains the same resident corrector for a controlled intervention;
    total resident parameters/cost must still be counted.
    """
    def __init__(self, corrector, corrected=True, hidden=16):
        super().__init__()
        self.corrector = corrector.requires_grad_(False).eval()
        self.corrected = bool(corrected)
        self.gate = nn.Sequential(nn.Conv2d(19, hidden, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(hidden, 3, 1))
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor([.9, .05, .05]).log())

    def train(self, mode=True):
        super().train(mode)
        self.corrector.eval()
        return self

    @torch.no_grad()
    def prepare(self, features):
        rgb, raw, context = features[:, :3], features[:, 3:6], features[:, 9:11]
        z = self.corrector.trunk(rgb)
        v = self.corrector.head(torch.cat([z, rgb, context], 1))
        gain = torch.exp(math.log(2)*torch.tanh(v[:, :3]))
        offset = self.corrector.shift*torch.tanh(v[:, 3:])
        corrected = self.corrector.transform(raw, gain, offset)
        return torch.cat([z, features], 1), corrected

    def from_prepared(self, inputs, candidates):
        weights = torch.softmax(self.gate(inputs).float(), dim=1)
        return (weights*candidates.float()).sum(1, keepdim=True), weights

    def forward(self, features):
        inputs, corrected = self.prepare(features)
        candidates = corrected if self.corrected else features[:, 3:6]
        prediction, weights = self.from_prepared(inputs, candidates)
        return prediction, {'weights': weights, 'candidates': candidates}


class RGBPipeline(nn.Module):
    """Complete inference API: only a floating point N,3,H,W hazy RGB tensor."""
    def __init__(self, network, calibration=None, uses_priors=False):
        super().__init__()
        self.network = network
        self.uses_priors = uses_priors
        if uses_priors:
            if calibration is None: raise ValueError('Frozen calibration required')
            self.extractor = FrozenCandidates(calibration)

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError('Expected N,3,H,W hazy RGB only')
        if self.uses_priors:
            features, _ = self.extractor(rgb.float())
            features = torch.cat([features, observed_context(rgb.float())], 1)
        else:
            features = rgb
        return self.network(features)[0]
