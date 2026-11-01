"""V2: separately calibrated, frozen candidate maps and convex RGB gating.

There is no learned residual or direct RGB-to-density output. Truth is used
only by training/diagnostics; this module's inference never receives truth.
"""
import torch
from .calibrated_prior import raw_cues, fixed_cues, apply_monotone
from .guided_filter import guided_filter


def single_scores(image, calibration):
    raw = raw_cues(image, calibration['prior']['window_size'])
    cues = fixed_cues(raw, calibration['limits'])
    gray = (image*image.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    cfg = calibration['prior']
    # guided_filter intentionally averages multi-channel src: call separately.
    maps = [guided_filter(gray,cues[:,i:i+1].pow(1.5),cfg['guided_radius'],cfg['guided_eps']) for i in range(3)]
    return raw, torch.cat(maps,1)


class FrozenCandidates(torch.nn.Module):
    def __init__(self, calibration):
        super().__init__()
        self.calibration = calibration

    def forward(self, image):
        raw, scores = single_scores(image,self.calibration)
        candidates = torch.cat([apply_monotone(scores[:,i:i+1],self.calibration['candidate_tables'][i]) for i in range(3)],1)
        lo = image.new_tensor([v[0] for v in self.calibration['limits']])[None,:,None,None]
        hi = image.new_tensor([v[1] for v in self.calibration['limits']])[None,:,None,None]
        # No per-image scaling, no clipping here: keep out-of-range raw intensity.
        raw_standard = (raw-lo)/(hi-lo).clamp_min(1e-8)
        features = torch.cat([image,candidates,raw_standard],1)
        return features,candidates


class ConvexFusion(torch.nn.Module):
    def __init__(self, mode='dynamic', hidden=16):
        super().__init__()
        self.mode = mode
        initial = torch.tensor([.5,.3,.2]).log()
        if mode == 'fixed':
            self.register_buffer('logits',initial)
        elif mode == 'global':
            self.logits = torch.nn.Parameter(initial)
        elif mode == 'dynamic':
            self.gate = torch.nn.Sequential(torch.nn.Conv2d(9,hidden,3,padding=1),torch.nn.ReLU(),
                torch.nn.Conv2d(hidden,hidden,3,padding=1),torch.nn.ReLU(),torch.nn.Conv2d(hidden,3,1))
            # Equal initial predictions across A/B/C; hidden features can learn.
            torch.nn.init.zeros_(self.gate[-1].weight)
            with torch.no_grad(): self.gate[-1].bias.copy_(initial)
        else:
            raise ValueError('mode must be fixed/global/dynamic')

    def forward(self, features, candidates=None):
        if candidates is None: candidates = features[:,3:6]
        logits = self.gate(features) if self.mode == 'dynamic' else self.logits[None,:,None,None]
        weights = torch.softmax(logits,dim=1).expand_as(candidates)
        return (weights*candidates).sum(1,keepdim=True),weights


def convex_oracle(candidates, truth):
    """Exact attainable scalar interval optimum for freely varying weights.

    This is a bound on unconstrained pointwise convex fusion; a finite RGB gate
    may not attain it. Softmax reaches interval endpoints only as a limit.
    """
    lo,hi = candidates.amin(1,keepdim=True),candidates.amax(1,keepdim=True)
    return torch.maximum(lo,torch.minimum(hi,truth))
