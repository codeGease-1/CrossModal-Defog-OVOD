"""Restricted mapping and inference leakage contracts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.models.haze_density.learned_correction import RestrictedCorrector, RGBReference, observed_context


def main():
    torch.set_num_threads(1); torch.manual_seed(42)
    rgb = torch.rand(2, 3, 32, 32)
    p = torch.rand(2, 3, 32, 32)
    x = torch.cat([rgb, p, torch.rand_like(p), observed_context(rgb)], 1)
    m = RestrictedCorrector(16)
    c, aux = m(x)
    assert torch.allclose(c, p, atol=1e-4)
    c.mean().backward()
    assert m.head[-1].weight.grad.abs().sum() > 0
    m.head[-1].weight.data.normal_(0, .1)
    c, aux = m(x)
    assert aux['gain'].min() >= .5 and aux['gain'].max() <= 2
    assert aux['offset'].abs().max() <= 3
    # Context fixed: changing prior 0 cannot alter corrected priors 1/2.
    x2 = x.clone(); x2[:, 3] = (x2[:, 3]+.1).clamp(max=1)
    c2, _ = m(x2)
    assert torch.equal(c[:, 1:], c2[:, 1:])
    assert (c2[:, 0] >= c[:, 0]-1e-7).all()
    assert (c2[:, 0]-c[:, 0]).mean() > .01
    assert c.min() >= 0 and c.max() <= 1
    for base in [16, 32]:
        r = RGBReference(base, with_context=base == 16)
        y, _ = r(x)
        assert y.shape == (2, 1, 32, 32)
        print('RGB', base, sum(q.numel() for q in r.parameters()))
    print('Corrector', sum(q.numel() for q in m.parameters()))
    assert sum(q.numel() for q in m.parameters()) < sum(q.numel() for q in RGBReference(32).parameters())
    print('PASS: identity initialization, finite gradients, bounded coefficients, own-cue isolation, monotonic response, parameter count')


if __name__ == '__main__': main()
