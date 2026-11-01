import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import unittest
import torch
from src.models.haze_density.dynamic_prior import ConvexFusion,convex_oracle


class GateTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(42)

    def test_initial_equal_and_no_residual(self):
        x=torch.randn(2,9,16,16); x[:,3:6]=torch.rand(2,3,16,16)
        pred0,_=ConvexFusion('fixed')(x)
        for name in ['global','dynamic']:
            p,w=ConvexFusion(name)(x)
            torch.testing.assert_close(p,pred0)
            torch.testing.assert_close(w.sum(1),torch.ones(2,16,16))
            self.assertTrue((w>=0).all())
            self.assertTrue((p>=x[:,3:6].amin(1,keepdim=True)-1e-7).all())
            self.assertTrue((p<=x[:,3:6].amax(1,keepdim=True)+1e-7).all())
        x[:,3:6]=.4
        p,_=ConvexFusion('dynamic')(x)
        torch.testing.assert_close(p,torch.full_like(p,.4))

    def test_oracle_bound_and_gradient(self):
        c=torch.tensor([.2,.4,.8]).reshape(1,3,1,1).expand(1,3,1,3)
        truth=torch.tensor([.0,.5,1.]).reshape(1,1,1,3)
        torch.testing.assert_close(convex_oracle(c,truth),torch.tensor([.2,.5,.8]).reshape(1,1,1,3))
        x=torch.rand(2,9,12,12); model=ConvexFusion('dynamic')
        p,_=model(x); p.square().mean().backward()
        self.assertTrue(torch.isfinite(model.gate[-1].weight.grad).all())
        self.assertGreater(float(model.gate[-1].weight.grad.abs().sum()),0)


if __name__=='__main__': unittest.main()
