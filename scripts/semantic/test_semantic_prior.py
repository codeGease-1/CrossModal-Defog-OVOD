"""Contract tests for leakage, sparse fallback and scalar fusion limits."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import unittest
import numpy as np
import torch
from src.models.haze_density.semantic_prior import (
    fit_calibration, SemanticCalibration, candidate_diagnostics,
    SemanticWeightGate, SemanticDirectRegressor)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(7)
        self.x=rng.random((100,3));self.y=rng.random(100)
        self.p=np.tile([.1,.9],(100,1));self.groups=np.repeat(['a','b','c','d'],25)

    def test_refuses_nontrain(self):
        with self.assertRaises(ValueError):
            fit_calibration(self.x,self.y,self.p,self.groups,split='development')

    def test_scene_support_and_uncertainty_fallback(self):
        cal=fit_calibration(self.x,self.y,self.p,self.groups,split='train',min_pixels=10,min_scenes=4)
        self.assertEqual(set(cal['classes']),{'1'})
        m=SemanticCalibration(cal);scores=torch.rand(1,3,8,8)
        uncertain=torch.full((1,2,8,8),.5)
        self.assertTrue(torch.equal(m(scores,uncertain),m(scores,uncertain,conditional=False)))
        cal=fit_calibration(self.x,self.y,self.p,['same']*100,split='train',min_pixels=10,min_scenes=4)
        self.assertEqual(cal['classes'],{})

    def test_unspecified_other_always_falls_back(self):
        p=np.tile([.99,.01],(100,1))
        cal=fit_calibration(self.x,self.y,p,self.groups,split='train',min_pixels=10,min_scenes=4)
        self.assertFalse(cal['support']['0']['eligible'])
        self.assertEqual(cal['classes'],{})

    def test_oracle_is_exact_interval_projection(self):
        c=np.array([[[.2,.4,.1]],[[.4,.5,.2]],[[.6,.8,.3]]]);y=np.array([[.1,.6,.7]])
        d=candidate_diagnostics(c,y,.03)
        self.assertAlmostEqual(d['offline_oracle_mae'],(.1+0+.4)/3)
        self.assertAlmostEqual(d['outside_envelope_fraction'],2/3)
        self.assertAlmostEqual(d['joint_high_fraction'],1/3)
        self.assertAlmostEqual(d['joint_low_fraction'],1/3)

    def test_D_same_candidates_convex_and_semantic_regression(self):
        torch.manual_seed(9)
        f=torch.rand(1,13,32,32);c=f[:,3:6].clone()
        gate=SemanticWeightGate(base_channels=8).eval()
        pred,w=gate(f,c)
        self.assertTrue(torch.allclose(w.sum(1),torch.ones_like(w[:,0]),atol=1e-6))
        self.assertTrue((pred>=c.amin(1,keepdim=True)-1e-6).all())
        self.assertTrue((pred<=c.amax(1,keepdim=True)+1e-6).all())
        # A single CPU fixture step checks independent H supervision/backprop;
        # this is not a fitted haze model or an efficacy experiment.
        loss=(pred-torch.full_like(pred,.25)).square().mean()
        loss.backward()
        self.assertGreater(float(gate.head[-1].weight.grad.abs().sum()),0.)
        optimizer=torch.optim.SGD(gate.parameters(),lr=.001);optimizer.step()
        for i in range(3):
            chosen=torch.zeros_like(c);chosen[:,i]=1
            self.assertTrue(torch.equal((chosen*c).sum(1,keepdim=True),c[:,i:i+1]))
        rgb_sem=torch.cat([f[:,:3],f[:,9:]],1)
        reg=SemanticDirectRegressor(base_channels=8).eval()
        out,_=reg(rgb_sem)
        self.assertEqual(out.shape,(1,1,32,32));self.assertTrue(torch.isfinite(out).all())


if __name__=='__main__':unittest.main()
