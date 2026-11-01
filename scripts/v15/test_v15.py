"""Checks for information loss, exact legacy control, and frozen inference."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.models.haze_density.calibrated_prior import (
    raw_cues, normalized_cues, fixed_cues, fuse, fit_monotone, apply_monotone, CalibratedPrior)
from src.models.haze_density.physical_prior import generate_s_final


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.image=torch.rand(1,3,40,40)*.7

    def test_legacy_exact_and_uniform_haze_information_loss(self):
        r=raw_cues(self.image)
        n=normalized_cues(r)
        torch.testing.assert_close(fuse(self.image,n),generate_s_final(self.image))
        hazy=.5*self.image+.5
        rh=raw_cues(hazy)
        torch.testing.assert_close(normalized_cues(rh),n,atol=2e-5,rtol=2e-5)
        self.assertGreater(float(rh[:,0].mean()),float(r[:,0].mean())+.2)

    def test_monotone_is_finite_bounded_and_constant_safe(self):
        table=fit_monotone([0,1,2,3],[.1,.8,.2,.9])
        self.assertTrue(np.all(np.diff(table['y'])>=0))
        result=apply_monotone(torch.linspace(-1,4,50),table)
        self.assertTrue(torch.all(torch.diff(result)>=0))
        self.assertTrue(torch.all((result>=0)&(result<=1)))
        table=fit_monotone([1,1,1],[0,.5,1])
        torch.testing.assert_close(apply_monotone(torch.tensor([0.,1.,2.]),table),torch.full((3,),.5))
        with self.assertRaises(ValueError): fit_monotone([float('nan')],[0])

    def test_inference_is_batch_independent_and_never_refits(self):
        cal={'prior':{'window_size':15,'guided_radius':15,'guided_eps':.01},
             'limits':[[0,.8],[-.2,0],[-.6,0]],'tables':{'fixed':{'x':[0,1],'y':[0,1]}}}
        before=json.dumps(cal,sort_keys=True)
        model=CalibratedPrior(cal)
        one=model(self.image)
        batch=model(torch.cat([self.image,torch.ones_like(self.image)]))[:1]
        torch.testing.assert_close(one,batch)
        self.assertEqual(before,json.dumps(cal,sort_keys=True))


if __name__=='__main__': unittest.main()
