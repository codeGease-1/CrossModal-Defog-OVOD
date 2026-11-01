"""Verify refactored student produces only convex weights and trainable features."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from src.models.haze_density.student_weight_gate import StudentWeightGate,TemperatureGate
from src.models.haze_density.haze_density_net import HazeDensityNet
from scripts.v2.rgb_control import RGBControl

class GateTests(unittest.TestCase):
    def test_student_contract_and_trunk(self):
        torch.set_num_threads(1);torch.manual_seed(0)
        model=StudentWeightGate(base_channels=8);old=HazeDensityNet(base_channels=8)
        self.assertEqual(set(model.encoder.state_dict()),set(old.encoder.state_dict()))
        self.assertEqual(set(model.multiscale.state_dict()),set(old.multiscale.state_dict()))
        self.assertEqual(set(model.fusion.state_dict()),set(old.fusion.state_dict()))
        self.assertFalse(hasattr(model,'decoder'))
        features=torch.rand(2,9,33,35);target=torch.rand(2,1,33,35);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        for step in range(2):
            optimizer.zero_grad();pred,w=model(features)
            self.assertEqual(tuple(w.shape),(2,3,33,35));self.assertTrue(torch.allclose(w.sum(1),torch.ones_like(w[:,0]),atol=1e-6))
            self.assertTrue(torch.allclose(pred,(w*features[:,3:6]).sum(1,keepdim=True)))
            self.assertTrue((pred>=features[:,3:6].amin(1,keepdim=True)-1e-6).all())
            self.assertTrue((pred<=features[:,3:6].amax(1,keepdim=True)+1e-6).all())
            (pred-target).square().mean().backward()
            if step==1:self.assertGreater(float(model.encoder.conv1.conv.weight.grad.abs().sum()),0.)
            optimizer.step()

    def test_equal_initialization(self):
        features=torch.rand(1,9,32,32);reference=(features[:,3:6]*torch.tensor([.5,.3,.2])[None,:,None,None]).sum(1,keepdim=True)
        for model in (TemperatureGate(),StudentWeightGate(base_channels=8)):
            pred,_=model(features);self.assertTrue(torch.allclose(pred,reference,atol=1e-6))

    def test_rgb_control_cannot_access_prior_channels(self):
        model=RGBControl().eval();features=torch.rand(1,9,32,32);altered=features.clone();altered[:,3:]=torch.rand_like(altered[:,3:])*100
        with torch.no_grad():a,_=model(features);b,_=model(altered);c,_=model(features[:,:3])
        self.assertTrue(torch.equal(a,b) and torch.equal(a,c))

if __name__=='__main__':unittest.main()
