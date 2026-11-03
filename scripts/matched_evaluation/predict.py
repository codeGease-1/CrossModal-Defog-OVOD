"""Portable RGB-only inference, fixed output scale and explicit FP32 preprocessing."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.models.haze_density.learned_correction import RestrictedCorrector,RGBReference,observed_context
from src.models.haze_density.matched_fusion import MatchedFusion
from src.models.haze_density.dynamic_prior import FrozenCandidates


class Predictor(torch.nn.Module):
    def __init__(self,package,device='cpu',amp=True):
        super().__init__();mode=package['mode'];self.mode=mode;self.amp=amp and device.startswith('cuda')
        if mode.endswith('_gate'):self.network=MatchedFusion(RestrictedCorrector(16),mode=='corrected_gate',package['gate_hidden'])
        else:self.network=RGBReference(32 if mode=='rgb32' else 16,with_priors=mode=='direct_same16')
        self.network.load_state_dict(package['model'])
        self.uses_priors=mode not in ['rgb16','rgb32']
        if self.uses_priors:self.extractor=FrozenCandidates(package['calibration'])
        self.to(device).eval()

    def forward(self,rgb):
        if rgb.ndim!=4 or rgb.shape[1]!=3 or not rgb.is_floating_point():raise ValueError('Expected N,3,H,W floating RGB')
        if not torch.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:raise ValueError('Expected finite display RGB in [0,1]')
        device=rgb.device.type
        with torch.autocast(device,enabled=False):
            if self.uses_priors:
                features,_=self.extractor(rgb.float());features=torch.cat([features,observed_context(rgb.float())],1)
            else:features=rgb.float()
        with torch.autocast(device,dtype=torch.float16,enabled=self.amp):prediction,_=self.network(features)
        return prediction.float()


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    npy=a.output.with_suffix('.npy');png=a.output.with_suffix('.png')
    if npy.exists() or png.exists():raise FileExistsError('Existing predictions protected; choose a new output prefix')
    with Image.open(a.image) as image:
        if image.mode!='RGB':raise ValueError('Verify sensor/bit-depth conversion first; input must be RGB')
        array=np.asarray(image,np.float32)/255
    model=Predictor(torch.load(a.weights,map_location='cpu',weights_only=False),a.device)
    with torch.no_grad():h=model(torch.from_numpy(array.copy()).permute(2,0,1)[None].to(a.device))[0,0].cpu().numpy()
    npy.parent.mkdir(parents=True,exist_ok=True);np.save(npy,h)
    from matplotlib import colormaps
    rgba=(colormaps['viridis'](np.clip(h,0,1))*255).astype('uint8');Image.fromarray(rgba).save(png)
    print('Saved raw H and fixed [0,1] color map:',npy,png)


if __name__=='__main__':main()
