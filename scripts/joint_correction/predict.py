"""Portable joint model inference: hazy RGB input, raw H and shared-scale PNG."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.models.haze_density.joint_correction import JointCorrection,InferencePipeline


def main():
    a=argparse.ArgumentParser();a.add_argument('--weights',type=Path,required=True);a.add_argument('--image',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--device',default='cpu');args=a.parse_args();torch.set_num_threads(1)
    npy=args.output.with_suffix('.npy');png=args.output.with_suffix('.png')
    for p in [npy,png]:
        if p.exists():raise FileExistsError('Existing prediction protected: '+str(p))
    pack=torch.load(args.weights,map_location='cpu',weights_only=False)
    net=JointCorrection([.9,.05,.05],pack['mode']=='joint_dynamic');net.load_state_dict(pack['model'])
    pipeline=InferencePipeline(net,pack['calibration']).to(args.device).eval()
    with Image.open(args.image) as im:
        if im.mode!='RGB':raise ValueError('Verify sensor encoding first: input must be RGB')
        image=np.asarray(im,dtype=np.float32)/255
    with torch.no_grad():h=pipeline(torch.from_numpy(image.copy()).permute(2,0,1)[None].to(args.device))[0,0].cpu().numpy()
    npy.parent.mkdir(parents=True,exist_ok=True);np.save(npy,h)
    from matplotlib import colormaps
    Image.fromarray((colormaps['viridis'](np.clip(h,0,1))*255).astype('uint8')).save(png)
    print('Saved raw H and fixed [0,1] color PNG',npy,png)


if __name__=='__main__':main()
