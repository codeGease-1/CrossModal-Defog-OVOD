"""Run a frozen research gate on one RGB image; never consumes truth.

The output is effective attenuation under its training protocol, not measured
atmospheric concentration. The default 256px matches calibration resolution.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read,write,digest,tensor
from scripts.v2.investigate import make_model
from src.models.haze_density.dynamic_prior import FrozenCandidates

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--output',required=True)
    p.add_argument('--experiment',default='experiments/v2_investigation_20261101');p.add_argument('--mode',default='student',choices=['student','temperature']);p.add_argument('--device',default='cuda');p.add_argument('--size',type=int,default=256)
    args=p.parse_args();torch.set_num_threads(1);out=Path(args.output)
    if out.exists():raise FileExistsError('Choose a new output directory; existing inference results are preserved')
    exp=Path(args.experiment);protocol=read(exp/'protocol.json');cfg=protocol['config'];frozen=read(exp/'frozen.json');seed=frozen['selected_seeds'][args.mode]
    calibration=Path(cfg['previous_directory'])/'calibration.json';assert digest(calibration)==frozen['calibration_sha256']
    path=exp/f'{args.mode}_seed{seed}/best.pth';assert digest(path)==frozen['checkpoints'][str(path.relative_to(exp))]
    image=Image.open(args.image).convert('RGB');original_size=image.size;image=image.resize((args.size,args.size),Image.Resampling.BILINEAR);rgb=np.asarray(image,np.float32)/255
    model=make_model(args.mode,cfg).to(args.device);model.load_state_dict(torch.load(path,map_location=args.device,weights_only=False)['model']);model.eval();extractor=FrozenCandidates(read(calibration))
    with torch.no_grad():
        f,c=extractor(tensor(rgb,args.device));h,w=model(f,c)
    out.mkdir(parents=True);np.savez_compressed(out/'prediction.npz',H=h[0,0].cpu().numpy(),weights=w[0].cpu().numpy(),candidates=c[0].cpu().numpy())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,5,figsize=(15,3));ax[0].imshow(rgb);ax[0].set_title('RGB')
    for a,title,v in zip(ax[1:],['H','weight D','weight C','weight K'],[h[0,0].cpu().numpy()]+list(w[0].cpu().numpy())):
        im=a.imshow(v,cmap='viridis',vmin=0,vmax=1);a.set_title(title)
    for a in ax:a.axis('off')
    fig.colorbar(im,ax=ax.tolist(),fraction=.015);fig.savefig(out/'prediction.png',dpi=150);plt.close(fig)
    write(out/'provenance.json',{'image':str(Path(args.image).resolve()),'image_sha256':digest(args.image),'original_size':original_size,'evaluation_size':[args.size,args.size],
        'checkpoint':str(path),'checkpoint_sha256':digest(path),'calibration_sha256':digest(calibration),'mode':args.mode,'seed':seed,'color_range':[0,1],
        'warning':'Research output conditional on synthesis/calibration domain; no guarantee of real absolute fog concentration. Other resolutions are unvalidated.'})
    print('Saved',out,'mean H',float(h.mean()))

if __name__=='__main__':main()
