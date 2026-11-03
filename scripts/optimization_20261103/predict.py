"""Frozen joint model file inference, eager or captured CUDA; shared [0,1] scale."""
import argparse,json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from PIL import Image
from src.models.haze_density.joint_correction import JointCorrection,InferencePipeline
from src.models.haze_density.captured_inference import CapturedInference
from src.models.haze_density.approximate_inference import ApproxCapturedInference
def main():
    a=argparse.ArgumentParser();a.add_argument('--weights',type=Path,required=True);a.add_argument('--image',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--device',default='cuda');a.add_argument('--execution',choices=['eager','captured','approximate'],default='captured');args=a.parse_args()
    if args.execution!='eager' and not args.device.startswith('cuda'):a.error('captured execution requires CUDA; select eager for CPU')
    paths=[args.output.with_suffix(s) for s in ['.npy','.png','.json']]
    for p in paths:
        if p.exists():raise FileExistsError(str(p))
    torch.set_num_threads(1);pack=torch.load(args.weights,map_location='cpu',weights_only=False)
    n=JointCorrection([.9,.05,.05],pack['mode']=='joint_dynamic');n.load_state_dict(pack['model'])
    pipe=InferencePipeline(n,pack['calibration']).to(args.device).eval()
    with Image.open(args.image) as im:
        if im.mode!='RGB':raise ValueError('Verify encoding and supply an RGB image')
        arr=np.array(im,dtype=np.float32)/255
    x=torch.from_numpy(arr).permute(2,0,1)[None].to(args.device);start=time.monotonic()
    runner=ApproxCapturedInference(pipe,x) if args.execution=='approximate' else (CapturedInference(pipe,x) if args.execution=='captured' else pipe)
    with torch.no_grad():h=runner(x)[0,0].cpu().numpy()
    paths[0].parent.mkdir(parents=True,exist_ok=True);np.save(paths[0],h)
    from matplotlib import colormaps
    Image.fromarray((colormaps['viridis'](np.clip(h,0,1))*255).astype('uint8')).save(paths[1])
    with paths[2].open('x') as f:json.dump({'weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),'execution':args.execution,
        'seconds_including_build_and_first_inference':time.monotonic()-start,'color_scale':[0,1],
        'input_contract':'Display RGB as produced by linear-domain haze synthesis, NOT arbitrary sensor calibration; no automatic domain recovery'},f,indent=2)
    print('Saved',*[str(p) for p in paths])
if __name__=='__main__':main()
