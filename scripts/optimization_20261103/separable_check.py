"""Separable local-mean feasibility check; reject if frozen tolerances fail."""
import sys,json,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,numpy as np
import torch.nn.functional as F
import src.models.haze_density.captured_inference as ci
from src.models.haze_density.captured_inference import CapturedInference,CaptureCore,TrustedCandidates
from scripts.optimization_20261103.latency import load,checked,timed,OLD
from scripts.v15.run import procedural,cases,tensor
OUT=ROOT/'experiments/optimization_20261103/separable_check'
original_mean=ci._mean_filter
def separable_mean(x,kernel_size):
    ref=x[:,:,:1,:1];p=kernel_size//2
    v=F.avg_pool2d(x-ref,(1,kernel_size),stride=1,padding=(0,p),count_include_pad=False)
    return F.avg_pool2d(v,(kernel_size,1),stride=1,padding=(p,0),count_include_pad=False)+ref
def main():
    OUT.mkdir();torch.set_num_threads(1);torch.manual_seed(20261103);start=time.monotonic();rows=[];maxf=0.;maxh=0.
    with torch.no_grad():
        pipe=load('joint_dynamic')
        ci._mean_filter=separable_mean
        core=CaptureCore(copy.deepcopy(pipe),'cuda')
        ext=TrustedCandidates(pipe.extractor.calibration).cuda();ext.load_state_dict(pipe.extractor.state_dict());core.extractor=ext
        for size in [1,32,65,256,512]:
            for kind in ['zero','one','constant','random']:
                x=torch.rand(1,3,size,size,device='cuda')
                if kind=='zero':x.zero_()
                if kind=='one':x.fill_(1)
                if kind=='constant':x.fill_(.3)
                f1,_=pipe.extractor(x);f2,_=ext(x);df=float((f1-f2).abs().max());maxf=max(maxf,df)
                if size>=32:dh=float((pipe(x)-core(x)).abs().max());maxh=max(maxh,dh)
                else:dh=None
                rows.append(dict(size=size,kind=kind,feature_difference=df,H_difference=dh))
        p=json.loads((OLD/'protocol.json').read_text())
        for scene in [s for s in p['scenes'] if s['split']=='development']:
            clear,_=procedural(scene['seed'],256)
            for meta,rgb,y in cases(clear,scene,p['base']):
                x=tensor(rgb,'cuda');f1,_=pipe.extractor(x);f2,_=ext(x);df=float((f1-f2).abs().max())
                dh=float((pipe(x)-core(x)).abs().max());maxf=max(maxf,df);maxh=max(maxh,dh)
        accepted=maxf<=1e-6 and maxh<=1e-5
        costs=[]
        if accepted:
            for mode in ['joint_dynamic','rgb16','direct_same16']:
                q=load(mode)
                for size in [256,512]:
                    x=torch.rand(1,3,size,size,device='cuda');run=CapturedInference(q,x)
                    costs.append(dict(method=mode,size=size,**timed(lambda:run(x),60)))
        result=dict(accepted=accepted,feature_max_difference=maxf,H_max_difference=maxh,tolerances=dict(feature=1e-6,H=1e-5),
                    seconds=time.monotonic()-start,boundary_checks=rows,costs=costs,existing_development_cases=512,
                    interpretation='No relaxation: failed numerical tolerances means this is diagnostic only, not released inference path.')
        (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
