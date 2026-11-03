"""CPU/CUDA, edge-case and existing development-scene numerical/cost checks."""
import copy
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.v15.run import read,write,procedural,cases,tensor
from src.models.haze_density.dynamic_prior import FrozenCandidates
from src.models.haze_density.fast_candidates import FastCandidates
from src.models.haze_density.learned_correction import RestrictedCorrector,observed_context,RGBReference
from scripts.matched_fusion.run import settings


def timing(fn,n=40):
    with torch.no_grad():
        for _ in range(8):fn()
        torch.cuda.synchronize();values=[];torch.cuda.reset_peak_memory_stats()
        for _ in range(n):
            torch.cuda.synchronize();s=time.perf_counter();fn();torch.cuda.synchronize();values.append((time.perf_counter()-s)*1000)
    return {'p50_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'repeats':n}


def main():
    settings();start=time.monotonic();out=ROOT/'experiments/joint_correction_20261102'
    cal=read('/mnt/CrossModal-Defog-OVOD/experiments/v2_20261101/calibration.json')
    p=read('/mnt/CrossModal-Defog-OVOD/experiments/learned_correction_20261102/formal_amp/protocol.json')
    rows=[];maxdiff=0.
    for device in ['cpu','cuda']:
        old=FrozenCandidates(cal).to(device);fast=FastCandidates(cal).to(device);torch.manual_seed(842)
        for size,batch in [(1,1),(32,2),(65,1),(256,1),(512,1)]:
            for kind in ['random','zero','one','constant']:
                x=torch.rand(batch,3,size,size,device=device) if kind=='random' else torch.full((batch,3,size,size),{'zero':0.,'one':1.,'constant':.3}[kind],device=device)
                with torch.no_grad():a=old(x)[0];b=fast(x)[0]
                delta=float((a-b).abs().max());maxdiff=max(maxdiff,delta)
                rows.append({'device':device,'size':size,'batch':batch,'kind':kind,'feature_max_abs_difference':delta})
        for invalid in [float('nan'),-1.,2.]:
            try:fast(torch.full((1,3,32,32),invalid,device=device));raise AssertionError('Invalid input accepted')
            except ValueError:pass
    old=FrozenCandidates(cal).cuda();fast=FastCandidates(cal).cuda();realmax=0.;count=0
    with torch.no_grad():
        for scene in [s for s in p['scenes'] if s['split']=='development']:
            clear,_=procedural(scene['seed'],p['base']['image_size'])
            for meta,rgb,y in cases(clear,scene,p['base']):
                x=tensor(rgb,'cuda');a=old(x)[0];b=fast(x)[0]
                realmax=max(realmax,float((a-b).abs().max()));count+=1
    # A small FP32 rounding tolerance is registered here before any training.
    assert maxdiff<=1e-6 and realmax<=1e-6,(maxdiff,realmax)
    model=RestrictedCorrector(16).cuda().eval()
    model.load_state_dict(torch.load('/mnt/CrossModal-Defog-OVOD/experiments/learned_correction_20261102/formal_amp/corrector16_seed42/best.pth',map_location='cpu',weights_only=False)['model'])
    costs=[];outdiff=0.
    for size in [256,512]:
        rgb=torch.rand(1,3,size,size,device='cuda')
        def forward(ext):
            f,_=ext(rgb);f=torch.cat([f,observed_context(rgb)],1)
            with torch.autocast('cuda',dtype=torch.float16):return model(f)[0]
        with torch.no_grad():outdiff=max(outdiff,float((forward(old)-forward(fast)).abs().max()))
        costs.append({'size':size,'legacy_prior':timing(lambda:old(rgb)),'fast_prior':timing(lambda:fast(rgb)),
                      'legacy_full_corrector':timing(lambda:forward(old)),'fast_full_corrector':timing(lambda:forward(fast))})
    assert outdiff<=1e-5,outdiff
    result={'cases':rows,'existing_development_cases':count,'existing_development_feature_max_difference':realmax,
        'feature_max_difference':maxdiff,'trained_corrector_max_difference':outdiff,'feature_tolerance':1e-6,'network_tolerance':1e-5,
        'costs':costs,'seconds':time.monotonic()-start,'definition':'Same formulas, FP32 prior; shared guide statistics, cached batched exact table interpolation. No source/calibration refit.'}
    write(out/'fast_checks.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
