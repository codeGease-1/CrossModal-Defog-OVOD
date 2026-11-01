"""Inference ablations: spatial alignment vs per-image adaptation.

No retraining/selection. Truth-based per-image constant-weight oracle is an
offline bound only, never an input to a learned predictor.
"""
import argparse,csv,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.v15.run import read,write,digest,procedural,cases,tensor,csv_write
from scripts.v2.run import record,summarize
from scripts.v2.investigate import make_model,charge,remaining
from scripts.v2.global_optimum import solve_simplex
from src.models.haze_density.dynamic_prior import FrozenCandidates

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);cfg=read('configs/v2_investigation.json');out=Path(cfg['output_directory']);dest=out/'spatial_controls';dest.mkdir(exist_ok=True)
    if not (dest/'plan.json').exists():write(dest/'plan.json',{'variants':['full spatial weights','per-image mean weights','per-image spatially shuffled weight vectors','truth-based best constant weights per image OFFLINE ONLY'],
        'data':'new nominal confirmation only','seeds':cfg['seeds'],'source_sha256':digest(__file__),
        'purpose':'Distinguish actual within-image weighting from merely image-dependent constant fusion','training':False,'selection':False,'budget':'same shared ledger'})
    assert digest(__file__)==read(dest/'plan.json')['source_sha256']
    if args.register:print('Spatial-control plan registered');return
    if (dest/'summary.json').exists():return
    assert remaining(out)>60;start=time.monotonic();p=read(out/'protocol.json');frozen=read(out/'frozen.json');old=Path(cfg['previous_directory']);ext=FrozenCandidates(read(old/'calibration.json'));models={}
    for seed in cfg['seeds']:
        path=out/f'student_seed{seed}/best.pth';assert digest(path)==frozen['checkpoints'][str(path.relative_to(out))]
        m=make_model('student',cfg).cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval();models[seed]=m
    rows=[];index=0
    with torch.no_grad():
        for s in read(out/'confirmation_manifest.json'):
            clear,_=procedural(s['seed'],p['base']['image_size'])
            for meta,rgb,y in cases(clear,s,p['base']):
                f,c=ext(tensor(rgb,'cuda'));cc=c[0].cpu().numpy().astype(np.float64).reshape(3,-1);target=y.ravel().astype(np.float64)
                best=solve_simplex(cc@cc.T/len(target),cc@target/len(target));oracle=(cc*best[:,None]).sum(0).reshape(y.shape)
                r=record(meta,'oracle_constant_OFFLINE_ONLY',oracle,y,rgb=clear);r['pixel_mse']=float(np.mean((oracle-y)**2));rows.append(r)
                for seed,m in models.items():
                    full,w=m(f,c);mean=w.mean((2,3),keepdim=True)
                    generator=torch.Generator(device='cuda').manual_seed(865000+index)
                    permutation=torch.randperm(y.size,device='cuda',generator=generator);shuffled=w.flatten(2)[:,:,permutation].reshape_as(w)
                    variants={'full':full,'image_mean':(c*mean).sum(1,keepdim=True),'shuffled':(c*shuffled).sum(1,keepdim=True)}
                    for name,pred in variants.items():
                        value=pred[0,0].cpu().numpy();r=record(meta,f'{name}_seed{seed}',value,y,rgb=clear);r['pixel_mse']=float(np.mean((value-y)**2));rows.append(r)
                index+=1
    csv_write(dest/'per_image.csv',rows);comparison={}
    for subset in ('all','uniform','spatial'):
        rr=[r for r in rows if subset=='all' or (r['pattern']=='uniform')==(subset=='uniform')];comparison[subset]={}
        for name in ('full','image_mean','shuffled','oracle_constant_OFFLINE_ONLY'):
            a=[r for r in rr if r['method'].startswith(name+'_seed') or r['method']==name]
            comparison[subset][name]={k:float(np.mean([r[k] for r in a])) for k in ('pixel_mae','pixel_mse','global_abs_error')}
    uncertainty={};scenes=sorted({r['scene_id'] for r in rows})
    for variant in ('image_mean','shuffled'):
        delta=[]
        for s in scenes:
            a=[r['pixel_mae'] for r in rows if r['scene_id']==s and r['method'].startswith('full_seed')]
            b=[r['pixel_mae'] for r in rows if r['scene_id']==s and r['method'].startswith(variant+'_seed')];delta.append(np.mean(a)-np.mean(b))
        rng=np.random.default_rng(42);boot=np.array(delta)[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
        uncertainty[variant]={'full_minus_variant_pixel_mae':float(np.mean(delta)),'scene_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
    write(dest/'summary.json',{'metrics':summarize(rows),'stratified':comparison,'paired_uncertainty':uncertainty,
        'oracle_scope':'Per-image constant oracle minimizes pixel MSE, not MAE; actual oracle weights use truth OFFLINE ONLY. A spatial gate need not beat an unattainable constant oracle to show alignment benefit.'})
    charge(out,'spatial_controls',start);print('Spatial controls complete')

if __name__=='__main__':main()
