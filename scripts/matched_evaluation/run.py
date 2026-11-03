"""Development diagnostics and complete inference cost; never opens a fresh confirmation."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from scripts.v15.run import read,write,digest,csv_write,procedural,cases,tensor
from scripts.learned_correction.run import diagnostic
from scripts.matched_fusion.run import settings,make
from src.models.haze_density.dynamic_prior import FrozenCandidates
from src.models.haze_density.learned_correction import observed_context,RGBReference,RestrictedCorrector
from src.models.haze_density.calibrated_prior import raw_cues,fixed_cues,apply_monotone
from src.models.haze_density.dynamic_prior import single_scores
from benchmarks.haze_density.metrics import spatial_metrics,ranking_metrics,correlation


def load(out,cfg,seed):
    nets={}
    for mode in cfg['new_modes']+cfg['continued_controls']:
        assert (out/f'{mode}_seed{seed}/complete.json').exists()
        net=make(mode,cfg,seed);net.load_state_dict(torch.load(out/f'{mode}_seed{seed}/best.pth',map_location='cpu',weights_only=False)['model'])
        nets[mode]=net.eval()
    old=Path(cfg['old_output'])
    for mode in cfg['continued_controls']:
        net=make(mode,cfg,seed)
        net.load_state_dict(torch.load(old/f'{mode}_seed{seed}/best.pth',map_location='cpu',weights_only=False)['model'])
        nets[mode+'_18']=net.eval()
    return nets


def candidate_diagnostic(c,y):
    a=diagnostic(c,y);lo,hi=c.min(0),c.max(0)
    distance=np.maximum(lo-y,0)+np.maximum(y-hi,0);outside=distance>0
    a['outside_amplitude_conditional_mean']=float(distance[outside].mean()) if outside.any() else 0.
    a['outside_amplitude_mean']=float(distance.mean())
    return a


def row_for(name,seed,pred,y,meta,proxies):
    err=pred-y
    row={'method':name,'seed':seed,'case_id':meta['case_id'],'scene_id':meta['scene_id'],'group_id':meta['group_id'],
        'pattern':meta['pattern'],'uniform_t':meta['uniform_t'],'airlight_index':meta['airlight_index'],
        'brightness_index':meta['brightness_index'],'truth_mean':float(y.mean()),'prediction_mean':float(pred.mean()),
        'pixel_mae':float(np.abs(err).mean()),'pixel_mse':float((err**2).mean()),'signed_bias':float(err.mean()),
        'global_mae':float(abs(err.mean())),'region_spearman':spatial_metrics(1-y,pred)['region_spearman'],
        'clear_response':float(pred.mean()) if meta['uniform_t']==1. else None}
    for key,mask in proxies.items():row[key+'_mae']=float(np.abs(err)[mask].mean()) if mask.any() else None
    return row


def summarize(rows):
    result={}
    keys=['pixel_mae','pixel_mse','signed_bias','global_mae','region_spearman','clear_response','dark_mae','bright_mae','low_texture_mae']
    for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==method];values={}
        for key in keys:
            a=[r[key] for r in rr if r[key] is not None];values[key]=float(np.mean(a)) if a else None
        groups={}
        for r in rr:
            if r['pattern']=='uniform':groups.setdefault((r['seed'],r['scene_id'],r['airlight_index'],r['brightness_index']),[]).append(r)
        ranks=[ranking_metrics([q['truth_mean'] for q in a],[q['prediction_mean'] for q in a])['pairwise_accuracy'] for a in groups.values()]
        values['same_scene_rank_accuracy']=float(np.mean(ranks)) if ranks else None
        values['cross_scene_mean_spearman']=correlation([r['truth_mean'] for r in rr],[r['prediction_mean'] for r in rr],rank=True)
        values['cases']=len(rr);result[method]=values
    return result


def bootstrap(rows,left,right,draws=3000):
    scenes=sorted({r['scene_id'] for r in rows});rng=np.random.default_rng(20261102)
    metrics=['pixel_mae','region_spearman','clear_response'];arrays={}
    for method in [left,right]:
        arrays[method]=np.array([[np.mean([r[k] for r in rows if r['method']==method and r['scene_id']==scene and r[k] is not None]) for k in metrics] for scene in scenes])
    sample=rng.integers(0,len(scenes),(draws,len(scenes)))
    l=arrays[left][sample].mean(1);r=arrays[right][sample].mean(1)
    dif=np.stack([l[:,0]/r[:,0]-1,r[:,1]-l[:,1],l[:,2]-r[:,2]],1)
    actual_l=arrays[left].mean(0);actual_r=arrays[right].mean(0)
    point=[actual_l[0]/actual_r[0]-1,actual_r[1]-actual_l[1],actual_l[2]-actual_r[2]]
    return {'left':left,'right':right,'scene_count':len(scenes),'bootstrap_draws':draws,
        'quantities':['relative_MAE_increase','rho_drop','clear_increase'],'point':point,
        'CI95':np.quantile(dif,[.025,.975],axis=0).T.tolist(),
        'margins':[.05,.01,.005],'all_upper_bounds_within_margins':bool(np.all(np.quantile(dif,.975,axis=0)<=np.array([.05,.01,.005]))),
        'scope':'Old development only, fixed three training seeds; scene bootstrap does not establish optimizer-seed robustness or fresh confirmation'}


def evaluate(out,cfg,p):
    assert (out/'training_complete.json').exists()
    start=time.monotonic();dst=out/'development';dst.mkdir(exist_ok=True)
    old=Path(cfg['old_output']);fits=read(old/'fusion_weights.json')
    ext=FrozenCandidates(read(cfg['calibration'])).cuda().eval()
    networks={seed:load(out,cfg,seed) for seed in cfg['seeds']}
    for condition in ['nominal','unseen_A','display_rgb']:
        d=dst/condition;d.mkdir(exist_ok=True)
        if (d/'complete.json').exists():continue
        base=copy.deepcopy(p['base'])
        if condition=='unseen_A':base['airlights']=read(old/'protocol.json')['config']['unseen_airlights']
        if condition=='display_rgb':base['domain']='display_rgb'
        rows=[];diags=[];weights=[];manifest=[]
        with torch.no_grad():
            for scene in [s for s in p['scenes'] if s['split']=='development']:
                clear,_=procedural(scene['seed'],base['image_size'])
                gray=clear@np.array([.299,.587,.114]);dy,dx=np.gradient(gray)
                proxies={'dark':gray<.2,'bright':gray>.8,'low_texture':np.hypot(dx,dy)<.01}
                items=list(cases(clear,scene,base))
                for first in range(0,len(items),8):
                    chunk=items[first:first+8]
                    image=torch.cat([tensor(rgb,'cuda') for _,rgb,_ in chunk]);f,raw=ext(image)
                    f=torch.cat([f,observed_context(image)],1)
                    for seed,nets in networks.items():
                        with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp']):
                            inputs,q=nets['corrected_gate'].prepare(f)
                            outputs={'original_fixed':(raw*raw.new_tensor(fits['original']['weights'])[None,:,None,None]).sum(1,keepdim=True),
                                'corrected_fixed':(q*q.new_tensor(fits[f'corrector16_seed{seed}']['weights'])[None,:,None,None]).sum(1,keepdim=True),
                                'corrected_D':q[:,:1]}
                            for mode,net in nets.items():
                                if mode.endswith('_gate'):
                                    c=q if mode=='corrected_gate' else raw
                                    pred,w=net.from_prepared(inputs,c);outputs[mode]=pred
                                    outputs[mode+'_mean']=(w.mean((2,3),keepdim=True)*c).sum(1,keepdim=True)
                                    generator=torch.Generator(device='cuda').manual_seed(90210+seed+first)
                                    perm=torch.randperm(w.shape[-1]*w.shape[-2],generator=generator,device='cuda')
                                    shuffled=w.flatten(2)[:,:,perm].reshape_as(w)
                                    outputs[mode+'_shuffle']=(shuffled*c).sum(1,keepdim=True)
                                    for j,(meta,_,_) in enumerate(chunk):
                                        weights.append({'method':mode,'seed':seed,'case_id':meta['case_id'],
                                            'mean_weights':w[j].mean((1,2)).cpu().tolist(),
                                            'spatial_std':w[j].std((1,2)).cpu().tolist(),
                                            'mean_entropy':float(-(w[j]*w[j].clamp_min(1e-12).log()).sum(0).mean())})
                                else:outputs[mode]=net(f)[0].float()
                        arrays={key:value[:,0].cpu().numpy() for key,value in outputs.items()}
                        raw_arrays=raw.cpu().numpy();q_arrays=q.cpu().numpy()
                        for j,(meta,rgb,y) in enumerate(chunk):
                            for name,pred in arrays.items():rows.append(row_for(name,seed,pred[j],y,meta,proxies))
                            for name,c in [('original',raw_arrays[j]),('corrected',q_arrays[j])]:
                                diags.append({'method':name,'seed':seed,'case_id':meta['case_id'],**candidate_diagnostic(c,y)})
                    for meta,rgb,y in chunk:
                        manifest.append({**meta,'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'target_sha256':hashlib.sha256(y.tobytes()).hexdigest()})
                print('EVAL',condition,scene['scene_id'],flush=True)
        csv_write(d/'per_case.csv',rows);write(d/'candidate_diagnostics.json',diags);write(d/'weight_diagnostics.json',weights)
        write(d/'data_manifest.json',manifest);write(d/'summary.json',summarize(rows))
        write(d/'by_seed.json',{str(seed):summarize([r for r in rows if r['seed']==seed]) for seed in cfg['seeds']})
        comparisons=[bootstrap(rows,'corrected_gate',ref) for ref in ['rgb16','rgb32','direct_same16','corrected_fixed','raw_gate']]
        write(d/'paired_scene_bootstrap.json',comparisons)
        write(d/'complete.json',{'condition':condition,'only_old_development':True,'cases_per_model_per_seed':len(manifest),
             'proxy_definition':'Clean source brightness/gradient masks, evaluation only; not land-cover semantics; masks before brightness augmentation'})
        print('SUMMARY',condition,json.dumps(read(d/'summary.json')),flush=True)
    write(out/'evaluation_complete.json',{'seconds':time.monotonic()-start,'new_confirmation_scenes':0})


def timed(fn,repeats=40,warmup=10):
    for _ in range(warmup):fn()
    torch.cuda.synchronize();times=[];torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated()
    for _ in range(repeats):
        torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    return {'p50_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
            'resident_allocated_bytes':resident,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'incremental_peak_bytes':torch.cuda.max_memory_allocated()-resident,'repeats':repeats}


def costs(out,cfg):
    assert (out/'training_complete.json').exists()
    other=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
    assert len(other)<=1,'Do not benchmark concurrently with another GPU process'
    start=time.monotonic();results=[];cal=read(cfg['calibration']);ext=FrozenCandidates(cal).cuda().eval()
    fits=read(Path(cfg['old_output'])/'fusion_weights.json');seed=cfg['seeds'][0]
    methods=['original_fixed','corrected_fixed','raw_gate','corrected_gate','rgb16','direct_same16','rgb32']
    with torch.no_grad():
        for mode in methods:
            if mode=='original_fixed':net=None
            elif mode=='corrected_fixed':
                net=RestrictedCorrector(16).cuda();net.load_state_dict(torch.load(Path(cfg['old_output'])/f'corrector16_seed{seed}/best.pth',map_location='cpu',weights_only=False)['model']);net.eval()
            else:
                net=make(mode,cfg,seed);net.load_state_dict(torch.load(out/f'{mode}_seed{seed}/best.pth',map_location='cpu',weights_only=False)['model']);net.eval()
            count=sum(v.numel() for v in net.parameters()) if net is not None else 0
            for size in [256,512]:
                rgb=torch.rand(1,3,size,size,device='cuda')
                def forward():
                    if mode in ['rgb16','rgb32']:
                        with torch.autocast('cuda',dtype=torch.float16,enabled=True):return net(rgb)[0].float()
                    with torch.autocast('cuda',enabled=False):
                        f,c=ext(rgb.float())
                        if mode!='original_fixed':f=torch.cat([f,observed_context(rgb.float())],1)
                    if mode=='original_fixed':return (c*c.new_tensor(fits['original']['weights'])[None,:,None,None]).sum(1,keepdim=True)
                    with torch.autocast('cuda',dtype=torch.float16,enabled=True):pred=net(f)[0]
                    if mode=='corrected_fixed':pred=(pred*pred.new_tensor(fits[f'corrector16_seed{seed}']['weights'])[None,:,None,None]).sum(1,keepdim=True)
                    return pred.float()
                cost=timed(forward)
                buf=io.BytesIO();Image.fromarray((rgb[0].permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(buf,format='PNG');png=buf.getvalue()
                def decoded():
                    a=np.asarray(Image.open(io.BytesIO(png)).convert('RGB'),np.float32)/255
                    cpu=torch.from_numpy(a.copy()).permute(2,0,1)[None].cuda()
                    if mode in ['rgb16','rgb32']:
                        with torch.autocast('cuda',dtype=torch.float16):return net(cpu)[0]
                    with torch.autocast('cuda',enabled=False):
                        f,c=ext(cpu);f=torch.cat([f,observed_context(cpu)],1) if mode!='original_fixed' else f
                    if mode=='original_fixed':return (c*c.new_tensor(fits['original']['weights'])[None,:,None,None]).sum(1,keepdim=True)
                    with torch.autocast('cuda',dtype=torch.float16):pred=net(f)[0]
                    if mode=='corrected_fixed':pred=(pred*pred.new_tensor(fits[f'corrector16_seed{seed}']['weights'])[None,:,None,None]).sum(1,keepdim=True)
                    return pred
                full=timed(decoded,20,5)
                cpu_output=timed(lambda:decoded().cpu(),20,5)
                results.append({'method':mode,'size':size,'network_parameters_total':count,'GPU_resident_RGB_to_H':cost,
                                'PNG_bytes_decode_H2D_to_H':full,'PNG_bytes_to_CPU_H':cpu_output,
                                'prior_statistics_and_mapping_scalars':sum(len(t['x'])+len(t['y']) for t in cal['candidate_tables'])+6 if mode not in ['rgb16','rgb32'] else 0,
                                'fixed_fusion_scalars':3 if mode in ['original_fixed','corrected_fixed'] else 0,
                                'precision':'FP32 priors/calibration/context and AMP network, output FP32'})
                print('COST',mode,size,json.dumps(cost),flush=True)
                del rgb
            del net;torch.cuda.empty_cache()
        rgb=torch.rand(1,3,256,256,device='cuda');raw,scores=single_scores(rgb,cal)
        components={
          'raw_three_cues':timed(lambda:raw_cues(rgb,cal['prior']['window_size'])),
          'raw_to_power_guided_scores':timed(lambda:single_scores(rgb,cal)),
          'supervised_monotone_mapping_only':timed(lambda:torch.cat([apply_monotone(scores[:,i:i+1],cal['candidate_tables'][i]) for i in range(3)],1)),
          'complete_frozen_candidate_extraction':timed(lambda:ext(rgb)),
          'observed_brightness_texture_context':timed(lambda:observed_context(rgb))}
    write(out/'complete_inference_cost.json',{'methods':results,'components_256':components,'seconds':time.monotonic()-start,
          'batch_size':1,'seed':seed,'CPU_threads':1,'GPU':torch.cuda.get_device_name(),
          'scope':'Measured after training, no concurrent GPU job; includes all actual validation and prior work. PNG path excludes disk/network fetch and includes decode/host transfer.'})


def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['evaluate','cost'],required=True)
    a.add_argument('--output',type=Path,default=ROOT/'experiments/matched_fusion_20261102');args=a.parse_args()
    settings();p=read(args.output/'protocol.json');cfg=p['config']
    for rel,sha in p['source_sha256'].items():assert digest(ROOT/rel)==sha,rel
    for rel,sha in p['protected_previous_checkpoints'].items():assert digest(rel)==sha,rel
    manifest={str(q.relative_to(ROOT)):digest(q) for q in [Path(__file__),ROOT/'scripts/matched_fusion/run.py']}
    receipt=args.output/f'{args.stage}_source.json'
    if receipt.exists():assert read(receipt)==manifest,'Evaluation code changed; use separate output'
    else:write(receipt,manifest)
    if args.stage=='evaluate':evaluate(args.output,cfg,p)
    else:costs(args.output,cfg)


if __name__=='__main__':main()
