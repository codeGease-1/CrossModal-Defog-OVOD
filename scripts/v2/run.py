"""V2 bounded, scene-disjoint experiment. All inference inputs are RGB-derived."""
import argparse
import copy
import csv
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read,write,digest,procedural,original_scenes,cases,tensor,csv_write
from benchmarks.haze_density.metrics import spatial_metrics,ranking_metrics
from src.models.haze_density.calibrated_prior import CalibratedPrior,fit_monotone
from src.models.haze_density.dynamic_prior import FrozenCandidates,ConvexFusion,single_scores,convex_oracle


def source_hashes():
    paths=list((ROOT/'scripts/v2').glob('*.py'))+[ROOT/'src/models/haze_density/dynamic_prior.py',ROOT/'src/models/haze_density/calibrated_prior.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def prepare(out,cfg):
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json')['config']==cfg
        return
    old=Path(cfg['v15_directory']); base=read(old/'protocol.json')['config']
    write(out/'protocol.json',{'config':cfg,'v15_config':base,'source_sha256':source_hashes(),
         'v15_calibration_sha256':digest(old/'calibration.json'),'v15_protocol_sha256':digest(old/'protocol.json'),
         'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},
         'warning':'Prior tests were observed. Historical sets remain diagnostic. Real geography cannot be split safely from available metadata.'})
    scenes=read(old/'scene_manifest.json')
    fresh=[]
    for i in range(cfg['fresh_confirmation_scenes']):
        seed=base['seed']+cfg['fresh_confirmation_seed_offset']+i
        rgb,kind=procedural(seed,base['image_size'])
        import hashlib
        fresh.append({'scene_id':f'confirmation_{i:03d}','split':'confirmation','group_id':f'procedural_original_{seed}',
                      'seed':seed,'source_kind':'unused_procedural_original','land_cover_design':kind,
                      'original_pixel_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
    assert not ({s['group_id'] for s in scenes}&{s['group_id'] for s in fresh})
    assert not ({s['original_pixel_sha256'] for s in scenes}&{s['original_pixel_sha256'] for s in fresh})
    write(out/'scene_manifest.json',scenes+fresh)
    write(out/'budget.json',{'training_seconds':0.,'calibration_seconds':0.,'limit_seconds':cfg['training_budget_seconds']})


def calibrate(out,cfg,base,device):
    if (out/'calibration.json').exists(): return
    cal=read(Path(cfg['v15_directory'])/'calibration.json')
    cache=out/'calibration_cache';cache.mkdir(exist_ok=True)
    ix=np.linspace(0,base['image_size']-1,16,dtype=int)
    for s in original_scenes(base,'train'):
        p=cache/(s['scene_id']+'.npz')
        if p.exists():continue
        started=time.monotonic();rgb,_=procedural(s['seed'],base['image_size']);xx=[];yy=[]
        for _,im,truth in cases(rgb,s,base):
            _,scores=single_scores(tensor(im,device),cal)
            xx.append(scores.cpu().numpy()[0][:,ix[:,None],ix].reshape(3,-1))
            yy.append(truth[ix[:,None],ix].ravel())
        np.savez_compressed(p,x=np.concatenate(xx,axis=1),y=np.concatenate(yy))
        budget=read(out/'budget.json');budget['calibration_seconds']+=time.monotonic()-started;write(out/'budget.json',budget)
        assert budget['calibration_seconds']<cfg['training_budget_seconds']
    arrays=[np.load(cache/(s['scene_id']+'.npz')) for s in original_scenes(base,'train')]
    x=np.concatenate([a['x'] for a in arrays],axis=1);y=np.concatenate([a['y'] for a in arrays])
    cal['candidate_tables']=[fit_monotone(x[i],y,base['isotonic_bins']) for i in range(3)]
    cal['v2_definition']='each candidate predicts independent H=1-t before convex fusion; no post-fusion mapping'
    write(out/'calibration.json',cal)


def cache_split(out,base,split,extractor,device):
    xx=[];yy=[];meta=[]
    for s in original_scenes(base,split):
        rgb,kind=procedural(s['seed'],base['image_size'])
        for m,im,truth in cases(rgb,s,base):
            features,_=extractor(tensor(im,device))
            xx.append(features.cpu());yy.append(torch.from_numpy(truth.copy())[None,None]);meta.append(dict(m,kind=kind))
        print('CACHE',split,s['scene_id'],flush=True)
    return torch.cat(xx),torch.cat(yy),meta


def diagnostics(out,cfg,base,device):
    if (out/'oracle_development.json').exists():return
    started=time.perf_counter();extractor=FrozenCandidates(read(out/'calibration.json'))
    sums=np.zeros(6); count=critical=biased=0; tables={};winners=np.zeros(3);rows=[]
    for s in original_scenes(base,'development'):
        rgb,kind=procedural(s['seed'],base['image_size'])
        for m,im,truth in cases(rgb,s,base):
            f,p=extractor(tensor(im,device));y=torch.from_numpy(truth)[None,None].to(device)
            fixed=(p*p.new_tensor([.5,.3,.2])[None,:,None,None]).sum(1,keepdim=True)
            oracle=convex_oracle(p,y);errors=(p-y).abs()
            c=(fixed-y).abs()>cfg['oracle_critical_error']
            b=(p.amin(1,keepdim=True)>y+cfg['oracle_bias_tolerance'])|(p.amax(1,keepdim=True)<y-cfg['oracle_bias_tolerance'])
            critical+=int(c.sum());biased+=int((b&c).sum());count+=y.numel()
            sums+=np.array([float((fixed-y).square().sum()),float((oracle-y).square().sum()),float((fixed-y).abs().sum()),
                            float((oracle-y).abs().sum()),float((p.amin(1,keepdim=True)>y).sum()),float((p.amax(1,keepdim=True)<y).sum())])
            win=errors.argmin(1).flatten().cpu().numpy();winners+=np.bincount(win,minlength=3)
            key=kind+'_'+m['pattern'];tables.setdefault(key,[]).append(errors.mean((0,2,3)).cpu().numpy().tolist())
            rows.append({'case_id':m['case_id'],'kind':kind,'fixed_mae':float((fixed-y).abs().mean()),'oracle_mae':float((oracle-y).abs().mean()),
                         **{f'prior_{i}_mae':float(errors[:,i].mean()) for i in range(3)}})
    ratio=biased/max(critical,1)
    write(out/'oracle_development.json',{'meaning':'Exact pixelwise MAE/MSE minimum over arbitrary convex weights on final H candidates. Not an achievable RGB network accuracy.',
          'fixed_mse':sums[0]/count,'oracle_mse_lower_bound':sums[1]/count,'fixed_mae':sums[2]/count,'oracle_mae_lower_bound':sums[3]/count,
          'critical_pixels':critical,'critical_all_biased_fraction':ratio,'bias_tolerance':cfg['oracle_bias_tolerance'],
          'winner_fractions':(winners/count).tolist(),'per_kind_pattern_prior_mae':{k:np.mean(v,axis=0).tolist() for k,v in tables.items()},
          'pilot_only':ratio>cfg['oracle_majority_threshold'],'elapsed_seconds':time.perf_counter()-started})
    csv_write(out/'oracle_per_case.csv',rows)
    print('ORACLE',json.dumps(read(out/'oracle_development.json')),flush=True)


def loss_fn(pred,y,cfg):
    pixel=(pred-y).square().mean()
    global_loss=(pred.mean((2,3))-y.mean((2,3))).square().mean()
    clear=y.mean((1,2,3))<1e-7
    extra=pred[clear].square().mean() if clear.any() else pred.sum()*0
    return pixel+cfg['global_loss_weight']*global_loss+cfg['clear_loss_extra_weight']*extra


def dev_loss(model,x,y,cfg,device):
    model.eval();values=[]
    with torch.no_grad():
        for i in range(0,len(x),cfg['batch_size']):
            p,_=model(x[i:i+cfg['batch_size']].to(device));target=y[i:i+cfg['batch_size']].to(device)
            values.append(float(loss_fn(p,target,cfg)))
    return float(np.mean(values))


def train(out,cfg,base,device):
    if (out/'training_complete.json').exists():return
    extractor=FrozenCandidates(read(out/'calibration.json'))
    with torch.no_grad():
        x,y,_=cache_split(out,base,'train',extractor,device)
        vx,vy,_=cache_split(out,base,'development',extractor,device)
    # Feature tensors are kept in RAM; no multi-GB disk cache or dataset extraction.
    pilot=read(out/'oracle_development.json')['pilot_only']; summaries=[]
    for mode in ('global','dynamic'):
        for seed in cfg['seeds']:
            directory=out/f'{mode}_seed{seed}';directory.mkdir(exist_ok=True)
            if (directory/'complete.json').exists():summaries.append(read(directory/'complete.json'));continue
            torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed)
            model=ConvexFusion(mode,cfg['hidden']).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr_'+mode])
            epoch0=0;step=0;best=float('inf');hist=[]
            if (directory/'latest.pth').exists():
                state=torch.load(directory/'latest.pth',map_location=device,weights_only=False)
                assert state['calibration_sha256']==digest(out/'calibration.json')
                model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
                epoch0=state['epoch'];step=state['step'];best=state['best'];hist=state['history']
                rng.set_state(state['shuffle_rng'].cpu());torch.set_rng_state(state['torch_rng'].cpu())
                if state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
            for epoch in range(epoch0,cfg['epochs']):
                budget=read(out/'budget.json')
                if budget['training_seconds']+budget['calibration_seconds']>=cfg['training_budget_seconds']:raise RuntimeError('Training budget exhausted; resume files saved')
                started=time.monotonic();model.train();order=torch.randperm(len(x),generator=rng); losses=[]
                for start in range(0,len(order),cfg['batch_size']):
                    if pilot and step>=cfg['diagnostic_only_pilot_steps']:break
                    indices=order[start:start+cfg['batch_size']]
                    features=x[indices].to(device);truth=y[indices].to(device)
                    optimizer.zero_grad(set_to_none=True);prediction,weights=model(features)
                    loss=loss_fn(prediction,truth,cfg);loss.backward();optimizer.step();losses.append(float(loss.detach()));step+=1
                validation=dev_loss(model,vx,vy,cfg,device)
                budget['training_seconds']+=time.monotonic()-started;write(out/'budget.json',budget)
                hist.append({'epoch':epoch+1,'step':step,'training_loss':float(np.mean(losses)) if losses else None,'development_loss':validation})
                improved=validation<best;best=min(best,validation)
                state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'step':step,'best':best,'history':hist,
                       'shuffle_rng':rng.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None,
                       'calibration_sha256':digest(out/'calibration.json'),'mode':mode,'seed':seed}
                torch.save(state,directory/'latest.pth')
                if improved:torch.save(state,directory/'best.pth')
                write(directory/'history.json',hist)
                print('TRAIN',mode,seed,epoch+1,validation,'budget',round(budget['training_seconds'],1),flush=True)
                if pilot and step>=cfg['diagnostic_only_pilot_steps']:break
            final={'mode':mode,'seed':seed,'best_development_loss':best,'steps':step,'pilot_only':pilot,
                   'trainable_parameters':sum(p.numel() for p in model.parameters())}
            write(directory/'complete.json',final);summaries.append(final)
    write(out/'training_complete.json',summaries)


def freeze(out,cfg):
    if (out/'frozen.json').exists():return
    summary=read(out/'training_complete.json')
    selected={mode:min([s for s in summary if s['mode']==mode],key=lambda v:v['best_development_loss'])['seed'] for mode in ('global','dynamic')}
    checkpoints={str(p.relative_to(out)):digest(p) for p in sorted(out.glob('*_seed*/best.pth'))}
    write(out/'frozen.json',{'selected_seeds':selected,'checkpoint_sha256':checkpoints,'calibration_sha256':digest(out/'calibration.json'),
          'selection':'best development loss; all three seeds will be reported, never selected on confirmation/external results',
          'post_fusion_transform':'identity for ALL three v2 methods','global_head_added':False})


def external_manifest(out,cfg,archive):
    path=out/'external_manifest.json'
    if path.exists():return read(path)
    old=Path(cfg['v15_directory']);historical=read(old/'external_manifest.json');excluded={s['pixel_sha256'] for s in historical}
    used={s['archive_member'] for s in historical}
    used.update(s['member'] for s in read(old/'audit/archive_audit.json')['samples'])
    records=[]
    import hashlib
    with zipfile.ZipFile(archive) as z:
        for subset in ('G','S','L'):
            names=sorted(n for n in z.namelist() if f'RSHaze_{subset}/test/cleanpng/' in n and n.endswith('.png') and n not in used)
            random.Random(cfg['external_selection_seed']).shuffle(names);count=0
            for n in names:
                data=z.read(n);im=Image.open(io.BytesIO(data)).convert('RGB');pixelhash=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
                if pixelhash in excluded:continue
                excluded.add(pixelhash);sid=f'new_{subset}_{Path(n).stem}'
                records.append({'scene_id':sid,'split':'external','group_id':'RSHaze_all_geography_unverified','seed':cfg['external_selection_seed']+50000+len(records),
                       'source_kind':'new_member_same_unverified_geographic_supergroup','archive_member':n,'pixel_sha256':pixelhash,'file_sha256':hashlib.sha256(data).hexdigest(),'subset':subset})
                count+=1
                if count>=cfg['new_external_per_subset']:break
    write(path,records);return records


def model_set(out,cfg,device):
    models={'fixed_aligned':ConvexFusion('fixed').to(device)}
    frozen=read(out/'frozen.json')
    for mode in ('global','dynamic'):
        for seed in cfg['seeds']:
            p=out/f'{mode}_seed{seed}/best.pth';assert digest(p)==frozen['checkpoint_sha256'][str(p.relative_to(out))]
            state=torch.load(p,map_location=device,weights_only=False);model=ConvexFusion(mode,cfg['hidden']).to(device);model.load_state_dict(state['model']);model.eval()
            models[f'{mode}_seed{seed}']=model
    return models


def summarize(rows):
    result={}
    for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==method];rank_groups={}
        for r in rr:
            if r['pattern']=='uniform':rank_groups.setdefault((r['scene_id'],r['airlight_index'],r['brightness_index']),[]).append(r)
        rankings=[ranking_metrics([r['truth_mean'] for r in v],[r['prediction_mean'] for r in v])['pairwise_accuracy'] for v in rank_groups.values()]
        uniform=[r for r in rr if r['pattern']=='uniform'];good=total=0
        for i,a in enumerate(uniform):
            for b in uniform[i+1:]:
                d=a['truth_mean']-b['truth_mean']
                if a['scene_id']==b['scene_id'] or abs(d)<1e-6:continue
                total+=1;good+=int((a['prediction_mean']-b['prediction_mean'])*np.sign(d)>1e-6)
        spatial=[r['region_spearman'] for r in rr if r['region_spearman'] is not None]
        clear=[r['prediction_mean'] for r in rr if r['pattern']=='uniform' and r['uniform_t']==1]
        result[method]={'n_cases':len(rr),'global_mae':float(np.mean([r['global_abs_error'] for r in rr])),
                'pixel_mae':float(np.mean([r['pixel_mae'] for r in rr])),'same_scene_rank':float(np.mean(rankings)) if rankings else None,
                'cross_condition_cross_scene_rank':good/total if total else None,
                'region_spearman':float(np.mean(spatial)) if spatial else None,'clear_response':float(np.mean(clear)) if clear else None}
    return result


def record(meta,method,pred,truth,weights=None,rgb=None):
    spatial=spatial_metrics(1-truth,pred)
    row={k:meta.get(k) for k in ('scene_id','group_id','pattern','uniform_t','airlight_index','brightness_index','case_id')}
    row.update(method=method,truth_mean=float(truth.mean()),prediction_mean=float(pred.mean()),
               global_abs_error=float(abs(pred.mean()-truth.mean())),pixel_mae=float(np.abs(pred-truth).mean()),region_spearman=spatial['region_spearman'])
    if weights is not None:
        w=weights
        row.update({f'weight_{i}':float(w[i].mean()) for i in range(3)})
        row.update({f'argmax_fraction_{i}':float((w.argmax(0)==i).mean()) for i in range(3)})
        row['entropy']=float(-(w*np.log(np.maximum(w,1e-8))).sum(0).mean())
    else:
        row.update({f'weight_{i}':None for i in range(3)});row.update({f'argmax_fraction_{i}':None for i in range(3)});row['entropy']=None
    if rgb is not None:
        gray=rgb@np.array([.299,.587,.114]);dy,dx=np.gradient(gray)
        for label,mask in [('bright',gray>.8),('low_texture',np.hypot(dx,dy)<.01)]:
            row[label+'_pixels']=int(mask.sum());row[label+'_mae']=float(np.abs(pred-truth)[mask].mean()) if mask.any() else None
    return row


def plot(path,rgb,truth,predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n=2+len(predictions);fig,axes=plt.subplots(1,n,figsize=(3*n,3.2))
    axes[0].imshow(rgb);axes[0].set_title('RGB')
    for ax,(name,value) in zip(axes[1:],[('Known H',truth)]+list(predictions.items())):
        im=ax.imshow(value,vmin=0,vmax=1,cmap='viridis');ax.set_title(f'{name}\nmean={value.mean():.3f}',fontsize=9)
    for ax in axes:ax.axis('off')
    fig.colorbar(im,ax=axes.tolist(),fraction=.012,pad=.01);fig.savefig(path,dpi=100,bbox_inches='tight');plt.close(fig)


def evaluate(out,cfg,base,device,split,archive):
    dest=out/split;dest.mkdir(exist_ok=True)
    if (dest/'summary.json').exists():return
    frozen=read(out/'frozen.json');assert digest(out/'calibration.json')==frozen['calibration_sha256']
    extractor=FrozenCandidates(read(out/'calibration.json'));models=model_set(out,cfg,device)
    oldcal=read(Path(cfg['v15_directory'])/'calibration.json');oldmodels={n:CalibratedPrior(oldcal,n) for n in ('normalized','fixed')}
    if split=='confirmation':scenes=[s for s in read(out/'scene_manifest.json') if s['split']=='confirmation']
    elif split=='external':scenes=external_manifest(out,cfg,archive)
    else:scenes=original_scenes(base,split)
    rows=[];oracle_rows=[];figures=dest/'figures';figures.mkdir(exist_ok=True)
    z=zipfile.ZipFile(archive) if split=='external' else None
    try:
        for si,s in enumerate(scenes):
            if z is not None:clear=np.asarray(Image.open(io.BytesIO(z.read(s['archive_member']))).convert('RGB').resize((base['image_size'],base['image_size']),Image.Resampling.BILINEAR),np.float32)/255
            else:clear,_=procedural(s['seed'],base['image_size'])
            for meta,rgb,truth in cases(clear,s,base):
                image=tensor(rgb,device);features,candidates=extractor(image);predictions={}
                for name,model in models.items():
                    pred,weights=model(features,candidates);pred=pred.cpu().numpy()[0,0];w=weights.cpu().numpy()[0]
                    rows.append(record(meta,name,pred,truth,w,clear));predictions[name]=pred
                for name,model in oldmodels.items():
                    pred=model(image).cpu().numpy()[0,0];name='v15_'+name;rows.append(record(meta,name,pred,truth,rgb=clear));predictions[name]=pred
                oracle=convex_oracle(candidates,torch.from_numpy(truth)[None,None].to(device)).cpu().numpy()[0,0]
                oracle_rows.append(record(meta,'oracle_OFFLINE_ONLY',oracle,truth,rgb=clear))
                if si<2 and meta['airlight_index']==0 and meta['brightness_index']==1 and (meta['pattern']=='gradient' or meta['uniform_t'] in (1.,.4)):
                    picks={'v15 fixed':predictions['v15_fixed'],'fixed aligned':predictions['fixed_aligned'],
                           'global':predictions[f'global_seed{frozen["selected_seeds"]["global"]}'],
                           'dynamic':predictions[f'dynamic_seed{frozen["selected_seeds"]["dynamic"]}'],'oracle':oracle}
                    plot(figures/(meta['case_id']+'.png'),rgb,truth,picks)
            print('EVAL',split,s['scene_id'],flush=True)
    finally:
        if z is not None:z.close()
    csv_write(dest/'per_image.csv',rows);csv_write(dest/'oracle_per_image.csv',oracle_rows)
    metrics=summarize(rows)
    failures=sorted([r for r in rows if r['method']==f'dynamic_seed{frozen["selected_seeds"]["dynamic"]}'],key=lambda r:r['pixel_mae'],reverse=True)[:8]
    write(dest/'failure_cases.json',failures)
    write(dest/'summary.json',{'metrics':metrics,'oracle_pixel_mae_lower_bound':float(np.mean([r['pixel_mae'] for r in oracle_rows])),
          'n_originals':len(scenes),'n_geographic_or_original_groups':len({s['group_id'] for s in scenes}),
          'role':'fresh member confirmation; geography remains unverified' if split=='external' else 'fresh controlled confirmation' if split=='confirmation' else 'historical diagnostic only'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['prepare','calibrate','diagnose','train','freeze','development','test','confirmation','external'])
    p.add_argument('--config',default='configs/v2_dynamic.json');p.add_argument('--output',default='experiments/v2_20261101');p.add_argument('--device',default='cuda');p.add_argument('--archive',default='/mnt/RSHaze+.zip')
    a=p.parse_args();cfg=read(a.config);out=Path(a.output);out.mkdir(exist_ok=True,parents=True);torch.set_num_threads(1)
    if a.stage=='prepare':prepare(out,cfg);return
    protocol=read(out/'protocol.json');assert protocol['config']==cfg
    for rel,sha in protocol['source_sha256'].items():assert digest(ROOT/rel)==sha, f'Source changed: {rel}; use new experiment directory'
    assert digest(Path(cfg['v15_directory'])/'calibration.json')==protocol['v15_calibration_sha256']
    base=protocol['v15_config']
    if a.stage=='train':train(out,cfg,base,a.device)
    elif a.stage=='freeze':freeze(out,cfg)
    else:
        with torch.no_grad():
            if a.stage=='calibrate':calibrate(out,cfg,base,a.device)
            elif a.stage=='diagnose':diagnostics(out,cfg,base,a.device)
            else:evaluate(out,cfg,base,a.device,a.stage,a.archive)


if __name__=='__main__':main()
