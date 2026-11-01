"""Frozen-model secondary spectral test, cost, weight and paired uncertainty."""
import io
from pathlib import Path
import sys
import time
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read,write,digest,tensor,csv_write,procedural,cases
from scripts.v2.run import model_set,record,summarize,plot
from src.models.haze_density.dynamic_prior import FrozenCandidates,convex_oracle,ConvexFusion
from src.models.haze_density.calibrated_prior import CalibratedPrior


def main():
    out=Path('experiments/v2_20261101');protocol=read(out/'protocol.json');cfg=protocol['config'];base=protocol['v15_config'];torch.set_num_threads(1)
    frozen=read(out/'frozen.json');models=model_set(out,cfg,'cuda');extractor=FrozenCandidates(read(out/'calibration.json'))
    rank=out/'ranking_ablation/best.pth'
    if rank.exists():
        state=torch.load(rank,map_location='cuda',weights_only=False);model=ConvexFusion('dynamic',cfg['hidden']).cuda();model.load_state_dict(state['model']);model.eval();models['dynamic_rank_seed42']=model
        unranked=torch.load(out/'dynamic_seed42/best.pth',map_location='cpu',weights_only=False)
        maxdiff=max(float((value.cpu()-unranked['model'][key]).abs().max()) for key,value in state['model'].items())
        write(out/'ranking_ablation/parameter_comparison.json',{'max_absolute_parameter_difference_from_unranked_seed42':maxdiff,
              'meaning':'Identical parameters imply no net training effect for this seed/protocol, not that ranking is universally useless.'})
    oldcal=read(Path(cfg['v15_directory'])/'calibration.json');oldmodels={n:CalibratedPrior(oldcal,n) for n in ('normalized','fixed')}
    dest=out/'original_spectral';dest.mkdir(exist_ok=True);rows=[];oracle_rows=[];checks=[]
    with torch.no_grad(),zipfile.ZipFile('/mnt/RSHaze+.zip') as z:
        for si,s in enumerate([s for s in read(out/'external_manifest.json') if s['subset']=='S']):
            n=s['archive_member'];names=[n,n.replace('/cleanpng/','/synhazypng/'),n.replace('/cleanpng/','/transpng/'),n.replace('/cleanpng/','/airpng/')]
            imgs=[Image.open(io.BytesIO(z.read(p))) for p in names];j,h,t,a=[np.asarray(im,np.float64)/255 for im in imgs]
            if not(j.shape==h.shape==t.shape==a.shape):checks.append({'scene':s['scene_id'],'status':'REJECT shapes'});continue
            residual=float(np.abs(j*t+a*(1-t)-h).mean());checks.append({'scene':s['scene_id'],'residual_mae':residual,'status':'PASS' if residual<.005 else 'REJECT residual'})
            if residual>=.005:continue
            size=base['image_size'];rgb=np.asarray(imgs[1].convert('RGB').resize((size,size),Image.Resampling.BILINEAR),np.float32)/255
            truth=np.asarray(Image.fromarray((1-t.mean(2)).astype(np.float32)).resize((size,size),Image.Resampling.BILINEAR));image=tensor(rgb,'cuda');features,candidates=extractor(image)
            meta=dict(s,pattern='original_spectral',case_id=s['scene_id']);predictions={}
            for name,model in models.items():
                pred,w=model(features,candidates);pred=pred.cpu().numpy()[0,0];predictions[name]=pred;rows.append(record(meta,name,pred,truth,w.cpu().numpy()[0],rgb))
            for name,model in oldmodels.items():
                pred=model(image).cpu().numpy()[0,0];predictions['v15_'+name]=pred;rows.append(record(meta,'v15_'+name,pred,truth,rgb=rgb))
            oracle=convex_oracle(candidates,torch.from_numpy(truth)[None,None].cuda()).cpu().numpy()[0,0];oracle_rows.append(record(meta,'oracle_OFFLINE_ONLY',oracle,truth,rgb=rgb))
            if si<3:plot(dest/(s['scene_id']+'.png'),rgb,truth,{'v15 fixed':predictions['v15_fixed'],'fixed aligned':predictions['fixed_aligned'],
                'global':predictions[f'global_seed{frozen["selected_seeds"]["global"]}'],'dynamic':predictions[f'dynamic_seed{frozen["selected_seeds"]["dynamic"]}'],'oracle':oracle})
        csv_write(dest/'per_image.csv',rows);csv_write(dest/'oracle_per_image.csv',oracle_rows)
        write(dest/'summary.json',{'metrics':summarize(rows),'checks':checks,'oracle_pixel_mae_lower_bound':float(np.mean([r['pixel_mae'] for r in oracle_rows])),
              'endpoint':'H_eff=1-mean_RGB(t_c), display-RGB original S synthesis; secondary endpoint, one conservative geographic group'})
        # Full pipeline cost includes candidate construction, not just tiny gating.
        costs=[]
        for size in (256,512):
            image=torch.rand(1,3,size,size,device='cuda')
            for name in ('fixed_aligned',f'global_seed{frozen["selected_seeds"]["global"]}',f'dynamic_seed{frozen["selected_seeds"]["dynamic"]}'):
                m=models[name]
                def forward():
                    f,c=extractor(image);return m(f,c)
                for _ in range(5):forward()
                torch.cuda.synchronize();times=[]
                for _ in range(20):
                    start=time.perf_counter();forward();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
                costs.append({'method':name,'size':size,'batch':1,'p50_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
                              'trainable_parameters':sum(p.numel() for p in m.parameters()),'includes':'all frozen priors, calibration, fusion'})
        if 'dynamic_rank_seed42' in models:
            rank_rows=[]
            for scene in [s for s in read(out/'scene_manifest.json') if s['split']=='confirmation']:
                clear,_=procedural(scene['seed'],base['image_size'])
                for meta,rgb,truth in cases(clear,scene,base):
                    f,c=extractor(tensor(rgb,'cuda'));pred,w=models['dynamic_rank_seed42'](f,c)
                    rank_rows.append(record(meta,'dynamic_rank_seed42',pred.cpu().numpy()[0,0],truth,w.cpu().numpy()[0],clear))
            csv_write(out/'ranking_ablation/confirmation_per_image.csv',rank_rows)
            write(out/'ranking_ablation/confirmation_summary.json',summarize(rank_rows))
    # Paired bootstrap over original scenes, averaged over the three initialization seeds.
    uncertainty={};collapse={};difficulty={}
    for split in ('development','test','confirmation','external'):
        with (out/split/'per_image.csv').open(encoding='utf-8') as f:rr=list(__import__('csv').DictReader(f))
        for mode in ('global','dynamic'):
            for seed in cfg['seeds']:
                r=[r for r in rr if r['method']==f'{mode}_seed{seed}']
                collapse[f'{split}_{mode}_{seed}']={'mean_weights':[float(np.mean([float(v[f'weight_{i}']) for v in r])) for i in range(3)],
                     'argmax_fractions':[float(np.mean([float(v[f'argmax_fraction_{i}']) for v in r])) for i in range(3)],
                     'entropy':float(np.mean([float(v['entropy']) for v in r]))}
        for method in sorted({r['method'] for r in rr}):
            r=[r for r in rr if r['method']==method]
            difficulty[f'{split}_{method}']={label:float(np.average([float(v[label+'_mae']) for v in r if v[label+'_mae']!=''],
                weights=[int(v[label+'_pixels']) for v in r if v[label+'_mae']!=''])) if any(v[label+'_mae']!='' for v in r) else None for label in ('bright','low_texture')}
        if split=='external':uncertainty[split]='No geographic confidence interval: one unverified supergroup';continue
        names=sorted({r['scene_id'] for r in rr});res={}
        for key in ('global_abs_error','pixel_mae','region_spearman'):
            delta=[]
            for scene in names:
                vals={}
                for mode in ('global','dynamic'):
                    r=[float(r[key]) for r in rr if r['scene_id']==scene and r['method'].startswith(mode+'_seed') and r[key]!='']
                    vals[mode]=np.mean(r)
                delta.append(vals['dynamic']-vals['global'])
            rng=np.random.default_rng(42);boot=np.array(delta)[rng.integers(0,len(names),(2000,len(names)))].mean(1)
            res[key]={'dynamic_minus_global':float(np.mean(delta)),'scene_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
        uncertainty[split]=res
    write(out/'supplement.json',{'cost':costs,'paired_uncertainty':uncertainty,'weight_collapse':collapse,'difficulty_proxy_mae':difficulty,
                                'difficulty_warning':'RGB high brightness/low texture masks are proxies, not verified roof/water semantic annotations'})


if __name__=='__main__':main()
