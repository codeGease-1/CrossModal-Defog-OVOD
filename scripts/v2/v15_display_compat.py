"""Source-protocol-aware V1.5 repair; separate from frozen V2 comparison.

Triggered by a paired DEVELOPMENT domain probe. Fixed statistics and scalar
calibration are fitted only to the original 48 procedural training originals,
with display-RGB synthesis. A second, unused controlled confirmation and unused
S members are held out. This is NOT automatic domain selection from test truth.
"""
import copy
import hashlib
import io
from pathlib import Path
import random
import sys
import time
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read,write,digest,procedural,original_scenes,cases,tensor,csv_write
from scripts.v2.run import record,summarize,plot
from src.models.haze_density.calibrated_prior import raw_cues,fixed_cues,fuse,fit_monotone,CalibratedPrior


def main():
    torch.set_num_threads(1);out=Path('experiments/v2_20261101');old=Path('experiments/v15_20261101');base=read(old/'protocol.json')['config'];cfg=dict(base,domain='display_rgb')
    dest=out/'v15_display_compat';dest.mkdir(exist_ok=True);cache=dest/'fit_cache';cache.mkdir(exist_ok=True)
    if (dest/'results.json').exists():return
    ix=np.linspace(0,base['image_size']-1,16,dtype=int)
    def sample(a):return a[...,ix[:,None],ix].reshape(-1)
    with torch.no_grad():
        if not (dest/'calibration.json').exists():
            start=time.monotonic()
            for s in original_scenes(cfg,'train'):
                p=cache/(s['scene_id']+'_raw.npz')
                if p.exists():continue
                clear,_=procedural(s['seed'],cfg['image_size']);vals=[]
                for _,rgb,y in cases(clear,s,cfg):
                    raw=raw_cues(tensor(rgb,'cuda')).cpu().numpy()[0]
                    vals.append(np.stack([sample(v) for v in raw]))
                np.savez_compressed(p,x=np.concatenate(vals,axis=1))
            data=np.concatenate([np.load(cache/(s['scene_id']+'_raw.npz'))['x'] for s in original_scenes(cfg,'train')],axis=1)
            cal=copy.deepcopy(read(old/'calibration.json'));cal['limits']=np.quantile(data,cfg['percentiles'],axis=1).T.tolist()
            for s in original_scenes(cfg,'train'):
                p=cache/(s['scene_id']+'_scores.npz')
                if p.exists():continue
                clear,_=procedural(s['seed'],cfg['image_size']);xx=[];yy=[]
                for _,rgb,y in cases(clear,s,cfg):
                    image=tensor(rgb,'cuda');score=fuse(image,fixed_cues(raw_cues(image),cal['limits']))
                    xx.append(sample(score.cpu().numpy()[0,0]));yy.append(sample(y))
                np.savez_compressed(p,x=np.concatenate(xx),y=np.concatenate(yy))
            arrays=[np.load(cache/(s['scene_id']+'_scores.npz')) for s in original_scenes(cfg,'train')]
            cal['tables']['fixed']=fit_monotone(np.concatenate([v['x'] for v in arrays]),np.concatenate([v['y'] for v in arrays]),cfg['isotonic_bins'])
            cal['target']='source-protocol-aware display-RGB controlled H=1-t; requires externally known synthesis convention'
            write(dest/'calibration.json',cal)
            budget=read(out/'budget.json');budget['calibration_seconds']+=time.monotonic()-start;write(out/'budget.json',budget)
        write(dest/'frozen.json',{'calibration_sha256':digest(dest/'calibration.json'),'source_sha256':digest(__file__),
                'trigger':'development paired-domain probe ONLY','primary_v2_models_unchanged':True,
                'confirmation_seeds':[base['seed']+60000+i for i in range(16)],'S_selection_seed':20261103,
                'warning':'Known source image-formation convention is required; never choose calibration by test error'})
        models={'v15_normalized':CalibratedPrior(read(old/'calibration.json'),'normalized'),
                'v15_fixed_linear':CalibratedPrior(read(old/'calibration.json'),'fixed'),
                'v15_fixed_display':CalibratedPrior(read(dest/'calibration.json'),'fixed')}
        all_rows=[]
        for domain in ('display_rgb','linear_srgb'):
            bc=dict(base,domain=domain)
            for i in range(16):
                seed=base['seed']+60000+i
                scene={'scene_id':f'confirmation2_{i:03d}','group_id':f'procedural_original_{seed}','seed':seed}
                clear,_=procedural(seed,base['image_size'])
                for meta,rgb,y in cases(clear,scene,bc):
                    image=tensor(rgb,'cuda')
                    for name,model in models.items():all_rows.append(dict(record(meta,name,model(image).cpu().numpy()[0,0],y),domain=domain))
        csv_write(dest/'confirmation2_per_image.csv',all_rows)
        old_members=set(s['archive_member'] for s in read(old/'external_manifest.json'))
        old_members.update(s['archive_member'] for s in read(out/'external_manifest.json'))
        old_members.update(s['member'] for s in read(old/'audit/archive_audit.json')['samples'])
        excluded=set(s['pixel_sha256'] for s in read(old/'external_manifest.json'))|set(s['pixel_sha256'] for s in read(out/'external_manifest.json'))
        real=[];manifest=[];checks=[]
        with zipfile.ZipFile('/mnt/RSHaze+.zip') as z:
            names=sorted(n for n in z.namelist() if 'RSHaze_S/test/cleanpng/' in n and n.endswith('.png') and n not in old_members)
            random.Random(20261103).shuffle(names)
            for n in names:
                data=z.read(n);im=Image.open(io.BytesIO(data)).convert('RGB');ph=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
                if ph in excluded:continue
                excluded.add(ph);sid='new2_S_'+Path(n).stem
                member={'scene_id':sid,'archive_member':n,'pixel_sha256':ph,'file_sha256':hashlib.sha256(data).hexdigest(),'group_id':'RSHaze_all_geography_unverified'}
                manifest.append(member)
                imgs=[im]+[Image.open(io.BytesIO(z.read(n.replace('/cleanpng/','/'+kind+'/')))) for kind in ('synhazypng','transpng','airpng')]
                j,h,t,a=[np.asarray(q,np.float64)/255 for q in imgs]
                error=float(np.abs(j*t+a*(1-t)-h).mean());checks.append({'scene_id':sid,'reconstruction_mae':error,'status':'PASS' if error<.005 else 'REJECT'})
                if error<.005:
                    size=base['image_size'];rgb=np.asarray(imgs[1].convert('RGB').resize((size,size),Image.Resampling.BILINEAR),np.float32)/255
                    y=np.asarray(Image.fromarray((1-t.mean(2)).astype(np.float32)).resize((size,size),Image.Resampling.BILINEAR));image=tensor(rgb,'cuda');preds={}
                    for name,model in models.items():
                        pred=model(image).cpu().numpy()[0,0];preds[name]=pred;real.append(record(dict(member,pattern='original_spectral',case_id=sid),name,pred,y))
                    if len(manifest)<=3:plot(dest/(sid+'.png'),rgb,y,preds)
                if len(manifest)>=8:break
        write(dest/'external_manifest.json',manifest);csv_write(dest/'original_S_per_image.csv',real)
        write(dest/'results.json',{'confirmation2':{d:summarize([r for r in all_rows if r['domain']==d]) for d in ('display_rgb','linear_srgb')},
            'original_S_new_members':summarize(real),'reconstruction_checks':checks,
            'adoption':'Compatibility prototype only; does not identify an unknown real image formation domain. No V2 retraining or retrospective replacement.',
            'remaining_limits':['Procedural training, same generator family','Spectral-average endpoint differs from scalar t','Real scenes share one unverified geographic group','Only one new source-specific calibration fit']})


if __name__=='__main__':main()
