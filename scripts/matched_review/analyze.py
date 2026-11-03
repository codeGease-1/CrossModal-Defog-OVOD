"""Inference-only development diagnostic: project matched RGB outputs into frozen cue intervals."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.v15.run import read,write,csv_write,procedural,cases,tensor,digest
from scripts.matched_fusion.run import settings,make
from src.models.haze_density.dynamic_prior import FrozenCandidates
from src.models.haze_density.learned_correction import observed_context

settings();start=time.monotonic();old=ROOT/'experiments/matched_fusion_20261102';out=ROOT/'experiments/matched_review_20261102'
p=read(old/'protocol.json');cfg=p['config'];ext=FrozenCandidates(read(cfg['calibration'])).cuda().eval();rows=[]
assert not __import__('subprocess').check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() or True
for seed in cfg['seeds']:
 c=make('corrected_gate',cfg,seed).eval();r=make('direct_same16',cfg,seed).eval()
 c.load_state_dict(torch.load(old/f'corrected_gate_seed{seed}/best.pth',map_location='cpu',weights_only=False)['model'])
 r.load_state_dict(torch.load(old/f'direct_same16_seed{seed}/best.pth',map_location='cpu',weights_only=False)['model'])
 with torch.no_grad():
  for scene in [v for v in p['scenes'] if v['split']=='development']:
   clear,_=procedural(scene['seed'],p['base']['image_size']);items=list(cases(clear,scene,p['base']))
   for first in range(0,len(items),8):
    chunk=items[first:first+8];image=torch.cat([tensor(rgb,'cuda') for _,rgb,_ in chunk]);f,_=ext(image);f=torch.cat([f,observed_context(image)],1)
    with torch.autocast('cuda',dtype=torch.float16):
     cp,aux=c(f);rp=r(f)[0].float()
    q=aux['candidates'];lo=q.amin(1,keepdim=True);hi=q.amax(1,keepdim=True)
    projection=torch.maximum(lo,torch.minimum(hi,rp))
    y=torch.from_numpy(np.stack([y for _,_,y in chunk])[:,None]).cuda()
    oracle=torch.maximum(lo,torch.minimum(hi,y))
    for j,(meta,_,_) in enumerate(chunk):
     record={'seed':seed,'case_id':meta['case_id'],'scene_id':meta['scene_id'],
       'rgb_outside_candidate_fraction':float(((rp[j]<lo[j])|(rp[j]>hi[j])).float().mean()),
       'projection_change_mean':float((projection[j]-rp[j]).abs().mean())}
     for name,pred in [('corrected_gate',cp),('direct_same16',rp),('projected_RGB',projection),('oracle',oracle)]:
      err=pred[j]-y[j];record[name+'_mae']=float(err.abs().mean());record[name+'_mse']=float(err.square().mean())
     rows.append(record)
 del c,r;torch.cuda.empty_cache()
summary={k:float(np.mean([v[k] for v in rows])) for k in rows[0] if k not in ['seed','case_id','scene_id']}
write(out/'projection_summary.json',{'metrics':summary,'by_seed':{str(seed):{k:float(np.mean([v[k] for v in rows if v['seed']==seed])) for k in summary} for seed in cfg['seeds']},
 'scope':'Old nominal development originals only. No fitting or new confirmation. Projection uses RGB prediction and frozen candidate interval, not truth.',
 'warning':'Projection combines two networks for diagnosis only, not a parameter/cost matched competitive model. Oracle alone uses truth offline.',
 'elapsed_seconds':time.monotonic()-start,'source_sha256':digest(__file__),'source_protocol_sha256':digest(old/'protocol.json')})
csv_write(out/'projection_per_case.csv',rows);print(json.dumps(read(out/'projection_summary.json')))
