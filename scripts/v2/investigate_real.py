"""Predeclared secondary endpoint on unused S members; no real-data fitting."""
from pathlib import Path
import hashlib,io,random,sys,time,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read,write,digest,tensor,csv_write
from scripts.v2.run import record,summarize,plot
from scripts.v2.investigate import make_model,charge,remaining
from src.models.haze_density.dynamic_prior import FrozenCandidates,ConvexFusion,convex_oracle

def main():
    torch.set_num_threads(1);cfg=read('configs/v2_investigation.json');out=Path(cfg['output_directory']);dest=out/'original_S';dest.mkdir(exist_ok=True)
    plan=dest/'plan.json'
    if not plan.exists():write(plan,{'selection_seed':20261104,'count':8,'endpoint':'H_eff=1-mean_RGB(t_c); secondary, not scalar t',
        'reconstruction_threshold':.005,'all_sources_one_external_geographic_group':True,'fitting_on_real_images':False,
        'role':'Predeclared failure-bound diagnostic, no change to model selection or nominal primary endpoint'})
    if not (out/'frozen.json').exists():print('Secondary plan frozen; waiting for training/freeze');return
    if (dest/'summary.json').exists():return
    assert remaining(out)>60;start=time.monotonic();old=Path(cfg['previous_directory']);frozen=read(out/'frozen.json')
    ext=FrozenCandidates(read(old/'calibration.json'));models={}
    for mode in cfg['candidates']:
        for seed in cfg['seeds']:
            name=f'{mode}_seed{seed}';path=out/name/'best.pth';assert digest(path)==frozen['checkpoints'][str(path.relative_to(out))]
            m=make_model(mode,cfg).cuda();m.load_state_dict(torch.load(path,weights_only=False,map_location='cuda')['model']);m.eval();models[name]=m
    seed=read(old/'frozen.json')['selected_seeds']['dynamic'];m=ConvexFusion('dynamic').cuda()
    m.load_state_dict(torch.load(old/f'dynamic_seed{seed}/best.pth',weights_only=False,map_location='cuda')['model']);m.eval();models['original_dynamic']=m
    weights=torch.tensor(read(old/'global_optimum/fit.json')['weights'],device='cuda')[None,:,None,None]
    excluded_members=set();excluded_pixels=set()
    for directory in ('experiments/v15_20261101',str(old),str(old/'v15_display_compat')):
        for s in read(Path(directory)/'external_manifest.json'):
            excluded_members.add(s['archive_member']);excluded_pixels.add(s['pixel_sha256'])
    excluded_members.update(s['member'] for s in read('experiments/v15_20261101/audit/archive_audit.json')['samples'])
    rows=[];oracle_rows=[];checks=[];manifest=[];size=read(out/'protocol.json')['base']['image_size']
    with torch.no_grad(),zipfile.ZipFile('/mnt/RSHaze+.zip') as z:
        names=sorted(n for n in z.namelist() if 'RSHaze_S/test/cleanpng/' in n and n.endswith('.png') and n not in excluded_members)
        random.Random(20261104).shuffle(names)
        for n in names:
            data=z.read(n);image=Image.open(io.BytesIO(data)).convert('RGB');ph=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
            if ph in excluded_pixels:continue
            excluded_pixels.add(ph);sid='new3_S_'+Path(n).stem
            s={'scene_id':sid,'archive_member':n,'pixel_sha256':ph,'file_sha256':hashlib.sha256(data).hexdigest(),'group_id':'RSHaze_all_geography_unverified'}
            imgs=[image]+[Image.open(io.BytesIO(z.read(n.replace('/cleanpng/','/'+k+'/')))) for k in ('synhazypng','transpng','airpng')]
            arrays=[np.asarray(im,np.float64)/255 for im in imgs];j,h,t,a=arrays
            err=float(np.abs(j*t+a*(1-t)-h).mean());reverse=float(np.abs(j*(1-t)+a*t-h).mean())
            assert all(ar.shape==arrays[0].shape for ar in arrays)
            passed=err<.005 and err<reverse;checks.append({'scene_id':sid,'reconstruction_mae':err,'reversed_direction_mae':reverse,'status':'PASS' if passed else 'REJECT'})
            manifest.append(s)
            if passed:
                rgb=np.asarray(imgs[1].convert('RGB').resize((size,size),Image.Resampling.BILINEAR),np.float32)/255
                y=np.asarray(Image.fromarray((1-t.mean(2)).astype(np.float32)).resize((size,size),Image.Resampling.BILINEAR));f,c=ext(tensor(rgb,'cuda'))
                truth=torch.from_numpy(y)[None,None].cuda();oracle=convex_oracle(c,truth);preds={}
                for name,m in models.items():
                    pred,w=m(f,c);v=pred[0,0].cpu().numpy();rows.append(record(dict(s,case_id=sid,pattern='original_spectral'),name,v,y,w[0].cpu().numpy(),rgb));preds[name]=v
                for name,pred in [('global_exact',(c*weights).sum(1,keepdim=True)),('single_dark',c[:,:1])]:
                    v=pred[0,0].cpu().numpy();rows.append(record(dict(s,case_id=sid,pattern='original_spectral'),name,v,y,rgb=rgb));preds[name]=v
                r=record(dict(s,case_id=sid,pattern='original_spectral'),'offline_oracle',oracle[0,0].cpu().numpy(),y,rgb=rgb)
                r['all_too_low_fraction']=float((c.amax(1,keepdim=True)<truth-.03).float().mean())
                r['all_too_high_fraction']=float((c.amin(1,keepdim=True)>truth+.03).float().mean());oracle_rows.append(r)
                if len(manifest)<=3:plot(dest/(sid+'.png'),rgb,y,{'global':preds['global_exact'],'old gate':preds['original_dynamic'],
                    'student weight':preds[f'student_seed{frozen["selected_seeds"]["student"]}'],'oracle':oracle[0,0].cpu().numpy()})
            if len(manifest)==8:break
    write(dest/'manifest.json',manifest);write(dest/'label_checks.json',checks);csv_write(dest/'per_image.csv',rows);csv_write(dest/'oracle_per_image.csv',oracle_rows)
    write(dest/'summary.json',{'metrics':summarize(rows),'oracle_pixel_mae':float(np.mean([r['pixel_mae'] for r in oracle_rows])),
        'all_too_low_fraction':float(np.mean([r['all_too_low_fraction'] for r in oracle_rows])),
        'all_too_high_fraction':float(np.mean([r['all_too_high_fraction'] for r in oracle_rows])),
        'scope':'Secondary spectral endpoint; eight new members, one unverified geographic group. No tuning or population CI.'})
    charge(out,'original_S',start);print('ORIGINAL S COMPLETE')

if __name__=='__main__':main()
