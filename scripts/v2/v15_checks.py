"""Continue unresolved V1.5 checks, without fitting to any real test image."""
import copy
import io
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read,write,digest,procedural,original_scenes,cases,tensor,csv_write
from scripts.v2.run import summarize,record
from src.models.haze_density.calibrated_prior import CalibratedPrior,fit_monotone,raw_cues,apply_monotone,fuse,fixed_cues
from benchmarks.haze_density.synthetic import synthesize


def main():
    torch.set_num_threads(1);out=Path('experiments/v2_20261101');old=Path('experiments/v15_20261101')
    base=read(old/'protocol.json')['config'];cal=read(old/'calibration.json');dest=out/'v15_continuation';dest.mkdir(exist_ok=True)
    if (dest/'results.json').exists():return
    models={k:CalibratedPrior(cal,k).cuda() for k in ('normalized','fixed')}
    domains=[]
    with torch.no_grad():
        for domain in ('linear_srgb','display_rgb'):
            cfg=dict(base,domain=domain)
            for s in original_scenes(base,'development')[:4]:
                clear,_=procedural(s['seed'],base['image_size'])
                for meta,im,truth in cases(clear,s,cfg):
                    image=tensor(im,'cuda')
                    for name,model in models.items():
                        domains.append(dict(record(meta,name,model(image).cpu().numpy()[0,0],truth),domain=domain))
        # Bootstrap only the monotone table with the fixed V1.5 cue ranges.
        # This isolates table sensitivity, not all uncertainty in percentile fits.
        cache=old/'fit_cache';arrays=[np.load(cache/(s['scene_id']+'_scores.npz')) for s in original_scenes(base,'train')]
        boot=[]
        for seed in (242,243,244):
            rng=np.random.default_rng(seed);indices=rng.integers(0,len(arrays),len(arrays))
            cc=copy.deepcopy(cal)
            y=np.concatenate([arrays[i]['truth'] for i in indices])
            for mode in ('normalized','fixed'):
                x=np.concatenate([arrays[i][mode] for i in indices]);cc['tables'][mode]=fit_monotone(x,y,base['isotonic_bins'])
            write(dest/f'calibration_bootstrap_{seed}.json',cc)
            model=CalibratedPrior(cc,'fixed').cuda();rows=[]
            scenes=[s for s in read(out/'scene_manifest.json') if s['split']=='confirmation']
            for s in scenes:
                clear,_=procedural(s['seed'],base['image_size'])
                for meta,im,truth in cases(clear,s,base):rows.append(record(meta,'v15_fixed_bootstrap',model(tensor(im,'cuda')).cpu().numpy()[0,0],truth))
            boot.append({'seed':seed,'resampled_original_indices':indices.tolist(),'confirmation':summarize(rows)})
        # Original S spectral reconstruction and support diagnostics; same endpoint as V1.5.
        original_rows=[];checks=[]
        with zipfile.ZipFile('/mnt/RSHaze+.zip') as z:
            for s in [s for s in read(out/'external_manifest.json') if s['subset']=='S']:
                n=s['archive_member'];names=[n,n.replace('/cleanpng/','/synhazypng/'),n.replace('/cleanpng/','/transpng/'),n.replace('/cleanpng/','/airpng/')]
                imgs=[Image.open(io.BytesIO(z.read(q))) for q in names];j,h,t,a=[np.asarray(im,np.float64)/255 for im in imgs]
                mae=float(np.abs(j*t+a*(1-t)-h).mean())
                check={'scene_id':s['scene_id'],'display_reconstruction_mae':mae,'reverse_mae':float(np.abs(j*(1-t)+a*t-h).mean()),
                       'channel_max_difference':float(np.abs(t-t[...,:1]).max()),'status':'PASS' if mae<.005 else 'REJECT'}
                checks.append(check)
                if check['status']!='PASS':continue
                size=base['image_size'];rgb=np.asarray(imgs[1].convert('RGB').resize((size,size),Image.Resampling.BILINEAR),np.float32)/255
                truth=np.asarray(Image.fromarray((1-t.mean(2)).astype(np.float32)).resize((size,size),Image.Resampling.BILINEAR))
                image=tensor(rgb,'cuda');raw=raw_cues(image);cues=fixed_cues(raw,cal['limits']);score=fuse(image,cues)
                knots=cal['tables']['fixed']['x']
                check['fixed_fused_score_below_calibration_support']=float((score<knots[0]).float().mean())
                check['fixed_fused_score_above_calibration_support']=float((score>knots[-1]).float().mean())
                for name,model in models.items():
                    original_rows.append(record(dict(s,pattern='original_spectral',case_id=s['scene_id']),name,model(image).cpu().numpy()[0,0],truth))
        csv_write(dest/'domain_probe.csv',domains);csv_write(dest/'original_spectral_v15.csv',original_rows)
        metrics={}
        for domain in ('linear_srgb','display_rgb'):metrics[domain]=summarize([r for r in domains if r['domain']==domain])
        write(dest/'results.json',{'domain_probe':metrics,'fixed_range_table_bootstrap':boot,
             'original_S_new_member_checks':checks,'original_S_v15_metrics':summarize(original_rows),
             'endpoint':'Original S: H_eff=1-mean_RGB(t_c); not scalar physical t. Real references are one unverified geographic group.',
             'unresolved':['No verified real geographic training/development split','No absolute real-haze ground truth outside checked synthetic spectral pairs',
                           'Global density head and InstanceNorm retraining not justified or tested here; no claim of necessity',
                           'Bootstrap varies table fit only, not the fixed cue percentile ranges or source generator family']})


if __name__=='__main__':main()
