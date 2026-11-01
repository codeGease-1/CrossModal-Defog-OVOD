"""Predeclared secondary diagnostics; never changes the frozen candidate.

Cross-condition ranking includes different brightnesses AND airlights, unlike
the primary within-condition cross-original comparison. Scene bootstrap is
only reported for genuinely independent procedural originals, never for the
single conservatively grouped RSHaze source.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read,write,digest,procedural,cases,tensor,scores,plot_case,METHODS
from src.models.haze_density.calibrated_prior import CalibratedPrior, apply_monotone
from src.models.haze_density.physical_prior import generate_s_final


def cross_conditions(rows):
    output={}
    for method in METHODS:
        rr=[r for r in rows if r['method']==method and r['pattern']=='uniform']
        good=ties=total=0
        for i,a in enumerate(rr):
            for b in rr[i+1:]:
                if a['scene_id']==b['scene_id']: continue
                dt=float(a['truth_mean'])-float(b['truth_mean'])
                if abs(dt)<1e-6: continue
                dp=(float(a['prediction_mean'])-float(b['prediction_mean']))*np.sign(dt)
                total+=1; good+=int(dp>1e-6); ties+=int(abs(dp)<=1e-6)
        output[method]={'accuracy':good/total,'tie_rate':ties/total,'comparable_pairs':total}
    return output


def bootstrap(rows,seed):
    names=sorted({r['scene_id'] for r in rows})
    rng=np.random.default_rng(seed)
    output={}
    for field in ('global_abs_error','region_spearman'):
        deltas=[]
        for name in names:
            vals={}
            for method in ('legacy','fixed_calibrated'):
                vv=[float(r[field]) for r in rows if r['method']==method and r['scene_id']==name and r[field]!='']
                vals[method]=np.mean(vv)
            deltas.append(vals['fixed_calibrated']-vals['legacy'])
        resampled=np.array(deltas)[rng.integers(0,len(names),(2000,len(names)))].mean(1)
        output[field]={'fixed_minus_legacy':float(np.mean(deltas)),
                       'scene_bootstrap_95_percentile_interval':np.quantile(resampled,[.025,.975]).tolist(),
                       'n_independent_procedural_originals':len(names)}
    return output


def cost(calibration,device):
    torch.manual_seed(20261101)
    methods={'legacy':lambda x:generate_s_final(x,**calibration['prior']),
             'normalized_calibrated':CalibratedPrior(calibration,'normalized').to(device),
             'fixed_calibrated':CalibratedPrior(calibration,'fixed').to(device)}
    output=[]
    for size in (256,512):
        image=torch.rand((1,3,size,size),device=device)
        for name,method in methods.items():
            for _ in range(5): method(image)
            if device=='cuda': torch.cuda.synchronize()
            values=[]
            for _ in range(20):
                start=time.perf_counter(); method(image)
                if device=='cuda': torch.cuda.synchronize()
                values.append((time.perf_counter()-start)*1000)
            output.append({'method':name,'size':size,'device':device,'batch':1,'warmup':5,'repeats':20,
                           'p50_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95))})
    return output


def failure_plots(out,cfg,calibration,device,split):
    failures=read(out/split/'failure_cases.json')
    selected=[r for r in failures if r['method']=='fixed_calibrated'][:3]
    if split=='external':
        manifest={s['scene_id']:s for s in read(out/'external_manifest.json')}
    for row in selected:
        if split=='external':
            scene=manifest[row['scene_id']]
            clear=np.asarray(Image.open(out/scene['path']).convert('RGB').resize((cfg['image_size'],cfg['image_size']),Image.Resampling.BILINEAR),np.float32)/255
        else:
            scene={k:row[k] for k in ('scene_id','split','group_id','seed','source_kind')}
            clear,_=procedural(scene['seed'],cfg['image_size'])
        for meta,rgb,truth in cases(clear,scene,cfg):
            if meta['case_id']!=row['case_id']: continue
            _,_,old,fix=scores(tensor(rgb,device),cfg,calibration['limits'])
            predictions={'legacy':old,'normalized_calibrated':apply_monotone(old,calibration['tables']['normalized']),
                         'fixed_calibrated':apply_monotone(fix,calibration['tables']['fixed'])}
            arrays={k:v.cpu().numpy()[0,0] for k,v in predictions.items()}
            plot_case(out/split/'figures'/('FAIL_'+meta['case_id']+'.png'),rgb,truth,arrays)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='experiments/v15_20261101'); p.add_argument('--device',default='cuda')
    a=p.parse_args();out=Path(a.output);protocol=read(out/'protocol.json');cfg=protocol['config'];torch.set_num_threads(1)
    cal=read(out/'calibration.json'); frozen=read(out/'frozen_selection.json')
    assert digest(out/'calibration.json')==frozen['calibration_sha256']
    result={'frozen_selected':frozen['selected'],'secondary_metric_plan':__doc__,'phases':{}}
    with torch.no_grad():
        for split in ('development','test','external'):
            with (out/split/'per_image.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
            result['phases'][split]={'cross_condition_cross_original_rank':cross_conditions(rows),
                'uncertainty':bootstrap(rows,cfg['seed']) if split!='external' else 'No geographic CI: all real sources form one supergroup.'}
            failure_plots(out,cfg,cal,a.device,split)
        result['cost']=cost(cal,a.device)
    write(out/'secondary_diagnostics.json',result)
    # Full-environment package versions; no environment variables or credentials.
    import subprocess
    (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    manifests={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*'))
               if p.is_file() and p.name!='artifacts_sha256.json' and 'fit_cache' not in p.parts}
    write(out/'artifacts_sha256.json',manifests)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
