"""Secondary original-S label check, after candidate freeze; never model selection.

S supplies channel-dependent t_R,t_G,t_B, not a single achromatic t. We explicitly
define t_eff=mean_RGB(t_c) for this SECONDARY endpoint; this does not silently
reinterpret a colorful PNG as grayscale transmission. Each held-out pair must
pass the image-formation residual check before this endpoint is evaluated.
"""
import argparse
import io
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read,write,digest,tensor,scores,plot_case,csv_write
from src.models.haze_density.calibrated_prior import apply_monotone
from benchmarks.haze_density.metrics import spatial_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',default='/mnt/RSHaze+.zip');p.add_argument('--output',default='experiments/v15_20261101');p.add_argument('--device',default='cuda')
    args=p.parse_args();out=Path(args.output);cfg=read(out/'protocol.json')['config'];torch.set_num_threads(1)
    cal=read(out/'calibration.json');frozen=read(out/'frozen_selection.json')
    assert digest(out/'calibration.json')==frozen['calibration_sha256']
    dest=out/'verified_spectral';dest.mkdir(exist_ok=True)
    if (dest/'results.json').exists(): return
    checks=[]; rows=[]; size=cfg['image_size']
    selected=[s for s in read(out/'external_manifest.json') if s['subset']=='S']
    with zipfile.ZipFile(args.archive) as z, torch.no_grad():
        for scene in selected:
            n=scene['archive_member']
            names=[n,n.replace('/cleanpng/','/synhazypng/'),n.replace('/cleanpng/','/transpng/'),n.replace('/cleanpng/','/airpng/')]
            imgs=[Image.open(io.BytesIO(z.read(q))) for q in names]
            j,h,t,a=[np.asarray(q,dtype=np.float64)/255 for q in imgs]
            if not (j.shape==h.shape==t.shape==a.shape):
                checks.append({'scene':scene['scene_id'],'status':'REJECT','reason':'shape mismatch'});continue
            mae=float(np.abs(j*t+a*(1-t)-h).mean())
            check={'scene':scene['scene_id'],'status':'PASS' if mae<.005 else 'REJECT','display_rgb_reconstruction_mae':mae,
                   'reverse_direction_mae':float(np.abs(j*(1-t)+a*t-h).mean()),
                   't_channel_max_difference':float(np.abs(t-t[...,:1]).max())}
            checks.append(check)
            if check['status']!='PASS':continue
            rgb=np.asarray(imgs[1].convert('RGB').resize((size,size),Image.Resampling.BILINEAR),np.float32)/255
            truth=np.asarray(Image.fromarray((1-t.mean(2)).astype(np.float32)).resize((size,size),Image.Resampling.BILINEAR))
            _,_,old,fix=scores(tensor(rgb,args.device),cfg,cal['limits'])
            preds={'legacy':old,'normalized_calibrated':apply_monotone(old,cal['tables']['normalized']),
                   'fixed_calibrated':apply_monotone(fix,cal['tables']['fixed'])}
            arrays={k:v.cpu().numpy()[0,0] for k,v in preds.items()}
            for method,pred in arrays.items():
                rows.append({'scene':scene['scene_id'],'method':method,'truth_mean':float(truth.mean()),
                    'prediction_mean':float(pred.mean()),'global_abs_error':float(abs(pred.mean()-truth.mean())),
                    'pixel_mae':float(np.abs(pred-truth).mean()),'region_spearman':spatial_metrics(1-truth,pred)['region_spearman']})
            if len(checks)<=3: plot_case(dest/(scene['scene_id']+'.png'),rgb,truth,arrays)
    metrics={}
    for method in preds if rows else []:
        rr=[r for r in rows if r['method']==method]
        metrics[method]={key:float(np.mean([r[key] for r in rr if r[key] is not None])) for key in ('global_abs_error','pixel_mae','region_spearman')}
    csv_write(dest/'per_image.csv',rows)
    write(dest/'results.json',{'interpretation':__doc__,'checks':checks,'metrics':metrics,
          'endpoint':'H_eff=1-mean_RGB(t_c), original display-RGB synthesis, secondary only',
          'n_conservative_geographic_groups':1,'selected_candidate_unchanged':frozen['selected']})
    print(metrics)


if __name__=='__main__':main()
