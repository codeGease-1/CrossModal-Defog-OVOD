"""Offline development diagnostics on previously exposed real references.

Known t describes ADDED synthetic attenuation of reference RGB, not the
reference's natural haze. No fitting or confirmatory scene selection here.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read,write,tensor,digest,csv_write
from benchmarks.haze_density.synthetic import synthesize
from src.models.haze_density.dynamic_prior import FrozenCandidates,single_scores
from src.models.haze_density.semantic_prior import SemanticCalibration,candidate_diagnostics
from scripts.v2.run import record,summarize


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='experiments/semantic_calibration_20261101');args=parser.parse_args()
    out=Path(args.output);cfg=read('configs/semantic_calibration.json');torch.set_num_threads(cfg['torch_threads'])
    ext=FrozenCandidates(read('experiments/v2_20261101/calibration.json'))
    correction=SemanticCalibration(read(out/'conditional_calibration.json'))
    weights=read(out/'fixed_weights.json')['weights'];rows=[];records=[];inputs=[]
    sources=read('experiments/v15_20261101/external_manifest.json')
    selected=[s for subset in ('G','S','L')
              for s in [s for s in sources if s['subset']==subset][:cfg['probe_real_per_subset']]]
    with torch.inference_mode():
        for source in selected:
            sid=source['scene_id']
            if not (out/'probe_cache'/f'{sid}_t1.0.npz').exists():continue
            path=Path('experiments/v15_20261101')/source['path']
            rgb0=np.asarray(Image.open(path).convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255
            inputs.append({'scene_id':sid,'group_id':source['group_id'],'source':path.as_posix(),
                           'sha256':digest(path),'role':'previously exposed external DEVELOPMENT reference, one geography-unverified supergroup'})
            gray=rgb0.mean(2);dy,dx=np.gradient(gray)
            masks={'all':np.ones_like(gray,dtype=bool),'bright_proxy':gray>.8,'low_texture_proxy':np.hypot(dx,dy)<.01}
            for t in cfg['probe_t']:
                target=np.full((256,256),1-t,dtype=np.float32)
                _,rgb,_=synthesize(rgb0,1-target,[1.,1.,1.],1.,'linear_srgb')
                image=tensor(rgb,'cpu');_,original=ext(image);_,score=single_scores(image,ext.calibration)
                prob=torch.from_numpy(np.load(out/'probe_cache'/f'{sid}_t{t}.npz')['probabilities'])[None]
                pooled=correction(score,prob,conditional=False)[0].numpy();c=correction(score,prob)[0].numpy()
                if not correction.calibration['classes']:
                    assert np.array_equal(pooled,c),'Empty class maps must reproduce pooled fallback'
                meta={'scene_id':sid,'group_id':source['group_id'],'case_id':sid+f'_added_t{t}',
                      'pattern':'uniform','uniform_t':t,'airlight_index':0,'brightness_index':0}
                for name,candidate in [('A_original',original[0].numpy()),('global_refit',pooled),('C_semantic',c)]:
                    pred=np.einsum('i,ihw->hw',weights[name],candidate)
                    records.append(record(meta,name,pred,target,rgb=rgb0))
                    for region,mask in masks.items():
                        if not mask.any():continue
                        # Reshape region pixels so oracle sees exactly the same pixels per method.
                        diag=candidate_diagnostics(candidate[:,mask][:,None,:],target[mask][None,:],cfg['bias_tolerance'])
                        rows.append({'scene_id':sid,'group_id':source['group_id'],'t_added':t,'version':name,
                                     'region':region,'pixels':int(mask.sum()),**diag})
    write(out/'real_reference_inputs.json',inputs)
    write(out/'real_reference_boundaries.json',rows)
    csv_write(out/'real_reference_per_image.csv',records)
    write(out/'real_reference_metrics.json',{'summary':summarize(records),
          'scope':'6 previously exposed real RGB backgrounds, known added synthetic t, uniform haze only; no geography generalization, no natural atmospheric t truth, no semantic truth'})


if __name__=='__main__':main()
