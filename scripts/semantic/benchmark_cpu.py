"""Full-pipeline CPU cost, including frozen semantic extraction; no training."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import psutil
import torch
from scripts.v15.run import read, write
from scripts.semantic.frozen_semantics import FrozenSemantics
from src.models.haze_density.dynamic_prior import FrozenCandidates
from src.models.haze_density.student_weight_gate import StudentWeightGate
from src.models.haze_density.semantic_prior import FrozenSemanticCandidates, SemanticWeightGate, SemanticDirectRegressor


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='experiments/semantic_calibration_20261101');a=p.parse_args()
    out=Path(a.output);cfg=read('configs/semantic_calibration.json');torch.set_num_threads(cfg['torch_threads'])
    sem=FrozenSemantics(out/'semantic_model',cfg['semantic_labels']).eval()
    cal=read(out/'conditional_calibration.json')
    original=FrozenCandidates(read('experiments/v2_20261101/calibration.json'))
    corrected=FrozenSemanticCandidates(original.calibration,cal)
    f=np.load(next((out/'diagnostic_cache').glob('development_*.npz')))['features']
    image=torch.from_numpy(f[:3])[None]
    w=read(out/'fixed_weights.json')['weights']
    wa=torch.tensor(w['A_original'])[None,:,None,None]
    wc=torch.tensor(w['C_semantic'])[None,:,None,None]
    gate=StudentWeightGate().eval();d=SemanticWeightGate().eval()
    direct=SemanticDirectRegressor().eval();light=SemanticDirectRegressor(base_channels=8).eval()
    rgb=SemanticDirectRegressor(semantic_channels=0).eval();rgb_light=SemanticDirectRegressor(semantic_channels=0,base_channels=8).eval()
    def C():
        _,c=corrected(image,sem(image));return (wc*c).sum(1,keepdim=True)
    def D():
        p=sem(image);f,c=corrected(image,p);return d(torch.cat([f,p],1),c)[0]
    def R(m):
        return m(torch.cat([image,sem(image)],1))[0]
    jobs=[('A_original_fixed',lambda:(wa*original(image)[1]).sum(1,keepdim=True),None,False),
          ('B_original_dynamic',lambda:gate(original(image)[0])[0],gate,False),
          ('C_semantic_fixed',C,None,True),('D_semantic_dynamic',D,d,True),
          ('RGB_semantic',lambda:R(direct),direct,True),('RGB_semantic_light',lambda:R(light),light,True),
          ('RGB_only',lambda:rgb(image)[0],rgb,False),('RGB_only_light',lambda:rgb_light(image)[0],rgb_light,False)]
    rows=[]
    with torch.inference_mode():
        for name,fn,m,uses_sem in jobs:
            for _ in range(2):fn()
            times=[]
            for _ in range(5):
                start=time.perf_counter();pred=fn();times.append(time.perf_counter()-start)
                assert torch.isfinite(pred).all()
            trainable=sum(p.numel() for p in m.parameters()) if m is not None else 0
            rows.append({'model':name,'trainable_neural_parameters':trainable,
                         'total_neural_parameters':trainable+(sum(p.numel() for p in sem.parameters()) if uses_sem else 0),
                         'median_seconds':float(np.median(times)),'p95_seconds':float(np.quantile(times,.95))})
            print(name,rows[-1],flush=True)
    memory=psutil.Process().memory_info()
    lut_values=sum(len(t['x'])+len(t['y']) for t in cal['global'])
    lut_values+=sum(1+sum(len(t['x'])+len(t['y']) for t in v['tables']) for v in cal['classes'].values())
    write(out/'full_cpu_cost.json',{'shape':[1,3,256,256],'threads':cfg['torch_threads'],'warmup':2,'repeats':5,'pipelines':rows,
          'timing_scope':'RGB tensor in memory -> full pipeline output; includes semantic RGB resize/normalize/512 model/upsample and priors/calibration/gate where needed; excludes disk I/O',
          'calibration_table_scalars':lut_values,'fixed_weights_scalars':3,
          'models':'Random initialized downstream networks; inference cost only, not efficacy results',
          'cpu_process_rss_bytes':memory.rss,'cpu_process_peak_bytes':getattr(memory,'peak_wset',None),
          'memory_limit':'One process loaded all candidate networks and libraries; peak cannot be attributed per model',
          'gpu_peak_memory_bytes':None,'gpu_timing':None,'gpu_used':False,
          'cache_bytes':{name:sum(p.stat().st_size for p in (out/name).rglob('*') if p.is_file()) for name in ['semantic_model','probe_cache','diagnostic_cache','frozen_candidate_cache']}})


if __name__=='__main__':main()
