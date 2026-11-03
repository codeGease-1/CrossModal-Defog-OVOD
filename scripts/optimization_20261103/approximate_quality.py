"""Distinct approximate-execution quality gate, never relabel exactness failure."""
import sys,time,json,csv,copy,hashlib,subprocess,statistics,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from scripts.v15.run import procedural,cases,tensor
from scripts.optimization_20261103.latency import load,timed,OLD
from src.models.haze_density.captured_inference import CapturedInference
from src.models.haze_density.approximate_inference import ApproxCapturedInference
from benchmarks.haze_density.metrics import spatial_metrics
O=ROOT/'experiments/optimization_20261103';OUT=O/'approximate_quality'
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    while not (O/'POSTPROCESS_COMPLETE.json').exists():
        if (O/'airlight_pilot/status.json').exists() and json.loads((O/'airlight_pilot/status.json').read_text())['stage']=='failed':raise RuntimeError('pilot failed')
        time.sleep(10)
    OUT.mkdir();cfg=json.loads((ROOT/'configs/approximate_quality_20261103.json').read_text());torch.set_num_threads(1);torch.manual_seed(20261103)
    save(OUT/'protocol.json',dict(config=cfg,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/models/haze_density/approximate_inference.py']},
        numerical_equivalence=False,prior_strict_check='failed and preserved in separable_check; this study uses separate predefined quality margins, NOT relaxed exactness tolerances'))
    proto=json.loads((OLD/'protocol.json').read_text());rows=[];costs=[];maxdiff=0.;start=time.monotonic()
    with torch.no_grad():
        for mode in cfg['methods']:
            pipe=load(mode)
            for size in [256,512]:
                x=torch.rand(1,3,size,size,device='cuda');exact=CapturedInference(pipe,x);ap=ApproxCapturedInference(pipe,x)
                for turn in range(3):
                    for name,fn in ([('exact',exact),('approximate',ap)] if turn%2==0 else [('approximate',ap),('exact',exact)]):
                        val=timed(lambda:fn(x),60)
                        b=io.BytesIO();Image.fromarray((x[0].permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(b,format='PNG');data=b.getvalue()
                        def full():
                            with Image.open(io.BytesIO(data)) as im:a=np.array(im,dtype=np.float32)/255
                            return fn(torch.from_numpy(a).permute(2,0,1)[None].cuda()).cpu()
                        costs.append(dict(method=mode,size=size,round=turn,execution=name,GPU_RGB_to_H=val,PNG_bytes_to_CPU_H=timed(full,20)))
                if size==256:
                    for cond in ['nominal','unseen_A','display_rgb']:
                        base=copy.deepcopy(proto['base'])
                        if cond=='unseen_A':base['airlights']=[[.55,.6,.65],[.9,.78,.65]]
                        if cond=='display_rgb':base['domain']='display_rgb'
                        for scene in [s for s in proto['scenes'] if s['split']=='development']:
                            clear,_=procedural(scene['seed'],256)
                            for meta,rgb,y in cases(clear,scene,base):
                                image=tensor(rgb,'cuda');preds={'exact':exact(image)[0,0].cpu().numpy(),'approximate':ap(image)[0,0].cpu().numpy()}
                                maxdiff=max(maxdiff,float(np.max(np.abs(preds['exact']-preds['approximate']))))
                                for name,pred in preds.items():
                                    rows.append(dict(method=mode,condition=cond,execution=name,scene=scene['scene_id'],case_id=meta['case_id'],
                                        pixel_mae=float(np.abs(pred-y).mean()),region_spearman=spatial_metrics(1-y,pred)['region_spearman'],
                                        clear_response=float(pred.mean()) if meta['uniform_t']==1. else None))
                del exact,ap
            del pipe;torch.cuda.empty_cache()
    summary={};passed=True;groups=sorted({a['scene'] for a in rows});draw=np.random.default_rng(20261103).integers(0,16,(3000,16))
    for mode in cfg['methods']:
        for cond in ['nominal','unseen_A','display_rgb']:
            vals={}
            for name in ['exact','approximate']:
                rr=[a for a in rows if a['method']==mode and a['condition']==cond and a['execution']==name]
                vals[name]={k:statistics.mean(a[k] for a in rr if a[k] is not None) for k in ['pixel_mae','region_spearman','clear_response']}
                vals[name]['scene_mae']=np.array([statistics.mean(a['pixel_mae'] for a in rr if a['scene']==g) for g in groups])
            e,a=vals['exact'],vals['approximate'];change=a['scene_mae'][draw].mean(1)/e['scene_mae'][draw].mean(1)-1;CI=np.quantile(change,[.025,.975]).tolist()
            one=CI[1]<=cfg['relative_MAE_margin'] and e['region_spearman']-a['region_spearman']<=cfg['rho_margin'] and a['clear_response']-e['clear_response']<=cfg['clear_margin'];passed=passed and one
            for v in vals.values():del v['scene_mae']
            summary[mode+'_'+cond]=dict(**vals,relative_MAE_CI95=CI,passes_quality_gate=one)
    with (OUT/'per_case.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(OUT/'costs.json',costs);save(OUT/'summary.json',summary);save(OUT/'COMPLETE.json',dict(all_gates_pass=passed,max_output_difference=maxdiff,numerically_identical=False,seconds=time.monotonic()-start,scope='seed42 exposed development only; not confirmation or arbitrary-scene guarantee'))
    print('APPROXIMATE QUALITY COMPLETE',passed,maxdiff,flush=True)
if __name__=='__main__':main()
