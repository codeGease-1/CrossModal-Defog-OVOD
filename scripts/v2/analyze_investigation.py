"""Summarize seeds, scene-paired uncertainty and actual gains/failures."""
import csv,hashlib,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.v15.run import read,write,cases,procedural,tensor
from scripts.v2.run import plot
from scripts.v2.investigate import make_model,charge,remaining
from src.models.haze_density.dynamic_prior import FrozenCandidates,convex_oracle

def getrows(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    torch.set_num_threads(1);cfg=read('configs/v2_investigation.json');out=Path(cfg['output_directory']);p=read(out/'protocol.json');frozen=read(out/'frozen.json');old=Path(cfg['previous_directory'])
    results={};worst={};bootstrap={};collapse={};difficulty={}
    for condition in ('nominal','unseen_airlight','display_rgb','original_S'):
        summary=read(out/condition/'summary.json');rows=getrows(out/condition/'per_image.csv');results[condition]={}
        for mode in cfg['candidates']:
            results[condition][mode]={}
            for metric in ('global_mae','pixel_mae','region_spearman','clear_response','same_scene_rank','cross_condition_cross_scene_rank'):
                values=[summary['metrics'][f'{mode}_seed{s}'][metric] for s in cfg['seeds']]
                if any(v is None for v in values):results[condition][mode][metric]=None
                else:results[condition][mode][metric]={'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)),'seeds':values}
            for seed in cfg['seeds']:
                rr=[r for r in rows if r['method']==f'{mode}_seed{seed}']
                collapse[f'{condition}_{mode}_{seed}']={'mean_weights':[float(np.mean([float(r[f'weight_{i}']) for r in rr])) for i in range(3)],
                    'argmax_fractions':[float(np.mean([float(r[f'argmax_fraction_{i}']) for r in rr])) for i in range(3)],
                    'entropy':float(np.mean([float(r['entropy']) for r in rr]))}
        for method in ['global_exact','original_dynamic']+[f'{mode}_seed{frozen["selected_seeds"][mode]}' for mode in cfg['candidates']]:
            rr=[r for r in rows if r['method']==method];difficulty[f'{condition}_{method}']={}
            for label in ('bright','low_texture'):
                valid=[r for r in rr if r[label+'_mae']!=''];n=sum(int(r[label+'_pixels']) for r in valid)
                difficulty[f'{condition}_{method}'][label]=sum(float(r[label+'_mae'])*int(r[label+'_pixels']) for r in valid)/n if n else None
        names=sorted({r['scene_id'] for r in rows})
        if condition!='original_S':
            bootstrap[condition]={}
            for mode in cfg['candidates']:
                bootstrap[condition][mode]={}
                for metric in ('pixel_mae','global_abs_error','region_spearman'):
                    deltas=[]
                    for name in names:
                        vals=[float(r[metric]) for r in rows if r['scene_id']==name and r['method'].startswith(mode+'_seed') and r[metric]!='']
                        baseline=[float(r[metric]) for r in rows if r['scene_id']==name and r['method']=='global_exact' and r[metric]!='']
                        deltas.append(np.mean(vals)-np.mean(baseline))
                    rng=np.random.default_rng(42);boots=np.array(deltas)[rng.integers(0,len(names),(2000,len(names)))].mean(1)
                    bootstrap[condition][mode][metric]={'delta_to_global_exact':float(np.mean(deltas)),'scene_bootstrap_95':np.quantile(boots,[.025,.975]).tolist()}
        selected=[r for r in rows if r['method']==f'student_seed{frozen["selected_seeds"]["student"]}'];base={r['case_id']:r for r in rows if r['method']=='global_exact'}
        for r in selected:r['delta_pixel_mae_to_global_exact']=float(r['pixel_mae'])-float(base[r['case_id']]['pixel_mae'])
        selected.sort(key=lambda r:r['delta_pixel_mae_to_global_exact'])
        worst[condition]={'best_gains':selected[:2],'worst_regressions':selected[-2:]}
    write(out/'analysis.json',{'seed_summary':results,'paired_scene_bootstrap':bootstrap,'collapse':collapse,'difficulty_proxy_mae':difficulty,
          'difficulty_limit':'RGB proxy masks only; no roof/water semantic labels','original_S_CI':'Not computed: one unverified geographic supergroup'})
    write(out/'failure_and_gain_cases.json',worst)
    # Benchmark and regenerate actual selected failures/gains after all tests freeze.
    if (out/'visualization_complete.json').exists():return
    start=time.monotonic();ext=FrozenCandidates(read(old/'calibration.json'));m=make_model('student',cfg).cuda()
    m.load_state_dict(torch.load(out/f'student_seed{frozen["selected_seeds"]["student"]}/best.pth',map_location='cuda',weights_only=False)['model']);m.eval()
    weights=torch.tensor(read(old/'global_optimum/fit.json')['weights'],device='cuda')[None,:,None,None];costs=[]
    with torch.no_grad():
        for mode in cfg['candidates']:
            model=make_model(mode,cfg).cuda();model.load_state_dict(torch.load(out/f'{mode}_seed{frozen["selected_seeds"][mode]}/best.pth',map_location='cuda',weights_only=False)['model']);model.eval()
            for size in (256,512):
                x=torch.rand(1,3,size,size,device='cuda')
                for _ in range(5):f,c=ext(x);model(f,c)
                torch.cuda.synchronize();times=[]
                for _ in range(20):
                    t=time.monotonic();f,c=ext(x);model(f,c);torch.cuda.synchronize();times.append((time.monotonic()-t)*1000)
                costs.append({'mode':mode,'size':size,'p50_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
                    'parameters':sum(q.numel() for q in model.parameters()),'includes':'frozen prior extraction + calibration + weight network + convex fusion'})
        for condition in ('nominal','unseen_airlight'):
            picks=worst[condition]['best_gains'][:1]+worst[condition]['worst_regressions'][-1:]
            scenes={s['scene_id']:s for s in read(out/'confirmation_manifest.json')};base=p['base'].copy()
            if condition=='unseen_airlight':base['airlights']=cfg['unseen_airlights']
            for pick in picks:
                s=scenes[pick['scene_id']];clear,_=procedural(s['seed'],base['image_size'])
                for meta,rgb,y in cases(clear,s,base):
                    if meta['case_id']!=pick['case_id']:continue
                    f,c=ext(tensor(rgb,'cuda'));pred,w=m(f,c);oracle=convex_oracle(c,torch.from_numpy(y)[None,None].cuda())
                    plot(out/condition/'figures'/('GAIN_OR_FAIL_'+meta['case_id']+'.png'),rgb,y,{'global':(c*weights).sum(1)[0].cpu().numpy(),
                        'student weight':pred[0,0].cpu().numpy(),'oracle':oracle[0,0].cpu().numpy()})
                    plot(out/condition/'figures'/('WEIGHTS_'+meta['case_id']+'.png'),rgb,y,{f'weight_{i}':w[0,i].cpu().numpy() for i in range(3)})
    write(out/'inference_cost.json',costs);write(out/'visualization_complete.json',{'status':'DONE','color_range':[0,1]});charge(out,'analysis_visualization',start)

if __name__=='__main__':main()
