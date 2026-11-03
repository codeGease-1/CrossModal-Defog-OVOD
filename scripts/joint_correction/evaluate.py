"""Train-only fixed refits, old-development diagnostics, and portable export."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scripts.v15.run import read,write,digest,csv_write,procedural,cases,tensor
from scripts.learned_correction.run import cache,diagnostic
from scripts.v2.global_optimum import solve_simplex
from scripts.joint_correction.run import make
from src.models.haze_density.fast_candidates import FastCandidates
from src.models.haze_density.learned_correction import observed_context,RestrictedCorrector,RGBReference
from src.models.haze_density.matched_fusion import MatchedFusion
from benchmarks.haze_density.metrics import spatial_metrics,ranking_metrics,correlation


def load(out,cfg,seed,device='cuda'):
    nets={}
    for mode in cfg['modes']:
        net=make(mode,cfg,seed,device);net.load_state_dict(torch.load(out/f'{mode}_seed{seed}/best.pth',map_location='cpu',weights_only=False)['model']);nets[mode]=net.eval()
    for mode in ['rgb16','rgb32','direct_same16']:
        net=RGBReference(32 if mode=='rgb32' else 16,with_priors=mode=='direct_same16').to(device)
        net.load_state_dict(torch.load(Path(cfg['matched_controls'])/f'{mode}_seed{seed}/best.pth',map_location='cpu',weights_only=False)['model']);nets[mode]=net.eval()
    net=MatchedFusion(RestrictedCorrector(16),True).to(device)
    net.load_state_dict(torch.load(Path(cfg['matched_controls'])/f'corrected_gate_seed{seed}/best.pth',map_location='cpu',weights_only=False)['model']);nets['frozen_gate']=net.eval()
    return nets


def refit(out,cfg,p,ext,nets):
    if (out/'fusion_refit.json').exists():return read(out/'fusion_refit.json')
    x,y,meta=cache(p,ext,'train');n=len(x);nc=sum(m['uniform_t']==1. for m in meta)
    stats={(seed,mode):[np.zeros((3,3)),np.zeros(3),np.zeros((3,3)),np.zeros(3),np.zeros((3,3))] for seed in cfg['seeds'] for mode in cfg['modes']}
    with torch.no_grad():
        for i in range(0,n,8):
            f=x[i:i+8].cuda();yy=y[i:i+8].cuda()
            for (seed,mode),values in stats.items():
                with torch.autocast('cuda',dtype=torch.float16):q=nets[seed][mode](f)[1]['candidates']
                for j in range(len(q)):
                    c=q[j].reshape(3,-1).double();t=yy[j,0].flatten().double();mean=c.mean(1).cpu().numpy();ym=float(t.mean())
                    Q,R,Qg,Rg,Qc=values;matrix=(c@c.T/c.shape[1]).cpu().numpy();r=(c@t/c.shape[1]).cpu().numpy()
                    Q+=matrix;R+=r;Qg+=np.outer(mean,mean);Rg+=mean*ym
                    if meta[i+j]['uniform_t']==1.:Qc+=matrix
    fits={}
    for (seed,mode),(Q,R,Qg,Rg,Qc) in stats.items():
        mat=Q/n+cfg['global_loss_weight']*Qg/n+cfg['clear_loss_extra_weight']*Qc/nc;rhs=R/n+cfg['global_loss_weight']*Rg/n
        fits[f'{mode}_seed{seed}']={'weights':solve_simplex(mat,rhs).tolist(),'Q':mat.tolist(),'r':rhs.tolist(),
            'fitted_on':'train only after network checkpoint freeze','checkpoint_sha256':digest(out/f'{mode}_seed{seed}/best.pth')}
    write(out/'fusion_refit.json',fits);return fits


def summary(rows):
    ans={}
    for mode in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==mode];v={}
        for key in ['pixel_mae','pixel_mse','global_mae','signed_bias','region_spearman','clear_response','dark_mae','bright_mae','low_texture_mae']:
            values=[r[key] for r in rr if r[key] is not None];v[key]=float(np.mean(values)) if values else None;v[key+'_defined_cases']=len(values)
        groups={};cross={}
        for r in rr:
            if r['pattern']=='uniform':
                groups.setdefault((r['seed'],r['scene_id'],r['airlight_index'],r['brightness_index']),[]).append(r)
                cross.setdefault((r['seed'],r['airlight_index'],r['brightness_index']),[]).append(r)
        acc=[ranking_metrics([r['truth_mean'] for r in g],[r['prediction_mean'] for r in g])['pairwise_accuracy'] for g in groups.values()]
        v['same_scene_rank_accuracy']=float(np.mean(acc)) if acc else None
        num=den=0
        for g in cross.values():
            truth=np.array([r['truth_mean'] for r in g]);pred=np.array([r['prediction_mean'] for r in g]);scene=np.array([r['scene_id'] for r in g])
            dt=truth[:,None]-truth[None,:];dp=pred[:,None]-pred[None,:]
            mask=np.triu(np.ones(dt.shape,dtype=bool),1)&(scene[:,None]!=scene[None,:])&(np.abs(dt)>1e-6)
            den+=int(mask.sum());num+=int(((dp*np.sign(dt)>1e-6)&mask).sum())
        v['cross_original_rank_accuracy']=num/den if den else None;v['cases']=len(rr);ans[mode]=v
    return ans


def paired_ci(rows,left,right):
    scenes=sorted({r['scene_id'] for r in rows});keys=['pixel_mae','region_spearman','clear_response'];arrays={}
    for mode in [left,right]:arrays[mode]=np.array([[np.mean([r[k] for r in rows if r['scene_id']==s and r['method']==mode and r[k] is not None]) for k in keys] for s in scenes])
    draw=np.random.default_rng(20261102).integers(0,len(scenes),(3000,len(scenes)));a=arrays[left][draw].mean(1);b=arrays[right][draw].mean(1)
    delta=np.stack([a[:,0]/b[:,0]-1,b[:,1]-a[:,1],a[:,2]-b[:,2]],1)
    aa=arrays[left].mean(0);bb=arrays[right].mean(0);point=[aa[0]/bb[0]-1,bb[1]-aa[1],aa[2]-bb[2]]
    return {'left':left,'right':right,'quantities':['relative_MAE_increase','rho_drop','clear_increase'],'point':point,
            'CI95':np.quantile(delta,[.025,.975],axis=0).T.tolist(),'margins':[.05,.01,.005],
            'all_upper_bounds_within_margins':bool(np.all(np.quantile(delta,.975,axis=0)<=[.05,.01,.005])),
            'scope':'Old development only; scene bootstrap of fixed three trained seeds, not a fresh confirmation or seed uncertainty interval'}


def evaluate_and_export(out,cfg,p):
    start=time.monotonic();assert (out/'training_complete.json').exists();ext=FastCandidates(read(cfg['calibration'])).cuda().eval()
    nets={seed:load(out,cfg,seed) for seed in cfg['seeds']};fits=refit(out,cfg,p,ext,nets)
    oldfits=read(Path(cfg['old_pretraining'])/'fusion_weights.json')
    frozen={'checkpoints':{f'{mode}_seed{seed}':digest(out/f'{mode}_seed{seed}/best.pth') for mode in cfg['modes'] for seed in cfg['seeds']},'fusion_refit_sha256':digest(out/'fusion_refit.json'),'new_confirmation_scenes':0}
    if (out/'frozen.json').exists():assert read(out/'frozen.json')==frozen
    else:write(out/'frozen.json',frozen)
    for condition in ['nominal','unseen_A','display_rgb']:
        d=out/'development'/condition;d.mkdir(parents=True,exist_ok=True)
        if (d/'complete.json').exists():continue
        base=copy.deepcopy(p['base'])
        if condition=='unseen_A':base['airlights']=read(Path(cfg['old_pretraining'])/'protocol.json')['config']['unseen_airlights']
        if condition=='display_rgb':base['domain']='display_rgb'
        rows=[];diags=[];weight_rows=[];manifest=[]
        with torch.no_grad():
            for scene in [s for s in p['scenes'] if s['split']=='development']:
                clear,_=procedural(scene['seed'],base['image_size']);gray=clear@np.array([.299,.587,.114]);dy,dx=np.gradient(gray)
                proxies={'dark':gray<.2,'bright':gray>.8,'low_texture':np.hypot(dx,dy)<.01};items=list(cases(clear,scene,base))
                for i in range(0,len(items),8):
                    chunk=items[i:i+8];image=torch.cat([tensor(rgb,'cuda') for _,rgb,_ in chunk]);f,raw=ext(image);f=torch.cat([f,observed_context(image)],1)
                    for seed,ms in nets.items():
                        with torch.autocast('cuda',dtype=torch.float16):
                            oldh,olda=ms['frozen_gate'](f)
                            outputs={'frozen_dynamic':oldh,'frozen_fixed':(olda['candidates']*raw.new_tensor(oldfits[f'corrector16_seed{seed}']['weights'])[None,:,None,None]).sum(1,keepdim=True),
                                     'original_fixed':(raw*raw.new_tensor(oldfits['original']['weights'])[None,:,None,None]).sum(1,keepdim=True)}
                            candidates={'original':raw,'frozen_corrected':olda['candidates']}
                            for mode in cfg['modes']:
                                h,a=ms[mode](f);q=a['candidates'];w=a['weights'];candidates[mode]=q
                                refit_h=(q*q.new_tensor(fits[f'{mode}_seed{seed}']['weights'])[None,:,None,None]).sum(1,keepdim=True)
                                if mode=='joint_fixed':outputs[mode]=refit_h;outputs['joint_fixed_selection_weights']=h
                                else:
                                    outputs[mode]=h;outputs['joint_dynamic_refit_fixed']=refit_h
                                    outputs['joint_dynamic_mean']=(w.mean((2,3),keepdim=True)*q).sum(1,keepdim=True)
                                    perm=torch.randperm(w.shape[-1]*w.shape[-2],generator=torch.Generator(device='cuda').manual_seed(842+seed+i),device='cuda')
                                    outputs['joint_dynamic_shuffle']=(w.flatten(2)[:,:,perm].reshape_as(w)*q).sum(1,keepdim=True)
                                for j,(meta,_,_) in enumerate(chunk):weight_rows.append({'method':mode,'seed':seed,'case_id':meta['case_id'],'mean_weights':w[j].mean((1,2)).cpu().tolist()})
                            for mode in ['rgb16','rgb32','direct_same16']:outputs[mode]=ms[mode](f)[0].float()
                        arrays={key:value[:,0].cpu().numpy() for key,value in outputs.items()};cs={key:value.cpu().numpy() for key,value in candidates.items()}
                        for j,(meta,_,y) in enumerate(chunk):
                            for mode,predictions in arrays.items():
                                pred=predictions[j];err=pred-y
                                row={'method':mode,'seed':seed,'scene_id':meta['scene_id'],'case_id':meta['case_id'],'group_id':meta['group_id'],
                                     'pattern':meta['pattern'],'airlight_index':meta['airlight_index'],'brightness_index':meta['brightness_index'],'uniform_t':meta['uniform_t'],
                                     'truth_mean':float(y.mean()),'prediction_mean':float(pred.mean()),'pixel_mae':float(np.abs(err).mean()),'pixel_mse':float((err**2).mean()),
                                     'global_mae':float(abs(err.mean())),'signed_bias':float(err.mean()),'region_spearman':spatial_metrics(1-y,pred)['region_spearman'],
                                     'clear_response':float(pred.mean()) if meta['uniform_t']==1. else None}
                                for key,mask in proxies.items():row[key+'_mae']=float(np.abs(err)[mask].mean()) if mask.any() else None
                                rows.append(row)
                            for mode,qs in cs.items():
                                q=qs[j];dd=diagnostic(q,y);distance=np.maximum(q.min(0)-y,0)+np.maximum(y-q.max(0),0)
                                dd['outside_amplitude_conditional_mean']=float(distance[distance>0].mean()) if (distance>0).any() else 0.
                                diags.append({'method':mode,'seed':seed,'case_id':meta['case_id'],**dd})
                    for meta,rgb,y in chunk:manifest.append({**meta,'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'target_sha256':hashlib.sha256(y.tobytes()).hexdigest()})
                print('EVAL',condition,scene['scene_id'],flush=True)
        csv_write(d/'per_case.csv',rows);write(d/'candidate_diagnostics.json',diags);write(d/'weight_diagnostics.json',weight_rows);write(d/'data_manifest.json',manifest)
        total=summary(rows);write(d/'summary.json',total);write(d/'by_seed.json',{str(seed):summary([r for r in rows if r['seed']==seed]) for seed in cfg['seeds']})
        write(d/'paired_scene_bootstrap.json',[paired_ci(rows,left,right) for left in cfg['modes'] for right in ['rgb16','rgb32','direct_same16','frozen_dynamic']])
        old=read(Path(cfg['matched_controls'])/'development'/condition/'summary.json')
        for mode in ['rgb16','rgb32','direct_same16']:assert abs(total[mode]['pixel_mae']-old[mode]['pixel_mae'])<1e-8,(condition,mode)
        write(d/'complete.json',{'cases_per_model_seed':len(manifest),'old_control_MAE_reproduced':True,'new_confirmation_scenes':0})
        print('SUMMARY',condition,json.dumps(total),flush=True)
    dest=out/'portable_weights';dest.mkdir(exist_ok=True);records=[]
    for seed in cfg['seeds']:
        for mode in cfg['modes']:
            net=nets[seed][mode];state=net.state_dict()
            if mode=='joint_fixed':state=copy.deepcopy(state);state['fixed_weights']=state['fixed_weights'].new_tensor(fits[f'{mode}_seed{seed}']['weights'])[None,:,None,None]
            package={'mode':mode,'seed':seed,'model':{k:v.cpu() for k,v in state.items()},'calibration':read(cfg['calibration']),
                     'parameters':sum(v.numel() for v in net.parameters()),'protocol_sha256':digest(out/'protocol.json'),
                     'source_checkpoint_sha256':digest(out/f'{mode}_seed{seed}/best.pth'),'target':'H=1-t; RGB-only inference; no clean/t/A/identity'}
            file=dest/f'{mode}_seed{seed}.pth'
            if not file.exists():torch.save(package,file)
            records.append({'file':str(file.relative_to(out)),'sha256':digest(file),'bytes':file.stat().st_size})
    write(out/'portable_manifest.json',records);write(out/'evaluation_complete.json',{'seconds':time.monotonic()-start,'new_confirmation_scenes':0})
