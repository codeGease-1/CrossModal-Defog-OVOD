"""Bounded causal follow-up to V2 collapse; stages preserve old artifacts."""
import argparse,copy,hashlib,io,json,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.v15.run import read,write,digest,cases,procedural,original_scenes,tensor,csv_write
from scripts.v2.run import cache_split,loss_fn,dev_loss,record,summarize,plot
from src.models.haze_density.dynamic_prior import FrozenCandidates,ConvexFusion,convex_oracle
from src.models.haze_density.student_weight_gate import TemperatureGate,StudentWeightGate


def make_model(mode,cfg):
    return TemperatureGate(cfg['temperature']) if mode=='temperature' else StudentWeightGate(cfg['student_base_channels'],cfg['temperature'])


def charge(out,stage,started):
    p=out/'budget.json';b=read(p);seconds=time.monotonic()-started
    b['new_gpu_wall_seconds']+=seconds;b['stages'].append({'stage':stage,'seconds':seconds});write(p,b)
    return b


def remaining(out):
    b=read(out/'budget.json');return b['limit_seconds']-b['new_gpu_wall_seconds']


def prepare(out,cfg):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():assert read(out/'protocol.json')['config']==cfg;return
    old=Path(cfg['previous_directory']);p=read(old/'protocol.json')
    sources=['scripts/v2/investigate.py','src/models/haze_density/student_weight_gate.py',
             'src/models/haze_density/dynamic_prior.py','scripts/v2/run.py','scripts/v15/run.py']
    sources += [str(q.relative_to(ROOT)) for q in (ROOT/'src/models/haze_density').glob('*.py') if str(q.relative_to(ROOT)) not in sources]
    write(out/'protocol.json',{'config':cfg,'base':p['v15_config'],'loss':p['config'],
        'candidate_calibration_sha256':digest(old/'calibration.json'),'previous_frozen_sha256':digest(old/'frozen.json'),
        'previous_budget':read(old/'budget.json'),'source_sha256':{s:digest(ROOT/s) for s in sources},
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'environment':{'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)},
        'purpose':'Separate optimization saturation, contextual capacity and candidate information limits; no old test tuning'})
    oldgroups={s['group_id'] for s in read(old/'scene_manifest.json')};scenes=[]
    for i in range(cfg['confirmation_scenes']):
        seed=p['v15_config']['seed']+cfg['confirmation_seed_offset']+i;rgb,kind=procedural(seed,p['v15_config']['image_size'])
        s={'scene_id':f'confirm3_{i:03d}','seed':seed,'group_id':f'procedural_original_{seed}',
           'land_cover_design':kind,'original_pixel_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}
        assert s['group_id'] not in oldgroups;scenes.append(s)
    write(out/'confirmation_manifest.json',scenes)
    write(out/'budget.json',{'new_gpu_wall_seconds':0.,'limit_seconds':cfg['new_gpu_wall_budget_seconds'],
         'stages':[],'accounting':'Conservative wall time of stages, including preprocessing/evaluation; no parallel GPU jobs. Previous V2 fit+calibration kept separately in protocol.'})


def preflight(out,cfg,p):
    if (out/'preflight.json').exists():return
    start=time.monotonic();rows=[];torch.manual_seed(123)
    x=torch.rand(cfg['batch_size'],9,256,256,device='cuda');y=torch.rand(cfg['batch_size'],1,256,256,device='cuda')
    for mode in cfg['candidates']:
        model=make_model(mode,cfg).cuda();opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);times=[]
        for step in range(10):
            torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True);pred,w=model(x)
            loss=(pred-y).square().mean();loss.backward();opt.step();torch.cuda.synchronize()
            if step>=2:times.append(time.monotonic()-t)
            assert torch.isfinite(loss) and torch.allclose(w.sum(1),torch.ones_like(w[:,0]),atol=1e-6)
        rows.append({'mode':mode,'parameters':sum(q.numel() for q in model.parameters()),'batch_step_median_seconds':float(np.median(times))})
        del model,opt
    write(out/'preflight.json',rows);b=charge(out,'preflight',start)
    estimated=sum(r['batch_step_median_seconds'] for r in rows)*192*cfg['epochs']*len(cfg['seeds'])
    write(out/'cost_forecast.json',{'training_step_estimate_seconds':estimated,'remaining_seconds':remaining(out),
        'note':'Excludes epoch validation and checkpoint IO; stop before evaluation reserve if necessary'})
    print('PREFLIGHT',rows,'forecast',estimated,flush=True)


def diagnostics(out,cfg,p):
    if (out/'diagnostics.json').exists():return
    start=time.monotonic();base=p['base'];ext=FrozenCandidates(read(Path(cfg['previous_directory'])/'calibration.json'))
    # A learnable three-choice fixture tests actual softmax/backprop, not haze validity.
    torch.manual_seed(704);x=torch.rand(8,9,32,32,device='cuda')
    band=(x[:,0]*3).long().clamp(0,2);c=x[:,3:6];y=c.gather(1,band[:,None])
    toy={}
    for label in ('original','temperature'):
        torch.manual_seed(42);m=ConvexFusion('dynamic').cuda() if label=='original' else make_model('temperature',cfg).cuda()
        opt=torch.optim.Adam(m.parameters(),lr=.001);hist=[]
        for step in range(301):
            opt.zero_grad(set_to_none=True);pred,w=m(x);loss=(pred-y).square().mean();loss.backward();opt.step()
            if step in (0,50,100,200,300):hist.append({'step':step,'mse':float(loss.detach()),'choice_accuracy':float((w.argmax(1)==band).float().mean())})
        toy[label]=hist
    # Single ORIGINAL training case: never use development/test to optimize.
    scene=original_scenes(base,'train')[0];clear,_=procedural(scene['seed'],base['image_size'])
    train_case_meta,rgb,truth=next(v for v in cases(clear,scene,base) if v[0]['pattern']=='gradient' and v[0]['brightness_index']==1)
    with torch.no_grad():features,candidates=ext(tensor(rgb,'cuda'));target=torch.from_numpy(truth)[None,None].cuda();oracle=convex_oracle(candidates,target)
    crop=(slice(None),slice(None),slice(64,128),slice(64,128));features=features[crop].contiguous();target=target[crop];candidates=features[:,3:6]
    overfit={};crop_bounds={'dark_mse':float((candidates[:,:1]-target).square().mean()),
        'oracle_mse':float((convex_oracle(candidates,target)-target).square().mean())}
    for label in ('original','temperature','student'):
        torch.manual_seed(42);m=ConvexFusion('dynamic').cuda() if label=='original' else make_model(label,cfg).cuda()
        opt=torch.optim.Adam(m.parameters(),lr=.001);hist=[]
        for step in range(201):
            opt.zero_grad(set_to_none=True);pred,w=m(features);loss=(pred-target).square().mean();loss.backward()
            grad=float(torch.sqrt(sum(q.grad.detach().square().sum() for q in m.parameters() if q.grad is not None)));opt.step()
            if step in (0,10,50,100,200):hist.append({'step':step,'mse':float(loss.detach()),'mean_weights':w.mean((0,2,3)).detach().cpu().tolist(),'gradient_l2':grad})
        overfit[label]=hist
    # RGB-derived regional ambiguity/gain, original development ONLY.
    rows=[]
    with torch.no_grad():
        for s in original_scenes(base,'development'):
            clear,kind=procedural(s['seed'],base['image_size'])
            for meta,rgb,truth in cases(clear,s,base):
                f,c=ext(tensor(rgb,'cuda'));y=torch.from_numpy(truth)[None,None].cuda();oracle=convex_oracle(c,y)
                e=(c-y).abs();d=e[:,:1];lower=(oracle-y).abs()
                gray=clear@np.array([.299,.587,.114]);dy,dx=np.gradient(gray)
                masks={'all':np.ones_like(gray,dtype=bool),'bright':gray>.8,'low_texture':np.hypot(dx,dy)<.01}
                for name,mask in masks.items():
                    if not mask.any():continue
                    mask=torch.from_numpy(mask).cuda()[None,None]
                    rows.append({'scene_id':s['scene_id'],'case_id':meta['case_id'],'kind':kind,'region':name,'pixels':int(mask.sum()),
                        'dark_mae':float(d[mask].mean()),'oracle_mae':float(lower[mask].mean()),
                        'outside_interval_fraction':float((((c.amin(1,keepdim=True)>y)|(c.amax(1,keepdim=True)<y))[mask]).float().mean()),
                        'non_dark_win_fraction':float((e.argmin(1,keepdim=True)[mask]!=0).float().mean())})
    csv_write(out/'regional_bounds.csv',rows)
    write(out/'diagnostics.json',{'toy_learnable_switch':toy,'single_train_case_crop_overfit':overfit,
        'crop_meta':{'scene_id':scene['scene_id'],'case_id':train_case_meta['case_id'],'crop':[64,128,64,128]},'crop_bounds':crop_bounds,
        'scope':'Toy/crop optimization diagnostics only; no generalization claim. Regional diagnostics use original development set.'})
    charge(out,'diagnostics',start);print('DIAGNOSTICS COMPLETE',flush=True)


def train(out,cfg,p):
    if (out/'training_complete.json').exists():return
    start=time.monotonic();ext=FrozenCandidates(read(Path(cfg['previous_directory'])/'calibration.json'));base=p['base']
    with torch.no_grad():
        x,y,_=cache_split(out,base,'train',ext,'cuda');vx,vy,_=cache_split(out,base,'development',ext,'cuda')
    charge(out,'train_feature_precompute',start);summary=[]
    probe=vx[::128].cuda()
    for mode in cfg['candidates']:
        for seed in cfg['seeds']:
            dest=out/f'{mode}_seed{seed}';dest.mkdir(exist_ok=True)
            if (dest/'complete.json').exists():summary.append(read(dest/'complete.json'));continue
            torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed);m=make_model(mode,cfg).cuda()
            opt=torch.optim.Adam(m.parameters(),lr=cfg['learning_rate']);epoch0=0;best=float('inf');hist=[];step=0
            if (dest/'latest.pth').exists():
                s=torch.load(dest/'latest.pth',map_location='cuda',weights_only=False);m.load_state_dict(s['model']);opt.load_state_dict(s['optimizer'])
                epoch0=s['epoch'];best=s['best'];hist=s['history'];step=s['step'];rng.set_state(s['shuffle_rng'].cpu())
                torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all(s['cuda_rng'])
            for epoch in range(epoch0,cfg['epochs']):
                if remaining(out)<cfg['evaluation_reserve_seconds']+30:raise RuntimeError('Budget reserve reached; progress saved. No automatic expansion.')
                start=time.monotonic();m.train();order=torch.randperm(len(x),generator=rng);losses=[];grads=[]
                # Epochs are short after preflight; deadline checkpoint is at epoch boundary.
                for k in range(0,len(x),cfg['batch_size']):
                    ix=order[k:k+cfg['batch_size']];opt.zero_grad(set_to_none=True);pred,w=m(x[ix].cuda());loss=loss_fn(pred,y[ix].cuda(),p['loss'])
                    loss.backward()
                    if k==0:grads.append(float(torch.sqrt(sum(q.grad.detach().square().sum() for q in m.parameters() if q.grad is not None))))
                    opt.step();losses.append(float(loss.detach()));step+=1
                val=dev_loss(m,vx,vy,p['loss'],'cuda');m.eval()
                with torch.no_grad():
                    _,w=m(probe);weight=w.mean((0,2,3)).cpu().tolist();entropy=float((-(w*w.clamp_min(1e-12).log()).sum(1)).mean())
                improved=val<best;best=min(best,val)
                hist.append({'epoch':epoch+1,'step':step,'train_loss':float(np.mean(losses)),'development_loss':val,
                    'probe_mean_weights':weight,'probe_entropy':entropy,'first_batch_gradient_l2':grads[0]})
                state={'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'step':step,'best':best,'history':hist,
                    'shuffle_rng':rng.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                    'mode':mode,'seed':seed,'calibration_sha256':p['candidate_calibration_sha256']}
                torch.save(state,dest/'latest.pth')
                if improved:torch.save(state,dest/'best.pth')
                write(dest/'history.json',hist);charge(out,f'train_{mode}_{seed}_epoch{epoch+1}',start)
                print('TRAIN',mode,seed,epoch+1,val,'weight',weight,'remaining',round(remaining(out),1),flush=True)
            row={'mode':mode,'seed':seed,'best_development_loss':best,'steps':step,'parameters':sum(q.numel() for q in m.parameters())}
            write(dest/'complete.json',row);summary.append(row)
    write(out/'training_complete.json',summary)


def freeze(out,cfg,p):
    if (out/'frozen.json').exists():return
    rows=read(out/'training_complete.json');selected={mode:min((r for r in rows if r['mode']==mode),key=lambda v:v['best_development_loss'])['seed'] for mode in cfg['candidates']}
    write(out/'frozen.json',{'selected_seeds':selected,'checkpoints':{str(q.relative_to(out)):digest(q) for q in out.glob('*_seed*/best.pth')},
          'calibration_sha256':p['candidate_calibration_sha256'],'selection':'original development only','no_test_selection':True})


def evaluate(out,cfg,p):
    start=time.monotonic();frozen=read(out/'frozen.json');old=Path(cfg['previous_directory']);ext=FrozenCandidates(read(old/'calibration.json'))
    weights=torch.tensor(read(old/'global_optimum/fit.json')['weights'],device='cuda')[None,:,None,None]
    models={}
    for mode in cfg['candidates']:
        for seed in cfg['seeds']:
            name=f'{mode}_seed{seed}';path=out/name/'best.pth';assert digest(path)==frozen['checkpoints'][str(path.relative_to(out))]
            m=make_model(mode,cfg).cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval();models[name]=m
    om=ConvexFusion('dynamic').cuda();os=read(old/'frozen.json')['selected_seeds']['dynamic'];path=old/f'dynamic_seed{os}/best.pth'
    assert digest(path)==read(old/'frozen.json')['checkpoint_sha256'][str(path.relative_to(old))]
    om.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);om.eval();models['original_dynamic']=om
    allsummary={}
    for condition in ('nominal','unseen_airlight','display_rgb'):
        if remaining(out)<60:raise RuntimeError('Evaluation budget exhausted; completed outputs retained')
        dest=out/condition;dest.mkdir(exist_ok=True);fig=dest/'figures';fig.mkdir(exist_ok=True)
        if (dest/'summary.json').exists():allsummary[condition]=read(dest/'summary.json');continue
        base=copy.deepcopy(p['base'])
        if condition=='unseen_airlight':base['airlights']=cfg['unseen_airlights']
        if condition=='display_rgb':base['domain']='display_rgb'
        rows=[];oracles=[]
        with torch.no_grad():
            for si,s in enumerate(read(out/'confirmation_manifest.json')):
                clear,_=procedural(s['seed'],base['image_size'])
                for meta,rgb,y in cases(clear,s,base):
                    f,c=ext(tensor(rgb,'cuda'));preds={}
                    for name,m in models.items():
                        pred,w=m(f,c);a=pred[0,0].cpu().numpy();rows.append(record(meta,name,a,y,w[0].cpu().numpy(),clear));preds[name]=a
                    for name,pred in [('global_exact',(weights*c).sum(1,keepdim=True)),('single_dark',c[:,:1])]:
                        a=pred[0,0].cpu().numpy();rows.append(record(meta,name,a,y,rgb=clear));preds[name]=a
                    oracle=convex_oracle(c,torch.from_numpy(y)[None,None].cuda())[0,0].cpu().numpy();oracles.append(record(meta,'offline_oracle',oracle,y,rgb=clear))
                    if si<2 and meta['pattern']=='gradient' and meta['brightness_index']==1 and meta['airlight_index']==0:
                        plot(fig/(meta['case_id']+'.png'),rgb,y,{'global':preds['global_exact'],'old gate':preds['original_dynamic'],
                            'temperature':preds[f'temperature_seed{frozen["selected_seeds"]["temperature"]}'],
                            'student weight':preds[f'student_seed{frozen["selected_seeds"]["student"]}'],'oracle':oracle})
                print('EVAL',condition,s['scene_id'],flush=True)
        csv_write(dest/'per_image.csv',rows);csv_write(dest/'oracle_per_image.csv',oracles)
        result={'metrics':summarize(rows),'oracle_pixel_mae':float(np.mean([r['pixel_mae'] for r in oracles])),
                'role':'primary fresh controlled confirmation' if condition=='nominal' else 'predeclared held-out domain stress; never tune on it'}
        write(dest/'summary.json',result);allsummary[condition]=result
    write(out/'evaluation_complete.json',allsummary);charge(out,'evaluate',start)


def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',required=True,choices=['prepare','preflight','diagnose','train','freeze','evaluate']);args=a.parse_args()
    torch.set_num_threads(1);cfg=read(ROOT/'configs/v2_investigation.json');out=Path(cfg['output_directory'])
    if args.stage=='prepare':prepare(out,cfg);return
    p=read(out/'protocol.json');assert p['config']==cfg
    for name,sha in p['source_sha256'].items():assert digest(ROOT/name)==sha,name
    assert digest(Path(cfg['previous_directory'])/'calibration.json')==p['candidate_calibration_sha256']
    if args.stage=='preflight':preflight(out,cfg,p)
    elif args.stage=='diagnose':diagnostics(out,cfg,p)
    elif args.stage=='train':train(out,cfg,p)
    elif args.stage=='freeze':freeze(out,cfg,p)
    else:evaluate(out,cfg,p)

if __name__=='__main__':main()
