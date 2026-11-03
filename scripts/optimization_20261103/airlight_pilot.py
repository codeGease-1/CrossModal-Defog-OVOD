"""A-coverage pilot: frozen calibration/architecture, identical augmented pairs."""
import sys,copy,json,time,hashlib,subprocess,traceback,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.v15.run import procedural,cases,tensor
from scripts.learned_correction.run import task_loss,loss_for,atomic_torch_save,diagnostic
from src.models.haze_density.fast_candidates import FastCandidates
from src.models.haze_density.learned_correction import observed_context,RGBReference
from src.models.haze_density.joint_correction import JointCorrection
from benchmarks.haze_density.metrics import spatial_metrics
OLD=Path('/mnt/CrossModal-Defog-OVOD-joint-correction-20261102/experiments/joint_correction_20261102')
MATCH=Path('/mnt/CrossModal-Defog-OVOD-matched-gate-20261102/experiments/matched_fusion_20261102')
OUT=ROOT/'experiments/optimization_20261103/airlight_pilot'
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def status(stage,**kw):(OUT/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
def init(mode,seed):
    if mode=='joint_dynamic':
        n=JointCorrection([.9,.05,.05],True);path=OLD/'portable_weights'/f'joint_dynamic_seed{seed}.pth';state=read_package(path)['model']
    else:
        n=RGBReference(16,with_priors=mode=='direct_same16');path=MATCH/f'{mode}_seed{seed}/best.pth';state=read_package(path)['model']
    n.load_state_dict(state);return n.cuda(),str(path),digest(path)
def read_package(path):return torch.load(path,map_location='cpu',weights_only=False)
def cache(protocol,ext,split,condition):
    xs=[];ys=[];metas=[]
    for scene in [s for s in protocol['scenes'] if s['split']==split]:
        base=copy.deepcopy(protocol['base'])
        if condition=='augmented_A':
            rng=np.random.default_rng(20261103+scene['seed'])
            base['airlights']=rng.uniform(.55,1.,(2,3)).tolist()
        elif condition=='unseen_A':base['airlights']=[[.55,.60,.65],[.90,.78,.65]]
        elif condition=='display_rgb':base['domain']='display_rgb'
        clear,_=procedural(scene['seed'],base['image_size'])
        with torch.no_grad():
            items=list(cases(clear,scene,base))
            for i in range(0,len(items),8):
                chunk=items[i:i+8];rgb=torch.cat([tensor(im,'cuda') for _,im,_ in chunk]);f,_=ext(rgb);f=torch.cat([f,observed_context(rgb)],1)
                xs.append(f.cpu());ys.append(torch.from_numpy(np.stack([y for _,_,y in chunk]))[:,None])
                for m,im,y in chunk:
                    metas.append(dict(m,airlight=base['airlights'][m['airlight_index']],domain=base['domain'],
                        RGB_sha256=hashlib.sha256(im.tobytes()).hexdigest(),H_sha256=hashlib.sha256(y.tobytes()).hexdigest()))
    return torch.cat(xs),torch.cat(ys),metas
def val(n,caches,cfg):
    n.eval();losses=[];maes=[]
    with torch.no_grad():
        for x,y,_ in caches:
            for i in range(0,len(x),8):
                yy=y[i:i+8].cuda()
                with torch.autocast('cuda',dtype=torch.float16):
                    h,_=n(x[i:i+8].cuda());loss=task_loss(h,yy,cfg)
                losses.append(float(loss));maes.append(float((h-yy).abs().mean()))
    return float(np.mean(losses)),float(np.mean(maes))
def train(mode,seed,trainset,dev,cfg,protocol_hash):
    d=OUT/f'{mode}_seed{seed}';d.mkdir(exist_ok=True)
    if (d/'complete.json').exists():return
    torch.manual_seed(seed);n,path,hsh=init(mode,seed);opt=torch.optim.Adam(n.parameters(),lr=cfg['learning_rate']);scale=torch.amp.GradScaler('cuda')
    gen=torch.Generator().manual_seed(seed);hist=[];epoch0=0;step=0;best=float('inf')
    if (d/'latest.pth').exists():
        s=read_package(d/'latest.pth');assert s['protocol_hash']==protocol_hash
        n.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);scale.load_state_dict(s['scaler'])
        gen.set_state(s['shuffle_rng']);torch.set_rng_state(s['CPU_rng']);torch.cuda.set_rng_state_all(s['CUDA_rng'])
        hist=s['history'];epoch0=s['epoch'];step=s['step'];best=s['best']
    x,y,_=trainset
    for epoch in range(epoch0,cfg['epochs']):
        status('training',mode=mode,seed=seed,epoch=epoch+1,epochs=cfg['epochs'],complete_runs=len(list(OUT.glob('*_seed*/complete.json'))),total_runs=9)
        start=time.monotonic();n.train();losses=[];order=torch.randperm(len(x),generator=gen)
        for i in range(0,len(x),8):
            idx=order[i:i+8];target=y[idx].cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                h,a=n(x[idx].cuda());loss=task_loss(h,target,cfg)
                if mode=='joint_dynamic':loss=loss+cfg['auxiliary_weight']*loss_for('corrector16',a['candidates'],target,cfg)
            assert torch.isfinite(loss)
            scale.scale(loss).backward();scale.unscale_(opt);torch.nn.utils.clip_grad_norm_(n.parameters(),5.)
            scale.step(opt);scale.update();step+=1;losses.append(float(loss.detach()))
        value,mae=val(n,dev,cfg);improved=value<best;best=min(best,value)
        hist.append(dict(epoch=epoch+1,steps=step,train_loss=float(np.mean(losses)),dev_selection_loss=value,dev_selection_MAE=mae,seconds=time.monotonic()-start))
        state=dict(model=n.state_dict(),optimizer=opt.state_dict(),scaler=scale.state_dict(),shuffle_rng=gen.get_state(),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all(),
                   history=hist,epoch=epoch+1,step=step,best=best,protocol_hash=protocol_hash,initial_weight_path=path,initial_weight_sha256=hsh)
        atomic_torch_save(state,d/'latest.pth')
        if improved:atomic_torch_save(state,d/'best.pth')
        (d/'history.json').write_text(json.dumps(hist,indent=2));print('TRAIN',mode,seed,json.dumps(hist[-1]),flush=True)
    save(d/'complete.json',dict(steps=step,epochs=cfg['epochs'],parameters=sum(p.numel() for p in n.parameters()),initial_weight_path=path,initial_weight_sha256=hsh))
    del n,opt;torch.cuda.empty_cache()
def evaluate(cfg,devs):
    status('evaluation');summary={};manifest=[]
    for condition,(x,y,meta) in devs.items():
        d=OUT/'evaluation'/condition;d.mkdir(parents=True,exist_ok=True);rows=[];diags=[]
        for seed in cfg['seeds']:
            for mode in cfg['modes']:
                for stage in ['before','after']:
                    n,_,_=init(mode,seed)
                    if stage=='after':n.load_state_dict(read_package(OUT/f'{mode}_seed{seed}/best.pth')['model'])
                    n.eval()
                    with torch.no_grad():
                        for i in range(0,len(x),8):
                            with torch.autocast('cuda',dtype=torch.float16):h,a=n(x[i:i+8].cuda())
                            preds=h[:,0].float().cpu().numpy();qs=a['candidates'].float().cpu().numpy() if mode=='joint_dynamic' else None
                            for j,pred in enumerate(preds):
                                yy=y[i+j,0].numpy();m=meta[i+j];err=pred-yy
                                rows.append(dict(seed=seed,mode=mode,stage=stage,case_id=m['case_id'],group_id=m['group_id'],pattern=m['pattern'],airlight=m['airlight_index'],brightness=m['brightness_index'],
                                    pixel_mae=float(np.abs(err).mean()),pixel_mse=float((err**2).mean()),signed_bias=float(err.mean()),
                                    global_mae=float(abs(err.mean())),region_spearman=spatial_metrics(1-yy,pred)['region_spearman'],
                                    clear_response=float(pred.mean()) if m['uniform_t']==1. else None))
                                if qs is not None:diags.append(dict(seed=seed,stage=stage,case_id=m['case_id'],**diagnostic(qs[j],yy)))
                    del n;torch.cuda.empty_cache()
        with (d/'per_case.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        save(d/'candidate_diagnostics.json',diags);s={}
        for mode in cfg['modes']:
            for stage in ['before','after']:
                rr=[a for a in rows if a['mode']==mode and a['stage']==stage]
                v={k:float(np.mean([a[k] for a in rr if a[k] is not None])) for k in ['pixel_mae','pixel_mse','signed_bias','global_mae','region_spearman','clear_response']}
                v['by_seed_MAE']={str(seed):float(np.mean([a['pixel_mae'] for a in rr if a['seed']==seed])) for seed in cfg['seeds']}
                s[mode+'_'+stage]=v
        save(d/'summary.json',s);summary[condition]=s
        save(d/'candidate_summary.json',{stage:{k:(np.mean([a[k] for a in diags if a['stage']==stage],axis=0).tolist()) for k in ['candidate_mae','candidate_bias','joint_low','joint_high','outside_fraction','oracle_mae','oracle_mse']} for stage in ['before','after']})
        print('EVAL',condition,json.dumps(s),flush=True)
    save(OUT/'summary.json',summary)
    portable=OUT/'portable_weights';portable.mkdir()
    for seed in cfg['seeds']:
        path=OUT/f'joint_dynamic_seed{seed}'/'best.pth'
        n,_,_=init('joint_dynamic',seed);n.load_state_dict(read_package(path)['model'])
        cal=read(read(OLD/'protocol.json')['config']['calibration'])
        f=portable/f'joint_dynamic_seed{seed}.pth'
        torch.save(dict(model={k:v.cpu() for k,v in n.state_dict().items()},calibration=cal,mode='joint_dynamic',seed=seed,source_checkpoint_sha256=digest(path)),f)
        manifest.append(dict(path=str(f.relative_to(OUT)),sha256=digest(f),bytes=f.stat().st_size));del n
    save(OUT/'portable_manifest.json',manifest)
def main():
    cfg=read(ROOT/'configs/airlight_pilot_20261103.json')
    if (OUT/'COMPLETE.json').exists():print('Already complete');return
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(20261103);start=time.monotonic()
    p=read(OLD/'protocol.json');calpath=Path(p['config']['calibration'])
    paths=[Path(__file__),ROOT/'configs/airlight_pilot_20261103.json']
    protocol=dict(config=cfg,base_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources={str(q.relative_to(ROOT)):digest(q) for q in paths},frozen_calibration_path=str(calpath),frozen_calibration_sha256=digest(calpath),
        environment=dict(torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name()),
        fairness='Each mode/seed starts its registered previous 36-opportunity best checkpoint, then same 1536 augmented pairs and 6 epochs. Optimization losses differ as previously disclosed; pilot only.',
        selection='Equal nominal+unseen_A OLD development task loss. Exposed dev, not confirmation.',
        scope='Isolate A coverage: old calibration remains fixed, no mixture of synthesis domains, no new architecture')
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:
        save(OUT/'protocol.json',protocol)
        snap=OUT/'source';snap.mkdir()
        for q in paths:(snap/q.name).write_bytes(q.read_bytes())
    ext=FastCandidates(read(calpath)).cuda().eval();status('cache_train')
    tr=cache(p,ext,'train','augmented_A');save_train=OUT/'train_manifest.json'
    if not save_train.exists():save(save_train,tr[2])
    dev={}
    for cond in ['nominal','unseen_A','display_rgb']:
        status('cache_development',condition=cond);dev[cond]=cache(p,ext,'development',cond)
        f=OUT/(cond+'_manifest.json')
        if not f.exists():save(f,dev[cond][2])
    assert not ({m['group_id'] for m in tr[2]} & {m['group_id'] for m in dev['nominal'][2]})
    data=dict(train_pairs=len(tr[0]),dev_pairs_each=len(dev['nominal'][0]),groups_disjoint=True,train_features_sha256=hashlib.sha256(tr[0].numpy().tobytes()).hexdigest())
    if (OUT/'data_check.json').exists():assert read(OUT/'data_check.json')==data
    else:save(OUT/'data_check.json',data)
    del ext;torch.cuda.empty_cache()
    for seed in cfg['seeds']:
        for mode in cfg['modes']:train(mode,seed,tr,[dev['nominal'],dev['unseen_A']],cfg,digest(OUT/'protocol.json'))
    evaluate(cfg,dev);save(OUT/'COMPLETE.json',dict(seconds=time.monotonic()-start,new_confirmation_scenes=0,all_nine_runs_complete=True));status('complete')
    print('AIRLIGHT PILOT COMPLETE',time.monotonic()-start,flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():status('failed',error=repr(e))
        traceback.print_exc();raise
