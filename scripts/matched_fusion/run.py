"""Finite, resumable matched-opportunity gate/RGB experiment on existing development data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.v15.run import read,write,digest,csv_write
from scripts.learned_correction.run import cache,task_loss,atomic_torch_save,diagnostic
from src.models.haze_density.dynamic_prior import FrozenCandidates
from src.models.haze_density.learned_correction import RestrictedCorrector,RGBReference
from src.models.haze_density.matched_fusion import MatchedFusion
from scripts.matched_fusion.test_contracts import check


def settings():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def source_paths():
    return sorted([*ROOT.glob('src/**/*.py'),*ROOT.glob('benchmarks/haze_density/*.py'),
       *ROOT.glob('scripts/matched_fusion/*.py'),ROOT/'scripts/v15/run.py',
       ROOT/'scripts/v2/global_optimum.py',ROOT/'scripts/learned_correction/run.py',
       ROOT/'configs/matched_fusion_20261102.json'])


def setup(out,cfg):
    old=Path(cfg['old_output']);prior=read(old/'protocol.json')
    assert digest(cfg['calibration'])==prior['calibration_sha256']
    for rel,sha in prior['source_sha256'].items():
        if (ROOT/rel).exists():assert digest(ROOT/rel)==sha,rel
    hashes={str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        p=read(protocol_path);assert p['config']==cfg and p['source_sha256']==hashes
        for rel,sha in p['protected_previous_checkpoints'].items():assert digest(rel)==sha,rel
        return p
    p={'config':cfg,'base':prior['base'],'scenes':prior['scenes'],
       'source_sha256':hashes,'old_protocol_sha256':digest(old/'protocol.json'),
       'calibration_sha256':digest(cfg['calibration']),
       'protected_previous_checkpoints':{str(q):digest(q) for q in old.glob('*_seed*/*.pth')},
       'old_fusion_sha256':digest(old/'fusion_weights.json'),
       'base_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
                      'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},
       'command':sys.argv,'confirmation_state':'No fresh confirmation; old development only',
       'supervised_exposures_per_final_model':36*1536,'updates_opportunity_per_final_model':36*192,
       'precompute':'RAM only, train and development separate; regenerate deterministically on resume',
       'scope':'Procedural structural screening; no natural fog or real geographic generalization claim'}
    write(protocol_path,p)
    for rel in hashes:
        dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
    (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    return p


def charge(out,label,seconds):
    path=out/'budget.json';b=read(path) if path.exists() else {'stage_wall_seconds':0.,'stages':[]}
    b['stages'].append({'stage':label,'seconds':seconds});b['stage_wall_seconds']+=seconds;write(path,b)


def corrector(cfg,seed):
    c=RestrictedCorrector(16).cuda()
    c.load_state_dict(torch.load(Path(cfg['old_output'])/f'corrector16_seed{seed}'/'best.pth',map_location='cpu',weights_only=False)['model'])
    return c.eval()


def make(mode,cfg,seed):
    if mode.endswith('_gate'):return MatchedFusion(corrector(cfg,seed),mode=='corrected_gate',cfg['gate_hidden']).cuda()
    return RGBReference(32 if mode=='rgb32' else 16,with_priors=mode=='direct_same16').cuda()


def prepare(net,x,cfg):
    inputs=torch.empty(len(x),19,*x.shape[-2:]);qs=torch.empty(len(x),3,*x.shape[-2:])
    with torch.no_grad():
        for i in range(0,len(x),cfg['batch_size']):
            f=x[i:i+cfg['batch_size']].cuda()
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp']):
                z,q=net.prepare(f)
                if i==0:assert torch.equal(q,net.corrector(f)[0]),'Candidate reconstruction changed'
            inputs[i:i+len(f)].copy_(z.cpu());qs[i:i+len(f)].copy_(q.cpu())
    return inputs,qs


def tensor_digest(x):
    h=hashlib.sha256()
    for i in range(0,len(x),8):h.update(x[i:i+8].numpy().tobytes())
    return h.hexdigest()


def val(net,mode,x,y,cfg,prepared=None):
    net.eval();losses=[];maes=[]
    with torch.no_grad():
        for i in range(0,len(x),cfg['batch_size']):
            target=y[i:i+cfg['batch_size']].cuda()
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp']):
                if prepared is None:pred,_=net(x[i:i+len(target)].cuda())
                else:
                    a,q=prepared
                    candidates=q[i:i+len(target)] if mode=='corrected_gate' else x[i:i+len(target),3:6]
                    pred,_=net.from_prepared(a[i:i+len(target)].cuda(),candidates.cuda())
                loss=task_loss(pred,target,cfg)
            losses.append(float(loss));maes.append(float((pred-target).abs().mean()))
    return float(np.mean(losses)),float(np.mean(maes))


def train_one(out,cfg,p,seed,mode,x,y,vx,vy,prepared=None,vprepared=None):
    d=out/f'{mode}_seed{seed}';d.mkdir(exist_ok=True)
    if (d/'complete.json').exists():return
    torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed)
    net=make(mode,cfg,seed)
    parameters=list(net.gate.parameters()) if mode.endswith('_gate') else list(net.parameters())
    optimizer=torch.optim.Adam(parameters,lr=cfg['learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'])
    end=cfg['gate_epochs'] if mode.endswith('_gate') else cfg['total_RGB_epochs']
    start_epoch,step,best,hist=0,0,float('inf'),[]
    previous=None
    if (d/'latest.pth').exists():previous=d/'latest.pth'
    elif mode in cfg['continued_controls']:
        previous=Path(cfg['old_output'])/f'{mode}_seed{seed}'/'latest.pth'
        carry=torch.load(previous.with_name('best.pth'),map_location='cpu',weights_only=False)
        carry['protocol_sha256']=digest(out/'protocol.json');carry['carried_from']=str(previous.with_name('best.pth'))
        atomic_torch_save(carry,d/'best.pth')
    if previous is not None:
        state=torch.load(previous,map_location='cpu',weights_only=False)
        if previous.parent==d:assert state['protocol_sha256']==digest(out/'protocol.json')
        net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start_epoch,step,best,hist=state['epoch'],state['step'],state['best'],state['history']
        rng.set_state(state['shuffle_rng'].cpu());torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    for epoch in range(start_epoch,end):
        started=time.monotonic();net.train();losses=[];order=torch.randperm(len(x),generator=rng)
        torch.cuda.reset_peak_memory_stats()
        for j in range(0,len(x),cfg['batch_size']):
            ix=order[j:j+cfg['batch_size']];target=y[ix].cuda();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['amp']):
                if prepared is None:pred,_=net(x[ix].cuda())
                else:
                    a,q=prepared;candidates=q[ix] if mode=='corrected_gate' else x[ix,3:6]
                    pred,_=net.from_prepared(a[ix].cuda(),candidates.cuda())
                loss=task_loss(pred,target,cfg)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; preserve latest checkpoint')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters,5.);scaler.step(optimizer);scaler.update()
            losses.append(float(loss.detach()));step+=1
        vl,mae=val(net,mode,vx,vy,cfg,vprepared)
        improved=vl<best;best=min(best,vl)
        row={'epoch':epoch+1,'step':step,'train_loss':float(np.mean(losses)),
             'development_loss':vl,'development_pixel_mae':mae,
             'seconds':time.monotonic()-started,'train_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        hist.append(row)
        state={'model':net.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
               'epoch':epoch+1,'step':step,'best':best,'history':hist,'shuffle_rng':rng.get_state(),
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
               'mode':mode,'seed':seed,'protocol_sha256':digest(out/'protocol.json')}
        atomic_torch_save(state,d/'latest.pth')
        if improved:atomic_torch_save(state,d/'best.pth')
        write(d/'history.json',hist);charge(out,f'{mode}_seed{seed}_epoch{epoch+1}',time.monotonic()-started)
        print('TRAIN',mode,seed,json.dumps(row),flush=True)
    if mode.endswith('_gate'):
        old=corrector(cfg,seed)
        assert all(torch.equal(v,old.state_dict()[k]) for k,v in net.corrector.state_dict().items())
    write(d/'complete.json',{'mode':mode,'seed':seed,'parameters_total':sum(v.numel() for v in net.parameters()),
           'parameters_trainable_this_stage':sum(v.numel() for v in net.parameters() if v.requires_grad),
           'steps_this_network':step,'total_supervised_updates_opportunity':step+(3456 if mode.endswith('_gate') else 0),
           'best_development_loss':best,'frozen_corrector_unchanged':True if mode.endswith('_gate') else None})
    del net,optimizer;torch.cuda.empty_cache()


def train(out,cfg,p):
    start=time.monotonic();ext=FrozenCandidates(read(cfg['calibration'])).cuda().eval()
    x,y,meta=cache(p,ext,'train');vx,vy,vm=cache(p,ext,'development')
    old=Path(cfg['old_output']);assert meta==read(old/'train_manifest.json');assert vm==read(old/'development_manifest.json')
    assert not ({m['group_id'] for m in meta}&{m['group_id'] for m in vm})
    write(out/'train_manifest.json',meta);write(out/'development_manifest.json',vm)
    write(out/'data_check.json',{'same_original_case_metadata_as_previous':True,'train_cases':len(x),'development_cases':len(vx),
          'train_originals':len({m['group_id'] for m in meta}),'development_originals':len({m['group_id'] for m in vm}),
          'group_overlap':False,'train_features_sha256':tensor_digest(x),'train_target_sha256':tensor_digest(y),
          'development_features_sha256':tensor_digest(vx),'development_target_sha256':tensor_digest(vy),
          'target':'H=1-t; generated scalar transmission, no old teacher scores','storage':'RAM only; no disk cache'})
    charge(out,'shared_RGB_prior_context_precompute_and_data_checks',time.monotonic()-start)
    for seed in cfg['seeds']:
        gates=['raw_gate','corrected_gate']
        if not all((out/f'{mode}_seed{seed}'/'complete.json').exists() for mode in gates):
            start=time.monotonic();net=make('corrected_gate',cfg,seed).eval()
            a,q=prepare(net,x,cfg);va,vq=prepare(net,vx,cfg)
            qhash=tensor_digest(q);vhash=tensor_digest(vq)
            write(out/f'candidate_cache_seed{seed}.json',{'train_corrected_sha256':qhash,'development_corrected_sha256':vhash,
                  'prepared_input_channels':19,'same_inputs_both_gate_arms':True,'frozen_checkpoint_sha256':digest(old/f'corrector16_seed{seed}'/'best.pth')})
            del net;charge(out,f'frozen_feature_candidate_precompute_seed{seed}',time.monotonic()-start)
            for mode in gates:train_one(out,cfg,p,seed,mode,x,y,vx,vy,(a,q),(va,vq))
            assert tensor_digest(q)==qhash and tensor_digest(vq)==vhash
            del a,q,va,vq
        for mode in ['rgb16',*cfg['continued_controls']]:train_one(out,cfg,p,seed,mode,x,y,vx,vy)
    for rel,sha in p['protected_previous_checkpoints'].items():assert digest(rel)==sha,rel
    write(out/'training_complete.json',{'all_modes':cfg['new_modes']+cfg['continued_controls'],'seeds':cfg['seeds'],
          'old_checkpoints_unchanged':True,'new_confirmation_scenes':0})
    print('ALL TRAINING COMPLETE',flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['check','train'],required=True)
    a.add_argument('--config',type=Path,default=ROOT/'configs/matched_fusion_20261102.json')
    a.add_argument('--output',type=Path,default=ROOT/'experiments/matched_fusion_20261102')
    args=a.parse_args();settings();cfg=read(args.config)
    if args.stage=='check':
        started=time.monotonic();checks=[]
        for device in ['cpu','cuda']:
            checks.append(check(Path(cfg['old_output'])/'corrector16_seed42/best.pth',read(cfg['calibration']),device))
        write(args.output/'contract_checks.json',checks);charge(args.output,'CPU_CUDA_contract_and_resume_checks',time.monotonic()-started)
        print(json.dumps(checks),flush=True)
    else:
        assert (args.output/'contract_checks.json').exists(),'Run meaningful checks first'
        p=setup(args.output,cfg);train(args.output,cfg,p)


if __name__=='__main__':main()
