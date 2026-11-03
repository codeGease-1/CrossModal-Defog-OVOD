"""Joint second-stage restricted calibration: two modes, three seeds, no new holdout."""
import argparse,copy,json,hashlib,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.v15.run import read,write,digest
from scripts.learned_correction.run import cache,task_loss,loss_for,atomic_torch_save
from scripts.matched_fusion.run import settings,tensor_digest
from src.models.haze_density.dynamic_prior import FrozenCandidates
from src.models.haze_density.joint_correction import JointCorrection
from scripts.joint_correction.check_model import check


def status(out,stage,**extra):
    old=read(out/'status.json') if (out/'status.json').exists() else {}
    old.update(stage=stage,updated_UTC=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),**extra)
    write(out/'status.json',old)


def charge(out,label,start):
    p=out/'budget.json';b=read(p) if p.exists() else {'stage_wall_seconds':0.,'stages':[]}
    elapsed=time.monotonic()-start;b['stages'].append({'stage':label,'seconds':elapsed});b['stage_wall_seconds']+=elapsed;write(p,b)


def paths():
    return sorted([*ROOT.glob('src/**/*.py'),*ROOT.glob('benchmarks/haze_density/*.py'),
      *ROOT.glob('scripts/joint_correction/*.py'),*ROOT.glob('scripts/matched_fusion/*.py'),
      ROOT/'scripts/learned_correction/run.py',ROOT/'scripts/v15/run.py',ROOT/'scripts/v2/global_optimum.py',ROOT/'configs/joint_correction_20261102.json'])


def setup(out,cfg):
    old=read(Path(cfg['old_pretraining'])/'protocol.json');protection=read(out/'inherited_source_and_protection.json')
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths()}
    if (out/'protocol.json').exists():
        p=read(out/'protocol.json');assert p['config']==cfg and p['source_sha256']==hashes,'Frozen source/config mismatch'
    else:
        for rel,sha in old['source_sha256'].items():
            if (ROOT/rel).exists():assert digest(ROOT/rel)==sha,rel
        p={'config':cfg,'base':old['base'],'scenes':old['scenes'],'source_sha256':hashes,
           'calibration_sha256':digest(cfg['calibration']),'old_protocol_sha256':digest(Path(cfg['old_pretraining'])/'protocol.json'),
           'matched_control_protocol_sha256':digest(Path(cfg['matched_controls'])/'protocol.json'),
           'protected_checkpoints':protection['old_checkpoints'],
           'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__,'GPU':torch.cuda.get_device_name()},
           'base_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'command':sys.argv,
           'supervised_update_opportunity':6912,'fresh_confirmation_scenes':0,
           'data':'48/16 original scenes,1536/512 pairs,original groups disjoint; synthetic H=1-t',
           'user_override':cfg['user_override'],'precision':'FP32 frozen priors/context; AMP networks; no new input/target'}
        write(out/'protocol.json',p)
        for rel in hashes:
            dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
        (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    assert digest(cfg['calibration'])==p['calibration_sha256']
    for f,sha in p['protected_checkpoints'].items():assert digest(f)==sha,f
    return p


def make(mode,cfg,seed,device='cuda'):
    fits=read(Path(cfg['old_pretraining'])/'fusion_weights.json')
    return JointCorrection(fits[f'corrector16_seed{seed}']['weights'],mode=='joint_dynamic').to(device)


def validation(net,x,y,cfg):
    net.eval();totals=[];auxs=[];maes=[]
    with torch.no_grad():
        for i in range(0,len(x),cfg['batch_size']):
            target=y[i:i+cfg['batch_size']].cuda()
            with torch.autocast('cuda',dtype=torch.float16):
                pred,a=net(x[i:i+len(target)].cuda());loss=task_loss(pred,target,cfg)
                aux=loss_for('corrector16',a['candidates'],target,cfg)
            totals.append(float(loss));auxs.append(float(aux));maes.append(float((pred-target).abs().mean()))
    return float(np.mean(totals)),float(np.mean(auxs)),float(np.mean(maes))


def train(out,cfg,p):
    if (out/'training_complete.json').exists():return
    status(out,'precomputing');start=time.monotonic()
    ext=FrozenCandidates(read(cfg['calibration'])).cuda().eval()
    x,y,metas=cache(p,ext,'train');vx,vy,vm=cache(p,ext,'development')
    previous=Path(cfg['matched_controls']);data=read(previous/'data_check.json')
    assert metas==read(previous/'train_manifest.json') and vm==read(previous/'development_manifest.json')
    hashes={'train_features_sha256':tensor_digest(x),'train_target_sha256':tensor_digest(y),
            'development_features_sha256':tensor_digest(vx),'development_target_sha256':tensor_digest(vy)}
    for k,v in hashes.items():assert v==data[k],k
    write(out/'data_check.json',dict(hashes,identical_to_matched_round=True,train_cases=len(x),development_cases=len(vx),no_group_overlap=not({m['group_id'] for m in metas}&{m['group_id'] for m in vm})))
    write(out/'train_manifest.json',metas);write(out/'development_manifest.json',vm);charge(out,'legacy_identical_prior_context_cache',start)
    for seed in cfg['seeds']:
        for mode in cfg['modes']:
            d=out/f'{mode}_seed{seed}';d.mkdir(exist_ok=True)
            if (d/'complete.json').exists():continue
            torch.manual_seed(seed);shuffle=torch.Generator().manual_seed(seed);net=make(mode,cfg,seed)
            net.corrector.load_state_dict(torch.load(Path(cfg['old_pretraining'])/f'corrector16_seed{seed}/best.pth',map_location='cpu',weights_only=False)['model'])
            opt=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate']);scale=torch.amp.GradScaler('cuda')
            epoch0,step,best,hist=0,0,float('inf'),[]
            if (d/'latest.pth').exists():
                saved=torch.load(d/'latest.pth',map_location='cpu',weights_only=False);assert saved['protocol_sha256']==digest(out/'protocol.json')
                net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scale.load_state_dict(saved['scaler'])
                epoch0,step,best,hist=saved['epoch'],saved['step'],saved['best'],saved['history']
                shuffle.set_state(saved['shuffle_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            for epoch in range(epoch0,cfg['epochs']):
                status(out,'training',mode=mode,seed=seed,epoch=epoch+1,epochs=cfg['epochs'],completed_runs=len(list(out.glob('*_seed*/complete.json'))),total_runs=6)
                start=time.monotonic();net.train();losses=[];order=torch.randperm(len(x),generator=shuffle);torch.cuda.reset_peak_memory_stats()
                for j in range(0,len(x),cfg['batch_size']):
                    ix=order[j:j+cfg['batch_size']];target=y[ix].cuda();opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.float16):
                        h,a=net(x[ix].cuda());loss=task_loss(h,target,cfg)+cfg['auxiliary_weight']*loss_for('corrector16',a['candidates'],target,cfg)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss, preserve latest')
                    scale.scale(loss).backward();scale.unscale_(opt);torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
                    scale.step(opt);scale.update();step+=1;losses.append(float(loss.detach()))
                val,aux,mae=validation(net,vx,vy,cfg);improved=val<best;best=min(best,val)
                hist.append({'epoch':epoch+1,'total_data_exposure_epoch':18+epoch+1,'step':step,'train_loss':float(np.mean(losses)),
                             'development_final_task_loss':val,'development_candidate_aux_loss':aux,'development_mae':mae,
                             'seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
                saved={'model':net.state_dict(),'optimizer':opt.state_dict(),'scaler':scale.state_dict(),'epoch':epoch+1,'step':step,'best':best,'history':hist,
                       'shuffle_rng':shuffle.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                       'mode':mode,'seed':seed,'protocol_sha256':digest(out/'protocol.json')}
                atomic_torch_save(saved,d/'latest.pth')
                if improved:atomic_torch_save(saved,d/'best.pth')
                write(d/'history.json',hist);charge(out,f'{mode}_seed{seed}_epoch{epoch+1}',start)
                print('TRAIN',mode,seed,json.dumps(hist[-1]),flush=True)
            write(d/'complete.json',{'mode':mode,'seed':seed,'total_parameters':sum(p.numel() for p in net.parameters()),
                  'second_stage_steps':step,'total_supervised_update_opportunity':3456+step,'best_final_task_loss':best,
                  'candidate_calibration_and_fusion_remain_separate_outputs':True})
            del net,opt;torch.cuda.empty_cache()
    write(out/'training_complete.json',{'all_six_runs_complete':True,'new_confirmation_scenes':0})


def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['check','train','all'],required=True)
    a.add_argument('--output',type=Path,default=ROOT/'experiments/joint_correction_20261102');args=a.parse_args();settings();out=args.output
    cfg=read(ROOT/'configs/joint_correction_20261102.json')
    try:
        if args.stage=='check':
            start=time.monotonic();rows=check(out,cfg);charge(out,'joint_CPU_CUDA_gradient_resume_checks',start);print(json.dumps(rows));return
        assert (out/'fast_checks.json').exists() and (out/'joint_checks.json').exists()
        p=setup(out,cfg);train(out,cfg,p)
        if args.stage=='all':
            from scripts.joint_correction.evaluate import evaluate_and_export
            status(out,'evaluating',completed_runs=6);start=time.monotonic();evaluate_and_export(out,cfg,p);charge(out,'evaluation_export',start)
            from scripts.joint_correction.cost_report import cost_and_report
            status(out,'measuring_cost');start=time.monotonic();cost_and_report(out,cfg,p);charge(out,'cost_and_report',start)
            for path,sha in p['protected_checkpoints'].items():assert digest(path)==sha,path
            write(out/'COMPLETE.json',{'all_stages_complete':True,'new_confirmation_scenes':0,'user_override_honored':True})
            status(out,'complete',completed_runs=6,report='docs/joint_correction_results_20261102.md');print('ALL STAGES COMPLETE',flush=True)
    except Exception as e:
        status(out,'failed',error_type=type(e).__name__,error=str(e));traceback.print_exc();raise


if __name__=='__main__':main()
