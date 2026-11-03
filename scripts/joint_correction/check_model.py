"""Behavior checks for joint gradients, candidate independence and actual disk resume."""
import copy,inspect
from pathlib import Path
import torch
from scripts.v15.run import read,write
from src.models.haze_density.learned_correction import RestrictedCorrector
from src.models.haze_density.joint_correction import JointCorrection,InferencePipeline
from scripts.learned_correction.run import task_loss,loss_for,atomic_torch_save


def check(out,cfg):
    rows=[];fits=read(Path(cfg['old_pretraining'])/'fusion_weights.json');weights=fits['corrector16_seed42']['weights']
    for device in ['cpu','cuda']:
        for mode in cfg['modes']:
            torch.manual_seed(842);net=JointCorrection(weights,mode=='joint_dynamic').to(device)
            old=RestrictedCorrector(16).to(device)
            state=torch.load(Path(cfg['old_pretraining'])/'corrector16_seed42/best.pth',map_location='cpu',weights_only=False)['model']
            old.load_state_dict(state);net.corrector.load_state_dict(state);old.eval();net.eval()
            x=torch.rand(2,11,32,32,device=device);y=torch.rand(2,1,32,32,device=device)
            with torch.no_grad(),torch.autocast(device,dtype=torch.float16,enabled=device=='cuda'):
                pred,aux=net(x);expected=old(x)[0];assert torch.equal(expected,aux['candidates'])
                changed=x.clone();changed[:,4:5]=(changed[:,4:5]+.01).clamp_max(1)
                other=net(changed)[1]['candidates'];assert torch.equal(other[:,[0,2]],aux['candidates'][:,[0,2]])
                assert torch.all(other[:,1:2]>=aux['candidates'][:,1:2])
                assert float((other[:,1:2]-aux['candidates'][:,1:2]).abs().sum())>0
                assert torch.allclose(aux['weights'].sum(1),torch.ones_like(pred[:,0]),atol=1e-6)
                assert torch.all(pred>=expected.amin(1,keepdim=True)-1e-6) and torch.all(pred<=expected.amax(1,keepdim=True)+1e-6)
            opt=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate']);scale=torch.amp.GradScaler(device,enabled=device=='cuda')
            def step(m,o,s):
                m.train();o.zero_grad(set_to_none=True)
                with torch.autocast(device,dtype=torch.float16,enabled=device=='cuda'):
                    h,a=m(x);loss=task_loss(h,y,cfg)+cfg['auxiliary_weight']*loss_for('corrector16',a['candidates'],y,cfg)
                s.scale(loss).backward();s.unscale_(o);torch.nn.utils.clip_grad_norm_(m.parameters(),5.);s.step(o);s.update()
            step(net,opt,scale)
            assert any(v.grad is not None and float(v.grad.abs().sum())>0 for v in net.corrector.trunk.parameters())
            assert any(v.grad is not None and float(v.grad.abs().sum())>0 for v in net.corrector.head.parameters())
            if net.dynamic:assert any(v.grad is not None and float(v.grad.abs().sum())>0 for v in net.gate.parameters())
            save={'model':net.state_dict(),'optimizer':opt.state_dict(),'scaler':scale.state_dict(),'cpu_rng':torch.get_rng_state()}
            if device=='cuda':save['cuda_rng']=torch.cuda.get_rng_state_all()
            path=out/f'check_resume_{mode}_{device}.pth';assert not path.exists();atomic_torch_save(save,path)
            step(net,opt,scale)
            loaded=torch.load(path,map_location='cpu',weights_only=False);other=JointCorrection(weights,net.dynamic).to(device);other.load_state_dict(loaded['model'])
            op=torch.optim.Adam(other.parameters(),lr=cfg['learning_rate']);op.load_state_dict(loaded['optimizer'])
            sc=torch.amp.GradScaler(device,enabled=device=='cuda');sc.load_state_dict(loaded['scaler']);torch.set_rng_state(loaded['cpu_rng'])
            if device=='cuda':torch.cuda.set_rng_state_all(loaded['cuda_rng'])
            step(other,op,sc)
            delta=max(float((v-other.state_dict()[k]).abs().max()) for k,v in net.state_dict().items());assert delta==0,delta
            pipe=InferencePipeline(net.eval(),read(cfg['calibration'])).to(device)
            with torch.no_grad():a=pipe(torch.rand(1,3,65,65,device=device))
            assert a.shape==(1,1,65,65) and torch.isfinite(a).all()
            rows.append({'mode':mode,'device':device,'initial_candidates_identical':True,'each_candidate_uses_own_prior':True,
                         'joint_corrector_gradients':True,'convex_outputs':True,'actual_disk_resume_max_difference':delta,
                         'inference_signature':str(inspect.signature(pipe.forward)),'total_network_parameters':sum(p.numel() for p in net.parameters())})
    write(out/'joint_checks.json',rows);return rows
