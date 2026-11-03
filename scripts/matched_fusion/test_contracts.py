"""Behavioral checks using actual frozen checkpoints, including saved resume."""
import copy
import inspect
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from src.models.haze_density.learned_correction import RestrictedCorrector,RGBReference
from src.models.haze_density.matched_fusion import MatchedFusion,RGBPipeline


def check(checkpoint, calibration, device='cuda'):
    torch.manual_seed(842); torch.set_num_threads(1)
    c=RestrictedCorrector(16).to(device)
    c.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
    c.eval();m=MatchedFusion(c).to(device)
    checks=[]
    for size in [32,65,256]:
        f=torch.rand(1,11,size,size,device=device)
        with torch.no_grad(),torch.autocast(device,dtype=torch.float16,enabled=device=='cuda'):
            expected=c(f)[0]; inputs,actual=m.prepare(f);pred,aux=m(f)
        delta=float((expected-actual).abs().max());assert delta==0,delta
        assert torch.allclose(aux['weights'].sum(1),torch.ones_like(pred[:,0]),atol=1e-6)
        assert torch.all(pred>=actual.amin(1,keepdim=True)-1e-6)
        assert torch.all(pred<=actual.amax(1,keepdim=True)+1e-6)
        checks.append({'size':size,'candidate_max_difference':delta})
    f=torch.rand(2,11,32,32,device=device);y=torch.rand(2,1,32,32,device=device)
    with torch.no_grad():inputs,q=m.prepare(f)
    before={k:v.detach().clone() for k,v in c.state_dict().items()}
    opt=torch.optim.Adam(m.gate.parameters(),lr=.001)
    scaler=torch.amp.GradScaler(device,enabled=device=='cuda')
    def step(net,optimizer,scale):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device,dtype=torch.float16,enabled=device=='cuda'):
            p,_=net.from_prepared(inputs,q);loss=(p-y).square().mean()
        scale.scale(loss).backward();scale.step(optimizer);scale.update()
    step(m,opt,scaler)
    assert all(p.grad is None and not p.requires_grad for p in c.parameters())
    assert all(torch.equal(before[k],v) for k,v in c.state_dict().items())
    assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in m.gate.parameters())
    m.train();assert not c.training
    state=copy.deepcopy({'model':m.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict()})
    step(m,opt,scaler)
    other=MatchedFusion(copy.deepcopy(c)).to(device);other.load_state_dict(state['model'])
    op=torch.optim.Adam(other.gate.parameters(),lr=.001);op.load_state_dict(state['optimizer'])
    scale=torch.amp.GradScaler(device,enabled=device=='cuda');scale.load_state_dict(state['scaler'])
    step(other,op,scale)
    resume_delta=max(float((v-other.state_dict()[k]).abs().max()) for k,v in m.state_dict().items())
    assert resume_delta==0,resume_delta
    pipeline=RGBPipeline(m,calibration,True).to(device).eval()
    with torch.no_grad():p=pipeline(torch.rand(1,3,64,64,device=device))
    assert p.shape==(1,1,64,64) and torch.isfinite(p).all()
    rgb=RGBReference(16).to(device).eval();x=torch.rand(1,11,32,32,device=device)
    changed=x.clone();changed[:,3:]=torch.randn_like(changed[:,3:])*10
    with torch.no_grad():assert torch.equal(rgb(x)[0],rgb(changed)[0])
    return {'candidate_equivalence':checks,'frozen_weights_unchanged':True,
            'only_gate_gradients':True,'convex_outputs':True,'optimizer_resume_max_delta':resume_delta,
            'RGB16_ignores_derived_channels':True,'inference_signature':str(inspect.signature(pipeline.forward)),
            'device':device,'parameters_total':sum(p.numel() for p in m.parameters()),
            'parameters_trainable':sum(p.numel() for p in m.parameters() if p.requires_grad)}
