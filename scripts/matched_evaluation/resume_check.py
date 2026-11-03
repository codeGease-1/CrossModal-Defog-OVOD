"""Actual disk/optimizer/scaler/RNG resume check without changing experiment weights."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from scripts.v15.run import read,write
from scripts.matched_fusion.run import settings,make
from scripts.learned_correction.run import task_loss,atomic_torch_save


def main():
    settings();out=ROOT/'experiments/matched_fusion_20261102';cfg=read(out/'protocol.json')['config'];checks=[]
    for mode in ['corrected_gate','rgb16']:
        seed=cfg['seeds'][0];original=out/f'{mode}_seed{seed}'/'latest.pth'
        state=torch.load(original,map_location='cpu',weights_only=False)
        net=make(mode,cfg,seed);net.load_state_dict(state['model'])
        params=list(net.gate.parameters()) if mode.endswith('_gate') else list(net.parameters())
        opt=torch.optim.Adam(params,lr=cfg['learning_rate']);opt.load_state_dict(state['optimizer'])
        scale=torch.amp.GradScaler('cuda');scale.load_state_dict(state['scaler'])
        torch.manual_seed(842);shuffle=torch.Generator().manual_seed(842)
        x=torch.rand(2,11,32,32,device='cuda');y=torch.rand(2,1,32,32,device='cuda')
        def step(m,o,s):
            m.train();o.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):pred=m(x)[0];loss=task_loss(pred,y,cfg)
            s.scale(loss).backward();s.unscale_(o);torch.nn.utils.clip_grad_norm_([v for v in m.parameters() if v.requires_grad],5.)
            s.step(o);s.update()
        step(net,opt,scale)
        saved={'model':net.state_dict(),'optimizer':opt.state_dict(),'scaler':scale.state_dict(),
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'shuffle_rng':shuffle.get_state()}
        path=out/f'disk_resume_{mode}.pth';assert not path.exists();atomic_torch_save(saved,path)
        next_order=torch.randperm(17,generator=shuffle);step(net,opt,scale)
        other=make(mode,cfg,seed);loaded=torch.load(path,map_location='cpu',weights_only=False);other.load_state_dict(loaded['model'])
        pr=list(other.gate.parameters()) if mode.endswith('_gate') else list(other.parameters())
        op=torch.optim.Adam(pr,lr=cfg['learning_rate']);op.load_state_dict(loaded['optimizer'])
        sc=torch.amp.GradScaler('cuda');sc.load_state_dict(loaded['scaler'])
        torch.set_rng_state(loaded['torch_rng']);torch.cuda.set_rng_state_all(loaded['cuda_rng']);shuffle.set_state(loaded['shuffle_rng'])
        assert torch.equal(next_order,torch.randperm(17,generator=shuffle));step(other,op,sc)
        delta=max(float((v-other.state_dict()[k]).abs().max()) for k,v in net.state_dict().items())
        assert delta==0,delta
        checks.append({'mode':mode,'actual_disk_save_load':True,'optimizer_scaler_RNG_restored':True,'max_parameter_difference':delta,
                       'original_latest_read_only':str(original),'smoke_updates_are_not_deployed':True})
        del net,other,opt,op;torch.cuda.empty_cache()
    write(out/'disk_resume_checks.json',checks);print(checks,flush=True)


if __name__=='__main__':main()
