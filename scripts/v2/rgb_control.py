"""Third and final training variant: RGB-only capacity-matched control.

This branch is diagnostic and never automatically enters the main pipeline.
It shares the refactored student's contextual trunk but not its prior inputs
or convex restriction. No old teacher checkpoint is loaded.
"""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.v15.run import read,write,digest
from scripts.v2.run import cache_split,loss_fn,dev_loss
from scripts.v2.investigate import charge,remaining
from src.models.haze_density.dynamic_prior import FrozenCandidates
from src.models.haze_density.student_weight_gate import StudentWeightGate


class RGBControl(nn.Module):
    def __init__(self):
        super().__init__();source=StudentWeightGate()
        self.encoder=source.encoder;self.multiscale=source.multiscale;self.fusion=source.fusion;self.up=source.up
        self.head=nn.Sequential(nn.Conv2d(16+3,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1),nn.Sigmoid())

    def forward(self,features,candidates=None):
        rgb=features[:,:3];f=self.encoder(rgb);a,b,c,_=self.multiscale(f,return_separate=True);f=self.up(self.fusion(a,b,c))
        if f.shape[-2:]!=rgb.shape[-2:]:f=F.interpolate(f,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
        return self.head(torch.cat([f,rgb],1)),None


def main():
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--register',action='store_true');args=a.parse_args()
    torch.set_num_threads(1);cfg=read('configs/v2_investigation.json');out=Path(cfg['output_directory']);dest=out/'rgb_control';dest.mkdir(exist_ok=True)
    plan=dest/'plan.json'
    if not plan.exists():write(plan,{'seeds':cfg['seeds'],'epochs':cfg['epochs'],'lr':cfg['learning_rate'],
        'source_sha256':digest(__file__),'inputs':'RGB only; no candidate/raw-cue features',
        'scope':'Third/final follow-up training variant, diagnostic only; no adoption or test-based tuning',
        'trigger':'Need to distinguish value of contextual neural regression from value of prior-constrained weighting; registered before any new confirmation results',
        'differences':'Same feature trunk and loss; head emits sigmoid H with direct RGB skip. Different output restriction and initialization.',
        'budget':'Same shared 3000s follow-up ledger, no extra allocation'})
    assert read(plan)['source_sha256']==digest(__file__)
    if args.register:print('RGB control plan registered');return
    if (dest/'frozen.json').exists():return
    start=time.monotonic();p=read(out/'protocol.json');ext=FrozenCandidates(read(Path(cfg['previous_directory'])/'calibration.json'))
    with torch.no_grad():
        x,y,_=cache_split(out,p['base'],'train',ext,'cuda');vx,vy,_=cache_split(out,p['base'],'development',ext,'cuda')
    # Discard six channels explicitly so candidates cannot leak through the network.
    x=x[:,:3].contiguous();vx=vx[:,:3].contiguous();charge(out,'rgb_feature_precompute',start);summary=[]
    for seed in cfg['seeds']:
        d=dest/f'seed{seed}';d.mkdir(exist_ok=True)
        if (d/'complete.json').exists():summary.append(read(d/'complete.json'));continue
        if remaining(out)<cfg['evaluation_reserve_seconds']+100:raise RuntimeError('Insufficient shared budget for RGB control; report blocked control, no expansion')
        torch.manual_seed(seed);m=RGBControl().cuda();opt=torch.optim.Adam(m.parameters(),lr=cfg['learning_rate']);rng=torch.Generator().manual_seed(seed)
        best=float('inf');hist=[];epoch0=step=0
        if (d/'latest.pth').exists():
            s=torch.load(d/'latest.pth',map_location='cuda',weights_only=False);m.load_state_dict(s['model']);opt.load_state_dict(s['optimizer'])
            epoch0=s['epoch'];step=s['step'];best=s['best'];hist=s['history'];rng.set_state(s['shuffle_rng'].cpu());torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all(s['cuda_rng'])
        for epoch in range(epoch0,cfg['epochs']):
            if remaining(out)<cfg['evaluation_reserve_seconds']+30:raise RuntimeError('Shared budget reserve reached; progress retained')
            start=time.monotonic();m.train();order=torch.randperm(len(x),generator=rng);losses=[]
            for k in range(0,len(x),cfg['batch_size']):
                ix=order[k:k+cfg['batch_size']];opt.zero_grad(set_to_none=True);pred,_=m(x[ix].cuda());loss=loss_fn(pred,y[ix].cuda(),p['loss']);loss.backward();opt.step();losses.append(float(loss.detach()));step+=1
            val=dev_loss(m,vx,vy,p['loss'],'cuda');improved=val<best;best=min(best,val)
            hist.append({'epoch':epoch+1,'step':step,'train_loss':float(np.mean(losses)),'development_loss':val})
            state={'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'step':step,'best':best,'history':hist,
                'shuffle_rng':rng.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'seed':seed}
            torch.save(state,d/'latest.pth')
            if improved:torch.save(state,d/'best.pth')
            write(d/'history.json',hist);charge(out,f'rgb_{seed}_epoch{epoch+1}',start)
            print('RGB',seed,epoch+1,val,'remaining',round(remaining(out),1),flush=True)
        r={'seed':seed,'best_development_loss':best,'steps':step,'parameters':sum(q.numel() for q in m.parameters())};write(d/'complete.json',r);summary.append(r)
    write(dest/'frozen.json',{'selected_seed':min(summary,key=lambda r:r['best_development_loss'])['seed'],'all_seeds':summary,
        'checkpoint_sha256':{str(q.relative_to(out)):digest(q) for q in dest.glob('seed*/best.pth')},'selection':'original development only'})

if __name__=='__main__':main()
