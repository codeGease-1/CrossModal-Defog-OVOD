"""One predeclared seed-42 loss ablation; never substitutes for 3-seed evidence.

Same shuffled main batches as the unranked gate. Extra same-original, same-A,
same-brightness uniform pairs only supply a margin ranking loss (weight .1).
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import time
import numpy as np
import torch
from scripts.v15.run import read,write,digest
from scripts.v2.run import cache_split,loss_fn,dev_loss
from src.models.haze_density.dynamic_prior import FrozenCandidates,ConvexFusion


def main():
    torch.set_num_threads(1);out=Path('experiments/v2_20261101');protocol=read(out/'protocol.json');cfg=protocol['config'];base=protocol['v15_config']
    dest=out/'ranking_ablation';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return
    if read(out/'oracle_development.json')['pilot_only']:
        write(dest/'complete.json',{'status':'NOT RUN: majority bound limitation, no objective expansion'});return
    extractor=FrozenCandidates(read(out/'calibration.json'))
    with torch.no_grad():x,y,_=cache_split(out,base,'train',extractor,'cuda');vx,vy,_=cache_split(out,base,'development',extractor,'cuda')
    torch.manual_seed(42);model=ConvexFusion('dynamic',cfg['hidden']).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr_dynamic'])
    shuffle=torch.Generator().manual_seed(42);pair_rng=torch.Generator().manual_seed(4042);epoch0=0;best=float('inf');history=[]
    if (dest/'latest.pth').exists():
        s=torch.load(dest/'latest.pth',map_location='cuda',weights_only=False);assert s['calibration_sha256']==digest(out/'calibration.json')
        model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);epoch0=s['epoch'];best=s['best'];history=s['history']
        shuffle.set_state(s['shuffle_rng'].cpu());pair_rng.set_state(s['pair_rng'].cpu());torch.set_rng_state(s['torch_rng'].cpu())
    for epoch in range(epoch0,cfg['epochs']):
        budget=read(out/'budget.json')
        if budget['training_seconds']+budget['calibration_seconds']>cfg['training_budget_seconds']-30:raise RuntimeError('No ranking-ablation budget; saved progress retained')
        started=time.monotonic();model.train();order=torch.randperm(len(x),generator=shuffle);losses=[]
        for k in range(0,len(x),cfg['batch_size']):
            ix=order[k:k+cfg['batch_size']];optimizer.zero_grad(set_to_none=True);p,_=model(x[ix].cuda());target=y[ix].cuda();loss=loss_fn(p,target,cfg)
            # Uniform entries 0..4 of each 8-case block; all pairs share original, A and brightness.
            block=(int(ix[0])//8)*8;levels=torch.randperm(5,generator=pair_rng)[:2].sort().values
            pair=torch.tensor([block+int(levels[0]),block+int(levels[1])]);pp,_=model(x[pair].cuda())
            strength=y[pair].mean((1,2,3)).cuda();predstrength=pp.mean((1,2,3))
            rank=torch.relu(.5*(strength[1]-strength[0])-(predstrength[1]-predstrength[0]))
            total=loss+.1*rank;total.backward();optimizer.step();losses.append(float(total.detach()))
        validation=dev_loss(model,vx,vy,cfg,'cuda');budget['training_seconds']+=time.monotonic()-started;write(out/'budget.json',budget)
        improved=validation<best;best=min(best,validation);history.append({'epoch':epoch+1,'development_primary_loss':validation,'training_total_loss':float(np.mean(losses))})
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'best':best,'history':history,
               'shuffle_rng':shuffle.get_state(),'pair_rng':pair_rng.get_state(),'torch_rng':torch.get_rng_state(),'calibration_sha256':digest(out/'calibration.json')}
        torch.save(state,dest/'latest.pth')
        if improved:torch.save(state,dest/'best.pth')
        write(dest/'history.json',history);print('RANK_ABLATION',epoch+1,validation,flush=True)
    base_loss=read(out/'dynamic_seed42/complete.json')['best_development_loss']
    write(dest/'complete.json',{'seed':42,'ranking_weight':.1,'best_development_primary_loss':best,'no_rank_seed42_development_loss':base_loss,
          'development_prefers_ranking':best<base_loss,'adoption':'One seed ablation only; no automatic promotion into main flow',
          'checkpoint_sha256':digest(dest/'best.pth')})


if __name__=='__main__':main()
