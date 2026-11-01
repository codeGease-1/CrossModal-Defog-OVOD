"""Disposable minimal optimization and latency check before bounded training.

Deliberately not named profile.py: that would shadow Python's stdlib module.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import time
import torch
from scripts.v15.run import read,write
from src.models.haze_density.dynamic_prior import ConvexFusion

torch.set_num_threads(1)
out=Path('experiments/v2_20261101');results={};total=0.
for mode in ('global','dynamic'):
    torch.manual_seed(42);model=ConvexFusion(mode).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    x=torch.rand(8,9,256,256,device='cuda');target=torch.rand(8,1,256,256,device='cuda')
    times=[]
    for i in range(12):
        torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        pred,w=model(x);loss=(pred-target).square().mean();loss.backward();optimizer.step();torch.cuda.synchronize()
        elapsed=time.perf_counter()-start;total+=elapsed
        if i>=2:times.append(elapsed)
        assert torch.isfinite(loss)
    results[mode]={'median_step_seconds':sorted(times)[len(times)//2],
                   'parameters':sum(p.numel() for p in model.parameters()),'batch_size':8,
                   'steps':12,'purpose':'synthetic disposable features, NOT an accuracy experiment'}
budget=read(out/'budget.json');budget['training_seconds']+=total;write(out/'budget.json',budget)
write(out/'smoke.json',results);print(results)
