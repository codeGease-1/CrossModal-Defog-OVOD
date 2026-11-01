"""Train-only exact active-set solution of the 3-weight convex quadratic.

Strengthens candidate B, including zero-weight faces unavailable at finite
softmax logits. No test value enters Q, r, the fit, or solver selection.
"""
import itertools
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.v15.run import read,write,digest,procedural,original_scenes,cases,tensor
from src.models.haze_density.dynamic_prior import FrozenCandidates


def solve_simplex(Q,r):
    solutions=[]
    for n in (1,2,3):
        for active in itertools.combinations(range(3),n):
            a=np.array(active);matrix=np.block([[2*Q[np.ix_(a,a)],np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
            rhs=np.r_[2*r[a],1.]
            ans=np.linalg.lstsq(matrix,rhs,rcond=None)[0][:n]
            if np.min(ans)<-1e-8 or abs(ans.sum()-1)>1e-6:continue
            w=np.zeros(3);w[a]=np.maximum(ans,0);w/=w.sum();solutions.append((float(w@Q@w-2*r@w),w))
    return min(solutions,key=lambda x:x[0])[1]


def main():
    out=Path('experiments/v2_20261101');dest=out/'global_optimum';dest.mkdir(exist_ok=True)
    if (dest/'fit.json').exists():return
    protocol=read(out/'protocol.json');cfg=protocol['config'];base=protocol['v15_config'];torch.set_num_threads(1)
    extractor=FrozenCandidates(read(out/'calibration.json'));Q=np.zeros((3,3));r=np.zeros(3);Qg=Q.copy();rg=r.copy();Qc=Q.copy();nc=n=0
    start=time.monotonic()
    with torch.no_grad():
        for scene in original_scenes(base,'train'):
            clear,_=procedural(scene['seed'],base['image_size'])
            for meta,rgb,truth in cases(clear,scene,base):
                _,p=extractor(tensor(rgb,'cuda'));p=p[0].reshape(3,-1).double();y=torch.from_numpy(truth.ravel()).cuda().double()
                q=(p@p.T/p.shape[1]).cpu().numpy();rr=(p@y/p.shape[1]).cpu().numpy();pm=p.mean(1).cpu().numpy();ym=float(y.mean())
                Q+=q;r+=rr;Qg+=np.outer(pm,pm);rg+=pm*ym;n+=1
                if meta['uniform_t']==1. and meta['pattern']=='uniform':Qc+=q;nc+=1
    Q=Q/n+cfg['global_loss_weight']*Qg/n+cfg['clear_loss_extra_weight']*Qc/nc
    r=r/n+cfg['global_loss_weight']*rg/n
    weights=solve_simplex(Q,r)
    budget=read(out/'budget.json');budget['training_seconds']+=time.monotonic()-start;write(out/'budget.json',budget)
    write(dest/'fit.json',{'weights':weights.tolist(),'quadratic_Q':Q.tolist(),'linear_r':r.tolist(),'n_training_cases':n,'n_clear_cases':nc,
          'objective':'dataset pixel MSE + 0.5 image-mean MSE + 0.25 conditional clear MSE; additive y-square constant omitted',
          'solver':'enumerate all seven nonempty simplex faces and solve equality-constrained normal equations',
          'calibration_sha256':digest(out/'calibration.json'),'fitted_on':'V1.5 procedural training originals ONLY',
          'refinement_trigger':'development-only inspection showed near-boundary weights; no confirmation/external result used',
          'elapsed_seconds':time.monotonic()-start})
    print('EXACT GLOBAL WEIGHTS',weights)


if __name__=='__main__':main()
