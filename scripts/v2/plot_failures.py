"""Reconstruct actual worst confirmation cases, with a common [0,1] scale."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.v15.run import read, procedural, cases, tensor
from scripts.v2.run import model_set
from src.models.haze_density.dynamic_prior import FrozenCandidates, convex_oracle

def main():
    torch.set_num_threads(1)
    out=Path('experiments/v2_20261101');p=read(out/'protocol.json');frozen=read(out/'frozen.json')
    models=model_set(out,p['config'],'cuda');extractor=FrozenCandidates(read(out/'calibration.json'))
    selected=read(out/'confirmation/failure_cases.json')[:2]
    scenes={s['scene_id']:s for s in read(out/'scene_manifest.json')}
    with torch.no_grad():
        for fail in selected:
            s=scenes[fail['scene_id']];clear,_=procedural(s['seed'],p['v15_config']['image_size'])
            for meta,rgb,y in cases(clear,s,p['v15_config']):
                if meta['case_id']!=fail['case_id']:continue
                features,candidates=extractor(tensor(rgb,'cuda'))
                pred,w=models[f'dynamic_seed{frozen["selected_seeds"]["dynamic"]}'](features,candidates)
                panels=[('H target',y)]+[(f'Candidate {i}',candidates[0,i].cpu().numpy()) for i in range(3)]
                panels += [('Dynamic',pred[0,0].cpu().numpy()),('Offline oracle',convex_oracle(candidates,tensor(y[:,:,None],'cuda')).cpu().numpy()[0,0])]
                panels += [(f'Weight {i}',w[0,i].cpu().numpy()) for i in range(3)]
                fig,axes=plt.subplots(2,5,figsize=(16,6));axes=axes.ravel();axes[0].imshow(rgb);axes[0].set_title('RGB')
                for ax,(name,value) in zip(axes[1:],panels):im=ax.imshow(value,vmin=0,vmax=1,cmap='viridis');ax.set_title(name)
                for ax in axes:ax.axis('off')
                fig.colorbar(im,ax=axes.tolist(),fraction=.015);fig.suptitle(meta['case_id'])
                fig.savefig(out/'confirmation/figures'/('FAIL_'+meta['case_id']+'.png'),dpi=120);plt.close(fig)

if __name__=='__main__':main()
