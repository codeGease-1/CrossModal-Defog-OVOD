"""Scientific plot of original-development losses; no image-specific rescaling."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.v15.run import read

def main():
    cfg=read('configs/v2_investigation.json');out=Path(cfg['output_directory']);fig,ax=plt.subplots(figsize=(9,5))
    for label,color in [('temperature','#b15a38'),('student','#2266a1'),('rgb','#2d8659')]:
        for i,seed in enumerate(cfg['seeds']):
            path=out/f'{label}_seed{seed}/history.json' if label!='rgb' else out/f'rgb_control/seed{seed}/history.json'
            if not path.exists():continue
            h=read(path);ax.plot([r['epoch'] for r in h],[r['development_loss'] for r in h],color=color,alpha=.7,label=label if i==0 else None)
    ref=read(Path(cfg['previous_directory'])/'global_seed44/complete.json')['best_development_loss']
    ax.axhline(ref,color='#555555',linestyle='--',label='shared weights (Adam)')
    ax.set(xlabel='Epoch',ylabel='Original development objective',title='Three initialization seeds per follow-up model');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=150)

if __name__=='__main__':main()
