"""Scene-paired comparison to the RGB-only control; no selection here."""
import csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.v15.run import read,write

def rows(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    cfg=read('configs/v2_investigation.json');out=Path(cfg['output_directory']);result={}
    for condition in ('nominal','unseen_airlight','display_rgb','original_S'):
        a=rows(out/condition/'per_image.csv');b=rows(out/'rgb_control'/condition/'per_image.csv');names=sorted({r['scene_id'] for r in a});result[condition]={}
        for metric in ('pixel_mae','global_abs_error','region_spearman'):
            delta=[]
            for s in names:
                aa=[float(r[metric]) for r in a if r['scene_id']==s and r['method'].startswith('student_seed') and r[metric]!='']
                bb=[float(r[metric]) for r in b if r['scene_id']==s and r[metric]!=''];delta.append(np.mean(aa)-np.mean(bb))
            r={'student_minus_rgb':float(np.mean(delta))}
            if condition!='original_S':
                rng=np.random.default_rng(42);boot=np.array(delta)[rng.integers(0,len(delta),(2000,len(delta)))].mean(1);r['scene_bootstrap_95']=np.quantile(boot,[.025,.975]).tolist()
            else:r['CI']=None;r['reason']='one unverified geographic group'
            result[condition][metric]=r
    write(out/'rgb_comparison.json',result)

if __name__=='__main__':main()
