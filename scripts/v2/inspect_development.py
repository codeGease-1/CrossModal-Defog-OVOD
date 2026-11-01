"""CPU-only representative development probe of a completed seed (no tuning)."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.v15.run import read,write,procedural,original_scenes,cases,tensor
from scripts.v2.investigate import make_model
from src.models.haze_density.dynamic_prior import FrozenCandidates

def main():
    torch.set_num_threads(1);start=time.monotonic();cfg=read('configs/v2_investigation.json');out=Path(cfg['output_directory']);p=read(out/'protocol.json')
    assert (out/'student_seed42/complete.json').exists()
    model=make_model('student',cfg);model.load_state_dict(torch.load(out/'student_seed42/best.pth',map_location='cpu',weights_only=False)['model']);model.eval()
    ext=FrozenCandidates(read(Path(cfg['previous_directory'])/'calibration.json'));rows=[]
    with torch.no_grad():
        for s in original_scenes(p['base'],'development')[:4]:
            clear,kind=procedural(s['seed'],p['base']['image_size'])
            for meta,rgb,y in cases(clear,s,p['base']):
                if meta['brightness_index']!=1 or meta['airlight_index']!=0 or meta['pattern'] not in ('gradient','uniform'):continue
                f,c=ext(tensor(rgb,'cpu'));pred,w=model(f,c)
                rows.append({'case_id':meta['case_id'],'kind':kind,'mean_weights':w.mean((0,2,3)).tolist(),
                    'argmax_fractions':[(w.argmax(1)==i).float().mean().item() for i in range(3)],
                    'weight_std_spatial':w[0].flatten(1).std(1).tolist(),
                    'pixel_mae':float((pred[0,0]-torch.from_numpy(y)).abs().mean()),'dark_mae':float((c[0,0]-torch.from_numpy(y)).abs().mean())})
    write(out/'development_probe_seed42.json',{'device':'cpu','role':'development-only descriptive probe, not independent evidence','cases':rows,'elapsed_seconds':time.monotonic()-start})
    print('Development probe complete',len(rows))

if __name__=='__main__':main()
