"""CPU-only checkpoint observer; never modifies training or uses test data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import time
import pickle
import torch
from scripts.v15.run import read,write,procedural,cases,tensor
from src.models.haze_density.dynamic_prior import ConvexFusion,FrozenCandidates

torch.set_num_threads(1);out=Path('experiments/v2_20261101');protocol=read(out/'protocol.json');cfg=protocol['config'];base=protocol['v15_config']
torch.manual_seed(42);extractor=FrozenCandidates(read(out/'calibration.json'))
scene={'scene_id':'weight_observer_dev','seed':base['seed']+10000}
clear,_=procedural(scene['seed'],base['image_size'])
features=[]
with torch.no_grad():
    for index,(meta,rgb,truth) in enumerate(cases(clear,scene,base)):
        if index in (0,2,4,5):features.append(extractor(tensor(rgb,'cpu'))[0])
x=torch.cat(features)
rows=read(out/'weight_trajectory.json') if (out/'weight_trajectory.json').exists() else []
seen={(r['seed'],r['epoch']) for r in rows};start=time.monotonic()
while time.monotonic()-start<1800:
    with torch.no_grad():
        for seed in cfg['seeds']:
            p=out/f'dynamic_seed{seed}/latest.pth'
            if not p.exists():continue
            try:state=torch.load(p,map_location='cpu',weights_only=False)
            except (RuntimeError,EOFError,ValueError,OSError,pickle.UnpicklingError):continue
            key=(seed,state['epoch'])
            if key in seen:continue
            seen.add(key);model=ConvexFusion('dynamic',cfg['hidden']);model.load_state_dict(state['model']);model.eval();_,w=model(x)
            rows.append({'seed':seed,'epoch':state['epoch'],'mean_weights':w.mean((0,2,3)).tolist(),
                         'argmax_fraction':[(w.argmax(1)==i).float().mean().item() for i in range(3)],
                         'entropy':float(-(w*w.clamp_min(1e-8).log()).sum(1).mean()),
                         'sample':'four fixed development cases; CPU observation, not accuracy selection'})
            write(out/'weight_trajectory.json',rows)
    if (out/'training_complete.json').exists():break
    time.sleep(.25)
print('OBSERVED',len(rows),'checkpoints')
