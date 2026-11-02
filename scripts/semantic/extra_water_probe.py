"""Additional exposed development water scenes; never alters original screen."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read,write,tensor,digest
from scripts.semantic.frozen_semantics import FrozenSemantics
from benchmarks.haze_density.synthetic import synthesize


def main():
    out=Path('experiments/semantic_calibration_20261101');cfg=read('configs/semantic_calibration.json')
    torch.set_num_threads(cfg['torch_threads']);m=FrozenSemantics(out/'semantic_model',cfg['semantic_labels']).eval()
    rows=[]
    # Visually selected from previously exposed contact sheet; not labelled truth.
    plan={'scenes':['external_G_1293','external_G_1297'],
          'purpose':'Supplement missing water coverage after inspecting old reference contact sheet; development proxy only; no effect on frozen 6-scene screen',
          'G1293_water_like_rectangle_256':[175,0,245,18],
          'annotation':'Visual rectangular proxy; not reliable semantic ground truth'}
    write(out/'extra_water_plan.json',plan)
    for sid in plan['scenes']:
        path=Path('experiments/v15_20261101/external_sources')/(sid+'.png')
        clear=np.asarray(Image.open(path).convert('RGB').resize((256,256),Image.Resampling.BILINEAR),np.float32)/255
        ref=m(tensor(clear,'cpu'))[0].numpy();label=ref.argmax(0);conf=ref.max(0)
        masks={'all':np.ones((256,256),bool),'water_reference_prediction':(label==3)&(conf>=.6)}
        if sid=='external_G_1293':
            mask=np.zeros((256,256),bool);mask[0:18,175:245]=True;masks['visual_water_like_proxy']=mask
        for t in cfg['probe_t']:
            _,rgb,_=synthesize(clear,np.full((256,256),t,np.float32),[1.,1.,1.],1.,'linear_srgb')
            p=m(tensor(rgb,'cpu'))[0].numpy();pred=p.argmax(0)
            np.savez_compressed(out/'probe_cache'/f'{sid}_t{t}.npz',probabilities=p)
            for region,mask in masks.items():
                if not mask.any():continue
                rows.append({'scene_id':sid,'t':t,'region':region,'pixels':int(mask.sum()),
                             'water_prediction_fraction':float((pred[mask]==3).mean()),
                             'agreement_with_clear_prediction':float((pred[mask]==label[mask]).mean()),
                             'source_sha256':digest(path)})
    write(out/'extra_water_probe.json',rows)


if __name__=='__main__':main()
