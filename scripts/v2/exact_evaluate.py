"""Evaluate strengthened B and quantify dynamic dependence after all fits freeze."""
import io
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read,write,procedural,original_scenes,cases,tensor,csv_write
from scripts.v2.run import record,summarize,model_set
from src.models.haze_density.dynamic_prior import FrozenCandidates


def main():
    torch.set_num_threads(1);out=Path('experiments/v2_20261101');protocol=read(out/'protocol.json');cfg=protocol['config'];base=protocol['v15_config']
    dest=out/'global_optimum';weights=torch.tensor(read(dest/'fit.json')['weights'],device='cuda')[None,:,None,None]
    models=model_set(out,cfg,'cuda');frozen=read(out/'frozen.json');gate=models[f'dynamic_seed{frozen["selected_seeds"]["dynamic"]}']
    extractor=FrozenCandidates(read(out/'calibration.json'))
    with torch.no_grad(),zipfile.ZipFile('/mnt/RSHaze+.zip') as z:
        for split in ('development','confirmation','external'):
            if (dest/(split+'.json')).exists():continue
            if split=='development':scenes=original_scenes(base,split)
            elif split=='confirmation':scenes=[s for s in read(out/'scene_manifest.json') if s['split']==split]
            else:scenes=read(out/'external_manifest.json')
            rows=[];differences=[]
            for s in scenes:
                if split=='external':clear=np.asarray(Image.open(io.BytesIO(z.read(s['archive_member']))).convert('RGB').resize((base['image_size'],base['image_size']),Image.Resampling.BILINEAR),np.float32)/255
                else:clear,_=procedural(s['seed'],base['image_size'])
                for meta,rgb,truth in cases(clear,s,base):
                    f,c=extractor(tensor(rgb,'cuda'));p=(c*weights).sum(1,keepdim=True);dynamic,w=gate(f,c)
                    for name,v in [('global_exact',p),('single_dark_candidate',c[:,:1])]:rows.append(record(meta,name,v.cpu().numpy()[0,0],truth))
                    differences.append({'case_id':meta['case_id'],'dynamic_vs_single_dark_mean_abs':float((dynamic-c[:,:1]).abs().mean()),
                                        'dynamic_vs_single_dark_max_abs':float((dynamic-c[:,:1]).abs().max()),'weight_non_dark_mean':float((1-w[:,:1]).mean())})
            csv_write(dest/(split+'_per_image.csv'),rows);csv_write(dest/(split+'_context_gap.csv'),differences)
            write(dest/(split+'.json'),{'metrics':summarize(rows),'mean_dynamic_to_dark_candidate_absolute_difference':float(np.mean([r['dynamic_vs_single_dark_mean_abs'] for r in differences])),
                  'max_dynamic_to_dark_candidate_difference':float(max(r['dynamic_vs_single_dark_max_abs'] for r in differences))})


if __name__=='__main__':main()
