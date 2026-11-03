"""Export all seeds and verify actual trained inference dependencies and state."""
import argparse
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.v15.run import read,write,digest,procedural
from scripts.matched_fusion.run import settings
from scripts.matched_evaluation.predict import Predictor


def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=ROOT/'experiments/matched_fusion_20261102');args=a.parse_args()
    out=args.output;settings();p=read(out/'protocol.json');cfg=p['config'];assert (out/'training_complete.json').exists()
    portable=out/'portable_weights';portable.mkdir(exist_ok=True);records=[];checks=[]
    for scene in p['scenes']:
        rgb,_=procedural(scene['seed'],p['base']['image_size'])
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==scene['pixel_sha256']
    for mode in cfg['new_modes']+cfg['continued_controls']:
        for seed in cfg['seeds']:
            path=out/f'{mode}_seed{seed}'/'best.pth';state=torch.load(path,map_location='cpu',weights_only=False)
            package={'mode':mode,'seed':seed,'gate_hidden':cfg['gate_hidden'],'model':state['model'],
                     'calibration':read(cfg['calibration']),'source_checkpoint_sha256':digest(path),
                     'protocol_sha256':digest(out/'protocol.json'),'target':'H=1-t; not measured natural fog density',
                     'inputs':'Hazy display RGB only; calibration and frozen correction included; no clear/t/A/oracle'}
            dest=portable/f'{mode}_seed{seed}.pth'
            if dest.exists():assert torch.load(dest,map_location='cpu',weights_only=False)['source_checkpoint_sha256']==digest(path)
            else:torch.save(package,dest)
            records.append({'file':str(dest.relative_to(out)),'sha256':digest(dest),'bytes':dest.stat().st_size,'selected_epoch':state['epoch'],
                            'mode':mode,'seed':seed,'state_tensor_elements':sum(v.numel() for v in package['model'].values())})
            if seed==cfg['seeds'][0]:
                torch.manual_seed(842);x=torch.rand(1,3,64,64)
                cpu=Predictor(package,'cpu');gpu=Predictor(package,'cuda')
                with torch.no_grad():
                    yc=cpu(x);yg=gpu(x.cuda()).cpu();restored=Predictor(torch.load(dest,map_location='cpu',weights_only=False),'cpu')(x)
                assert torch.equal(yc,restored)
                assert torch.isfinite(yc).all() and torch.isfinite(yg).all()
                checks.append({'mode':mode,'CPU_save_load_max_difference':float((yc-restored).abs().max()),
                               'CPU_FP32_vs_CUDA_AMP_max_difference':float((yc-yg).abs().max()),
                               'signature':'forward(rgb)','device_and_precision_difference_not_an_accuracy_test':True})
                del cpu,gpu;torch.cuda.empty_cache()
    write(out/'portable_manifest.json',records);write(out/'trained_inference_checks.json',{'models':checks,'source_pixels_match_previous_manifest':True})
    for path,sha in p['protected_previous_checkpoints'].items():assert digest(path)==sha,path
    print('EXPORTED',len(records),'CPU/CUDA CHECKS',checks,flush=True)


if __name__=='__main__':main()
