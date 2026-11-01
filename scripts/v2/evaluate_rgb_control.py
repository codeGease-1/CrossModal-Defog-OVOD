"""Frozen RGB-only diagnostic, on the identical predeclared confirmation cases."""
import io,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read,write,digest,procedural,cases,tensor,csv_write
from scripts.v2.run import record,summarize
from scripts.v2.rgb_control import RGBControl
from scripts.v2.investigate import charge,remaining

def main():
    torch.set_num_threads(1);cfg=read('configs/v2_investigation.json');out=Path(cfg['output_directory']);dest=out/'rgb_control';frozen=read(dest/'frozen.json');p=read(out/'protocol.json')
    if (dest/'evaluation_complete.json').exists():return
    assert remaining(out)>60;start=time.monotonic();models={}
    for seed in cfg['seeds']:
        path=dest/f'seed{seed}/best.pth';assert digest(path)==frozen['checkpoint_sha256'][str(path.relative_to(out))]
        m=RGBControl().cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval();models[f'rgb_seed{seed}']=m
    allresults={}
    with torch.no_grad():
        for condition in ('nominal','unseen_airlight','display_rgb'):
            d=dest/condition;d.mkdir(exist_ok=True)
            if (d/'summary.json').exists():allresults[condition]=read(d/'summary.json');continue
            base=p['base'].copy()
            if condition=='unseen_airlight':base['airlights']=cfg['unseen_airlights']
            if condition=='display_rgb':base['domain']='display_rgb'
            rows=[]
            for s in read(out/'confirmation_manifest.json'):
                clear,_=procedural(s['seed'],base['image_size'])
                for meta,rgb,y in cases(clear,s,base):
                    x=tensor(rgb,'cuda')
                    for name,m in models.items():pred,_=m(x);rows.append(record(meta,name,pred[0,0].cpu().numpy(),y,rgb=clear))
            csv_write(d/'per_image.csv',rows);result={'metrics':summarize(rows),'role':'RGB-only necessary control, not main pipeline'};write(d/'summary.json',result);allresults[condition]=result
        d=dest/'original_S';d.mkdir(exist_ok=True);rows=[];size=p['base']['image_size']
        with zipfile.ZipFile('/mnt/RSHaze+.zip') as z:
            passed={r['scene_id'] for r in read(out/'original_S/label_checks.json') if r['status']=='PASS'}
            for s in read(out/'original_S/manifest.json'):
                if s['scene_id'] not in passed:continue
                n=s['archive_member'];rgb=np.asarray(Image.open(io.BytesIO(z.read(n.replace('/cleanpng/','/synhazypng/')))).convert('RGB').resize((size,size),Image.Resampling.BILINEAR),np.float32)/255
                t=np.asarray(Image.open(io.BytesIO(z.read(n.replace('/cleanpng/','/transpng/')))),np.float32)/255
                y=np.asarray(Image.fromarray((1-t.mean(2)).astype(np.float32)).resize((size,size),Image.Resampling.BILINEAR));x=tensor(rgb,'cuda')
                for name,m in models.items():pred,_=m(x);rows.append(record(dict(s,case_id=s['scene_id'],pattern='original_spectral'),name,pred[0,0].cpu().numpy(),y,rgb=rgb))
        csv_write(d/'per_image.csv',rows);result={'metrics':summarize(rows),'role':'Secondary spectral endpoint, one geographic group'};write(d/'summary.json',result);allresults['original_S']=result
        costs=[]
        model=models[f'rgb_seed{frozen["selected_seed"]}']
        for size in (256,512):
            x=torch.rand(1,3,size,size,device='cuda')
            for _ in range(5):model(x)
            torch.cuda.synchronize();times=[]
            for _ in range(20):
                t=time.monotonic();model(x);torch.cuda.synchronize();times.append((time.monotonic()-t)*1000)
            costs.append({'size':size,'p50_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),'parameters':sum(q.numel() for q in model.parameters())})
        write(dest/'inference_cost.json',costs)
    write(dest/'evaluation_complete.json',allresults);charge(out,'rgb_evaluate',start);print('RGB CONTROL EVALUATED')

if __name__=='__main__':main()
