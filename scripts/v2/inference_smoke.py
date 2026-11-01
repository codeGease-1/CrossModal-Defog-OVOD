"""Exercise the actual frozen RGB-file inference CLI, on a TRAINING fixture."""
from pathlib import Path
import subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
from scripts.v15.run import read,write,procedural,original_scenes,cases

def main():
    cfg=read('configs/v2_investigation.json');out=Path(cfg['output_directory']);p=read(out/'protocol.json');d=out/'inference_smoke';d.mkdir(exist_ok=True)
    if (d/'result.json').exists():return
    s=original_scenes(p['base'],'train')[0];clear,_=procedural(s['seed'],p['base']['image_size'])
    meta,rgb,_=next(v for v in cases(clear,s,p['base']) if v[0]['pattern']=='gradient' and v[0]['brightness_index']==1)
    image=d/'train_fixture.png';Image.fromarray(np.round(rgb*255).astype('uint8')).save(image)
    subprocess.run([sys.executable,'scripts/v2/predict_weight_gate.py','--image',str(image),'--experiment',str(out),'--output',str(d/'prediction'),'--device','cpu'],check=True)
    data=np.load(d/'prediction/prediction.npz');h,w,c=data['H'],data['weights'],data['candidates']
    assert np.isfinite(h).all() and np.isfinite(w).all() and np.all(w>=0)
    assert np.allclose(w.sum(0),1,atol=1e-6) and np.allclose(h,(w*c).sum(0),atol=1e-6)
    write(d/'result.json',{'status':'PASS','input_role':'TRAINING fixture, not an accuracy result','case_id':meta['case_id'],
        'H_shape':list(h.shape),'weight_shape':list(w.shape),'device':'cpu','CLI_exercised':True,'truth_used_for_inference':False})
    print('Actual RGB-file inference CLI: PASS')

if __name__=='__main__':main()
