"""PIL-only render; fixed categorical palette and fixed H scale [0,1]."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def main():
    a=argparse.ArgumentParser();a.add_argument('--output',default='experiments/semantic_calibration_20261101');args=a.parse_args()
    out=Path(args.output)
    cfg=json.loads(Path('configs/semantic_calibration.json').read_text())
    manifest=json.loads(Path('experiments/v15_20261101/external_manifest.json').read_text())
    palette=np.array([[160,160,160],[211,155,88],[60,146,87],[68,136,194]],dtype=np.uint8)
    for source in manifest:
        sid=source['scene_id']
        if not (out/'probe_cache'/f'{sid}_t1.0.npz').exists():continue
        clear=np.asarray(Image.open(Path('experiments/v15_20261101')/source['path']).convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255
        board=Image.new('RGB',(1024,560),'white');d=ImageDraw.Draw(board)
        for i,t in enumerate(cfg['probe_t']):
            # Same declared linear-sRGB synthesis, purely for this offline figure.
            linear=np.where(clear<=.04045,clear/12.92,((clear+.055)/1.055)**2.4)
            hazy=linear*t+1-t
            rgb=np.where(hazy<=.0031308,12.92*hazy,1.055*np.maximum(hazy,0)**(1/2.4)-.055)
            board.paste(Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8)),(256*i,24))
            p=np.load(out/'probe_cache'/f'{sid}_t{t}.npz')['probabilities']
            board.paste(Image.fromarray(palette[p.argmax(0)]),(256*i,280))
            d.text((256*i+8,5),f't={t}',fill='black')
        d.text((8,540),sid+' | gray=other, brown=built, green=vegetation, blue=water; PREDICTIONS',fill='black')
        board.save(out/f'probe_{sid}.png')
    for cache in sorted((out/'diagnostic_cache').glob('development_*gradient*.npz'))[:2]:
        data=np.load(cache)
        name=cache.name
        frozen=np.load(out/'frozen_candidate_cache'/name)['candidates']
        weights=json.loads((out/'fixed_weights.json').read_text())['weights']
        values=[data['truth'],np.einsum('i,ihw->hw',weights['A_original'],data['original']),
                np.einsum('i,ihw->hw',weights['C_semantic'],frozen),
                np.abs(np.maximum(frozen.min(0),np.minimum(frozen.max(0),data['truth']))-data['truth'])]
        board=Image.new('RGB',(1024,300),'white');d=ImageDraw.Draw(board)
        # Shared linear blue-to-red scale, 0=blue, 1=red. No per-image stretch.
        for i,(title,v) in enumerate(zip(['H truth','A fixed','C fixed','C oracle absolute error'],values)):
            v=np.clip(v,0,1);color=np.stack([v,np.zeros_like(v),1-v],-1)
            board.paste(Image.fromarray((color*255).astype(np.uint8)),(256*i,24))
            d.text((256*i+5,5),title,fill='black')
        d.text((5,284),'Shared absolute scale [0,1]: blue=0, red=1; procedural DEVELOPMENT only',fill='black')
        board.save(out/('bound_'+cache.stem+'.png'))


if __name__=='__main__':main()
