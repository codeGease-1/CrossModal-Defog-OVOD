"""Inspect transmission encoding without extracting or modifying the archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image, ImageDraw


def audit(archive, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, thumbs = [], []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        metadata = [n for n in sorted(names) if not n.endswith('/') and not n.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for subset in ('G', 'S', 'L'):
            clean = sorted(n for n in names if f'RSHaze_{subset}/train/cleanpng/' in n and n.endswith('.png'))
            chosen = random.Random(20261101).sample(clean, min(8, len(clean)))
            for n in chosen:
                raw = z.read(n)
                im = Image.open(io.BytesIO(raw))
                j = np.asarray(im).astype(np.float64) / 255
                row = {'member': n, 'sha256': hashlib.sha256(raw).hexdigest(), 'mode': im.mode,
                       'shape': list(j.shape), 'metadata_keys': list(im.info), 'exif_keys': list(im.getexif())}
                thumb = im.convert('RGB').resize((150, 150))
                thumbs.append((thumb, f'{subset}: {Path(n).name}'))
                tn, an, hn = (n.replace('/cleanpng/', '/' + k + '/') for k in ('transpng', 'airpng', 'synhazypng'))
                if all(k in names for k in (tn, an, hn)):
                    arrays = [np.asarray(Image.open(io.BytesIO(z.read(k)))) for k in (tn, an, hn)]
                    tr, air, hazy = arrays
                    row.update({'t_dtype': str(tr.dtype), 't_shape': list(tr.shape),
                                't_min_max': [int(tr.min()), int(tr.max())],
                                't_quantiles': np.quantile(tr, [0,.01,.5,.99,1]).tolist(),
                                'air_shape': list(air.shape), 'hazy_shape': list(hazy.shape)})
                    if tr.ndim == 3:
                        row['t_channel_max_diff'] = int(np.abs(tr.astype(int)-tr[..., :1].astype(int)).max())
                    if j.shape == air.shape == hazy.shape and tr.shape[:2] == j.shape[:2]:
                        a, h = air.astype(float)/255, hazy.astype(float)/255
                        t = tr.astype(float)/255
                        if t.ndim == 2:
                            t = t[..., None]
                        errors = {}
                        lin = lambda x: np.where(x<=.04045, x/12.92, ((x+.055)/1.055)**2.4)
                        for domain, jj, aa, hh in [('display',j,a,h), ('linear',lin(j),lin(a),lin(h))]:
                            for direction, tt in [('t',t), ('1-t',1-t)]:
                                errors[f'{domain}_{direction}_mae'] = float(np.abs(jj*tt+aa*(1-tt)-hh).mean())
                        row['reconstruction'] = errors
                        row['clean_hazy_mae'] = float(np.abs(j-h).mean())
                rows.append(row)
        result = {'archive': str(archive), 'archive_bytes': Path(archive).stat().st_size,
                  'non_image_entries': metadata, 'samples': rows,
                  'warning': 'Encoding plausibility is not proof of geographic independence or calibrated physical labels.'}
        (output/'archive_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        sheet = Image.new('RGB',(600, 180*((len(thumbs)+3)//4)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i,(im,label) in enumerate(thumbs):
            x,y=(i%4)*150,(i//4)*180
            sheet.paste(im,(x,y)); draw.text((x,y+152),label,fill='black')
        sheet.save(output/'audit_contact.png')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.archive, args.output)
