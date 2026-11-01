"""Reproducible, bounded V1.5 calibration experiment. No V2/V3 training.

Stages are individually restartable. External real images are deliberately
untouched until calibration and development selection have been frozen.
"""
import argparse
import csv
import hashlib
import io
import json
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
import torch
from benchmarks.haze_density.synthetic import synthesize, transmission_map
from benchmarks.haze_density.metrics import correlation, spatial_metrics, ranking_metrics
from src.models.haze_density.calibrated_prior import (
    raw_cues, normalized_cues, fixed_cues, fuse, fit_monotone, apply_monotone)
from src.models.haze_density.physical_prior import generate_s_final, min_max_normalize

METHODS = ['legacy', 'normalized_calibrated', 'fixed_calibrated']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def procedural(seed, size):
    """Independent artificial originals, NOT real remote-sensing training data."""
    rng = np.random.default_rng(seed)
    kind = ['urban', 'vegetation', 'water', 'soil'][seed % 4]
    bases = {'urban':[.38,.38,.37], 'vegetation':[.19,.31,.16],
             'water':[.12,.22,.27], 'soil':[.48,.38,.26]}
    low = rng.uniform(0,1,(12,12,3))
    smooth = np.asarray(Image.fromarray((low*255).astype('uint8')).resize((size,size), Image.Resampling.BILINEAR))/255
    texture = rng.normal(0, {'urban':.025,'vegetation':.05,'water':.004,'soil':.02}[kind], (size,size,1))
    rgb = np.clip(np.array(bases[kind]) + (smooth-.5)*.32 + texture, 0, 1)
    if kind == 'urban':
        im = Image.fromarray((rgb*255).astype('uint8')); draw = ImageDraw.Draw(im)
        for _ in range(65):
            x,y = rng.integers(0,size,2); w,h = rng.integers(4,25,2)
            v = float(rng.choice([.12,.35,.65,.92]))
            color = tuple(np.clip((v+rng.normal(0,.025,3))*255,0,255).astype(int))
            draw.rectangle([int(x),int(y),int(x+w),int(y+h)],fill=color)
        rgb = np.asarray(im)/255
    return rgb.astype(np.float32), kind


def original_scenes(cfg, split):
    return [{'scene_id': f'{split}_{i:03d}', 'split':split,
             'group_id': f'procedural_original_{cfg["seed"]+cfg["scene_seed_offsets"][split]+i}',
             'seed':cfg['seed']+cfg['scene_seed_offsets'][split]+i,
             'source_kind':'procedural_controlled_original'}
            for i in range(cfg['scene_counts'][split])]


def cases(clear, scene, cfg):
    for ai, air in enumerate(cfg['airlights']):
        for bi, brightness in enumerate(cfg['brightness']):
            settings = [('uniform',t) for t in cfg['uniform_t']] + [(p,None) for p in cfg['spatial_patterns']]
            for pattern, value in settings:
                # The spatial distribution is fixed for all brightness/airlight variants of this original.
                t, params = transmission_map(pattern, cfg['image_size'], scene['seed'], value)
                _, rgb, _ = synthesize(clear, t, air, brightness, cfg['domain'])
                meta = dict(scene, airlight_index=ai, brightness_index=bi, brightness=brightness,
                            pattern=pattern, uniform_t=value,
                            case_id=f'{scene["scene_id"]}_a{ai}_b{bi}_{pattern}_{value}',
                            t_parameters=params)
                yield meta, rgb, 1-t


def tensor(rgb, device):
    return torch.from_numpy(rgb.transpose(2,0,1).copy())[None].to(device)


def scores(image, cfg, limits=None):
    raw = raw_cues(image, cfg['prior']['window_size'])
    nc = normalized_cues(raw)
    legacy = fuse(image,nc,cfg['prior']['guided_radius'],cfg['prior']['guided_eps'])
    fixed = None if limits is None else fuse(image, fixed_cues(raw, limits),
                                             cfg['prior']['guided_radius'],cfg['prior']['guided_eps'])
    return raw, nc, legacy, fixed


def prepare(out, cfg):
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json')['config'] == cfg, 'Config changed; use a new experiment directory'
        return
    sources = [ROOT/'scripts/v15/run.py', ROOT/'scripts/v15/audit_archive.py',
               ROOT/'src/models/haze_density/calibrated_prior.py', ROOT/'src/models/haze_density/physical_prior.py',
               ROOT/'src/models/haze_density/decoder.py', ROOT/'benchmarks/haze_density/synthetic.py',
               ROOT/'src/models/haze_density/guided_filter.py']
    manifest = [s for split in cfg['scene_counts'] for s in original_scenes(cfg,split)]
    assert len({s['group_id'] for s in manifest}) == len(manifest)
    for s in manifest:
        rgb, kind = procedural(s['seed'], cfg['image_size'])
        s.update(land_cover_design=kind, original_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    assert len({s['original_pixel_sha256'] for s in manifest}) == len(manifest)
    write(out/'scene_manifest.json',manifest)
    write(out/'protocol.json', {'config':cfg,'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
          'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'environment':{'python':sys.version, 'torch':torch.__version__, 'numpy':np.__version__,
                         'cuda':torch.version.cuda, 'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                         'platform':platform.platform()},
          'data_policy':'All real RSHaze scenes are a single held-out supergroup. No existing random-file split is reused.',
          'limits':'Procedural-domain calibration is preliminary; external images test controlled additional attenuation, not naturally occurring absolute haze.'})


def smoke(out,cfg,device):
    if (out/'smoke.json').exists():
        return
    rgb,_ = procedural(cfg['seed']+10000,cfg['image_size'])
    image=tensor(rgb,device)
    start=time.perf_counter()
    with torch.no_grad():
        r,n,l,_=scores(image,cfg)
        reference=generate_s_final(image,**cfg['prior'])
    elapsed=time.perf_counter()-start
    error=float((reference-l).abs().max())
    assert error<1e-6, 'Legacy adapter does not match the frozen implementation'
    # Exact display-space affine attenuation is the mechanism test, distinct from the main linear-sRGB benchmark.
    invariance=[]
    for t in (1.,.9,.6,.4):
        rr, nn, ll,_=scores(image*t+1-t,cfg)
        invariance.append({'t':t,'raw_d_mean':float(rr[:,0].mean()),
                           'normalized_max_change':float((nn-n).abs().max()),
                           'legacy_mean':float(ll.mean())})
    z=torch.linspace(-1,1,256,device=device).reshape(1,1,16,16)
    norm=torch.nn.InstanceNorm2d(1).to(device)
    shifts=[]
    for shift in (-2.,0.,2.):
        v=z+shift; nv=norm(v)
        shifts.append({'input_mean':float(v.mean()), 'after_norm_mean':float(nv.mean()),
                       'sigmoid_mean':float(nv.sigmoid().mean()),'without_norm_sigmoid_mean':float(v.sigmoid().mean())})
    network_rows=[]
    checkpoint=ROOT/'experiments/spatial_v3_run01/checkpoints/best.pth'
    if checkpoint.exists():
        from src.models.haze_density.haze_density_net import HazeDensityNet
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=HazeDensityNet(base_channels=saved['config'].get('base_channels',32)).to(device).eval()
        model.load_state_dict(saved['model_state_dict'])
        captured={}
        def hook(module, inputs, output):
            captured['pre']=inputs[0].detach(); captured['post']=output.detach()
        handle=model.decoder.norm3.register_forward_hook(hook)
        for t in (1.,.9,.75,.6,.4):
            _,im,_=synthesize(rgb,np.full(rgb.shape[:2],t,np.float32),[1,1,1],1.,cfg['domain'])
            prediction=model(tensor(im,device))
            row={'t':t,'prediction_mean':float(prediction.mean())}
            for name,zv in captured.items():
                row[name+'_mean']=float(zv.mean()); row[name+'_std']=float(zv.std())
                row[name+'_p05']=float(torch.quantile(zv,.05)); row[name+'_p95']=float(torch.quantile(zv,.95))
            network_rows.append(row)
        handle.remove()
    write(out/'smoke.json',{'status':'PASS','legacy_max_abs_difference':error,
          'first_case_seconds_including_cuda_warmup':elapsed,
          'display_affine_invariance':invariance,'instance_norm_shift_test':shifts,
          'legacy_network_norm3_audit':network_rows,
          'legacy_network_checkpoint_sha256':digest(checkpoint) if checkpoint.exists() else None,
          'forecast_cases':sum(cfg['scene_counts'].values())*32+cfg['external_scenes_per_subset']*3*32,
          'training_budget_seconds':cfg['training_budget_seconds']})
    print('SMOKE',json.dumps(read(out/'smoke.json')),flush=True)


def fit(out,cfg,device):
    path=out/'calibration.json'
    if path.exists():
        return
    cache=out/'fit_cache'; cache.mkdir(exist_ok=True)
    ledger_path=out/'budget.json'
    ledger=read(ledger_path) if ledger_path.exists() else {'fit_elapsed_seconds':0.0,'neural_training_seconds':0.0}
    started=time.monotonic()
    def save_budget():
        state=dict(ledger,fit_elapsed_seconds=ledger['fit_elapsed_seconds']+time.monotonic()-started)
        write(ledger_path,state)
        if state['fit_elapsed_seconds']>=cfg['training_budget_seconds']:
            raise RuntimeError('Calibration budget exhausted; partial per-scene checkpoints preserved')
    train=original_scenes(cfg,'train')
    # Fixed stratified pixel samples, with identical pixels and cases for both calibrators.
    side=int(np.sqrt(cfg['pixels_per_case']))
    indices=np.linspace(0,cfg['image_size']-1,side,dtype=int)
    def sample(v):
        return v[...,indices[:,None],indices].reshape(-1)
    for s in train:
        p=cache/(s['scene_id']+'_raw.npz')
        if p.exists(): continue
        rgb,_=procedural(s['seed'],cfg['image_size']); vals=[]
        for _,im,truth in cases(rgb,s,cfg):
            rr=raw_cues(tensor(im,device),cfg['prior']['window_size']).cpu().numpy()[0]
            vals.append(np.stack([sample(v) for v in rr]))
        np.savez_compressed(p,raw=np.concatenate(vals,axis=1)); save_budget()
    raw=np.concatenate([np.load(cache/(s['scene_id']+'_raw.npz'))['raw'] for s in train],axis=1)
    limits=np.quantile(raw,cfg['percentiles'],axis=1).T.tolist()
    write(out/'limits_checkpoint.json',{'limits':limits,'train_group_ids':[s['group_id'] for s in train]})
    for s in train:
        p=cache/(s['scene_id']+'_scores.npz')
        if p.exists(): continue
        rgb,_=procedural(s['seed'],cfg['image_size']); a,b,y=[],[],[]
        for _,im,truth in cases(rgb,s,cfg):
            _,_,legacy,fixed=scores(tensor(im,device),cfg,limits)
            a.append(sample(legacy.cpu().numpy()[0,0])); b.append(sample(fixed.cpu().numpy()[0,0])); y.append(sample(truth))
        np.savez_compressed(p,normalized=np.concatenate(a),fixed=np.concatenate(b),truth=np.concatenate(y)); save_budget()
        print('FIT',s['scene_id'],flush=True)
    arrays={k:np.concatenate([np.load(cache/(s['scene_id']+'_scores.npz'))[k] for s in train]) for k in ('normalized','fixed','truth')}
    tables={k:fit_monotone(arrays[k],arrays['truth'],cfg['isotonic_bins']) for k in ('normalized','fixed')}
    calibration={'prior':cfg['prior'],'limits':limits,'tables':tables,'train_group_ids':[s['group_id'] for s in train],
                 'target':'effective additional attenuation H=1-t', 'scene_manifest_sha256':digest(out/'scene_manifest.json')}
    write(path,calibration); save_budget()


def external_scenes(archive,out,cfg):
    path=out/'external_manifest.json'
    if path.exists(): return read(path)
    sources=out/'external_sources'; sources.mkdir(exist_ok=True)
    records=[]; seen=set()
    with zipfile.ZipFile(archive) as z:
        for subset in ('G','S','L'):
            names=sorted(n for n in z.namelist() if f'RSHaze_{subset}/test/cleanpng/' in n and n.endswith('.png'))
            rng=random.Random(cfg['seed']); rng.shuffle(names)
            count=0
            for n in names:
                data=z.read(n); im=Image.open(io.BytesIO(data)).convert('RGB')
                dh=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
                if dh in seen: continue
                seen.add(dh)
                sid=f'external_{subset}_{Path(n).stem}'
                (sources/(sid+'.png')).write_bytes(data)
                records.append({'scene_id':sid,'split':'external','group_id':'RSHaze_all_geography_unverified',
                                'seed':cfg['seed']+30000+len(records),'source_kind':'real_reference_with_controlled_additional_haze',
                                'archive_member':n,'file_sha256':hashlib.sha256(data).hexdigest(),
                                'pixel_sha256':dh,'subset':subset,'path':str((sources/(sid+'.png')).relative_to(out))})
                count+=1
                if count>=cfg['external_scenes_per_subset']: break
    write(path,records)
    return records


def summary(rows):
    result={}
    for method in METHODS:
        rr=[r for r in rows if r['method']==method]
        groups={}
        for r in rr:
            if r['pattern']=='uniform':
                groups.setdefault((r['scene_id'],r['airlight_index'],r['brightness_index']),[]).append(r)
        ranks=[ranking_metrics([q['truth_mean'] for q in v],[q['prediction_mean'] for q in v])['pairwise_accuracy'] for v in groups.values()]
        # All different-original pairs under each nuisance condition. Equal strengths are excluded.
        numer=denom=ties=0
        for ai in (0,1):
            for bi in (0,1):
                subset=[r for r in rr if r['pattern']=='uniform' and r['airlight_index']==ai and r['brightness_index']==bi]
                for i,a in enumerate(subset):
                    for b in subset[i+1:]:
                        if a['scene_id']==b['scene_id'] or abs(a['truth_mean']-b['truth_mean'])<1e-6: continue
                        delta=(a['prediction_mean']-b['prediction_mean'])*np.sign(a['truth_mean']-b['truth_mean'])
                        denom+=1; numer+=int(delta>1e-6); ties+=int(abs(delta)<=1e-6)
        rho=[r['region_spearman'] for r in rr if r['region_spearman'] is not None]
        clear=[r for r in rr if r['pattern']=='uniform' and r['uniform_t']==1.]
        result[method]={'n_cases':len(rr),'global_mae':float(np.mean([r['global_abs_error'] for r in rr])),
             'pixel_mae':float(np.mean([r['pixel_mae'] for r in rr])),
             'same_scene_rank_accuracy':float(np.mean(ranks)),
             'cross_original_rank_accuracy':numer/denom if denom else None,
             'cross_original_tie_rate':ties/denom if denom else None,
             'region_spearman':float(np.mean(rho)) if rho else None,'spatial_defined_cases':len(rho),
             'clear_mean':float(np.mean([r['prediction_mean'] for r in clear])),
             'clear_bright_response':float(np.mean([r['bright_response'] for r in clear if r['bright_response'] is not None])) if any(r['bright_response'] is not None for r in clear) else None}
    return result


def evaluate(out,cfg,device,split,archive):
    dest=out/split; dest.mkdir(exist_ok=True)
    if (dest/'summary.json').exists(): return
    calibration=read(out/'calibration.json')
    if split=='test' or split=='external':
        frozen=read(out/'frozen_selection.json')
        assert frozen['calibration_sha256']==digest(out/'calibration.json')
    scenes=external_scenes(archive,out,cfg) if split=='external' else original_scenes(cfg,split)
    images=[]; audits=[]; figures=dest/'figures'; figures.mkdir(exist_ok=True)
    started=time.perf_counter()
    for si,s in enumerate(scenes):
        if split=='external':
            p=out/s['path']; assert digest(p)==s['file_sha256']
            clear=np.asarray(Image.open(p).convert('RGB').resize((cfg['image_size'],cfg['image_size']),Image.Resampling.BILINEAR),np.float32)/255
        else: clear,_=procedural(s['seed'],cfg['image_size'])
        for meta,rgb,truth in cases(clear,s,cfg):
            image=tensor(rgb,device); raw,nc,old,fix=scores(image,cfg,calibration['limits'])
            preds={'legacy':old,'normalized_calibrated':apply_monotone(old,calibration['tables']['normalized']),
                   'fixed_calibrated':apply_monotone(fix,calibration['tables']['fixed'])}
            arrays={k:v.cpu().numpy()[0,0] for k,v in preds.items()}
            gray=clear @ np.array([.299,.587,.114]); bright=gray>.8
            for name,pred in arrays.items():
                spatial=spatial_metrics(1-truth,pred)
                images.append({k:v for k,v in dict(meta,method=name,truth_mean=float(truth.mean()),prediction_mean=float(pred.mean()),
                    global_abs_error=float(abs(pred.mean()-truth.mean())),pixel_mae=float(np.abs(pred-truth).mean()),
                    region_spearman=spatial['region_spearman'],bright_response=float(pred[bright].mean()) if bright.any() else None).items() if k!='t_parameters'})
            weighted=(nc*nc.new_tensor([.5,.3,.2])[None,:,None,None]).sum(1,keepdim=True)
            for name,value in [('D_raw',raw[:,0]),('C_negative_raw',raw[:,1]),('K_negative_raw',raw[:,2]),
                               ('D_normalized',nc[:,0]),('C_normalized',nc[:,1]),('K_normalized',nc[:,2]),
                               ('legacy_weighted',weighted),('legacy_power',weighted.pow(1.5)),
                               ('legacy_final',old),('fixed_uncalibrated',fix)]:
                a=value.cpu().numpy().ravel(); q=np.quantile(a,[0,.05,.5,.95,1])
                audits.append({'case_id':meta['case_id'],'scene_id':s['scene_id'],'airlight_index':meta['airlight_index'],
                    'brightness_index':meta['brightness_index'],'pattern':meta['pattern'],'truth_mean':float(truth.mean()),
                    'signal':name,'mean':float(a.mean()),'p05':float(q[1]),'p50':float(q[2]),'p95':float(q[3]),'range':float(q[4]-q[0])})
            if si<3 and meta['airlight_index']==0 and meta['brightness_index']==1 and (meta['pattern']=='gradient' or meta['uniform_t'] in (1.,.4)):
                plot_case(figures/(meta['case_id']+'.png'),rgb,truth,arrays)
        print('EVAL',split,s['scene_id'],flush=True)
    csv_write(dest/'per_image.csv',images); csv_write(dest/'signal_audit.csv',audits)
    totals=summary(images)
    strata={}
    for ai in (0,1):
        for bi in (0,1):
            strata[f'a{ai}_b{bi}']=summary([r for r in images if r['airlight_index']==ai and r['brightness_index']==bi])
    audit_summary={}
    for sig in sorted({r['signal'] for r in audits}):
        rr=[r for r in audits if r['signal']==sig and r['pattern']=='uniform']
        audit_summary[sig]={stat:correlation([r['truth_mean'] for r in rr],[r[stat] for r in rr],rank=True) for stat in ('mean','p05','p50','p95','range')}
    failures=[]
    for method in METHODS:
        rr=[r for r in images if r['method']==method]
        failures.extend(sorted(rr,key=lambda r:r['global_abs_error'],reverse=True)[:5])
    write(dest/'failure_cases.json',failures)
    write(dest/'summary.json',{'metrics':totals,'condition_strata':strata,'signal_spearman':audit_summary,
          'elapsed_seconds':time.perf_counter()-started,'n_originals':len(scenes),
          'n_groups':len({s['group_id'] for s in scenes}),
          'calibration_sha256':digest(out/'calibration.json'),
          'interpretation':'controlled additional attenuation; external originals are not verified haze-free or separate geographic areas'})


def plot_case(path,rgb,truth,arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(15,3.3))
    axes[0].imshow(rgb); axes[0].set_title('Input RGB')
    for ax,(name,v) in zip(axes[1:],[('Known 1-t',truth)]+list(arrays.items())):
        im=ax.imshow(v,cmap='viridis',vmin=0,vmax=1)
        ax.set_title(f'{name}\nmean={v.mean():.3f}',fontsize=9)
    for ax in axes: ax.axis('off')
    fig.colorbar(im,ax=axes.tolist(),fraction=.018,pad=.01,label='Shared scale [0,1]')
    fig.savefig(path,dpi=110,bbox_inches='tight'); plt.close(fig)


def freeze(out,cfg):
    if (out/'frozen_selection.json').exists(): return
    m=read(out/'development/summary.json')['metrics']; base=m['legacy']
    valid=[k for k in METHODS if m[k]['region_spearman']>=base['region_spearman']-cfg['spatial_noninferiority_margin'] and m[k]['global_mae']<=base['global_mae']]
    selected=min(valid,key=lambda k:m[k]['global_mae'])
    write(out/'frozen_selection.json',{'selected':selected,'rule':cfg['selection_rule'],
          'calibration_sha256':digest(out/'calibration.json'),'development_metrics':m,
          'external_opened_for_model_selection':False,'extra_global_head':'NOT ADDED; structural need not established'})
    print('FROZEN',selected,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['prepare','smoke','fit','development','freeze','test','external'],required=True)
    p.add_argument('--config',default='configs/v15_calibration.json')
    p.add_argument('--output',default='experiments/v15_20261101')
    p.add_argument('--archive',default='/mnt/RSHaze+.zip')
    p.add_argument('--device',default='cuda')
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); cfg=read(a.config)
    torch.set_num_threads(1); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    if a.stage=='prepare': prepare(out,cfg)
    else:
        protocol=read(out/'protocol.json'); assert protocol['config']==cfg
        for rel,sha in protocol['source_sha256'].items():
            assert digest(ROOT/rel)==sha, f'Source changed after freeze: {rel}; use a new output directory'
        with torch.no_grad():
            if a.stage=='smoke': smoke(out,cfg,a.device)
            elif a.stage=='fit':
                assert (out/'smoke.json').exists(); fit(out,cfg,a.device)
            elif a.stage=='freeze': freeze(out,cfg)
            else: evaluate(out,cfg,a.device,a.stage,a.archive)


if __name__=='__main__': main()
