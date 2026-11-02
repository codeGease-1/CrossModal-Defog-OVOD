"""CPU feasibility stages only. No neural training or confirmation entry point."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from scripts.v15.run import read, write, digest, procedural, original_scenes, cases, tensor, csv_write
from benchmarks.haze_density.synthetic import synthesize
from scripts.v2.global_optimum import solve_simplex
from src.models.haze_density.dynamic_prior import FrozenCandidates, single_scores
from src.models.haze_density.semantic_prior import (
    fit_calibration, SemanticCalibration, candidate_diagnostics,
    SemanticWeightGate, SemanticDirectRegressor, FrozenSemanticCandidates)
from scripts.semantic.frozen_semantics import FrozenSemantics


def prepare(out, cfg):
    base = read(ROOT/'configs/v15_calibration.json')
    manifests = [dict(s, purpose='reduced development diagnostic')
                 for split, n in [('train', cfg['diagnostic_train_scenes']),
                                  ('development', cfg['diagnostic_development_scenes'])]
                 for s in original_scenes(base, split)[:n]]
    assert not ({s['group_id'] for s in manifests if s['split']=='train'} &
                {s['group_id'] for s in manifests if s['split']=='development'})
    for s in manifests:
        rgb, _ = procedural(s['seed'], base['image_size'])
        s['original_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
    write(out/'scene_manifest.json', manifests)
    old = read(ROOT/'experiments/v2_investigation_20261101/completion.json')
    used = old['conservative_combined_seconds'] + old['previous_v15_fit_not_included_here']
    write(out/'budget.json', {'original_gpu_limit_seconds': 3600, 'historical_conservative_seconds': used,
          'arithmetic_remainder_seconds': max(0, 3600-used),
          'new_gpu_authorized_seconds': cfg['new_gpu_training_authorized_seconds'],
          'this_run_gpu_seconds': 0, 'policy': 'No GPU work or neural training in this CPU diagnostic; historical remainder is not a new one-hour grant'})
    sources = [*Path('scripts/semantic').glob('*.py'), Path('src/models/haze_density/semantic_prior.py'),
               Path('configs/semantic_calibration.json'), Path('src/models/haze_density/dynamic_prior.py'),
               Path('scripts/v15/run.py'), Path('src/models/haze_density/calibrated_prior.py')]
    write(out/'protocol.json', {'config': cfg, 'base': base,
          'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'source_sha256': {p.as_posix(): digest(p) for p in sources},
          'candidate_calibration_sha256': digest(ROOT/'experiments/v2_20261101/calibration.json'),
          'model_sha256': {p.name: digest(p) for p in (out/'semantic_model').glob('*')},
          'environment': {'python': sys.version, 'torch': torch.__version__, 'cuda_used': False},
          'data_audit': {'procedural': 'known scalar t and disjoint seed originals; design names are NOT semantic labels',
                         'RSHaze': 'external only; geography unverified; original G labels failed reconstruction; S is channelwise, not scalar t',
                         'semantics': 'ADE20K predictions from input RGB; no task semantic truth; not an independent sensor',
                         'prior_confirmation': 'all earlier confirmations now development references; no new confirmation accessed'}})
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (out/'pip_freeze.txt').write_text(freeze, encoding='utf-8')


def model(out, cfg):
    return FrozenSemantics(out/'semantic_model', cfg['semantic_labels']).eval()


def probe(out, cfg):
    m = model(out, cfg)
    base = read(out/'protocol.json')['base']
    rows, inputs, examples = [], [], []
    manifest = read(ROOT/'experiments/v15_20261101/external_manifest.json')
    for subset in ('G', 'S', 'L'):
        for s in [s for s in manifest if s['subset']==subset][:cfg['probe_real_per_subset']]:
            path = ROOT/'experiments/v15_20261101'/s['path']
            rgb = np.asarray(Image.open(path).convert('RGB').resize((256,256),Image.Resampling.BILINEAR), dtype=np.float32)/255
            inputs.append((s['scene_id'], rgb, 'real_reference_proxy', s['group_id']))
    for s in original_scenes(base, 'development')[:cfg['probe_development_scenes']]:
        rgb, _ = procedural(s['seed'], 256)
        inputs.append((s['scene_id'], rgb, 'procedural_fixture', s['group_id']))
    cache = out/'probe_cache'
    cache.mkdir(exist_ok=True)
    with torch.inference_mode():
        m(tensor(inputs[0][1], 'cpu'))  # one warm-up excluded, explicitly recorded
        for sid, clear, source, group in inputs:
            reference = m(tensor(clear, 'cpu'))[0].numpy()
            ref_label, ref_conf = reference.argmax(0), reference.max(0)
            imgs, labels = [], []
            for t in cfg['probe_t']:
                _, rgb, _ = synthesize(clear, np.full((256,256),t,np.float32), [1.,1.,1.], 1., base['domain'])
                start = time.perf_counter()
                p = m(tensor(rgb, 'cpu'))[0].numpy()
                seconds = time.perf_counter()-start
                label, conf = p.argmax(0), p.max(0)
                certain = (ref_conf >= cfg['confidence_threshold']) & (ref_label != 0)
                gray = clear.mean(2)
                dy, dx = np.gradient(gray)
                masks = {'all': np.ones_like(gray, dtype=bool), 'targeted_reference_prediction': certain,
                         'bright_proxy': gray>.8, 'low_texture_proxy': np.hypot(dx,dy)<.01}
                masks.update({name+'_reference_prediction': (ref_label==k)&(ref_conf>=cfg['confidence_threshold'])
                              for k,name in enumerate(cfg['semantic_groups']) if k})
                for region, mask in masks.items():
                    if not mask.any():
                        continue
                    rows.append({'scene_id': sid, 'group_id': group, 'source': source, 't': t,
                                 'region': region, 'pixels': int(mask.sum()),
                                 'agreement_with_clear_prediction': float((label[mask]==ref_label[mask]).mean()),
                                 'confident_fraction': float((conf[mask]>=cfg['confidence_threshold']).mean()),
                                 'mean_confidence': float(conf[mask].mean()), 'seconds': seconds})
                np.savez_compressed(cache/f'{sid}_t{t}.npz', probabilities=p)
                imgs.append(rgb); labels.append(label)
            examples.append((sid, imgs, labels))
            print('PROBE', sid, flush=True)
    csv_write(out/'semantic_probe.csv', rows)
    targeted = [r for r in rows if r['source']=='real_reference_proxy' and r['t']==.6 and r['region']=='targeted_reference_prediction']
    all_real = [r for r in rows if r['source']=='real_reference_proxy' and r['t']==.6 and r['region']=='all']
    n = sum(r['pixels'] for r in targeted)
    agreement = sum(r['pixels']*r['agreement_with_clear_prediction'] for r in targeted)/n if n else None
    coverage = float(np.mean([r['confident_fraction'] for r in all_real]))
    stable = agreement is not None and agreement >= cfg['probe_min_agreement_t06'] and coverage >= cfg['probe_min_confident_fraction']
    timings = [r['seconds'] for r in rows if r['region']=='all']
    write(out/'semantic_probe.json', {'targeted_t06_pixel_weighted_agreement': agreement,
          't06_confident_fraction': coverage, 'proxy_stability_screen_passed': stable,
          'parameters_total': sum(p.numel() for p in m.parameters()), 'parameters_trainable': 0,
          'median_seconds': float(np.median(timings)), 'p95_seconds': float(np.quantile(timings,.95)),
          'warmup_calls': 1, 'timing': 'CPU, RGB tensor conversion + normalize/resize 512 + model + resize 256 + aggregation; no disk read',
          'limitations': 'Agreement with clear-input prediction is stability, NOT semantic accuracy; no reliable white-roof/water/vegetation annotations. No inference gets clear RGB.',
          'training_decision': 'No full training authorized. If screen fails, retain failure evidence before any full training.'})
    # PIL rendering is a separate stage to avoid this host's OpenMP collision.
    write(out/'probe_inputs.json', [{'scene_id':sid,'source':source,'group_id':group,
           'pixel_sha256':hashlib.sha256(clear.tobytes()).hexdigest()}
          for sid,clear,source,group in inputs])


def iter_cases(out, cfg, split):
    base = read(out/'protocol.json')['base']
    for scene in read(out/'scene_manifest.json'):
        if scene['split'] != split:
            continue
        clear, _ = procedural(scene['seed'], base['image_size'])
        for meta, rgb, truth in cases(clear, scene, base):
            if f"{meta['pattern']}_{meta['uniform_t']}" in cfg['diagnostic_patterns'] and meta['airlight_index']==0 and meta['brightness_index']==1:
                yield meta, rgb, truth


def diagnose(out, cfg):
    ext = FrozenCandidates(read(ROOT/'experiments/v2_20261101/calibration.json'))
    semantic = model(out, cfg)
    raw_cal = ext.calibration
    cache = out/'diagnostic_cache'; cache.mkdir(exist_ok=True)
    xyps, fit_groups = [], []
    with torch.inference_mode():
        for split in ('train','development'):
            for meta, rgb, y in iter_cases(out, cfg, split):
                path = cache/(meta['case_id']+'.npz')
                if not path.exists():
                    x = tensor(rgb, 'cpu'); _, scores = single_scores(x, raw_cal)
                    features, original = ext(x); prob = semantic(x)
                    np.savez_compressed(path, scores=scores[0].numpy(), probabilities=prob[0].numpy(),
                                        original=original[0].numpy(), features=features[0].numpy(), truth=y)
                if split=='train':
                    a = np.load(path); indices = np.linspace(0,y.size-1,cfg['pixels_per_case'],dtype=int)
                    xyps.append((a['scores'].reshape(3,-1)[:,indices].T,a['truth'].ravel()[indices],
                                 a['probabilities'].reshape(4,-1)[:,indices].T))
                    fit_groups.extend([meta['group_id']]*len(indices))
                print('CACHE', split, meta['case_id'], flush=True)
    xs, ys, ps = [np.concatenate([a[i] for a in xyps]) for i in range(3)]
    cal = fit_calibration(xs,ys,ps,fit_groups,split='train',confidence=cfg['confidence_threshold'],
                          min_pixels=cfg['min_class_pixels'],min_scenes=cfg['min_class_scenes'],
                          shrinkage=cfg['shrinkage_pixels'],bins=cfg['isotonic_bins'])
    write(out/'conditional_calibration.json', cal)
    correction = SemanticCalibration(cal)
    matrices = {name: [np.zeros((3,3)),np.zeros(3),0,np.zeros((3,3)),0]
                for name in ('A_original','global_refit','C_semantic')}
    rows, predictions, candidate_hashes = [], [], {}
    frozen_cache=out/'frozen_candidate_cache';frozen_cache.mkdir(exist_ok=True)
    with torch.inference_mode():
        for split in ('train','development'):
            for meta,rgb,y in iter_cases(out,cfg,split):
                data = np.load(cache/(meta['case_id']+'.npz'))
                score = torch.from_numpy(data['scores'])[None]
                prob = torch.from_numpy(data['probabilities'])[None]
                corrected = correction(score,prob)[0].numpy()
                global_only = correction(score,prob,conditional=False)[0].numpy()
                versions = {'A_original':data['original'],'global_refit':global_only,'C_semantic':corrected}
                frozen_path=frozen_cache/(meta['case_id']+'.npz')
                np.savez_compressed(frozen_path,candidates=corrected,probabilities=data['probabilities'])
                candidate_hashes[frozen_path.name]=digest(frozen_path)
                for name, c in versions.items():
                    if split=='train':
                        a = c.reshape(3,-1).astype(float); yy=y.ravel(); mean=a.mean(1)
                        q,r,n,qc,nc=matrices[name]
                        matrices[name]=[q+a@a.T/a.shape[1]+.5*np.outer(mean,mean),r+a@yy/a.shape[1]+.5*mean*yy.mean(),n+1,
                                        qc+(a@a.T/a.shape[1] if meta['uniform_t']==1. else 0),nc+int(meta['uniform_t']==1.)]
                    else:
                        diag = candidate_diagnostics(c,y,cfg['bias_tolerance'])
                        row = {'case_id':meta['case_id'],'scene_id':meta['scene_id'],'group_id':meta['group_id'],'version':name,**diag}
                        rows.append(row); predictions.append((meta,name,c,y))
    weights = {}
    for name,(q,r,n,qc,nc) in matrices.items():
        weights[name]=solve_simplex(q/n+.25*qc/nc,r/n).tolist()
    write(out/'fixed_weights.json', {'weights': weights,'fit_split':'train','objective':'pixel MSE + .5 image-mean MSE + .25 conditional clear MSE',
                                    'solver':'all simplex faces; same loss as historical v2; reduced dataset, not historical fair comparison'})
    metrics = []
    from scripts.v2.run import record, summarize
    record_rows = []
    for meta,name,c,y in predictions:
        pred=(c*np.asarray(weights[name])[:,None,None]).sum(0)
        record_rows.append(record(meta,name,pred,y))
        metrics.append({'case_id':meta['case_id'],'scene_id':meta['scene_id'],'version':name,
                        'global_mae':float(abs(pred.mean()-y.mean())), 'pixel_mae':float(abs(pred-y).mean()),
                        'clear_response':float(pred.mean()) if meta['uniform_t']==1. else None})
    write(out/'candidate_boundaries.json',rows)
    write(out/'frozen_candidate_manifest.json',{'files':candidate_hashes,
          'calibration_sha256':digest(out/'conditional_calibration.json'),
          'policy':'C and D must use these identical candidates; targets and clean references are absent from this cache'})
    write(out/'development_metrics.json',{'summary':summarize(record_rows),'per_case':metrics,
          'scope':'8 procedural training originals, 4 development originals, 4 cases each; no real semantic labels or fresh confirmation; A/C reoptimized on identical reduced training cases'})
    csv_write(out/'development_per_image.csv',record_rows)
    # Same frozen candidate object is passed into D. Random weights only test the inference contract.
    data=np.load(cache/(next(iter_cases(out,cfg,'development'))[0]['case_id']+'.npz'))
    scores=torch.from_numpy(data['scores'])[None]; prob=torch.from_numpy(data['probabilities'])[None]
    c=correction(scores,prob); f=torch.from_numpy(data['features'])[None].clone();f[:,3:6]=c
    torch.manual_seed(cfg['seed']); gate=SemanticWeightGate().eval()
    start=time.perf_counter()
    with torch.inference_mode():
        pred,w=gate(torch.cat([f,prob],1),c)
    gate_seconds=time.perf_counter()-start
    assert torch.allclose(w.sum(1),torch.ones_like(w[:,0]),atol=1e-6)
    assert (pred>=c.amin(1,keepdim=True)-1e-6).all() and (pred<=c.amax(1,keepdim=True)+1e-6).all()
    costs=[]
    for label,m in [('D',gate),('RGB_semantic',SemanticDirectRegressor()),('RGB_semantic_light',SemanticDirectRegressor(base_channels=8))]:
        costs.append({'model':label,'trainable_parameters':sum(p.numel() for p in m.parameters()),
                      'total_parameters_including_semantic':sum(p.numel() for p in m.parameters())+sum(p.numel() for p in semantic.parameters())})
    write(out/'inference_contract.json',{'status':'PASS; untrained models only',
          'C_D_candidate_sha256':hashlib.sha256(c.numpy().tobytes()).hexdigest(),
          'D_parameters':costs,'D_single_cpu_forward_seconds':gate_seconds,
          'prohibitions':'No truth, clean reference, free residual, dehazing or detection in inference',
          'weights':'fusion choices, not physical confidence truth',
          'gpu_memory':'not measured; CPU only', 'neural_checkpoints':'none; no neural training'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','probe','diagnose'],required=True)
    parser.add_argument('--output',default='experiments/semantic_calibration_20261101');args=parser.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    cfg=read(ROOT/'configs/semantic_calibration.json');torch.set_num_threads(cfg['torch_threads'])
    if args.stage=='prepare':
        if (out/'protocol.json').exists():
            raise RuntimeError('Protocol already exists; preserve evidence and choose another output')
        prepare(out,cfg)
    else:
        protocol=read(out/'protocol.json')
        assert protocol['config']==cfg,'Config changed; use new directory'
        for name,sha in protocol['source_sha256'].items():
            assert digest(ROOT/name)==sha, f'Source changed: {name}; use a new evidence directory'
        {'probe':probe,'diagnose':diagnose}[args.stage](out,cfg)


if __name__=='__main__':
    main()
