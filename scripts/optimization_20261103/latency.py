"""Frozen-weight complete latency study, public validation, numerical checks."""
import sys,json,time,copy,io,hashlib,subprocess,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from PIL import Image
from scripts.v15.run import procedural,cases,tensor
from src.models.haze_density.joint_correction import JointCorrection,InferencePipeline
from src.models.haze_density.learned_correction import RGBReference,observed_context
from src.models.haze_density.captured_inference import CapturedInference,validate_rgb
OUT=ROOT/'experiments/optimization_20261103/latency_run03'
OLD=Path('/mnt/CrossModal-Defog-OVOD-joint-correction-20261102/experiments/joint_correction_20261102')
MATCH=Path('/mnt/CrossModal-Defog-OVOD-matched-gate-20261102/experiments/matched_fusion_20261102')
def save(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def status(stage,**kw):(OUT/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
def load(mode):
    if mode=='joint_dynamic':
        pack=torch.load(OLD/'portable_weights/joint_dynamic_seed42.pth',map_location='cpu',weights_only=False)
        n=JointCorrection([.9,.05,.05],True);n.load_state_dict(pack['model'])
    else:
        n=RGBReference(16,with_priors=mode=='direct_same16')
        n.load_state_dict(torch.load(MATCH/f'{mode}_seed42/best.pth',map_location='cpu',weights_only=False)['model'])
    cal=json.loads((OLD/'protocol.json').read_text())['config']['calibration']
    return InferencePipeline(n,json.loads(Path(cal).read_text()),uses_priors=mode!='rgb16').cuda().eval()
def timed(fn,n=60):
    for _ in range(6):fn()
    torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(n):
        torch.cuda.synchronize();s=time.perf_counter();fn();torch.cuda.synchronize();times.append(1000*(time.perf_counter()-s))
    return {'p50_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),'repeats':n,
            'resident_allocated_bytes':base,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'incremental_peak_bytes':torch.cuda.max_memory_allocated()-base}
def checked(p,x):validate_rgb(x);return p(x)
def main():
    if OUT.exists():raise FileExistsError('Existing latency study protected')
    OUT.mkdir(parents=True);torch.set_num_threads(1);torch.manual_seed(20261103);start=time.monotonic()
    config=json.loads((ROOT/'configs/optimization_20261103.json').read_text())
    sources=[ROOT/'src/models/haze_density/captured_inference.py',Path(__file__),ROOT/'configs/optimization_20261103.json']
    save('protocol.json',{'base_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'config':config,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'torch':torch.__version__,'GPU':torch.cuda.get_device_name(),'CUDA':torch.version.cuda,
        'scope':'seed42 frozen weights; no training; every model gets identical capture and public range validation; serialized process retains eager/reference and captured implementations'})
    p=json.loads((OLD/'protocol.json').read_text());rows=[];checks=[];segments=[]
    with torch.no_grad():
        for mode in config['methods']:
            pipe=load(mode);parameters=sum(p.numel() for p in pipe.parameters());maxdev=0.
            for size in config['sizes']:
                status('capture',method=mode,size=size);x=torch.rand(1,3,size,size,device='cuda');s=time.monotonic()
                runner=CapturedInference(pipe,x);cold=time.monotonic()-s
                for kind in ['zero','one','constant','random']:
                    y={'zero':lambda:torch.zeros_like(x),'one':lambda:torch.ones_like(x),'constant':lambda:torch.full_like(x,.3),'random':lambda:torch.rand_like(x)}[kind]()
                    a=checked(pipe,y);b=runner(y);diff=float((a-b).abs().max());assert diff<=1e-5,(mode,size,kind,diff)
                    checks.append({'method':mode,'size':size,'case':kind,'max_abs_difference':diff})
                # Replay must not overwrite previously returned tensors.
                owned=runner(x);snapshot=owned.clone();runner(torch.zeros_like(x));assert torch.equal(owned,snapshot)
                rejected=0
                for val in [float('nan'),float('inf'),-1.,1.1]:
                    bad=x.clone();bad[0,0,0,0]=val
                    try:runner(bad)
                    except ValueError:rejected+=1
                assert rejected==4
                checks.append({'method':mode,'size':size,'invalid_inputs_rejected':rejected,'output_ownership':True})
                for repeat in range(3):
                    for impl in (['eager','captured'] if repeat%2==0 else ['captured','eager']):
                        fn=(lambda:checked(pipe,x)) if impl=='eager' else (lambda:runner(x))
                        val=timed(fn)
                        bio=io.BytesIO();Image.fromarray((x[0].permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(bio,format='PNG');data=bio.getvalue()
                        def png():
                            with Image.open(io.BytesIO(data)) as im:arr=np.array(im.convert('RGB'),dtype=np.float32)/255
                            img=torch.from_numpy(arr).permute(2,0,1)[None].cuda()
                            return (checked(pipe,img) if impl=='eager' else runner(img)).cpu()
                        full=timed(png,20)
                        rec={'method':mode,'implementation':impl,'size':size,'round':repeat,'all_network_parameters':parameters,
                             'GPU_RGB_to_H':val,'PNG_bytes_to_CPU_H':full,'capture_build_seconds':cold if impl=='captured' else 0.}
                        rows.append(rec);print('COST',json.dumps(rec),flush=True)
                if size==256:
                    status('equivalence_512_old_dev',method=mode)
                    for scene in [s for s in p['scenes'] if s['split']=='development']:
                        clear,_=procedural(scene['seed'],256)
                        for meta,rgb,target in cases(clear,scene,p['base']):
                            image=tensor(rgb,'cuda');delta=float((checked(pipe,image)-runner(image)).abs().max());maxdev=max(maxdev,delta);assert delta<=1e-5,(meta['case_id'],delta)
                    checks.append({'method':mode,'old_dev_cases':512,'max_abs_difference':maxdev})
                if mode=='joint_dynamic' and size==256:
                    f,_=pipe.extractor(x);f=torch.cat([f,observed_context(x)],1);z=pipe.network.corrector.trunk(x)
                    for name,fn in [('input_validation',lambda:validate_rgb(x)),('FP32_prior_calibration',lambda:pipe.extractor(x)),
                                    ('content_features',lambda:observed_context(x)),('RGB_trunk',lambda:pipe.network.corrector.trunk(x)),
                                    ('full_network_after_features',lambda:pipe.network(f))]:
                        with torch.autocast('cuda',dtype=torch.float16,enabled=name in ['RGB_trunk','full_network_after_features']):
                            segments.append({'stage':name,**timed(fn,40)})
                del runner;gc.collect();torch.cuda.empty_cache()
            del pipe;gc.collect();torch.cuda.empty_cache()
    save('costs.json',rows);save('equivalence.json',checks);save('segments.json',{'warning':'Stage times include Python dispatch/synchronization and are not additive causal decomposition','rows':segments})
    save('complete.json',{'seconds':time.monotonic()-start,'new_training_runs':0,'new_confirmation_scenes':0});status('complete')
    print('LATENCY COMPLETE',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
