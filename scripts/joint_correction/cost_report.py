"""Full inference cost with optimized priors for ALL prior-using controls; honest report."""
import copy,io,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scripts.v15.run import read,write,digest
from scripts.joint_correction.run import make
from src.models.haze_density.joint_correction import InferencePipeline,JointCorrection
from src.models.haze_density.learned_correction import RGBReference,RestrictedCorrector
from src.models.haze_density.matched_fusion import MatchedFusion


def timed(fn,n=40):
    for _ in range(8):fn()
    torch.cuda.synchronize();values=[];torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated()
    for _ in range(n):
        torch.cuda.synchronize();s=time.perf_counter();fn();torch.cuda.synchronize();values.append((time.perf_counter()-s)*1000)
    return {'p50_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95)),
            'resident_allocated_bytes':resident,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'incremental_peak_bytes':torch.cuda.max_memory_allocated()-resident,'repeats':n}


def cost_and_report(out,cfg,p):
    root=Path(__file__).resolve().parents[2];cal=read(cfg['calibration']);rows=[];checks=[]
    for mode in [*cfg['modes'],'frozen_dynamic','direct_same16','rgb16','rgb32']:
        torch.cuda.empty_cache();seed=cfg['seeds'][0]
        if mode in cfg['modes']:
            package=torch.load(out/'portable_weights'/f'{mode}_seed{seed}.pth',map_location='cpu',weights_only=False)
            net=JointCorrection([.9,.05,.05],mode=='joint_dynamic');net.load_state_dict(package['model'])
            cpu=InferencePipeline(copy.deepcopy(net),cal).eval();x=torch.rand(1,3,65,65)
            with torch.no_grad():yc=cpu(x)
            assert torch.isfinite(yc).all() and yc.shape==(1,1,65,65)
            restore=JointCorrection([.9,.05,.05],mode=='joint_dynamic');restore.load_state_dict(torch.load(out/'portable_weights'/f'{mode}_seed{seed}.pth',map_location='cpu',weights_only=False)['model'])
            with torch.no_grad():yr=InferencePipeline(restore,cal).eval()(x)
            assert torch.equal(yc,yr)
            checks.append({'mode':mode,'CPU_portable_save_load_max_difference':float((yc-yr).abs().max()),'RGB_only_shape':list(yc.shape)})
            del cpu,restore
        elif mode=='frozen_dynamic':
            net=MatchedFusion(RestrictedCorrector(16),True)
            net.load_state_dict(torch.load(Path(cfg['matched_controls'])/f'corrected_gate_seed{seed}/best.pth',map_location='cpu',weights_only=False)['model'])
        else:
            net=RGBReference(32 if mode=='rgb32' else 16,with_priors=mode=='direct_same16')
            net.load_state_dict(torch.load(Path(cfg['matched_controls'])/f'{mode}_seed{seed}/best.pth',map_location='cpu',weights_only=False)['model'])
        pipeline=InferencePipeline(net,cal,uses_priors=mode not in ['rgb16','rgb32']).cuda().eval()
        if mode in cfg['modes']:
            with torch.no_grad():yg=pipeline(x.cuda()).cpu()
            checks[-1]['CPU_FP32_vs_CUDA_AMP_max_difference']=float((yg-yc).abs().max())
        params=sum(v.numel() for v in pipeline.parameters());buffers=sum(v.numel() for v in pipeline.buffers())
        with torch.no_grad():
            for size in [256,512]:
                rgb=torch.rand(1,3,size,size,device='cuda')
                gpu=timed(lambda:pipeline(rgb))
                b=io.BytesIO();Image.fromarray((rgb[0].permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(b,format='PNG');data=b.getvalue()
                def decoded():
                    image=np.asarray(Image.open(io.BytesIO(data)).convert('RGB'),dtype=np.float32)/255
                    return pipeline(torch.from_numpy(image.copy()).permute(2,0,1)[None].cuda()).cpu()
                full=timed(decoded,20)
                rows.append({'method':mode,'size':size,'all_network_parameters':params,'all_registered_buffer_elements':buffers,
                   'GPU_RGB_to_H':gpu,'PNG_bytes_to_CPU_H':full,'semantic_parameters':0,
                   'unique_active_calibration_scalars':sum(len(t['x'])+len(t['y']) for t in cal['candidate_tables'])+6 if pipeline.uses_priors else 0,
                   'fixed_fusion_scalars':3 if mode=='joint_fixed' else 0,
                   'precision':'FP32 priors/context; AMP network; same fast implementation offered to every prior-using method'})
                print('COST',mode,size,json.dumps(rows[-1]),flush=True)
                del rgb
        del pipeline,net;torch.cuda.empty_cache()
    write(out/'full_inference_cost.json',{'methods':rows,'batch_size':1,'CPU_threads':1,'GPU':torch.cuda.get_device_name(),
          'scope':'Measured serially after training/evaluation; one method resident. PNG path includes decode,H2D,D2H, excludes disk/network fetch.'})
    write(out/'trained_portable_inference_checks.json',checks)
    metrics=read(out/'development/nominal/summary.json');seed_metrics=read(out/'development/nominal/by_seed.json')
    ci=read(out/'development/nominal/paired_scene_bootstrap.json');table=[]
    modes=['frozen_dynamic',*cfg['modes'],'rgb16','rgb32','direct_same16']
    for mode in modes:
        c=next(a for a in rows if a['method']==mode and a['size']==256)
        table.append(f"|{mode}|{c['all_network_parameters']}|{metrics[mode]['pixel_mae']:.6f}|{metrics[mode]['region_spearman']:.6f}|{metrics[mode]['clear_response']:.6f}|{c['GPU_RGB_to_H']['p50_ms']:.2f}|")
    result={'only_development':True,'fresh_confirmation_scenes':0,'user_override_honored':True,
       'automatic_research_stop':False,'automatic_RGB_switch':False,'real_source_audit_started':False,
       'comparisons':ci,'all_seeds':cfg['seeds'],'estimated_prior_stage_wall_seconds':read(out/'budget.json')['stage_wall_seconds'],
       'interpretation':'Do not infer success from lower parameter count alone; preserve differences in MAE, spatial/clear response and complete runtime separately.'}
    write(out/'decision.json',result)
    text=f"""# \u8054\u5408\u6821\u6b63\u6784\u5efa\u4e0e\u7ed3\u679c

\u57fa\u7ebf{p['base_HEAD']}\uff0c\u5168\u90e8\u5de5\u4f5c\u5728{root}\u3002\u6309\u7528\u6237\u4fee\u8ba2\uff0c\u672c\u8f6e\u4e0d\u89e6\u53d1\u7814\u7a76\u8def\u7ebf\u7ec8\u6b62\u3001\u4e0d\u5207\u6362\u7eafRGB\u6216\u7cbe\u7b80\u5148\u9a8c\u3001\u4e0d\u6838\u9a8c\u65b0\u771f\u5b9e\u6765\u6e90\u3001\u4e0d\u6253\u5f00\u65b0\u786e\u8ba4\u3002\u6ca1\u6709\u63d0\u4ea4\u6216\u63a8\u9001\u3002

## \u5b9e\u9a8c\u53d8\u5316\u4e0e\u516c\u5e73\u8fb9\u754c

\u4ec5\u4e24\u4e2a\u65b0\u6a21\u578b\u3001\u4e09\u4e2a\u79cd\u5b50\u3002\u6cbf\u7528\u539f18\u8f6e\u6821\u6b63\u5668\u68c0\u67e5\u70b9\uff0c\u5728\u540c1536\u914d\u5bf9\u4e0a\u8ffd\u52a018\u8f6e\u8054\u5408\u66f4\u65b0\uff0c\u548c\u76f8\u540c36\u8f6e\u76d1\u7763\u673a\u4f1a\u7684\u65e2\u6709RGB\u5bf9\u7167\u6bd4\u8f83\u3002\u6ca1\u6709\u65b0\u7684\u6e05\u6670\u56fe\u76ee\u6807\u3001\u8bed\u4e49\u6216\u7269\u7406\u9879\u3002\u8bad\u7ec3\u635f\u5931\u4e3a\u6700\u7ec8H\u4efb\u52a1\u635f\u5931\u52a0.25\u500d\u72ec\u7acb\u5019\u9009\u8f85\u52a9\u635f\u5931\uff0c\u8f85\u52a9\u5019\u9009\u6bd4\u4f8b1:.25:.25\u3002\u5f00\u53d1\u68c0\u67e5\u70b9\u6309\u6700\u7ec8H\u4efb\u52a1\u635f\u5931\u9009\u62e9\uff0c\u4e0eRGB\u9009\u62e9\u76ee\u6807\u4e00\u81f4\uff1b\u8bad\u7ec3\u76ee\u6807\u7ed3\u6784\u4ecd\u4e0d\u540c\uff0c\u4e0d\u5ba3\u79f0\u5b8c\u5168\u76f8\u540c\u4f18\u5316\u8fc7\u7a0b\u3002

\u56fa\u5b9a\u7ec4\u8bad\u7ec3\u53ca\u9009\u62e9\u671f\u95f4\u4f7f\u7528\u539f\u8bad\u7ec3\u96c6\u62df\u5408\u7684\u56fa\u5b9a\u6743\u91cd\uff0c\u7f51\u7edc\u51bb\u7ed3\u540e\u91cd\u65b0\u5728\u8bad\u7ec3\u96c6\u4f18\u5316\u5168\u5c40\u878d\u5408\uff1b\u53e6\u62a5\u539f\u9009\u62e9\u6743\u91cd\u4e0b\u7ed3\u679c\uff0c\u907f\u514d\u9690\u85cf\u91cd\u62df\u5408\u5f71\u54cd\u3002\u52a8\u6001\u7ec4\u53ea\u8f93\u51fa\u51f8\u6743\u91cd\uff0c\u5404\u5019\u9009\u4ecd\u53d7\u81ea\u8eab\u5148\u9a8c\u7684\u6709\u754c\u53d8\u6362\u7ea6\u675f\uff0c\u6ca1\u6709\u81ea\u7531H\u6b8b\u5dee\u3002\u65e7\u51bb\u7ed3\u5019\u9009\u5bf9\u7167\u4fdd\u6301\u539f\u6837\uff0c\u8054\u5408\u8bad\u7ec3\u5f62\u6210\u65b0\u7684\u5019\u9009\uff0c\u4e0d\u80fd\u79f0\u4e0e\u65e7\u51bb\u7ed3\u7ec4\u5019\u9009\u5b8c\u5168\u76f8\u540c\u3002

## \u540d\u4e49\u5f00\u53d1\u7ed3\u679c\uff08\u4e09\u79cd\u5b50\u5747\u503c\uff0c\u975e\u96c6\u6210\uff09

|\u65b9\u6cd5|\u5b8c\u6574\u7f51\u7edc\u53c2\u6570|\u50cf\u7d20MAE|\u5c40\u90e8\u76f8\u5173|\u6e05\u6670\u54cd\u5e94|256\u5b8c\u6574GPU p50\u6beb\u79d2|
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

\u5b8c\u6574\u6210\u672c\u5305\u542bFP32\u5148\u9a8c\u3001\u6807\u5b9a\u3001\u5185\u5bb9\u63cf\u8ff0\u548c\u5168\u90e8\u7f51\u7edc\uff1b\u6240\u6709\u4f7f\u7528\u5148\u9a8c\u7684\u63a7\u5236\u7ec4\u540c\u6837\u83b7\u5f97\u7b49\u4ef7\u4f18\u5316\u3002\u56fa\u5b9a\u5e38\u91cf\u3001\u6ce8\u518c\u7f13\u51b2\u3001PNG\u5230CPU\u56fe\u5b8c\u6574\u6210\u672c\u3001512\u5206\u8fa8\u7387\u548c\u663e\u5b58\u53e6\u89c1full_inference_cost.json\u3002\u4e0d\u80fd\u6df7\u7528\u4e0a\u4e00\u8f6e\u65e7\u5b9e\u73b0\u8017\u65f6\u6bd4\u8f83\u65b0\u6a21\u578b\u3002

## \u8bc1\u636e\u4e0e\u9650\u5236

fast_checks.json\u8bb0\u5f55CPU/CUDA\u8fb9\u754c\u6837\u4f8b\u4e0e512\u65e7\u5f00\u53d1\u6837\u672c\u6570\u503c\u4e00\u81f4\u6027\u3001\u65e2\u6709\u6a21\u578b\u8f93\u51fa\u548c\u6210\u672c\uff1bjoint_checks.json\u68c0\u67e5\u72ec\u7acb\u5019\u9009\u3001\u8054\u5408\u68af\u5ea6\u3001\u51f8\u7ec4\u5408\u3001\u5b9e\u9645\u78c1\u76d8\u4f18\u5316\u5668/scaler/RNG\u6062\u590d\u548cRGB\u63a8\u7406\u3002data_check.json\u4e0e\u4e0a\u4e00\u8f6e\u5b8c\u6574\u7279\u5f81/\u76ee\u6807\u54c8\u5e0c\u5339\u914d\u3002\u65e7\u6743\u91cd\u54c8\u5e0c\u4e0d\u53d8\u3002\u6240\u6709\u65b0\u6a21\u578b\u7684latest\u53ef\u6062\u590d\uff0cportable_weights\u5305\u542b\u4e09\u79cd\u5b50\u6743\u91cd\u3002

development\u4e09\u4e2a\u6761\u4ef6\u5747\u4fdd\u7559\u9010\u6837\u672c\u8bef\u5dee\u3001\u9010\u5019\u9009\u6709\u7b26\u53f7\u504f\u5dee\u3001\u5171\u540c\u9ad8\u4f4e\u4f30\u3001\u8303\u56f4\u8d8a\u754c\u3001oracle MAE/MSE\u4e0b\u754c\u3001\u7a7a\u95f4\u5e72\u9884\u3001\u6e05\u6670\u54cd\u5e94\u3001\u6392\u5e8f\u4e0e\u56f0\u96be\u5185\u5bb9\u4ee3\u7406\u3002\u4ee3\u7406\u4e0d\u662f\u5730\u7269\u8bed\u4e49\uff1boracle\u4ec5\u79bb\u7ebf\u8bca\u65ad\uff0c\u4e0d\u8fdb\u5165\u8bad\u7ec3\u6216\u63a8\u7406\u3002\u65e7RGB\u5bf9\u7167\u7684MAE\u88ab\u91cd\u65b0\u8fd0\u884c\u5e76\u6838\u5bf9\u3002bootstrap\u4e3a\u56fa\u5b9a\u4e09\u4e2a\u8bad\u7ec3\u79cd\u5b50\u768416\u539f\u666f\u914d\u5bf9\u533a\u95f4\uff0c\u4e0d\u4ee3\u8868\u65b0\u786e\u8ba4\u6216\u5168\u90e8\u4f18\u5316\u968f\u673a\u6027\u3002

\u672c\u8f6e\u53ea\u80fd\u68c0\u9a8c\u7a0b\u5e8f\u5316\u5f00\u53d1\u539f\u666f\u4e0b\u7684\u7ed3\u6784\u4e0e\u8ba1\u7b97\u5047\u8bbe\uff0c\u4e0d\u80fd\u8bc1\u660e\u771f\u5b9e\u9065\u611f\u6216\u81ea\u7136\u96fe\u6709\u6548\u6027\u3002\u6570\u503c\u672a\u8fbe\u9884\u767b\u8bb05%/.01/.005\u5bb9\u5dee\u4f1a\u5982\u5b9e\u4fdd\u7559\uff1b\u6309\u7528\u6237\u8981\u6c42\uff0c\u672c\u8f6e\u4e0d\u636e\u6b64\u81ea\u52a8\u7ec8\u6b62\u8def\u7ebf\u6216\u6269\u5f20\u641c\u7d22\u3002\u6ca1\u6709\u65b0\u589e\u771f\u5b9e\u6570\u636e\u6838\u9a8c\u3001\u53bb\u96fe\u6216\u68c0\u6d4b\u5de5\u4f5c\u3002
"""
    doc=root/'docs/joint_correction_results_20261102.md'
    if not doc.exists():
        with doc.open('x',encoding='utf-8') as f:f.write(text)
    elif doc.read_text(encoding='utf-8')!=text:raise RuntimeError('Existing result report protected; inspect before regeneration')
    write(out/'result_report.json',{'path':str(doc),'sha256':digest(doc)})
