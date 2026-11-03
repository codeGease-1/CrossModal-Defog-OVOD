"""Bounded post-training evaluation/export and candid Chinese report."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.v15.run import read,write,digest


def report(out):
    p=read(out/'protocol.json');cfg=p['config'];nom=read(out/'development/nominal/summary.json')
    seeds=read(out/'development/nominal/by_seed.json');cost=read(out/'complete_inference_cost.json')
    times={r['method']:r for r in cost['methods'] if r['size']==256}
    modes=['original_fixed','corrected_fixed','raw_gate','corrected_gate','rgb16','direct_same16','rgb32']
    lines=[]
    for mode in modes:
        v=nom[mode];c=times[mode]
        lines.append(f"|{mode}|{c['network_parameters_total']:,}|{v['pixel_mae']:.6f}|{v['region_spearman']:.6f}|{v['clear_response']:.6f}|{c['GPU_resident_RGB_to_H']['p50_ms']:.2f}|")
    comparisons=read(out/'development/nominal/paired_scene_bootstrap.json')
    per_seed=[]
    for seed in cfg['seeds']:
        v=seeds[str(seed)]
        per_seed.append({'seed':seed,'corrected_gate_mae':v['corrected_gate']['pixel_mae'],
           'corrected_fixed_mae':v['corrected_fixed']['pixel_mae'],'rgb16_mae':v['rgb16']['pixel_mae'],
           'rgb32_mae':v['rgb32']['pixel_mae'],'direct_same16_mae':v['direct_same16']['pixel_mae']})
    improvement=sum(r['corrected_gate_mae']<r['corrected_fixed_mae'] for r in per_seed)
    best=min(['rgb16','direct_same16','rgb32'],key=lambda k:nom[k]['pixel_mae'])
    ratio=nom['corrected_gate']['pixel_mae']/nom[best]['pixel_mae']-1
    ci=next(x for x in comparisons if x['right']==best)
    passed=bool(ci['all_upper_bounds_within_margins'] and improvement>=2)
    if passed:decision="\u5f00\u53d1\u7b5b\u67e5\u8fbe\u5230\u5df2\u767b\u8bb0\u76f8\u8fd1\u5bb9\u5dee\uff0c\u4ecd\u672a\u901a\u8fc7\u65b0\u7684\u786e\u8ba4\u6d4b\u8bd5\uff1b\u4e0b\u4e00\u6b65\u5148\u5ba1\u9605\u5b8c\u6574\u6210\u672c\u4e0e\u79cd\u5b50\u7a33\u5b9a\u6027\u3002"
    elif improvement>=2 and nom['corrected_gate']['pixel_mae']<nom['corrected_fixed']['pixel_mae']:
        decision="\u52a8\u6001\u878d\u5408\u6709\u5f00\u53d1\u6536\u76ca\uff0c\u4f46\u5c1a\u672a\u5728\u6700\u4f73\u5339\u914d\u56de\u5f52\u5bf9\u7167\u4e0b\u6ee1\u8db3\u76f8\u8fd1\u6807\u51c6\uff1b\u505c\u6b62\u6269\u5f20\u6a21\u5757\uff0c\u4fdd\u7559\u5dee\u8ddd\u548c\u5931\u8d25\u8bc1\u636e\u3002"
    else:decision="\u52a8\u6001\u878d\u5408\u672a\u5f62\u6210\u7a33\u5b9a\u6536\u76ca\uff1b\u505c\u6b62\u589e\u52a0\u95e8\u63a7\u590d\u6742\u5ea6\uff0c\u4e0d\u81ea\u52a8\u6269\u5f20\u641c\u7d22\u6216\u6253\u5f00\u65b0\u786e\u8ba4\u96c6\u3002"
    stages=read(out/'postprocess_status.json').get('stages',[])
    total=read(out/'budget.json')['stage_wall_seconds']+sum(a['seconds'] for a in stages)
    summary={'only_development':True,'new_confirmation_scenes':0,'three_seed_mean_not_ensemble':True,
       'best_development_regression':best,'relative_MAE_increase_vs_best_regression':ratio,
       'corrected_gate_improves_fixed_seed_count':improvement,'registered_margins_development_pass':passed,
       'decision':decision,'per_seed':per_seed,'measured_stage_wall_seconds_this_round':total,
       'old_pretraining_cost':'Prior first 18 epochs reused, accounted separately, not charged again',
       'stop_no_automatic_additional_candidates':True}
    write(out/'decision.json',summary)
    text=f"""# \u5339\u914d\u8bad\u7ec3\u673a\u4f1a\u7684\u5019\u9009\u6821\u6b63\u4e0e\u52a8\u6001\u878d\u5408\u5b9e\u9a8c

\u57fa\u7ebf\u63d0\u4ea4\uff1a{p['base_HEAD']}\u3002\u5de5\u4f5c\u6811\uff1a{ROOT}\u3002\u5168\u90e8\u7f16\u8f91\u3001\u8bad\u7ec3\u548c\u6743\u91cd\u5728/mnt\uff0c\u672a\u81ea\u52a8\u63d0\u4ea4\u6216\u63a8\u9001\u3002

## \u7ed3\u8bba

{decision}

\u8fd9\u662f48\u4e2a\u7a0b\u5e8f\u5316\u8bad\u7ec3\u539f\u666f\u300116\u4e2a\u5f00\u53d1\u539f\u666f\u7684\u7ed3\u6784\u7b5b\u67e5\uff0c\u4e0d\u80fd\u89c6\u4e3a\u771f\u5b9e\u9065\u611f\u96fe\u6216\u65b0\u786e\u8ba4\u6d4b\u8bd5\u7ed3\u8bba\u3002\u4e09\u4e2a\u79cd\u5b50\u5747\u503c\u4e0d\u662f\u96c6\u6210\u9884\u6d4b\u3002\u7a0b\u5e8f\u5316\u7c7b\u522b\u540d\u79f0\u4e0d\u662f\u771f\u5b9e\u5730\u7269\u8bed\u4e49\u3002

## \u76f8\u540c\u6570\u636e\u4e0e\u8bad\u7ec3\u673a\u4f1a

\u8bad\u7ec31536\u914d\u5bf9\u3001\u5f00\u53d1512\u914d\u5bf9\uff0c256\u00d7256\u3002\u540c\u539f\u666f\u7684\u96fe\u3001\u4eae\u5ea6\u548c\u5927\u6c14\u5149\u7248\u672c\u540c\u7ec4\u3002\u6cbf\u7528\u8bad\u7ec3\u96c6\u56fa\u5b9a\u6807\u5b9a\uff0c\u76ee\u6807H=1-t\uff0c\u4e0d\u84b8\u998f\u65e7\u8bc4\u5206\u3002

\u6821\u6b63\u5668\u5148\u8bad\u7ec318\u8f6e\u540e\u51bb\u7ed3\uff0c\u95e8\u63a7\u8ffd\u52a018\u8f6e\uff1bRGB\u5bf9\u7167\u603b\u8ba136\u8f6e\u30016912\u6b21\u6279\u6b21\u66f4\u65b0\u673a\u4f1a\u3002\u5df2\u6709RGB32\u53ca\u540c\u8f93\u5165\u56de\u5f52\u4ece\u539flatest\u8fde\u540c\u4f18\u5316\u5668\u3001scaler\u548c\u968f\u673a\u72b6\u6001\u7ee7\u7eed\u8bad\u7ec3\uff1b\u7eafRGB16\u4ece\u593436\u8f6e\u3002\u76f8\u540c\u914d\u5bf9\u3001\u76ee\u6807\u3001batch=8\u3001Adam\u5b66\u4e60\u7387.001\u3001AMP\u548c\u5f00\u53d1task loss\u9009\u62e9\u89c4\u5219\u3002\u4e24\u9636\u6bb5\u548c\u76f4\u63a5\u56de\u5f52\u7684\u76ee\u6807\u7ed3\u6784\u4e0d\u540c\uff0c\u4e0d\u58f0\u79f0\u76f8\u540c\u4f18\u5316\u8fc7\u7a0b\u3002

\u539f\u5019\u9009\u95e8\u63a7\u540c\u6837\u4f7f\u7528\u6821\u6b63\u9636\u6bb5\u8bad\u7ec3\u51fa\u7684\u51bb\u7ed3\u7279\u5f81\uff0c\u662f\u5019\u9009\u56fe\u6d88\u878d\uff0c\u4e0d\u7b49\u540c\u5386\u53f2v2\u3002\u95e8\u63a7\u53ea\u6709\u4e09\u901a\u9053softmax\uff0c\u6ca1\u6709\u81ea\u7531H\u6b8b\u5dee\uff0c\u4e09\u5019\u9009\u4e0d\u968f\u95e8\u63a7\u8bad\u7ec3\u53d8\u5316\u3002

## \u5f00\u53d1\u7ed3\u679c\u4e0e\u5b8c\u6574\u6210\u672c

|\u65b9\u6cd5|\u5168\u90e8\u795e\u7ecf\u7f51\u7edc\u53c2\u6570|\u50cf\u7d20MAE|\u5c40\u90e8\u76f8\u5173|\u6e05\u6670\u54cd\u5e94|256\u5b8c\u6574GPU\u63a8\u7406p50\u6beb\u79d2|
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

\u6210\u672c\u4eceGPU\u4e0a\u7684RGB\u5230H\uff0c\u5305\u542b\u5b9e\u9645\u5148\u9a8c\u3001\u6807\u5b9a\u3001\u5185\u5bb9\u63cf\u8ff0\u3001\u51bb\u7ed3\u4e0e\u53ef\u8bad\u7ec3\u7f51\u7edc\u53ca\u878d\u5408\u3002\u56fa\u5b9a\u8868\u548c\u878d\u5408\u5e38\u91cf\u53e6\u5217\uff0c\u4e0d\u7b97\u4f5c\u53ef\u8bad\u7ec3\u7f51\u7edc\u53c2\u6570\u3002PNG\u89e3\u7801\u81f3\u8f93\u51faCPU\u56fe\u3001512\u5206\u8fa8\u7387\u3001\u663e\u5b58\u548c\u7ec4\u4ef6\u6210\u672c\u89c1complete_inference_cost.json\u3002\u539f\u5019\u9009\u6d88\u878d\u4fdd\u7559\u76f8\u540c\u6821\u6b63\u5668\u5bf9\u8c61\u53ca\u8ba1\u7b97\uff0c\u672a\u6d4b\u7684\u90e8\u7f72\u7b80\u5316\u4e0d\u80fd\u4f5c\u4e3a\u7701\u65f6\u7ed3\u679c\u3002

\u9010\u79cd\u5b50\u7ed3\u679c\u89c1development/nominal/by_seed.json\u3002\u76f8\u5bf9\u6700\u4f73\u5f00\u53d1\u56de\u5f52{best}\uff0c\u6821\u6b63\u95e8\u63a7MAE\u53d8\u5316{ratio:+.2%}\uff1b\u6539\u5584\u56fa\u5b9a\u6821\u6b63\u878d\u5408\u7684\u79cd\u5b50{improvement}/3\u3002\u573a\u666f\u914d\u5bf9bootstrap\u53ea\u8986\u76d6\u56fa\u5b9a\u4e09\u4e2a\u8bad\u7ec3\u79cd\u5b50\u7684\u573a\u666f\u62bd\u6837\uff0c\u4e0d\u4ee3\u8868\u8bad\u7ec3\u968f\u673a\u6027\u6216\u65b0\u786e\u8ba4\u3002

## \u8bca\u65ad\u4e0e\u8fb9\u754c

\u9010\u5019\u9009\u8bef\u5dee/\u504f\u5dee\u3001\u5171\u540c\u9ad8\u4f4e\u4f30\u3001\u771f\u503c\u8d8a\u754c\u6bd4\u4f8b/\u5e45\u5ea6\u3001\u51f8\u878d\u5408MAE/MSE\u4e0b\u754c\u5747\u4fdd\u7559\u3002\u52a8\u6001\u878d\u5408\u4e0d\u6539\u53d8\u51bb\u7ed3\u5019\u9009\u6307\u6807\uff0c\u53ea\u68c0\u9a8c\u4e92\u8865\u80fd\u5426\u5229\u7528\u3002oracle\u4ec5\u79bb\u7ebf\u8bca\u65ad\uff0c\u4e0d\u8fdb\u5165\u8bad\u7ec3\u6216\u63a8\u7406\u3002\u5747\u503c\u5316/\u6253\u4e71\u6743\u91cd\u53ea\u4f5c\u63a8\u7406\u5e72\u9884\uff0c\u4e0d\u80fd\u4ee3\u66ff\u5148\u9a8c\u5fc5\u8981\u6027\u7684\u5339\u914d\u91cd\u8bad\u3002

\u56f0\u96be\u5185\u5bb9\u7528\u6e05\u6670\u6765\u6e90\u4eae\u5ea6/\u68af\u5ea6\u4f5c\u79bb\u7ebf\u4ee3\u7406\uff0c\u4e0d\u662f\u5730\u7269\u8bed\u4e49\u3002\u63a8\u7406\u6ca1\u6709\u6e05\u6670\u56fe\u3001\u771f\u5b9et/A\u3001\u573a\u666f\u8eab\u4efd\u6216oracle\u3002\u672a\u89c1A\u4e0e\u663e\u793a\u57df\u6df7\u5408\u53ea\u5728\u65e7\u5f00\u53d1\u539f\u666f\u68c0\u67e5\u3002\u5f53\u524d\u7269\u7406\u9879\u5173\u95ed\uff0c\u6ca1\u6709\u72ec\u7acb\u7269\u7406\u8d21\u732e\u3002

## \u9a8c\u8bc1\u3001\u6062\u590d\u4e0e\u9884\u7b97

contract_checks.json\u542bCPU/CUDA\u5019\u9009\u5b8c\u5168\u4e00\u81f4\u3001\u51bb\u7ed3\u68af\u5ea6\u3001\u51f8\u7ec4\u5408\u3001\u4f18\u5316\u5668\u6062\u590d\uff1btrained_inference_checks.json\u542b\u771f\u5b9e\u6743\u91cdCPU\u6062\u590d\u548cCUDA\u63a8\u7406\u3002\u4fdd\u7559\u6570\u636e\u6e05\u5355\u3001\u6e90\u7801\u5feb\u7167\u3001\u54c8\u5e0c\u3001\u73af\u5883\u3001\u547d\u4ee4\u53ca\u5168\u8f6e\u5386\u53f2\uff1b\u65e7best/latest\u6821\u9a8c\u4e0d\u53d8\u3002

\u6062\u590d\uff1a\u5728\u4e0a\u8ff0\u5de5\u4f5c\u6811\u8fd0\u884c python -u scripts/matched_fusion/run.py --stage train\u3002\u65b0latest\u5305\u542b\u6a21\u578b\u3001\u4f18\u5316\u5668\u3001scaler\u548cshuffle/CPU/CUDA\u968f\u673a\u72b6\u6001\u3002portable_weights\u542b\u6240\u6709\u79cd\u5b50\u5b8c\u6574\u63a8\u7406\u6743\u91cd\u3002\u63a8\u7406\u5165\u53e3scripts/matched_evaluation/predict.py\u63a5\u53d7RGB\uff0c\u8f93\u51fa\u539f\u59cbnpy\u548c\u56fa\u5b9a[0,1]\u8272\u6807PNG\u3002

\u672c\u8f6e\u9636\u6bb5\u5899\u949f\u7ea6{total/60:.2f}\u5206\u949f\uff1b\u65e7\u9636\u6bb5\u8bad\u7ec3\u53e6\u5217\uff0c\u9636\u6bb5\u5899\u949f\u4e0d\u540c\u4e8e\u7eafCUDA kernel\u65f6\u95f4\u3002\u4e0d\u4f1a\u81ea\u52a8\u65b0\u589e\u5019\u9009\u6216\u6253\u5f00\u65b0\u786e\u8ba4\u3002\u53ea\u6709\u7ed3\u679c\u652f\u6301\u65f6\uff0c\u624d\u8fdb\u4e00\u6b65\u505a\u5148\u9a8c\u5fc5\u8981\u6027\u5339\u914d\u91cd\u8bad\u6216\u771f\u5b9e\u6765\u6e90\u6570\u636e\u8d44\u683c\u6838\u9a8c\u3002
"""
    doc=ROOT/'docs/\u5339\u914d\u8bad\u7ec3\u673a\u4f1a\u7684\u5019\u9009\u6821\u6b63\u52a8\u6001\u878d\u5408\u62a5\u544a_20261102.md'
    doc.write_text(text,encoding='utf-8');(out/'README.md').write_text(text,encoding='utf-8')
    return summary


def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=ROOT/'experiments/matched_fusion_20261102')
    a.add_argument('--report-only',action='store_true');args=a.parse_args();out=args.output
    if args.report_only:print(json.dumps(report(out),ensure_ascii=True));return
    receipt=out/'postprocess_status.json'
    if receipt.exists():raise FileExistsError('Postprocess receipt exists; inspect individual stages before resuming')
    files=sorted((ROOT/'scripts/matched_evaluation').glob('*.py'))
    protocol={'sources':{str(p.relative_to(ROOT)):digest(p) for p in files},
              'training_protocol_sha256':digest(out/'protocol.json'),
              'scope':'Old development only, no automatic confirmation or candidate expansion',
              'commands':[['scripts/matched_evaluation/resume_check.py'],
                          ['scripts/matched_evaluation/run.py','--stage','evaluate'],
                          ['scripts/matched_evaluation/run.py','--stage','cost'],
                          ['scripts/matched_evaluation/export.py']]}
    write(out/'evaluation_protocol.json',protocol)
    for rel in protocol['sources']:
        dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
    state={'status':'waiting_for_training','stages':[]};write(receipt,state)
    pid=read(out/'launch.json')['pid'];deadline=time.monotonic()+3*3600
    while True:
        proc=Path(f'/proc/{pid}/stat');alive=proc.exists() and proc.read_text().split()[2]!='Z'
        if not alive:break
        if time.monotonic()>deadline:
            state.update(status='timeout_training_still_running');write(receipt,state);return
        time.sleep(5)
    if not (out/'training_complete.json').exists():
        state.update(status='training_failed_inspect_training_log');write(receipt,state);return
    for rel,sha in protocol['sources'].items():assert digest(ROOT/rel)==sha,rel
    for cmd in protocol['commands']:
        label=('resume_check' if cmd[0].endswith('resume_check.py') else ('evaluate' if 'evaluate' in cmd else ('cost' if 'cost' in cmd else 'export')))
        log=out/(label+'.log');assert not log.exists()
        state.update(status='running_'+label);write(receipt,state);started=time.monotonic()
        with log.open('wb') as f:
            p=subprocess.run([sys.executable,'-u',*cmd],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        state['stages'].append({'stage':label,'command':[sys.executable,*cmd],'returncode':p.returncode,'seconds':time.monotonic()-started})
        write(receipt,state)
        if p.returncode:
            state.update(status='failed_'+label);write(receipt,state);return
    state.update(status='reporting');write(receipt,state);summary=report(out)
    for path,sha in read(out/'protocol.json')['protected_previous_checkpoints'].items():assert digest(path)==sha,path
    state.update(status='complete',decision=summary);write(receipt,state);print(json.dumps(state,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
