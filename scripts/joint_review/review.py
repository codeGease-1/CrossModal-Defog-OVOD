"""Recompute the finished joint study from saved records; no training or GPU work."""
import csv,hashlib,json,math,statistics as st,subprocess,time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'experiments/joint_correction_20261102'
OUT=ROOT/'experiments/joint_review_20261102'
DOC=ROOT/'docs/joint_correction_review_20261102.md'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def paired(rows,left,right):
    groups={}
    for mode in [left,right]:
        groups[mode]={}
        for a in rows:
            if a['method']==mode:groups[mode].setdefault(a['scene_id'],[]).append(float(a['pixel_mae']))
    scenes=sorted(groups[left]);assert scenes==sorted(groups[right])
    a=np.array([st.mean(groups[left][s]) for s in scenes]);b=np.array([st.mean(groups[right][s]) for s in scenes])
    draw=np.random.default_rng(20261102).integers(0,len(scenes),(3000,len(scenes)))
    delta=a[draw].mean(1)/b[draw].mean(1)-1
    return {'left':left,'right':right,'relative_MAE_change':float(a.mean()/b.mean()-1),'CI95':np.quantile(delta,[.025,.975]).tolist(),
            'scope':'Old 16 development scenes, fixed three fitted seeds; descriptive post-hoc comparison, not confirmation'}
def main():
    start=time.monotonic()
    if OUT.exists() or DOC.exists():raise FileExistsError('Review output/report already exist; inspect instead of overwriting')
    assert read(RUN/'COMPLETE.json')['all_stages_complete']
    protocol=read(RUN/'protocol.json');protected=read(RUN/'inherited_source_and_protection.json')
    integrity={}
    for label,entries in [('frozen_source',protocol['source_sha256']),('old_checkpoints',protected['old_checkpoints'])]:
        for name,h in entries.items():
            path=Path(name) if name.startswith('/') else ROOT/name
            assert sha(path)==h,str(path)
        integrity[label+'_verified']=len(entries)
    assert sha(Path(protocol['config']['calibration']))==protocol['calibration_sha256']
    frozen=read(RUN/'frozen.json')
    for name,h in frozen['checkpoints'].items():assert sha(RUN/name/'best.pth')==h
    assert sha(RUN/'fusion_refit.json')==frozen['fusion_refit_sha256']
    for a in read(RUN/'portable_manifest.json'):assert sha(RUN/a['file'])==a['sha256']
    integrity.update(new_best_verified=len(frozen['checkpoints']),portable_verified=6,calibration_verified=True)
    result={'scope':'Completed old-development review only; no training, new scenes or GPU work','integrity':integrity,'conditions':{}}
    input_paths=[RUN/'protocol.json',RUN/'full_inference_cost.json',RUN/'decision.json',RUN/'frozen.json',RUN/'budget.json']
    for cond in ['nominal','unseen_A','display_rgb']:
        d=RUN/'development'/cond;s=read(d/'summary.json');rows=list(csv.DictReader((d/'per_case.csv').open()))
        assert len(rows)==len(s)*1536
        for mode,a in s.items():
            rr=[v for v in rows if v['method']==mode];assert len(rr)==1536
            assert abs(st.mean(float(v['pixel_mae']) for v in rr)-a['pixel_mae'])<1e-12
            assert len({(v['seed'],v['case_id']) for v in rr})==1536
            for key in ['pixel_mse','global_mae','signed_bias','region_spearman','clear_response','dark_mae','bright_mae','low_texture_mae']:
                vals=[float(v[key]) for v in rr if v[key] not in ('','None')]
                assert all(math.isfinite(v) for v in vals)
                if vals:assert abs(st.mean(vals)-a[key])<1e-12,(mode,key)
        cd=read(d/'candidate_diagnostics.json');cand={}
        for mode in ['original','frozen_corrected','joint_fixed','joint_dynamic']:
            rr=[a for a in cd if a['method']==mode];assert len(rr)==1536
            vals={k:st.mean(a[k] for a in rr) for k in ['joint_low','joint_high','outside_fraction','oracle_mae','oracle_mse','outside_amplitude_conditional_mean']}
            for k in ['candidate_mae','candidate_bias']:vals[k]=[st.mean(a[k][i] for a in rr) for i in range(3)]
            cand[mode]=vals
        strata={}
        for key in ['pattern','brightness_index','airlight_index']:
            strata[key]={}
            for val in sorted({a[key] for a in rows}):
                strata[key][val]={m:st.mean(float(a['pixel_mae']) for a in rows if a['method']==m and a[key]==val) for m in ['joint_fixed','joint_dynamic','rgb16','direct_same16']}
        wr=read(d/'weight_diagnostics.json')
        weights={mode:[st.mean(a['mean_weights'][i] for a in wr if a['method']==mode) for i in range(3)] for mode in ['joint_fixed','joint_dynamic']}
        result['conditions'][cond]={'summary':s,'candidate_means':cand,'by_seed':read(d/'by_seed.json'),'strata':strata,'average_gate_weights':weights,
            'posthoc_paired_comparisons':[paired(rows,'joint_dynamic',m) for m in ['joint_dynamic_refit_fixed','joint_dynamic_mean','joint_dynamic_shuffle','joint_fixed']],
            'csv_recomputation':{'rows':len(rows),'methods':len(s),'all_numeric_summary_columns_match':True}}
        input_paths += [d/n for n in ['per_case.csv','summary.json','candidate_diagnostics.json','weight_diagnostics.json','by_seed.json','paired_scene_bootstrap.json','data_manifest.json']]
    result['full_inference_cost']=read(RUN/'full_inference_cost.json')
    result['decision']=read(RUN/'decision.json')
    result['completed_update_opportunities']={p.parent.name:read(p) for p in RUN.glob('*_seed*/complete.json')}
    assert len(result['completed_update_opportunities'])==6
    assert all(a['second_stage_steps']==3456 and a['total_supervised_update_opportunity']==6912 for a in result['completed_update_opportunities'].values())
    result['actual_current_round_wall_seconds']=read(RUN/'budget.json')['stage_wall_seconds']
    result['review_seconds']=time.monotonic()-start
    OUT.mkdir()
    write(OUT/'audit.json',result)
    write(OUT/'input_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in input_paths})
    write(OUT/'execution.json',{'command':'python scripts/joint_review/review.py','base_HEAD':protocol['base_HEAD'],'script_sha256':sha(Path(__file__)),'CUDA_used':False,'training_started':False})
    nom=result['conditions']['nominal'];s=nom['summary'];cost={(a['method'],a['size']):a for a in result['full_inference_cost']['methods']}
    lines=['# 联合校正实验结果复核（2026-11-02）','',
    '六个任务、评估、完整成本和推理权重导出均已完成。受限校正加动态融合在反复使用的程序化开发场景上已超过同骨干纯 RGB；参数接近，完整推理仍更慢，尚未满足“相同完整推理成本”的总目标。不能据此宣称真实遥感或新场景有效。','',
    '本复核仅重算已保存记录、验证哈希，没有启动训练或新确认测试。云端工作树：'+str(ROOT)+'；基线：'+protocol['base_HEAD']+'。','',
    '## 比较口径','',
    '48 个训练原景、16 个旧开发原景；每个原景 32 个版本，训练 1536 对、开发 512 对。原景分组无交集，输入/目标哈希与上一轮匹配。H=1-t 是合成协议下的有效遮蔽。所有均值为三个独立种子成绩平均，不是三模型集成。','',
    '两个新模式各三个种子：18 轮候选预训练加 18 轮联合训练；纯 RGB16、RGB32、同输入直接回归沿用并重验既有 36 轮控制。每个任务提供 6912 次监督更新机会，第二阶段实际执行 3456 步。预训练从开发选出的 best 开始，种子 44 的预训练 best 在第 17 轮；因此不能把机会数说成最终权重必定经历了 6912 次累积更新。','',
    '最终 H 任务损失加 0.25 倍独立候选辅助损失；暗通道、局部绝对对比度、RGB 颜色差异的辅助比例 1:0.25:0.25。没有额外清晰图辅助目标。新模型仍使用内容特征，不含地物语义或物理约束。RGB 控制的训练配对、目标与机会一致，但没有同构的三候选辅助头，优化过程不完全相同。','',
    '校正器仅对各自先验做有界变换，融合为凸组合，无自由 H 残差。它的系数网络仍读取 RGB，因此结构受限和误差改善尚不能证明三先验不可替代；后续仍需同容量、同辅助监督机会的机制控制。','',
    '## 名义开发结果','',
    '|方法|全部网络参数|像素 MAE|局部相关|清晰响应|256 完整 GPU 毫秒|',
    '|---|---:|---:|---:|---:|---:|']
    labels={'frozen_dynamic':'冻结候选＋动态融合','joint_fixed':'联合校正＋固定融合','joint_dynamic':'联合校正＋动态融合','rgb16':'同骨干纯 RGB16','rgb32':'较宽纯 RGB32','direct_same16':'同输入直接回归'}
    for m in labels:
        a=s[m];c=cost[m,256];lines.append(f"|{labels[m]}|{c['all_network_parameters']}|{a['pixel_mae']:.6f}|{a['region_spearman']:.6f}|{a['clear_response']:.6f}|{c['GPU_RGB_to_H']['p50_ms']:.2f}|")
    lines+=['','动态联合模型比 RGB16 的 MAE 低 24.22%，参数多 1.46%；比同输入直接回归低 5.61%，参数多 0.92%。三个种子都优于各自 RGB16，和同输入直接回归相比是 2/3 个种子优于，不能说每个种子均胜出。','',
    '已登记的 MAE +5%、相关下降 0.01、清晰响应增加 0.005 容差：动态组对 RGB16、RGB32、同输入直接回归的场景重采样区间上界均通过。固定组对同输入直接回归点估计差 3.87%，但区间上界为 7.52%，不能按区间标准判通过。区间固定三个已训练种子，仅对 16 个旧开发场景重采样，不覆盖全部训练随机性，更不是新确认结论。','',
    '|种子|联合固定 MAE|联合动态 MAE|RGB16 MAE|同输入直接回归 MAE|','|---|---:|---:|---:|---:|']
    for seed,ss in nom['by_seed'].items():lines.append('|'+seed+'|'+ '|'.join(f"{ss[m]['pixel_mae']:.6f}" for m in ['joint_fixed','joint_dynamic','rgb16','direct_same16'])+'|')
    lines+=['','## 校正和动态融合各做了什么','',
    '|候选集合|暗通道 MAE|对比度 MAE|颜色差 MAE|共同低估 >0.03|共同高估 >0.03|凸融合 oracle MAE|凸融合 oracle MSE|越界比例|',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,a in nom['candidate_means'].items():
        lines.append('|'+m+'|'+ '|'.join(f'{v:.6f}' for v in a['candidate_mae'])+f"|{100*a['joint_low']:.3f}%|{100*a['joint_high']:.3f}%|{a['oracle_mae']:.6f}|{a['oracle_mse']:.8f}|{100*a['outside_fraction']:.2f}%|")
    lines+=['','有符号偏差、越界条件幅度及各压力条件的完整候选诊断见 audit.json。三张候选各自的平均误差均下降；共同低估和 oracle 下界也下降，所以这轮并非仅扩大包络或用低权重隐藏差候选。对比度和颜色差候选仍显著差于暗通道，不能声称三者等量有效。越界比例采用严格数值范围，清晰端点也会影响该数，不能单独评价好坏。','',
    '动态组的同一批冻结候选：训练集重拟合固定融合 MAE 0.008641，逐图权重均值化 0.007763，打乱空间位置 0.008158，正常动态权重 0.007261。正常动态比同候选固定融合低 15.97%；均值化、打乱分别使误差增加 6.91%、12.35%。这支持空间权重已被实际利用；不代表另行重训固定模型永远不能追上。','',
    '联合固定组训练/选权重时沿用原全局权重，网络冻结后才用训练集重拟合。重拟合前 MAE 0.008063、后 0.007990，主体改善并非来自最后一次全局拟合。联合与冻结阶段还同时改变了优化路径和检查点选择目标，不能把所有提升单独归因于“解冻”。','',
    '## 完整成本与剩余差距','',
    '|方法|256 GPU 毫秒|256 PNG到CPU毫秒|512 GPU 毫秒|512 PNG到CPU毫秒|256 额外峰值 MiB|512 额外峰值 MiB|',
    '|---|---:|---:|---:|---:|---:|---:|']
    for m in ['joint_fixed','joint_dynamic','rgb16','rgb32','direct_same16']:
        a,b=cost[m,256],cost[m,512]
        lines.append(f"|{labels[m]}|{a['GPU_RGB_to_H']['p50_ms']:.2f}|{a['PNG_bytes_to_CPU_H']['p50_ms']:.2f}|{b['GPU_RGB_to_H']['p50_ms']:.2f}|{b['PNG_bytes_to_CPU_H']['p50_ms']:.2f}|{a['GPU_RGB_to_H']['incremental_peak_bytes']/2**20:.2f}|{b['GPU_RGB_to_H']['incremental_peak_bytes']/2**20:.2f}|")
    lines+=['','V100、batch=1、网络 AMP，先验/内容/标定 FP32；40 次 GPU、20 次 PNG 路径测量，报告中位数。GPU 计时包含先验、固定标定、内容描述及整个网络。PNG 路径另含解码、CPU/GPU 传输，排除磁盘/网络读取。显存为 PyTorch allocated 指标，额外峰值扣除该次驻留基线，并非整卡 nvidia-smi 显存。','',
    '相对 RGB16，动态组 256 GPU 时延增加 48.95%，PNG 路径增加 29.35%；512 GPU 增加 54.11%。参数相近不等于更便宜。相对同输入直接回归的 256 GPU 时延仍增加 13.49%。先验实现已做等价加速：同权重 512 个旧开发样本的特征差为零，优化也提供给所有先验控制，不能拿旧慢版本 RGB/先验控制衬托新方法。','',
    '## 域变化下仍失败','',
    '|条件|联合固定 MAE|联合动态 MAE|RGB16 MAE|同输入直接回归 MAE|','|---|---:|---:|---:|---:|']
    for cond,a in result['conditions'].items():lines.append('|'+cond+'|'+ '|'.join(f"{a['summary'][m]['pixel_mae']:.6f}" for m in ['joint_fixed','joint_dynamic','rgb16','direct_same16'])+'|')
    lines+=['','未见大气光下动态模型比 RGB16 差约 14.0%，显示域条件差约 6.9%，均超出 5% 点估计容差。未见 A 的候选共同低估约 50.74%，oracle MAE 0.051751，较冻结校正的 0.045529 更差。显示域共同低估约 71.13%，oracle 0.106944，虽较旧冻结校正缓解，仍明显差于原候选 oracle 0.076598。不能将名义域提升解读成物理泛化。','',
    'RGB-only 仅指推理可获得信息：不读取清晰图、真实 t/A、场景身份或人工语义。输入仍必须符合训练成像域。当前名义训练是 linear_srgb，不能把任意显示域 PNG/JPEG 除以 255 后当作同一输入域；InferencePipeline 中 display RGB 的注释不能当成自动编码转换保证。旧压力实验已暴露这个风险，冻结文件本轮未修改。','',
    '同景/跨景排序在名义域均饱和为 1，不能用它们证明跨景数值已准确。亮区代理误差仍高于全图；这些亮度/纹理代理不是地物语义。本轮没有新的语义或物理模块成绩。','',
    '## 保存、核验与下一步','',
    f"本轮流水线阶段累计 {result['actual_current_round_wall_seconds']/60:.2f} 分钟，包括预计算、训练、评估和成本，不包含上一轮预训练，也不是精确 GPU 活跃秒账单。当前没有该实验 GPU 计算进程。旧 48 份 best/latest 权重哈希和新 6 份 best、6 份 portable 均通过；新 latest 保留优化器、scaler、CPU/CUDA RNG 和协议哈希。三条件各 18432 行，12 方法的 MAE/MSE/偏差/空间/清晰/代理误差与汇总一致。",'',
    '原始证据和权重保留在 experiments/joint_correction_20261102；复核在 experiments/joint_review_20261102/audit.json、input_sha256.json、execution.json。完整成功和失败记录都保留，未提交、未推送。','',
    '建议下一步仍保留三先验，先做推理开销分解和等价执行优化，再单独处理 A/成像域变化；改变训练分布必须同时重训 RGB 控制。机制归因还需同容量、同辅助监督机会的直接回归对照，并检查候选对其原始先验的依赖。每轮至多三个主要新候选，先冻结具体协议和预算。','',
    '遵照用户修订，本复核不据未达标结果终止路线、不切换纯 RGB 或精简先验、不启动真实来源核验或新的确认集。这次“完成了”只触发结果复核，没有自动追加训练。']
    with DOC.open('x',encoding='utf-8') as f:f.write(chr(10).join(lines)+chr(10))
    write(OUT/'complete.json',{'report':str(DOC.relative_to(ROOT)),'report_sha256':sha(DOC),'audit_sha256':sha(OUT/'audit.json'),'new_training_runs':0,'new_confirmation_scenes':0})
    print(json.dumps({'report':str(DOC),'audit':str(OUT/'audit.json'),'seconds':time.monotonic()-start,'integrity':integrity}))
if __name__=='__main__':main()
