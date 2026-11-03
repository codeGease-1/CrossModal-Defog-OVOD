"""Wait for the finite pilot, independently recompute metrics and write report."""
import sys,json,csv,time,hashlib,statistics,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
O=ROOT/'experiments/optimization_20261103';P=O/'airlight_pilot'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    while not (P/'COMPLETE.json').exists():
        if (P/'status.json').exists() and read(P/'status.json')['stage']=='failed':raise RuntimeError(read(P/'status.json'))
        time.sleep(10)
    rows=read(O/'latency_run03/costs.json');lat={}
    for m in ['joint_dynamic','rgb16','direct_same16']:
        lat[m]={}
        for size in [256,512]:
            lat[m][size]={}
            for impl in ['eager','captured']:
                rr=[a for a in rows if a['method']==m and a['size']==size and a['implementation']==impl]
                lat[m][size][impl]={k:statistics.mean(a[k]['p50_ms'] for a in rr) for k in ['GPU_RGB_to_H','PNG_bytes_to_CPU_H']}
                lat[m][size][impl]['cold_build_seconds']=statistics.mean(a['capture_build_seconds'] for a in rr)
    s=read(P/'summary.json');audit={'latency':lat,'airlight_summary':s,'recomputed':{},'paired_scene_intervals':{},'immutable_hash_checks':{}}
    for cond in ['nominal','unseen_A','display_rgb']:
        with (P/'evaluation'/cond/'per_case.csv').open() as f:rr=list(csv.DictReader(f))
        count=0
        for mode in ['joint_dynamic','rgb16','direct_same16']:
            for stage in ['before','after']:
                vals=[float(a['pixel_mae']) for a in rr if a['mode']==mode and a['stage']==stage]
                assert len(vals)==1536
                assert abs(statistics.mean(vals)-s[cond][mode+'_'+stage]['pixel_mae'])<1e-12
                count+=1
        groups=sorted({a['group_id'] for a in rr});draw=np.random.default_rng(20261103).integers(0,len(groups),(3000,len(groups)))
        contrasts={}
        for left,right in [('joint_dynamic_after','joint_dynamic_before'),('joint_dynamic_after','rgb16_after'),('joint_dynamic_after','direct_same16_after')]:
            arrays=[]
            for name in [left,right]:
                mode,stage=name.rsplit('_',1)
                arrays.append(np.array([statistics.mean(float(a['pixel_mae']) for a in rr if a['mode']==mode and a['stage']==stage and a['group_id']==g) for g in groups]))
            a,b=arrays;d=a[draw].mean(1)/b[draw].mean(1)-1
            contrasts[left+'_vs_'+right]=dict(relative_MAE_change=float(a.mean()/b.mean()-1),CI95=np.quantile(d,[.025,.975]).tolist(),scope='Old 16 scenes, fixed three fitted seeds; exploratory development only')
        audit['recomputed'][cond]=dict(rows=len(rr),method_stages=count);audit['paired_scene_intervals'][cond]=contrasts
    protected=read(O.parent.parent/'experiments/candidate_correction_delivery_20261102/joint_correction/inherited_source_and_protection.json')['old_checkpoints']
    for p,h in protected.items():assert sha(Path(p))==h
    audit['immutable_hash_checks']['old_checkpoints_verified']=len(protected)
    for rec in read(P/'portable_manifest.json'):assert sha(P/rec['path'])==rec['sha256']
    cfg=read(P/'protocol.json');assert sha(Path(cfg['frozen_calibration_path']))==cfg['frozen_calibration_sha256']
    audit['immutable_hash_checks']['frozen_calibration']=True
    # Full file CLI smoke test, on an exposed original; no target passed to inference.
    from scripts.v15.run import procedural
    from PIL import Image
    asset=O/'inference_smoke';asset.mkdir();clear,_=procedural(20271101,65);im=asset/'input.png'
    Image.fromarray((clear*255).astype('uint8')).save(im)
    weights=P/'portable_weights/joint_dynamic_seed42.pth'
    for impl in ['eager','captured']:
        subprocess.run(['python','scripts/optimization_20261103/predict.py','--weights',str(weights),'--image',str(im),'--output',str(asset/impl),'--execution',impl,'--device','cuda'],cwd=ROOT,check=True)
    a=np.load(asset/'eager.npy');b=np.load(asset/'captured.npy');delta=float(np.max(np.abs(a-b)));assert delta<=1e-5
    audit['actual_file_inference']={'max_abs_difference':delta,'output_shape':list(a.shape),'weights_sha256':sha(weights),'shared_color_scale':[0,1]}
    save(O/'audit.json',audit)
    lines=['# 2026-11-03 优化与大气光覆盖试验','',
        '昨天的精选归档已提交并推送为 ed4cc48c05e1f767e006d9f155af2f65b0f4aed3，今天的代码从该提交拉到 /mnt 独立工作树。简明贡献说明另见 research/昨日成果简报_20261102.md。','',
        '## 先给结论','',
        'CUDA 图使绝对推理时间下降，但公平加速 RGB 后，相对成本差距仍在。大气光增强的收益和名义域代价见下表；不能将六轮开发试验视为泛化问题已解决。没有新增确认、真实数据核验或语义/物理模块。','',
        '## 冻结权重的推理优化','',
        '|模型|256 原实现 ms|256 捕获 ms|512 原实现 ms|512 捕获 ms|256 PNG到CPU原实现/捕获 ms|','|---|---:|---:|---:|---:|---:|']
    for m in lat:
        a=lat[m][256];b=lat[m][512]
        lines.append(f"|{m}|{a['eager']['GPU_RGB_to_H']:.3f}|{a['captured']['GPU_RGB_to_H']:.3f}|{b['eager']['GPU_RGB_to_H']:.3f}|{b['captured']['GPU_RGB_to_H']:.3f}|{a['eager']['PNG_bytes_to_CPU_H']:.3f}/{a['captured']['PNG_bytes_to_CPU_H']:.3f}|")
    ratio=lat['joint_dynamic'][256]['captured']['GPU_RGB_to_H']/lat['rgb16'][256]['captured']['GPU_RGB_to_H']-1
    lines += ['',f"本次捕获实现的校正模型仍比同样优化的 RGB16 慢 {ratio:.1%}。不能拿新校正实现与旧 RGB 耗时混比。均为 V100、batch=1、三轮测量的中位数均值；网络 AMP、先验 FP32、公共范围校验和输出独立拷贝包含在计时中。PNG 路径含解码和双向传输，不含磁盘/网络读取。",'',
        '捕获前先把内容特征的常量注册为缓冲区；每次调用仍校验输入，再拷入静态缓冲、执行整个先验与网络、克隆输出。三个模型各 512 个旧开发样本、常量与边界检查均为输出差零；实际文件推理也通过。形状/设备/dtype 固定，单调用者，非线程安全；冷启动构建时间与每次调用分开记录，遇到新尺寸需另建执行器。对 CPU 使用 eager 路径。','',
        '成本原始表包含 p50/p95、驻留/峰值 allocated 显存、捕获构建成本。测量进程同时持有参考与捕获版本，不能把该显存数当成单模型整卡显存，图私有池和 CUDA 上下文也不能只用增量 allocated 解释。当前与昨日设置/实例可能不同，以同一次两种执行的公平对照为准。','',
        '可分离局部均值试验未通过固定数值门槛：特征最大差 0.0001457、H 最大差 0.0002991，高于 1e-6/1e-5；没有降低门槛，没有加入正式推理路径。两个早期构建问题（缺失导入、捕获中创建设备常量）及失败日志保留在云端，最终源码已修复。','',
        '## 固定结构和标定，追加六轮 A 增强','',
        '三个模型各三个种子从昨日各自的 36 轮机会 best 起点追加相同 1536 对、六轮，共每模型 1152 次额外更新。每个训练原景两个 A，RGB 各分量在 [0.55,1] 采样；亮度/t/加雾机制保持原协议。校正/标定没有获得额外真值，旧训练标定固定以隔离 A 覆盖；这是后续全面重拟合前的小试验。','',
        '检查点按旧 nominal 和 unseen_A 开发任务损失等权选择；这些场景已暴露，只是开发参考。各架构起点和优化目标不同，联合组保留候选辅助损失，并非同构辅助头的因果控制。','',
        '|条件|联合动态 前→后 MAE|RGB16 前→后 MAE|同输入回归 前→后 MAE|','|---|---:|---:|---:|']
    for cond in s:
        vals=[]
        for mode in ['joint_dynamic','rgb16','direct_same16']:
            vals.append(f"{s[cond][mode+'_before']['pixel_mae']:.6f}→{s[cond][mode+'_after']['pixel_mae']:.6f}")
        lines.append('|'+cond+'|'+'|'.join(vals)+'|')
    lines+=['','全部种子的 MAE、局部相关、清晰响应、有符号偏差、共同高低估、候选误差和 oracle 保存在原始表及 audit.json。表中是三种子平均，非集成。区间只对 16 个固定旧开发原景重采样，不覆盖完整训练随机性。','',
        '如果未见 A 改善却名义域退步，应记为分布覆盖带来的取舍，不能声称全部解决；如果 RGB 控制改善更多，也不能把数据收益归因于三先验独有贡献。显示域压力是直接在显示域混合造成的生成机制变化，而网络名义输入本来就是显示 RGB，简单逆 gamma 不是修复。','',
        '## 可运行入口与保存位置','',
        '云端项目：/mnt/CrossModal-Defog-OVOD-optimize-20261103。监控：python scripts/optimization_20261103/status.py --watch。训练恢复：python -u scripts/optimization_20261103/airlight_pilot.py，需原始工作树 HEAD 和匹配的源码/配置，不覆盖原 best/latest。完整恢复状态与小型 portable_weights 留在 experiments/optimization_20261103/airlight_pilot；不进 Git。','',
        '推理：python scripts/optimization_20261103/predict.py --weights <portable.pth> --image <RGB.png> --output <新前缀> --execution captured --device cuda；CPU 选 --execution eager --device cpu。输入需符合成像协议，输出原始 H 与共同 [0,1] 色标。','',
        '下一步优先根据完整成本和名义/变化 A 的取舍设计受控方案；先不增加自由 H 头、语义分割或硬物理反演。保留三先验，不据失败自动终止路线或打开新确认。']
    doc=ROOT/'docs/optimization_results_20261103.md';assert not doc.exists()
    with doc.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    save(O/'POSTPROCESS_COMPLETE.json',{'report':str(doc.relative_to(ROOT)),'report_sha256':sha(doc),'audit_sha256':sha(O/'audit.json'),'all_csv_recomputations':18,'actual_CLI_difference':delta})
    print('POSTPROCESS COMPLETE',flush=True)
if __name__=='__main__':main()
