"""Generate compact numerical tables without manual transcription."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.v15.run import read

def average(summary,prefix,seeds):
    result={}
    for key in ('global_mae','pixel_mae','region_spearman','clear_response','same_scene_rank','cross_condition_cross_scene_rank'):
        v=[summary['metrics'][f'{prefix}_seed{s}'][key] for s in seeds]
        result[key]=float(np.mean(v)) if all(q is not None for q in v) else None
    return result

def fmt(x):return 'N/A' if x is None else f'{x:.6f}'

def main():
    cfg=read('configs/v2_investigation.json');out=Path(cfg['output_directory']);a=read(out/'analysis.json');budget=read(out/'budget.json');cost=read(out/'inference_cost.json')
    lines=['# v2 继续探索：自动生成的数值表','',
        '此文件来自冻结模型的逐例结果；原始 JSON/CSV 是依据。三个种子取平均，未用新测试选择种子。原始 S 为不同的光谱均值次要终点。','']
    for condition in ('nominal','unseen_airlight','display_rgb','original_S'):
        summary=read(out/condition/'summary.json');rgb=read(out/'rgb_control'/condition/'summary.json')
        lines += [f'## {condition}','','| 方法 | 整体MAE | 像素MAE | 区域相关 | 清晰响应 | 同场景排序 | 跨图排序 |','|---|---:|---:|---:|---:|---:|---:|']
        rows=[('优化固定融合',summary['metrics']['global_exact']),('原小门控',summary['metrics']['original_dynamic']),
              ('温度门控，3种子均值',average(summary,'temperature',cfg['seeds'])),('重构学生权重，3种子均值',average(summary,'student',cfg['seeds'])),
              ('RGB直接回归，3种子均值',average(rgb,'rgb',cfg['seeds']))]
        for name,r in rows:lines.append('| '+name+' | '+' | '.join(fmt(r[k]) for k in ('global_mae','pixel_mae','region_spearman','clear_response','same_scene_rank','cross_condition_cross_scene_rank'))+' |')
        lines += ['',f"同尺度凸融合 oracle 像素 MAE 下界：{summary['oracle_pixel_mae']:.6f}。真值仅离线使用。",'']
    lines += ['## 三种子与成组区间','','完整三种子值和标准差见 `analysis.json`。以下 C−B 为重构学生减优化固定融合；误差负值更好，相关正值更好。','']
    for condition in ('nominal','unseen_airlight','display_rgb'):
        for metric,r in a['paired_scene_bootstrap'][condition]['student'].items():
            lines += [f"- {condition} {metric}: {r['delta_to_global_exact']:.7f}; 95% 场景 bootstrap [{r['scene_bootstrap_95'][0]:.7f}, {r['scene_bootstrap_95'][1]:.7f}]。"]
    lines += ['','## 完整推理时间','','| 模型 | 尺寸 | 参数量 | 中位耗时ms |','|---|---:|---:|---:|']
    for r in cost:lines.append(f"| {r['mode']} | {r['size']} | {r['parameters']} | {r['p50_ms']:.3f} |")
    for r in read(out/'rgb_control/inference_cost.json'):lines.append(f"| RGB直接回归 | {r['size']} | {r['parameters']} | {r['p50_ms']:.3f} |")
    lines += ['','前两者计时包括全部先验和标定，RGB对照没有先验计算；均batch1、预热5次、计时20次，不含图像读盘。',
        '',f"本轮 GPU 阶段保守墙钟计时（以最终 budget.json 为准）：{budget['new_gpu_wall_seconds']:.2f} 秒，上限 {budget['limit_seconds']} 秒。",'']
    (out/'numerical_tables.md').write_text('\n'.join(lines),encoding='utf-8');print('Numerical tables written')

if __name__=='__main__':main()
