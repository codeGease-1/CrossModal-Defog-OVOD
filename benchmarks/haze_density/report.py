"""Fixed-scale plots and descriptive tables, without test-set normalization."""
import csv
import io
import os
from pathlib import Path
import tempfile
import numpy as np
from .common import read_json
from .metrics import rank_residual


def csv_table(path, rows):
    if not rows:
        Path(path).write_text('status\nNOT RUN\n', encoding='utf-8')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render_report(output, max_figures=12):
    output = Path(output)
    data = read_json(output / 'results.json')
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'haze_benchmark_matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    summary = []
    for row in data['summary']:
        summary.append({k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in row.items()})
    csv_table(output / 'summary.csv', summary)
    csv_table(output / 'ranking.csv', data['rankings'])
    csv_table(output / 'condition_strata.csv', [
        {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in row.items()}
        for row in data['condition_strata']])
    csv_table(output / 'spatial_strata.csv', [
        {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in row.items()}
        for row in data.get('spatial_strata', [])])
    csv_table(output / 'per_image.csv', [{k: v for k, v in row.items() if k != 'confound_regions'} for row in data['images']])
    # Choose coverage examples by scene/pattern before looking at response values.
    selected, seen = [], set()
    for row in sorted(data['images'], key=lambda r: (r['pattern'], r['scene_id'], r['case_id'], r['method'])):
        key = (row['scene_id'], row['pattern'], row['method'])
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= max_figures:
            break
    image_links = []
    for index, row in enumerate(selected):
        with np.load(Path(data['dataset_directory']) / row['path'], allow_pickle=False) as arrays:
            original, clear, hazy, t = (arrays[k] for k in ('original_rgb', 'clear_rgb', 'hazy_rgb', 'transmission'))
        p = np.load(output / row['prediction_path'], allow_pickle=False)
        residual = rank_residual(t, p, data['config']['region_grid'])
        fig, axes = plt.subplots(1, 6, figsize=(18, 3.6), constrained_layout=True)
        panels = [(original, 'Original clear RGB'), (clear, 'Brightness-adjusted J'), (hazy, 'Synthetic I'),
                  (t, 'Known transmission t\nLower = denser haze'), (p, row['method'] + ' relative response\nHigher = denser by design')]
        for ax, (values, title) in zip(axes, panels):
            artist = ax.imshow(values, vmin=0, vmax=1, cmap='viridis' if values.ndim == 2 else None)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
            if values.ndim == 2:
                fig.colorbar(artist, ax=ax, shrink=.7)
        if residual is None:
            axes[5].text(.5, .5, 'N/A\nConstant spatial truth', ha='center', va='center')
        else:
            artist = axes[5].imshow(residual, vmin=-1, vmax=1, cmap='coolwarm', interpolation='nearest')
            fig.colorbar(artist, ax=axes[5], shrink=.7)
        axes[5].set_title('Regional rank residual\nDimensionless; NOT physical MAE', fontsize=9)
        axes[5].axis('off')
        fig.suptitle(f'{row["scene_id"]} | {row["land_cover"]} | {row["pattern"]} | A={row["airlight"]}, brightness={row["brightness"]}', fontsize=10)
        filename = f'comparison_{index:02d}.png'
        fig.savefig(figures / filename, dpi=130)
        plt.close(fig)
        image_links.append(f'![{row["scene_id"]} {row["method"]}](figures/{filename})')
    curve_groups = {}
    for row in data['rankings']:
        key = (row['scene_id'], tuple(row['airlight']), row['brightness'])
        curve_groups.setdefault(key, []).append(row)
    curve_links = []
    for index, (key, rows) in enumerate(list(curve_groups.items())[:max_figures]):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for row in rows:
            ax.plot(row['strengths'], row['scores'], marker='o', label=row['method'])
        ax.set(xlabel='Known synthetic degradation 1 - t', ylabel='Mean relative response (uncalibrated)',
               ylim=(0, 1), title=f'{key[0]} | A={key[1]} | brightness={key[2]}')
        ax.grid(alpha=.25)
        ax.legend()
        name = f'curve_{index:02d}.png'
        fig.savefig(figures / name, dpi=130)
        plt.close(fig)
        curve_links.append(f'![ranking curve {key[0]}](figures/{name})')
    def fmt(value):
        return 'N/A' if value is None else f'{value:.4f}' if isinstance(value, (int, float)) else str(value)
    lines = ['# 独立雾响应评测报告', '', f'执行状态：{data["status"]}；数据性质：{data["data_kind"]}；划分：{data["split"]}。',
             '', 'PASS 表示评测实现成功运行，不代表算法已经通过科学有效性验证。缺少预注册验收阈值时，以下指标均为描述性结果。',
             '', '## 方法可用性', '', '| 方法 | 状态 | 说明 |', '|---|---|---|']
    for method in data['methods']:
        lines.append(f'| {method["method"]} | {method["status"]} | {method.get("reason", "独立真值评测已执行")} |')
    lines += ['', '## 场景宏平均', '', '| 方法 | 地物 | 场景组数 | 排序准确率 | 平局率 | Spearman | 单调性违反率 | 无雾响应 | 空间区域 Spearman |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        keys = ('method', 'land_cover', 'n_scene_groups', 'pairwise_accuracy', 'tie_rate', 'spearman', 'monotonicity_violation_rate', 'clear_response', 'region_spearman')
        lines.append('| ' + ' | '.join(fmt(row[k]) for k in keys) + ' |')
    lines += ['', '各场景内先平均、再对场景组宏平均。详见 `condition_strata.csv` 的地物×大气光×亮度分层及 `results.json` 的有效场景数。未知地物不冒充城市/植被/水体/裸地。',
              '', '无雾响应阈值仅为各方法内部诊断，不是跨方法校准过的 FPR；预测与透射率没有直接计算物理 MAE。常量真值的空间相关为 N/A，常量预测的相关也为 N/A。',
              '', '## 典型排序失败候选', '']
    for row in data['failure_candidates']:
        if row['pairwise_accuracy'] is not None and row['pairwise_accuracy'] < 1:
            lines.append(f'- {row["method"]} / {row["scene_id"]}，A={row["airlight"]}，brightness={row["brightness"]}：accuracy={row["pairwise_accuracy"]:.3f}，ties={row["tie_rate"]:.3f}，strong−clear={row["strong_minus_clear_response"]:.4f}。可检验的解释包括反射率/颜色混淆与逐图归一化丢失幅值；当前实验未做因果归因。')
    lines += ['', '## 空间响应失败候选', '']
    for row in data.get('spatial_failure_candidates', []):
        lines.append(f'- {row["method"]} / {row["case_id"]}：区域 Spearman={row["region_spearman"]:.4f}，重雾−轻雾响应差={row["dense_minus_light_response"]:.4f}。负相关/负差表示该样例上与已知退化方向冲突；具体原因还需受控消融。')
    lines += ['', '## 固定尺度可视化', '', '所有预测图固定显示范围 0～1；透射率 t 越小表示雾越重。秩残差是无量纲空间诊断，不是真实透射率误差。', '', *curve_links, '', *image_links,
              '', '## 结论边界', '', '这些实验只检验给定显示 RGB、合成域、场景和参数下的响应。不能据此宣称得到真实消光系数、真实透射率精度或真实遥感雾场泛化能力。',
              '', '源码/环境/Git/种子/配置和所有原始数值见 `results.json`；未运行方法不补造结果。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output / 'report.md'
