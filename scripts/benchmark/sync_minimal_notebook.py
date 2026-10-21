"""Embed the current standalone benchmark bundle into the minimal Colab workflow."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'docs/haze_density_benchmark_colab.ipynb'
TARGET = ROOT / 'docs/雾密度网络_最简实验流程.ipynb'
TAG = 'stage2-independent-benchmark'


def cell(kind, source):
    return {'cell_type': kind, 'metadata': {'tags': [TAG]},
            **({'execution_count': None, 'outputs': []} if kind == 'code' else {}),
            'source': source.splitlines(keepends=True)}


def main():
    standalone = json.loads(SOURCE.read_text(encoding='utf-8'))
    target = json.loads(TARGET.read_text(encoding='utf-8'))
    target['cells'] = [item for item in target['cells'] if TAG not in item.get('metadata', {}).get('tags', [])]
    setup = ''.join(next(item['source'] for item in standalone['cells'] if item['cell_type'] == 'code'))
    intro = ''.join(target['cells'][0]['source'])
    intro = intro.replace('→ 可选测试集评估`', '→ 可选测试集评估 → 独立 Benchmark`')
    intro = intro.replace('此 notebook 不构建阶段 2 Benchmark。',
                          '第 8 节提供独立 Benchmark：用已知透射率的合成雾验证排序与空间响应，并测量运行成本。训练/验证中的伪标签拟合和独立正确性是两类结果。')
    target['cells'][0]['source'] = intro.splitlines(keepends=True)
    environment_note = ''.join(target['cells'][1]['source'])
    environment_note = environment_note.replace(
        '请先把本次修改推送到远端 `codex-v1`，再在 Colab 运行；只上传 notebook 不会更新训练源码。',
        '第 1 节的训练源码仍从远端 `codex-v1` 获取；如果远端尚无阶段 1 修复，需先更新该分支。第 8 节的 Benchmark 代码已内嵌在此 notebook，无需为评测单独推送源码。')
    target['cells'][1]['source'] = environment_note.splitlines(keepends=True)
    howto = ''.join(target['cells'][-1]['source'])
    howto = ''.join(line for line in howto.splitlines(keepends=True)
                    if not line.startswith('- **独立 Benchmark**：'))
    previous = '- **最终测试**：方案确定后运行第 7 节，并保留本次报告和输出目录。'
    addition = previous + '\n- **独立 Benchmark**：完成第 1、3、4 节后可直接运行第 8 节；已有当前版本 `best.pth` 时自动加入学生网络，否则仍能评测 DCP 和三先验。'
    if previous not in howto:
        raise ValueError('Minimal notebook usage section changed; inspect before sync')
    target['cells'][-1]['source'] = howto.replace(previous, addition).splitlines(keepends=True)
    heading = '''## 8. 独立 Benchmark：雾密度判断是否可信

本节与第 7 节的“拟合伪标签”评估不同：使用官方 RSHaze_G **测试集清晰图**作为背景，按已知透射率合成雾，分别检验雾强度排序、空间分布、无雾响应，并计时。可以在第 1、3、4 节后独立运行，不必重新训练；如果已有本实验当前先验版本的 `best.pth`，会加入学生网络。**只在方案确定后查看测试结果，不要据此反复调训练超参数。**

优先复用第 3 节已解压的数据，核对四张官方清晰图的 SHA256；缺少匹配图时，入口可从官方 ZIP 按字节范围获取小样本，硬限制 24 MiB。四个地物 ROI 属于一个保守地理组，结果不能证明跨地区泛化。实验会跑最小闭环、128 样本扩展评测和 CPU/GPU 性能；缺少 GPU 或合格权重会明确标为 NOT RUN/NOT EVALUATED。结果、真值、图和日志保存在本实验的 Google Drive `benchmarks/` 新目录。

下面第一个代码单元只解包经过校验的评测代码，不会替换训练中的模块。第二个单元运行评测，第三个单元读取结果。完整定义见 [Benchmark 使用说明](guides/haze_density_benchmark.md)。
'''
    setup += '\nBENCHMARK_ROOT = PROJECT\nprint("独立评测代码：", BENCHMARK_ROOT)\n'
    run = '''# 需要第 1、3、4 节提供的 PROJECT_ROOT、DATASET_ROOT、RUN_DIR 与 PRIOR_VERSION。
if not DATASET_ROOT.is_dir() or not RUN_DIR.is_dir():
    raise RuntimeError('请先完成第 3、4 节的数据和实验目录准备。')
ALLOW_SMALL_SAMPLE_DOWNLOAD = True  # 仅在本地数据缺少四张匹配的官方清晰图时读取，硬上限 24 MiB。
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
BENCHMARK_OUTPUT = RUN_DIR / 'benchmarks' / stamp
benchmark_command = [
    sys.executable, '-u', str(BENCHMARK_ROOT / 'scripts/benchmark/run_colab_benchmark.py'),
    '--data-root', str(DATASET_ROOT), '--output', str(BENCHMARK_OUTPUT),
]
if ALLOW_SMALL_SAMPLE_DOWNLOAD:
    benchmark_command.append('--fetch-small-sample')
candidate = CHECKPOINT_DIR / 'best.pth'
if candidate.is_file():
    saved = torch.load(candidate, map_location='cpu', weights_only=True)
    if saved.get('config', {}).get('prior_version') == PRIOR_VERSION and saved.get('epoch', 0) >= 1:
        benchmark_command += ['--checkpoint', str(candidate)]
        print('学生网络将使用本实验当前版本权重：', candidate)
    else:
        print('权重缺少当前先验版本或训练轮次，学生正确性将标记 NOT EVALUATED：', candidate)
else:
    print('未找到本实验权重；DCP/三先验仍可评测，学生正确性 NOT EVALUATED。')
print('独立评测结果将保存在：', BENCHMARK_OUTPUT)
run_command(benchmark_command, cwd=BENCHMARK_ROOT)
'''
    view = '''# 可单独重跑本单元查看已生成的结果，不重新计算。
from IPython.display import display, Markdown, Image
if 'BENCHMARK_OUTPUT' not in globals():
    raise RuntimeError('请先运行第 8 节评测单元，或把 BENCHMARK_OUTPUT 指向已有的 Drive 结果目录。')
status_file = BENCHMARK_OUTPUT / 'run_status.json'
if not status_file.is_file():
    raise FileNotFoundError(f'尚无实验状态记录：{status_file}')
print('各阶段执行状态：', json.loads(status_file.read_text(encoding='utf-8')))
accuracy = BENCHMARK_OUTPUT / 'full_results' / 'report.md'
performance = BENCHMARK_OUTPUT / 'performance' / 'report.md'
for path in (accuracy, performance):
    if path.is_file():
        text = path.read_text(encoding='utf-8')
        display(Markdown(text.split('## 固定尺度可视化')[0]))
for figure in sorted((BENCHMARK_OUTPUT / 'full_results/figures').glob('comparison_*.png'))[:4]:
    display(Image(filename=str(figure)))
print('完整 CSV、逐图结果、真值和固定尺度可视化：', BENCHMARK_OUTPUT)
'''
    insertion = len(target['cells']) - 1
    target['cells'][insertion:insertion] = [cell('markdown', heading), cell('code', setup),
                                              cell('code', run), cell('code', view)]
    TARGET.write_text(json.dumps(target, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Integrated standalone bundle {standalone["metadata"]["bundle_sha256"]} into {TARGET}')


if __name__ == '__main__':
    main()
