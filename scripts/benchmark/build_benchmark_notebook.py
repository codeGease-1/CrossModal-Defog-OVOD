"""Bundle public project code into a standalone notebook; no data, weights or credentials."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def build():
    paths = set()
    for folder in ('src', 'benchmarks', 'scripts/train', 'scripts/test', 'scripts/benchmark', 'scripts/utils'):
        paths.update((ROOT / folder).rglob('*.py'))
    paths.update([ROOT / 'configs/haze_benchmark.yaml', ROOT / 'configs/haze_benchmark_pilot_scenes.json',
                  ROOT / 'configs/haze_density.yaml', ROOT / 'requirements.txt',
                  ROOT / 'scripts/data/fetch_benchmark_samples.py'])
    # Existing stage-1 tests import the split-generation helper.
    paths.update((ROOT / 'scripts/data').glob('*.py'))
    origin = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
              'files': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.writestr(path.relative_to(ROOT).as_posix(), path.read_bytes())
        archive.writestr('bundle_origin.json', json.dumps(origin, indent=2))
    payload = stream.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode('ascii')
    setup = f'''# 选择已安装依赖的 Python / Jupyter 内核；代码包可解压到系统临时目录。
import base64, hashlib, io, pathlib, tempfile, zipfile, importlib.util
missing = [m for m in ['torch', 'numpy', 'PIL', 'yaml', 'matplotlib', 'scipy'] if importlib.util.find_spec(m) is None]
if missing:
    raise RuntimeError('NOT RUN: 缺少依赖 ' + ', '.join(missing) + '；请先安装项目 requirements.txt。')
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
PROJECT = pathlib.Path(tempfile.mkdtemp(prefix='stage2_haze_')).resolve()
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for entry in archive.infolist():
        target = (PROJECT / entry.filename).resolve()
        if not target.is_relative_to(PROJECT):
            raise ValueError('Invalid bundle member')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(entry))
print('项目代码已解包：', PROJECT)
'''
    run = '''# 输出必须指向持久化磁盘；每次使用新的空目录。
# 已有 RSHaze+ 时设置 DATA_ROOT；联网下载仅为缺少官方小样本时的显式选项。
import subprocess, sys
OUTPUT_ROOT = pathlib.Path('/path/to/persistent/benchmarks')  # 必改
DATA_ROOT = None  # 例如 pathlib.Path('/path/to/datasets/RSHaze+')
CHECKPOINT = None  # 当前 v3 的 best.pth；缺少时学生正确性 NOT EVALUATED
ALLOW_SMALL_SAMPLE_DOWNLOAD = False
if OUTPUT_ROOT.as_posix().startswith('/path/to/'):
    raise ValueError('请先将 OUTPUT_ROOT 改为实际的持久化磁盘目录。')
OUTPUT_ROOT = OUTPUT_ROOT.expanduser().resolve()
from datetime import datetime, timezone
BENCHMARK_OUTPUT = OUTPUT_ROOT / ('benchmark_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
command = [sys.executable, '-B', '-X', 'utf8', 'scripts/benchmark/run_benchmark.py',
           '--output', str(BENCHMARK_OUTPUT)]
if DATA_ROOT is not None:
    command += ['--data-root', str(pathlib.Path(DATA_ROOT).expanduser().resolve())]
if CHECKPOINT is not None:
    command += ['--checkpoint', str(pathlib.Path(CHECKPOINT).expanduser().resolve())]
if ALLOW_SMALL_SAMPLE_DOWNLOAD:
    command.append('--fetch-small-sample')
result = subprocess.run(command, cwd=PROJECT)
if result.returncode:
    raise RuntimeError('Benchmark 未完整完成，请检查上方日志和 run_status.json。')
'''
    display = '''from IPython.display import display, Markdown, Image
for path in (BENCHMARK_OUTPUT / 'full_results/report.md',):
    if not path.is_file():
        continue
    display(Markdown(path.read_text().split('## 固定尺度可视化')[0]))
    for figure in sorted((path.parent / 'figures').glob('comparison_*.png'))[:4]:
        display(Image(filename=str(figure)))
path = BENCHMARK_OUTPUT / 'performance/report.md'
if path.is_file():
    display(Markdown(path.read_text()))
print('完整结果目录（请备份）：', BENCHMARK_OUTPUT)
'''
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# 阶段 2：独立雾密度 Benchmark\n',
        '此 notebook 自带当前代码，无需远程 Git 分支或访问本地 Windows 路径。不会训练模型。\n',
        '先运行代码检查，再完成最小闭环、128 样本试验及 CPU/GPU 性能测试。输出是相对响应，不是物理浓度。\n',
        '仅一个地理区域的四种混合地物 ROI，不能证明跨区域泛化。GPU 缺失明确 NOT RUN。\n',
        '适用于通用 GPU 平台、本地与 Jupyter；请先在第二个单元配置数据和持久化输出路径。\n',
        '本文件未预填结果；运行时间取决于你分配到的硬件。']}]
    for code in (setup, run, display):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(True)})
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python'}, 'bundle_sha256': digest}, 'nbformat': 4, 'nbformat_minor': 5}
    target = ROOT / 'docs/haze_density_benchmark.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(target, len(payload), 'bundled bytes')


if __name__ == '__main__':
    build()
