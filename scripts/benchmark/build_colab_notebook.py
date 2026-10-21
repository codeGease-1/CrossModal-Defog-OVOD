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
    for folder in ('src', 'benchmarks', 'scripts/train', 'scripts/test', 'scripts/benchmark'):
        paths.update((ROOT / folder).rglob('*.py'))
    paths.update([ROOT / 'configs/haze_benchmark.yaml', ROOT / 'configs/haze_benchmark_pilot_scenes.json',
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
    setup = f'''# 在 VS Code 右上角选择 Colab 内核，再运行本单元。
import base64, hashlib, io, pathlib, tempfile, zipfile, importlib.util
if not pathlib.Path('/content').is_dir():
    raise RuntimeError('NOT RUN: 当前不是 Colab /content 环境，请选择 Colab 内核。')
missing = [m for m in ['torch', 'numpy', 'PIL', 'yaml', 'matplotlib'] if importlib.util.find_spec(m) is None]
if missing:
    raise RuntimeError('NOT RUN: 缺少依赖 ' + ', '.join(missing) + '；安装 torch numpy pillow pyyaml matplotlib 后重试。')
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
PROJECT = pathlib.Path(tempfile.mkdtemp(prefix='stage2_haze_', dir='/content'))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for entry in archive.infolist():
        target = (PROJECT / entry.filename).resolve()
        if not target.is_relative_to(PROJECT):
            raise ValueError('Invalid bundle member')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(entry))
print('项目代码已解包：', PROJECT)
'''
    run = '''# 优先探测常见数据目录；找不到时只下载官方小样本，硬上限 24 MiB，不下载整个数据集。
# 若已有数据，可加入 ['--data-root', '/content/drive/MyDrive/你的数据目录']。
# 修复后训练权重必须明确指定 --checkpoint；未指定则学生正确性 NOT EVALUATED。
import subprocess, sys
command = [sys.executable, '-B', '-X', 'utf8', 'scripts/benchmark/run_colab_benchmark.py', '--fetch-small-sample']
result = subprocess.run(command, cwd=PROJECT)
if result.returncode:
    raise RuntimeError('Benchmark 未完整完成，请检查上方日志和 run_status.json。')
'''
    display = '''from IPython.display import display, Markdown, Image
for path in sorted((PROJECT / 'experiments').glob('colab_benchmark_*/full_results/report.md')):
    display(Markdown(path.read_text().split('## 固定尺度可视化')[0]))
    for figure in sorted((path.parent / 'figures').glob('comparison_*.png'))[:4]:
        display(Image(filename=str(figure)))
for path in sorted((PROJECT / 'experiments').glob('colab_benchmark_*/performance/report.md')):
    display(Markdown(path.read_text()))
print('保存整个结果目录后再关闭 Colab：', PROJECT / 'experiments')
# 可选：在新的单元运行以下两行打包下载（可能包含较大的合成真值数组）。
# import shutil; bundle = shutil.make_archive('/content/haze_benchmark_results', 'zip', PROJECT / 'experiments')
# from google.colab import files; files.download(bundle)
'''
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# 阶段 2：独立雾密度 Benchmark\n',
        '此 notebook 自带当前代码，无需远程 Git 分支或访问本地 Windows 路径。不会训练模型。\n',
        '先运行代码检查，再完成最小闭环、128 样本试验及 CPU/GPU 性能测试。输出是相对响应，不是物理浓度。\n',
        '仅一个地理区域的四种混合地物 ROI，不能证明跨区域泛化。GPU 缺失明确 NOT RUN。\n',
        '本文件未预填任何 Colab 结果；运行时间取决于你分配到的硬件。']}]
    for code in (setup, run, display):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(True)})
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python'}, 'bundle_sha256': digest}, 'nbformat': 4, 'nbformat_minor': 5}
    target = ROOT / 'docs/haze_density_benchmark_colab.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(target, len(payload), 'bundled bytes')


if __name__ == '__main__':
    build()
