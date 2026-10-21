"""Provenance, strict JSON, and resource-status helpers."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    # Never hide undefined metrics behind NaN or zero.
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + '\n', encoding='utf-8')


def read_json(path):
    def invalid(value):
        raise ValueError(f'Nonstandard JSON constant: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output must be empty: {path}; choose a new run directory')
    path.mkdir(parents=True, exist_ok=True)
    return path


def environment(seed, config=None):
    import numpy as np
    import torch
    from . import BENCHMARK_VERSION
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    paths = []
    for folder in (ROOT / 'benchmarks/haze_density', ROOT / 'src/models/haze_density'):
        paths.extend(sorted(folder.glob('*.py')))
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'benchmark_version': BENCHMARK_VERSION, 'seed': seed, 'config': config,
        'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
        'bundle_origin': read_json(ROOT / 'bundle_origin.json') if (ROOT / 'bundle_origin.json').is_file() else None,
        'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in paths},
        'python': sys.version, 'platform': platform.platform(),
        'cpu': platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER', 'unknown'),
        'logical_cpu_count': os.cpu_count(), 'numpy': np.__version__,
        'torch': torch.__version__, 'cuda_build': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'cudnn': torch.backends.cudnn.version(), 'torch_threads': torch.get_num_threads(),
        'gpu': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
    }


def seed_runtime(seed, threads=1):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class NotAvailable(RuntimeError):
    """A required resource is absent; report NOT RUN, not a passing experiment."""
