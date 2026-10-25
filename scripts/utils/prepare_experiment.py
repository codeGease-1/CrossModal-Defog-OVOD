"""Freeze paths, training settings and a split manifest on a persistent disk.

Run from the project root; existing experiment settings are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from scripts.data.generate_rshazeplus_split import generate_split
from scripts.data.verify_rshazeplus_split import verify_split_json
from scripts.train.train_haze_density import DEFAULT_CONFIG, build_config
from src.data import HazeDensityDataset
from src.models.haze_density.physical_prior import PRIOR_VERSION

SUBSETS = ('RSHaze_G', 'RSHaze_L', 'RSHaze_S')


def prepare_experiment(dataset_root, output_root, run_name, *,
                       config_path=ROOT / 'configs/haze_density.yaml',
                       image_size=None, batch_size=None, num_workers=None, lr=None):
    if not run_name or run_name in ('.', '..') or any(c in run_name for c in '/\\:'):
        raise ValueError('run_name must be a single directory name')
    dataset_root = Path(dataset_root).expanduser().resolve()
    run_dir = Path(output_root).expanduser().resolve() / run_name
    for subset in SUBSETS:
        for official in ('train', 'test'):
            for kind in ('synhazypng', 'cleanpng'):
                directory = dataset_root / subset / official / kind
                if not directory.is_dir() or not any(directory.glob('*.png')):
                    raise FileNotFoundError(f'Missing or empty PNG directory: {directory}')

    overrides = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    if not isinstance(overrides, dict) or set(overrides) - set(DEFAULT_CONFIG):
        raise ValueError('Config must be a flat training mapping with supported keys')
    requested = {**DEFAULT_CONFIG, **overrides}
    requested.update(dataset_root=str(dataset_root),
                     split_file=str(run_dir / 'rshazeplus_split.json'),
                     checkpoint_dir=str(run_dir / 'checkpoints'),
                     result_dir=str(run_dir / 'results'))
    for key, value in dict(image_size=image_size, batch_size=batch_size,
                           num_workers=num_workers, lr=lr).items():
        if value is not None:
            requested[key] = value
    # Reuse the trainer's validation without adding bookkeeping keys to its YAML.
    build_config(SimpleNamespace(**requested, no_amp=False, resume=None))
    if not isinstance(requested['num_workers'], int) or requested['num_workers'] < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    if not math.isfinite(float(requested['lr'])):
        raise ValueError('lr must be finite')

    frozen = run_dir / 'train_config.yaml'
    if frozen.exists():
        saved = yaml.safe_load(frozen.read_text(encoding='utf-8'))
        if saved != requested:
            raise ValueError('Existing experiment config differs; restore settings or choose a new run_name')
    split_path = Path(requested['split_file'])
    checkpoint_dir = Path(requested['checkpoint_dir'])
    if not split_path.exists():
        if any((checkpoint_dir / name).exists() for name in ('best.pth', 'latest.pth')):
            raise FileNotFoundError('Weights exist but the original split manifest is missing; restore it')
        run_dir.mkdir(parents=True, exist_ok=True)
        _, success = generate_split(root=str(dataset_root), val_ratio=0.1,
                                    seed=requested['seed'], output_file=str(split_path))
        if not success:
            raise RuntimeError('Split generation failed')
    if not verify_split_json(str(split_path)):
        raise ValueError('Split manifest validation failed')
    manifest = json.loads(split_path.read_text(encoding='utf-8'))
    for split in ('train', 'val', 'test'):
        dataset = HazeDensityDataset(root=str(dataset_root), split=split,
                                    image_size=requested['image_size'], split_file=str(split_path))
        actual = {(item['subset'], item['filename']) for item in dataset.image_list}
        recorded = {(item['subset'], item['filename']) for item in manifest[split]}
        if not actual or actual != recorded:
            raise ValueError(f'{split}: manifest does not match the dataset')
        if split == 'train' and len(dataset) < requested['batch_size']:
            raise ValueError('Training samples fewer than batch_size')
        print(f'{split}: {len(dataset)} images')
    if not frozen.exists():
        # Exclusive creation also protects an experiment being prepared elsewhere.
        with frozen.open('x', encoding='utf-8') as handle:
            yaml.safe_dump(requested, handle, allow_unicode=True, sort_keys=False)

    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=ROOT, text=True,
                                           encoding='utf-8', stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return 'unavailable (uploaded source archive)'

    import torch
    sources = [p for folder in ('src', 'scripts', 'configs') for p in (ROOT / folder).rglob('*')
               if p.suffix in ('.py', '.yaml', '.json') and '__pycache__' not in p.parts]
    sources.append(ROOT / 'requirements.txt')
    info = dict(branch=git('branch', '--show-current'), commit=git('rev-parse', 'HEAD'),
                git_status=git('status', '--porcelain'), prior_version=PRIOR_VERSION,
                python=sys.version, torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(sources)})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    (run_dir / f'code_{stamp}.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n',
                                             encoding='utf-8')
    print(f'Experiment: {run_dir}\nConfig: {frozen}')
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path, help='Persistent experiment disk')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/haze_density.yaml')
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--lr', type=float)
    args = parser.parse_args()
    prepare_experiment(args.dataset_root, args.output_root, args.run_name, config_path=args.config,
                       image_size=args.image_size, batch_size=args.batch_size,
                       num_workers=args.num_workers, lr=args.lr)


if __name__ == '__main__':
    main()
