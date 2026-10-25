"""Portable experiment preparation and notebook execution using synthetic data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
import yaml

from scripts.utils.prepare_experiment import ROOT, SUBSETS, prepare_experiment
from scripts.utils.setup_environment import check_pytorch, setup_paths


class CloudWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name).resolve() / '测试 workspace'
        self.data = self.work / 'external data/RSHaze+'
        self.output = self.work / 'persistent runs'
        for subset in SUBSETS:
            for official, count in (('train', 10), ('test', 2)):
                for kind in ('synhazypng', 'cleanpng'):
                    folder = self.data / subset / official / kind
                    folder.mkdir(parents=True)
                    for index in range(count):
                        Image.new('RGB', (16, 16), color=(30 + index, 80, 120)).save(folder / f'{index}.png')

    def prepare(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_experiment(self.data, self.output, 'run01', **kwargs)

    def test_external_paths_are_absolute_and_existing_settings_are_frozen(self):
        path = self.prepare()
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        self.assertEqual(config['dataset_root'], str(self.data))
        for key in ('split_file', 'checkpoint_dir', 'result_dir'):
            self.assertTrue(Path(config[key]).is_relative_to(self.output / 'run01'))
        split = Path(config['split_file'])
        original = (path.read_bytes(), split.read_bytes(), path.stat().st_mtime_ns, split.stat().st_mtime_ns)
        self.prepare()
        self.assertEqual(original, (path.read_bytes(), split.read_bytes(), path.stat().st_mtime_ns, split.stat().st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'config differs'):
            self.prepare(batch_size=2)
        self.assertEqual(path.read_bytes(), original[0])
        provenance = json.loads(next(path.parent.glob('code_*.json')).read_text(encoding='utf-8'))
        self.assertIn('src/models/haze_density/physical_prior.py', provenance['source_sha256'])

    def test_missing_original_manifest_is_not_regenerated_for_existing_weights(self):
        path = self.prepare()
        checkpoint_dir = path.parent / 'checkpoints'
        checkpoint_dir.mkdir()
        torch.save({'epoch': 1}, checkpoint_dir / 'latest.pth')
        split = path.parent / 'rshazeplus_split.json'
        split.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'original split'):
            self.prepare()
        self.assertFalse(split.exists())

    def test_dataset_change_is_rejected_without_replacing_manifest(self):
        path = self.prepare()
        split = path.parent / 'rshazeplus_split.json'
        original = split.read_bytes()
        (self.data / 'RSHaze_G/train/synhazypng/0.png').unlink()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(split.read_bytes(), original)

    def test_invalid_data_and_run_name_do_not_create_experiment(self):
        with self.assertRaisesRegex(ValueError, 'single directory'):
            prepare_experiment(self.data, self.output, '../escape')
        (self.data / 'RSHaze_G/test/cleanpng/0.png').unlink()
        (self.data / 'RSHaze_G/test/cleanpng/1.png').unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_required_cuda_rejects_cpu_and_paths_do_not_depend_on_working_directory(self):
        with patch.object(torch.cuda, 'is_available', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(check_pytorch(require_cuda=True))
            self.assertTrue(check_pytorch(require_cuda=False))
        self.assertEqual(setup_paths(), ROOT)

    def test_standalone_notebook_unpacks_without_platform_directories(self):
        book = json.loads((ROOT / 'docs/haze_density_benchmark.ipynb').read_text(encoding='utf-8'))
        setup = ''.join(next(c['source'] for c in book['cells'] if c['cell_type'] == 'code'))
        with tempfile.TemporaryDirectory() as directory:
            with patch('tempfile.mkdtemp', return_value=directory), contextlib.redirect_stdout(io.StringIO()):
                namespace = {}
                exec(compile(setup, 'standalone_setup', 'exec'), namespace)
            project = namespace['PROJECT']
            origin = json.loads((project / 'bundle_origin.json').read_text(encoding='utf-8'))
            import hashlib
            for name, digest in origin['files'].items():
                self.assertEqual(hashlib.sha256((project / name).read_bytes()).hexdigest(), digest)
            self.assertTrue((project / 'scripts/benchmark/run_benchmark.py').is_file())
            # The placeholder must be rejected before any write on Windows and Linux.
            run_cell = ''.join(book['cells'][2]['source'])
            with patch('subprocess.run') as invoked, self.assertRaisesRegex(ValueError, 'OUTPUT_ROOT'):
                exec(compile(run_cell, 'standalone_run', 'exec'), namespace)
            invoked.assert_not_called()
        for name in ('haze_density_benchmark.ipynb', '雾密度网络_最简实验流程.ipynb'):
            book = json.loads((ROOT / 'docs' / name).read_text(encoding='utf-8'))
            self.assertNotIn('colab', book['metadata'])
            for cell in book['cells']:
                if cell['cell_type'] == 'code':
                    code = ''.join(cell['source'])
                    self.assertNotIn('google.colab', code)
                    self.assertNotIn('/content', code)


if __name__ == '__main__':
    unittest.main()
