#!/usr/bin/env python
"""Review regression checks; no real dataset is needed.

Run: python -m unittest discover -s scripts/test -p test_review_regressions.py -v
Split checks need only the standard library. Tensor/training checks explicitly
skip when PyTorch is unavailable.
"""
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.datasets import RSHazePlusDataset

HAS_TORCH = importlib.util.find_spec('torch') is not None


class SplitRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.split_file = self.root / 'split.json'
        for subset in ('RSHaze_G', 'RSHaze_L', 'RSHaze_S'):
            for official_split, count in (('train', 10), ('test', 2)):
                for kind in ('synhazypng', 'cleanpng'):
                    folder = self.root / subset / official_split / kind
                    folder.mkdir(parents=True)
                    for index in range(count):
                        (folder / f'{index}.png').touch()

    def dataset(self, split, use_file=True):
        return RSHazePlusDataset(
            str(self.root), split=split, transform=lambda image: image,
            split_file=str(self.split_file) if use_file else None,
        )

    def assert_partition(self, train, val):
        train_ids, val_ids = set(train.get_all_ids()), set(val.get_all_ids())
        self.assertFalse(train_ids & val_ids)
        self.assertEqual((len(train_ids), len(val_ids)), (27, 3))
        self.assertEqual(len(train_ids | val_ids), 30)

    def test_train_first_without_manifest(self):
        self.assert_partition(self.dataset('train', False), self.dataset('val', False))

    def test_train_first_creates_manifest_and_excludes_validation(self):
        train = self.dataset('train')
        self.assertTrue(self.split_file.exists())
        self.assert_partition(train, self.dataset('val'))
        missing_manifest = self.root / 'missing_data_split.json'
        with self.assertRaisesRegex(ValueError, 'No official training images'):
            RSHazePlusDataset(
                str(self.root / 'missing_dataset'), transform=lambda image: image,
                split_file=str(missing_manifest),
            )
        self.assertFalse(missing_manifest.exists())

    def test_validation_first_gives_same_partition(self):
        val = self.dataset('val')
        self.assert_partition(self.dataset('train'), val)
        self.assertEqual(val.get_all_ids(), self.dataset('val', False).get_all_ids())

    def test_partition_does_not_reset_global_random_state(self):
        random.seed(123)
        before = random.getstate()
        self.dataset('train', False)
        self.assertEqual(before, random.getstate())

    def test_overlapping_manifest_is_rejected(self):
        self.dataset('train')
        data = json.loads(self.split_file.read_text(encoding='utf-8'))
        data['train'].append(data['val'][0])
        self.split_file.write_text(json.dumps(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.dataset('train')

    def test_incomplete_manifest_is_rejected(self):
        self.dataset('train')
        original = self.split_file.read_text(encoding='utf-8')
        for issue in ('incomplete', 'duplicate', 'bad_format'):
            with self.subTest(issue=issue):
                data = json.loads(original)
                if issue == 'incomplete':
                    data['train'].pop()
                elif issue == 'duplicate':
                    data['train'].append(data['train'][0])
                else:
                    data['val'] = ['legacy_filename_only.png']
                saved = json.dumps(data)
                self.split_file.write_text(saved, encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.dataset('train')
                self.assertEqual(self.split_file.read_text(encoding='utf-8'), saved)

    def test_official_test_is_unchanged(self):
        self.assertEqual(len(self.dataset('test')), 6)
        self.assertTrue(all(item['official_split'] == 'test'
                            for item in self.dataset('test').image_list))


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class TensorRegressionTests(unittest.TestCase):
    def test_normalization_preserves_horizontal_gradient(self):
        import torch
        from src.models.haze_density.physical_prior import min_max_normalize
        image = torch.tensor([[[[0., 1., 2.], [0., 1., 2.]]]])
        torch.testing.assert_close(min_max_normalize(image), image / 2)

    def test_normalization_is_independent_per_image_and_channel(self):
        import torch
        from src.models.haze_density.physical_prior import min_max_normalize
        image = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3)
        expected = (image - image.amin((-2, -1), keepdim=True)) / 5
        torch.testing.assert_close(min_max_normalize(image), expected)
        self.assertTrue(torch.equal(min_max_normalize(torch.ones_like(image)),
                                    torch.zeros_like(image)))

    def test_config_file_is_read_and_cli_wins(self):
        from scripts.train.train_haze_density import parse_args, build_config
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.yaml'
            path.write_text('epochs: 3\nbase_channels: 8\namp: false\n', encoding='utf-8')
            with patch.object(sys, 'argv', ['train', '--config', str(path), '--epochs', '7']):
                config = build_config(parse_args())
            self.assertEqual(config['epochs'], 7)
            self.assertEqual(config['base_channels'], 8)
            self.assertFalse(config['amp'])

    def test_latest_checkpoint_retains_best_loss_and_can_resume(self):
        import torch
        from scripts.train.train_haze_density import save_checkpoint, load_checkpoint
        from src.models.haze_density.physical_prior import PRIOR_VERSION
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            save_checkpoint(model, optimizer, 2, .3, False,
                            {'prior_version': PRIOR_VERSION}, folder, best_val_loss=.1)
            epoch, best = load_checkpoint(model, optimizer, folder / 'latest.pth')
            self.assertEqual(epoch, 3)
            self.assertEqual(best, .1)


if __name__ == '__main__':
    unittest.main()
