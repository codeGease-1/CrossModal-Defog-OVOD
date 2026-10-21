"""Stage 1 implementation regression tests, not a correctness/performance benchmark.

Run with a real PyTorch runtime:
  python -m unittest discover -s scripts/test -p test_stage1_bugfixes.py -v
No dataset download, trained checkpoint, GPU, or training run is required.
"""
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from src.data import RSHazePlusDataset, build_rshazeplus_dataloader
from src.data.transforms import HazeTrainTransform
from src.models.haze_density.guided_filter import _mean_filter, guided_filter
from src.models.haze_density.physical_prior import (
    _local_mean, dark_channel, min_max_normalize, generate_s_final,
    WEIGHT_DARK, WEIGHT_CONTRAST, WEIGHT_COLOR, EXPONENT_MU,
)


def setUpModule():
    torch.set_num_threads(1)


class PriorImplementationTests(unittest.TestCase):
    def test_spatial_minmax_matches_full_image_reduction(self):
        for shape in ((2, 3, 4, 7), (1, 1, 1, 7), (1, 1, 7, 1)):
            with self.subTest(shape=shape):
                x = torch.arange(np.prod(shape), dtype=torch.float64).reshape(shape)
                low = x.amin((-2, -1), keepdim=True)
                high = x.amax((-2, -1), keepdim=True)
                expected = (x - low) / (high - low + 1e-8)
                torch.testing.assert_close(min_max_normalize(x), expected)

    def test_horizontal_gradient_is_not_erased(self):
        x = torch.tensor([[[[0., .5, 1.], [0., .5, 1.]]]])
        torch.testing.assert_close(min_max_normalize(x), x)

    def test_noncontiguous_input(self):
        x = torch.arange(24, dtype=torch.float64).reshape(1, 1, 4, 6).transpose(2, 3)
        torch.testing.assert_close(min_max_normalize(x), x / (23 + 1e-8))

    def test_constant_minmax_is_zero_for_supported_float_types(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for value in (0., .5, 1.):
                with self.subTest(dtype=dtype, value=value):
                    result = min_max_normalize(torch.full((2, 3, 5, 7), value, dtype=dtype))
                    self.assertTrue(torch.isfinite(result).all())
                    self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_local_means_preserve_constants_including_corners(self):
        for mean in (_local_mean, _mean_filter):
            for size in (1, 3, 15, 31):
                with self.subTest(mean=mean.__name__, size=size):
                    x = torch.full((2, 1, 5, 7), .5)
                    torch.testing.assert_close(mean(x, size), x)

    def test_local_means_match_existing_pixel_window_mean(self):
        x = torch.arange(35, dtype=torch.float64).reshape(1, 1, 5, 7) / 35
        expected = torch.empty_like(x)
        for row in range(5):
            for col in range(7):
                expected[0, 0, row, col] = x[
                    0, 0, max(0, row-1):min(5, row+2), max(0, col-1):min(7, col+2)
                ].mean()
        torch.testing.assert_close(_local_mean(x, 3), expected)
        torch.testing.assert_close(_mean_filter(x, 3), expected)

    def test_min_pool_does_not_invent_dark_border_pixels(self):
        x = torch.ones(1, 3, 5, 7)
        torch.testing.assert_close(dark_channel(x), torch.ones(1, 1, 5, 7))

    def test_guided_filter_preserves_constant_source_with_zero_guide(self):
        source = torch.full((1, 1, 9, 11), .5)
        torch.testing.assert_close(guided_filter(torch.zeros_like(source), source, 2), source)

    def test_guided_filter_preserves_constant_source_with_nonconstant_guide(self):
        source = torch.full((2, 1, 9, 11), .5, dtype=torch.float64)
        guide = torch.linspace(0, 1, source.numel(), dtype=source.dtype).reshape_as(source)
        torch.testing.assert_close(guided_filter(guide, source, 2), source)

    def test_radius_zero_is_identity(self):
        source = torch.linspace(0, 1, 35).reshape(1, 1, 5, 7)
        torch.testing.assert_close(guided_filter(torch.zeros_like(source), source, 0), source)

    def test_extreme_images_are_finite_and_keep_shape(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for height, width in ((1, 1), (3, 7), (17, 19)):
                for value in (0., .3, .5, 1.):
                    with self.subTest(dtype=dtype, shape=(height, width), value=value):
                        image = torch.full((1, 3, height, width), value, dtype=dtype)
                        results = generate_s_final(image, return_intermediate=True)
                        for name, result in results.items():
                            self.assertEqual(tuple(result.shape), (1, 1, height, width), name)
                            self.assertTrue(torch.isfinite(result).all(), name)
                            self.assertTrue(((result >= 0) & (result <= 1)).all(), name)

    def test_nonbinary_constant_does_not_become_false_contrast_after_minmax(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                image = torch.full((1, 3, 33, 37), .3, dtype=dtype)
                result = generate_s_final(image, return_intermediate=True)
                torch.testing.assert_close(result['C_hat'], torch.ones_like(result['C_hat']))
                # Original formula: D_hat=0, C_hat=K_hat=1 -> (0.3+0.2)^1.5.
                expected = torch.full_like(result['S_final'], .5 ** 1.5)
                torch.testing.assert_close(result['S_final'], expected)

    def test_nonfinite_or_out_of_range_images_are_rejected(self):
        for value in (float('nan'), float('inf'), -0.1, 1.1):
            with self.subTest(value=value):
                image = torch.full((1, 3, 5, 7), value)
                with self.assertRaises(ValueError):
                    generate_s_final(image)

    def test_invalid_shapes_are_rejected(self):
        for shape in ((3, 5, 7), (1, 1, 5, 7), (1, 3, 0, 7), (0, 3, 5, 7)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                generate_s_final(torch.zeros(shape))

    def test_invalid_windows_and_regularizers_are_rejected(self):
        image = torch.zeros(1, 3, 5, 7)
        for kwargs in ({'window_size': 0}, {'window_size': 2},
                       {'guided_radius': -1}, {'guided_radius': 1.5},
                       {'guided_eps': 0}, {'guided_eps': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_s_final(image, **kwargs)
        with self.assertRaises(ValueError):
            min_max_normalize(image, eps=0)

    def test_prior_constants_remain_unchanged(self):
        self.assertEqual((WEIGHT_DARK, WEIGHT_CONTRAST, WEIGHT_COLOR, EXPONENT_MU),
                         (.5, .3, .2, 1.5))

    def test_checkpoint_from_previous_target_version_cannot_resume_silently(self):
        from scripts.train.train_haze_density import load_checkpoint
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'older.pth'
            for config in ({}, {'prior_version': 'spatial_minmax_v2'}):
                torch.save({'config': config}, path)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                    ValueError, 'physical-prior version'
                ):
                    load_checkpoint(model, optimizer, path)


class DataPipelineImplementationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'split.json'
        # Position-coded RGB image: independent random crops/flips are observable.
        y, x = np.mgrid[:12, :16]
        pixels = np.stack((x * 10, y * 15, (x + y) * 7), axis=-1).astype('uint8')
        for subset in RSHazePlusDataset.ALL_SUBSETS:
            for official, count in (('train', 10), ('test', 2)):
                for kind in ('synhazypng', 'cleanpng'):
                    folder = self.root / subset / official / kind
                    folder.mkdir(parents=True)
                    for index in range(count):
                        Image.fromarray(pixels).save(folder / f'{index}.png')

    def dataset(self, split='train', **kwargs):
        return RSHazePlusDataset(str(self.root), split=split, image_size=6,
                                 split_file=str(self.manifest), **kwargs)

    def test_random_training_augmentation_is_shared_by_pair(self):
        dataset = self.dataset(return_clean=True)
        for seed in range(8):
            torch.manual_seed(seed)
            with self.subTest(seed=seed):
                sample = dataset[0]
                torch.testing.assert_close(sample['image'], sample['clean'], rtol=0, atol=0)

    def test_validation_pair_alignment(self):
        sample = self.dataset('val', return_clean=True)[0]
        torch.testing.assert_close(sample['image'], sample['clean'], rtol=0, atol=0)

    def test_missing_clean_is_rejected_when_pair_is_requested(self):
        for path in self.root.glob('*/train/cleanpng/*.png'):
            path.unlink()
        with self.assertRaises(ValueError):
            self.dataset(return_clean=True)
        # Hazy-only training remains usable and does not request missing labels.
        self.assertIn('image', self.dataset(return_clean=False)[0])

    def test_mismatched_native_pair_sizes_are_rejected(self):
        dataset = self.dataset(return_clean=True)
        Image.new('RGB', (17, 12)).save(dataset.image_list[0]['clear_path'])
        with self.assertRaises(ValueError):
            dataset[0]

    def test_custom_unpaired_transform_is_rejected_for_pairs(self):
        dataset = self.dataset(return_clean=True, transform=lambda image: image)
        with self.assertRaises(TypeError):
            dataset[0]

    def test_pair_augmentation_consumes_one_set_of_random_parameters(self):
        image = Image.open(self.root / 'RSHaze_G/train/synhazypng/0.png').convert('RGB')
        transform = HazeTrainTransform(6)
        torch.manual_seed(7)
        single = transform(image)
        after_single = torch.get_rng_state()
        torch.manual_seed(7)
        paired, _ = transform.apply_pair(image, image)
        torch.testing.assert_close(single, paired)
        self.assertTrue(torch.equal(torch.get_rng_state(), after_single))

    def test_clean_pair_loader_collates_consistent_keys(self):
        loader = build_rshazeplus_dataloader(
            str(self.root), image_size=6, batch_size=2, num_workers=0,
            pin_memory=False, return_clean=True, split_file=str(self.manifest),
        )
        batch = next(iter(loader))
        self.assertEqual(tuple(batch['clean'].shape), (2, 3, 6, 6))
        torch.testing.assert_close(batch['image'], batch['clean'], rtol=0, atol=0)

    def test_generated_manifest_loads_without_dropping_same_named_test_files(self):
        from scripts.data.generate_rshazeplus_split import generate_split
        state = random.getstate()
        with contextlib.redirect_stdout(io.StringIO()):
            data, success = generate_split(str(self.root), output_file=str(self.manifest))
        self.assertTrue(success)
        self.assertEqual((len(data['train']), len(data['val']), len(data['test'])), (27, 3, 6))
        train, val = self.dataset(), self.dataset('val')
        self.assertFalse(set(train.get_all_ids()) & set(val.get_all_ids()))
        self.assertEqual(len(train) + len(val), 30)
        self.assertEqual(random.getstate(), state)

    def test_manifest_verifier_uses_official_source_directory(self):
        from scripts.data.verify_rshazeplus_split import verify_split_json
        self.dataset()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(verify_split_json(str(self.manifest)))

    def test_manifest_verifier_still_rejects_train_val_overlap_and_duplicates(self):
        from scripts.data.verify_rshazeplus_split import verify_split_json
        self.dataset()
        original = self.manifest.read_text(encoding='utf-8')
        for issue in ('overlap', 'duplicate', 'empty', 'legacy'):
            with self.subTest(issue=issue):
                data = json.loads(original)
                if issue == 'overlap':
                    data['train'].append(data['val'][0])
                elif issue == 'duplicate':
                    data['train'].append(data['train'][0])
                elif issue == 'empty':
                    data['val'] = []
                else:
                    data['train'] = ['old_id']
                self.manifest.write_text(json.dumps(data), encoding='utf-8')
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertFalse(verify_split_json(str(self.manifest)))

    def test_split_generator_rejects_invalid_or_missing_data_without_writing(self):
        from scripts.data.generate_rshazeplus_split import generate_split
        for ratio in (0, 1, float('nan')):
            with self.subTest(ratio=ratio), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    generate_split(str(self.root), val_ratio=ratio, output_file=str(self.manifest))
            self.assertFalse(self.manifest.exists())
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            generate_split(str(self.root / 'missing'), output_file=str(self.manifest))
        self.assertFalse(self.manifest.exists())

    def test_path_checker_uses_metadata_without_decoding_or_random_augmentation(self):
        from scripts.data.check_data_leakage import (
            get_image_identifiers, check_leakage, check_internal_duplicates,
        )
        dataset = self.dataset()
        rng_before = torch.get_rng_state()
        with patch.object(RSHazePlusDataset, '__getitem__', side_effect=AssertionError('decoded')):
            self.assertEqual(len(get_image_identifiers(dataset)), len(dataset))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(check_leakage(str(self.root), str(self.manifest)))
                self.assertTrue(check_internal_duplicates(str(self.root), str(self.manifest)))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))


if __name__ == '__main__':
    unittest.main()
