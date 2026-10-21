"""Independent truth/metric regression checks, not a scientific dataset."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
import yaml
from benchmarks.haze_density.metrics import ranking_metrics, correlation, spatial_metrics, region_means, rank_residual
from benchmarks.haze_density.synthetic import synthesize, srgb_to_linear, linear_to_srgb, transmission_map, load_scenes, generate
from benchmarks.haze_density.baselines import build_method
from benchmarks.haze_density.common import NotAvailable, write_json, ROOT


def setUpModule():
    torch.set_num_threads(1)


class MetricTests(unittest.TestCase):
    def test_perfect_and_reversed_order(self):
        good = ranking_metrics([0, .2, .4, .6], [10, 20, 30, 40])
        bad = ranking_metrics([0, .2, .4, .6], [40, 30, 20, 10])
        self.assertEqual((good['pairwise_accuracy'], good['monotonicity_violation_rate']), (1, 0))
        self.assertEqual((bad['pairwise_accuracy'], bad['monotonicity_violation_rate']), (0, 1))
        self.assertAlmostEqual(good['spearman'], 1)
        self.assertAlmostEqual(bad['spearman'], -1)

    def test_prediction_ties_are_not_success(self):
        result = ranking_metrics([0, 1, 2], [.5, .5, .5])
        self.assertEqual(result['pairwise_accuracy'], 0)
        self.assertEqual(result['tie_rate'], 1)
        self.assertIsNone(result['spearman'])
        self.assertEqual(result['monotonicity_violation_rate'], 0)
        self.assertEqual(result['adjacent_tie_rate'], 1)

    def test_tied_truth_pairs_excluded(self):
        result = ranking_metrics([0, 0, 1], [0, .2, 1])
        self.assertEqual(result['comparable_pairs'], 2)
        self.assertEqual(result['pairwise_accuracy'], 1)

    def test_spearman_ties_match_known_value(self):
        self.assertAlmostEqual(correlation([1, 1, 2], [1, 2, 3], rank=True), np.sqrt(3) / 2)

    def test_scale_invariance(self):
        self.assertEqual(ranking_metrics([0, 1, 2], [.1, .4, .2]),
                         ranking_metrics([0, 1, 2], [101., 104., 102.]))

    def test_constant_spatial_truth_is_not_applicable(self):
        result = spatial_metrics(np.ones((8, 8)), np.zeros((8, 8)))
        self.assertEqual(result['status'], 'N/A')
        self.assertIsNone(result['region_spearman'])
        self.assertIsNone(rank_residual(np.ones((8, 8)), np.zeros((8, 8))))

    def test_spatial_direction_and_rank_residual(self):
        t = np.tile(np.linspace(.4, .9, 8), (8, 1))
        self.assertAlmostEqual(spatial_metrics(t, 1-t)['region_spearman'], 1)
        self.assertAlmostEqual(spatial_metrics(t, t)['region_spearman'], -1)
        self.assertGreater(spatial_metrics(t, 1-t)['dense_minus_light_response'], 0)
        np.testing.assert_allclose(rank_residual(t, 1-t), 0)

    def test_invalid_metrics_rejected(self):
        for x in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                ranking_metrics(x, x)
        with self.assertRaises(ValueError):
            correlation([1, 2], [1])
        with self.assertRaises(ValueError):
            spatial_metrics(np.ones((4, 4)), np.ones((4, 3)))


class SynthesisTests(unittest.TestCase):
    def test_srgb_roundtrip(self):
        x = np.linspace(0, 1, 101, dtype=np.float32)
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=2e-7)

    def test_no_haze_preserves_scene(self):
        j = np.random.default_rng(42).random((8, 9, 3)).astype('float32')
        clear, image, _ = synthesize(j, np.ones((8, 9)), [.9, .8, .7])
        np.testing.assert_allclose(image, j, atol=2e-7)
        np.testing.assert_array_equal(clear, image)

    def test_known_linear_mixture(self):
        j = np.full((8, 8, 3), .2, dtype='float32')
        _, image, _ = synthesize(j, np.full((8, 8), .4), [.8, .9, 1])
        np.testing.assert_allclose(srgb_to_linear(image), srgb_to_linear(j)*.4 + np.array([.8, .9, 1])*.6, atol=2e-7)

    def test_uniform_and_spatial_maps_are_reproducible(self):
        for kind in ('uniform', 'gradient', 'blob', 'regions'):
            a, p = transmission_map(kind, 17, 42, .6)
            b, q = transmission_map(kind, 17, 42, .6)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(p, q)
            self.assertTrue(((a >= 0) & (a <= 1)).all())

    def test_invalid_physics_inputs(self):
        j = np.zeros((4, 4, 3))
        for t in (np.full((4, 4), -1), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                synthesize(j, t, [1, 1, 1])


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        pixels = np.random.default_rng(42).integers(0, 256, (20, 24, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(self.root / 'a.png')
        self.data = {'data_kind': 'implementation_fixture', 'scenes': [
            {'scene_id': 'a', 'group_id': 'area_a', 'path': 'a.png', 'split': 'test',
             'land_cover': 'unknown', 'source': 'unit test fixture, NOT remote sensing'}]}
        self.path = self.root / 'scenes.json'
        write_json(self.path, self.data)
        self.config = yaml.safe_load((ROOT / 'configs/haze_benchmark.yaml').read_text())
        self.config.update({'image_size': 16, 'uniform_t': [1., .4], 'methods': ['dcp', 'prior', 'student']})

    def test_group_leakage_rejected(self):
        other = {**self.data['scenes'][0], 'scene_id': 'b', 'split': 'calibration'}
        self.data['scenes'].append(other)
        write_json(self.path, self.data)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            load_scenes(self.path)

    def test_same_content_under_different_group_rejected(self):
        other = {**self.data['scenes'][0], 'scene_id': 'b', 'group_id': 'other_area'}
        self.data['scenes'].append(other)
        write_json(self.path, self.data)
        with self.assertRaisesRegex(ValueError, 'Identical'):
            load_scenes(self.path)

    def test_generation_saves_independent_truth_and_repeatable_values(self):
        first = generate(self.path, self.root / 'first', self.config)
        second = generate(self.path, self.root / 'second', self.config)
        for a, b in zip(first['cases'], second['cases']):
            self.assertEqual((a['group_id'], a['split']), ('area_a', 'test'))
            with np.load(self.root / 'first' / a['path']) as left, np.load(self.root / 'second' / b['path']) as right:
                for key in left.files:
                    np.testing.assert_array_equal(left[key], right[key])
                np.testing.assert_allclose(left['transmission'], a['uniform_t'])
        with self.assertRaises(ValueError):
            generate(self.path, self.root / 'first', self.config)

    def test_old_checkpoint_is_not_a_v1_accuracy_result(self):
        checkpoint = self.root / 'old.pth'
        torch.save({'config': {}, 'epoch': 5}, checkpoint)
        self.config['checkpoint'] = str(checkpoint)
        with self.assertRaises(NotAvailable):
            build_method('student', 'cpu', self.config)

    def test_json_never_silently_serializes_nan(self):
        with self.assertRaises(ValueError):
            write_json(self.root / 'bad.json', {'metric': float('nan')})

    def test_small_pipeline_records_missing_student(self):
        from benchmarks.haze_density.evaluate import evaluate
        generate(self.path, self.root / 'dataset', self.config)
        result = evaluate(self.root / 'dataset', self.root / 'result', self.config)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['data_kind'], 'implementation_fixture')
        self.assertEqual([m['status'] for m in result['methods']], ['PASS', 'PASS', 'NOT RUN'])
        self.assertEqual(len(result['rankings']), 2)
        self.assertTrue((self.root / 'result/results.json').is_file())


class PerformanceTests(unittest.TestCase):
    def test_latency_statistics_and_throughput(self):
        from benchmarks.haze_density.performance import latency_summary
        result = latency_summary([.01, .02, .03, .04])
        self.assertAlmostEqual(result['p50_ms'], 25)
        self.assertAlmostEqual(result['p95_ms'], 38.5)
        self.assertAlmostEqual(result['images_per_second'], 40)

    def test_partial_macs_for_known_convolution(self):
        from benchmarks.haze_density.performance import convolution_linear_macs
        model = torch.nn.Conv2d(3, 2, 3, bias=False)
        result = convolution_linear_macs(model, torch.zeros(1, 3, 5, 5))
        self.assertEqual(result['macs'], 1 * 2 * 3 * 3 * 3 * 3 * 3)
        self.assertIn('NOT total FLOPs', result['coverage'])

    def test_warmup_and_cpu_memory_status(self):
        from benchmarks.haze_density.performance import measure
        calls = []
        result = measure(lambda: calls.append(1), torch.device('cpu'), 2, 3)
        self.assertEqual(len(calls), 5)
        self.assertEqual(result['iterations'], 3)
        self.assertEqual(result['memory']['status'], 'N/A')

    def test_unavailable_gpu_is_not_cpu_fallback(self):
        from unittest.mock import patch
        from benchmarks.haze_density.baselines import resolve_device
        with patch('torch.cuda.is_available', return_value=False), self.assertRaises(NotAvailable):
            resolve_device('cuda')

    def test_gpu_timer_synchronizes_before_and_after_each_sample(self):
        from unittest.mock import patch
        from benchmarks.haze_density.performance import measure
        calls = []
        with patch('torch.cuda.synchronize', side_effect=lambda *a: calls.append('sync')), \
             patch('torch.cuda.memory_allocated', return_value=100), \
             patch('torch.cuda.reset_peak_memory_stats'), \
             patch('torch.cuda.max_memory_allocated', return_value=150), \
             patch('torch.cuda.max_memory_reserved', return_value=200):
            result = measure(lambda: calls.append('operation'), torch.device('cuda'), 1, 2)
        self.assertEqual(calls, ['operation', 'sync', 'sync', 'operation', 'sync', 'sync', 'operation', 'sync'])
        self.assertEqual(result['memory']['incremental_allocated_bytes'], 50)


if __name__ == '__main__':
    unittest.main()
