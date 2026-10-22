"""Independent metric and student-adapter checks; fixtures are NOT trained models."""
import copy
import tempfile
from pathlib import Path
import unittest
import json
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
import yaml

from benchmarks.haze_density.common import ROOT, write_json
from benchmarks.haze_density.evaluate import evaluate
from benchmarks.haze_density.baselines import build_method
from benchmarks.haze_density.metrics import correlation, spatial_metrics
from benchmarks.haze_density.synthetic import generate
from src.models.haze_density import HazeDensityNet
from src.models.haze_density.physical_prior import PRIOR_VERSION


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        from PIL import Image
        Image.fromarray(np.random.default_rng(42).integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(self.root / 'a.png')
        manifest = {'data_kind': 'implementation_fixture', 'scenes': [
            {'scene_id': 'a', 'group_id': 'area', 'split': 'test', 'land_cover': 'unknown',
             'path': 'a.png', 'source': 'unit test fixture'}]}
        self.manifest = self.root / 'scenes.json'
        write_json(self.manifest, manifest)
        self.config = yaml.safe_load((ROOT / 'configs/haze_benchmark.yaml').read_text())
        self.config.update(image_size=16, methods=['student'])

    def checkpoint(self, **changes):
        model = HazeDensityNet(base_channels=4).eval()
        saved = {'config': {'prior_version': PRIOR_VERSION, 'base_channels': 4, 'use_sigmoid': True},
                 'epoch': 1, 'model_state_dict': model.state_dict()}
        saved.update(changes)
        path = self.root / 'fixture.pth'
        torch.save(saved, path)
        self.config['checkpoint'] = str(path)
        return model, saved

    def test_spearman_matches_independent_scipy_with_ties(self):
        rng = np.random.default_rng(42)
        arrays = rng.integers(0, 5, (50, 2, 16)).tolist()
        result = subprocess.run([sys.executable, '-c',
            'import json,sys; from scipy.stats import spearmanr; print(json.dumps([float(spearmanr(x,y).statistic) for x,y in json.load(sys.stdin)]))'],
            input=json.dumps(arrays), capture_output=True, text=True, check=True)
        for (x, y), expected in zip(arrays, json.loads(result.stdout)):
            self.assertAlmostEqual(correlation(x, y, rank=True), expected)

    def test_constant_prediction_is_not_spatial_success(self):
        t = np.tile(np.linspace(.4, .9, 16), (16, 1))
        result = spatial_metrics(t, np.full((16, 16), .5))
        self.assertIsNone(result['region_spearman'])
        self.assertEqual(result['region_pairwise_accuracy'], 0)

    def test_student_adapter_preserves_raw_prediction(self):
        expected, _ = self.checkpoint()
        loaded, info = build_method('student', 'cpu', self.config)
        x = torch.rand(1, 3, 16, 16)
        with torch.inference_mode():
            torch.testing.assert_close(loaded(x), expected(x), rtol=0, atol=0)
        self.assertIn('not physical transmission', info['output_meaning'])
        self.assertIn('NOT VERIFIED', info['training_overlap_status'])

    def test_nonfinite_weights_rejected(self):
        _, saved = self.checkpoint()
        key = next(k for k in saved['model_state_dict'] if k.endswith('weight'))
        saved['model_state_dict'][key].view(-1)[0] = float('nan')
        torch.save(saved, self.config['checkpoint'])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            build_method('student', 'cpu', self.config)

    def test_student_fixture_uses_independent_truth_and_fair_inputs(self):
        # Oracle adapter verifies evaluator direction/grouping independently of
        # student training. A fixture checkpoint is never scientific evidence.
        self.checkpoint()
        data = generate(self.manifest, self.root / 'data', self.config, profile='full')
        maps = []
        for case in data['cases']:
            with np.load(self.root / 'data' / case['path']) as z:
                maps.append((z['hazy_rgb'].copy(), z['transmission'].copy()))
        class Oracle(torch.nn.Module):
            def forward(inner, x):
                rgb, t = maps.pop(0)
                np.testing.assert_array_equal(x[0].permute(1, 2, 0).numpy(), rgb)
                return torch.from_numpy(1-t)[None, None]
        with patch('benchmarks.haze_density.evaluate.build_method', return_value=(Oracle(), {'method': 'student'})):
            result = evaluate(self.root / 'data', self.root / 'result', self.config)
        self.assertEqual(len(maps), 0)
        summary = result['summary'][0]
        self.assertEqual(summary['pairwise_accuracy']['value'], 1)
        self.assertAlmostEqual(summary['spearman']['value'], 1)
        self.assertAlmostEqual(summary['region_spearman']['value'], 1)
        self.assertEqual(summary['clear_response']['value'], 0)
        self.assertEqual(result['physical_mae']['status'], 'N/A')

    def test_evaluation_rejects_leakage_before_predictor(self):
        data = generate(self.manifest, self.root / 'data', self.config)
        data['cases'][0]['split'] = 'calibration'
        write_json(self.root / 'data/dataset.json', data)
        with patch('benchmarks.haze_density.evaluate.build_method') as build:
            with self.assertRaisesRegex(ValueError, 'leakage'):
                evaluate(self.root / 'data', self.root / 'result', self.config)
            build.assert_not_called()

    def test_evaluation_rejects_config_mismatch(self):
        generate(self.manifest, self.root / 'data', self.config)
        other = copy.deepcopy(self.config)
        other['domain'] = 'display_rgb'
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            evaluate(self.root / 'data', self.root / 'result', other)


if __name__ == '__main__':
    unittest.main()
