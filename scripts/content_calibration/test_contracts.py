"""Necessary mechanism and no-leakage contracts, NOT model efficacy tests."""
import copy
import inspect
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.models.haze_density.content_calibration import (
    fit_content_calibration, ContentCalibratedCandidates, FixedCandidateFusion,
    content_features, response_objective, weighted_monotone, ContentDirectReference)
from src.models.haze_density.student_weight_gate import StudentWeightGate
from src.models.haze_density.semantic_prior import candidate_diagnostics


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        self.x = rng.uniform(-.1, .8, (1200, 3))
        self.f = rng.random((1200, 2))
        self.y = np.clip(self.x[:, 0] + .2*(self.f[:, 0] > .5), 0, 1)
        self.g = np.tile(np.array(['a', 'b', 'c', 'd']), 300)
        self.cfg = dict(split='train', min_pixels=20, min_scenes=4, bins=16, shrinkage=30, cue_window=3, content_window=5)

    def fit(self, **kwargs):
        return fit_content_calibration(self.x, self.y, self.f, self.g, **{**self.cfg, **kwargs})

    def test_train_only_and_original_support(self):
        with self.assertRaises(ValueError): self.fit(split='development')
        with self.assertRaises(ValueError): self.fit(split='test')
        cal = fit_content_calibration(self.x, self.y, self.f, np.repeat('one_original', len(self.y)), **self.cfg)
        self.assertEqual(cal['cells'], {})
        self.assertTrue(len(self.fit()['cells']) > 0)

    def test_candidate_dependency_and_bounded_correction(self):
        model = ContentCalibratedCandidates(self.fit())
        x = torch.tensor(self.x[:64].T.reshape(1, 3, 8, 8), dtype=torch.float32)
        f = torch.tensor(self.f[:64].T.reshape(1, 2, 8, 8), dtype=torch.float32)
        out, used = model.correct(x, f)
        changed = x.clone(); changed[:, 0] += .1
        other = model.correct(changed, f)[0]
        self.assertTrue(torch.equal(out[:, 1:], other[:, 1:]))
        self.assertGreater(float((out[:, 0]-other[:, 0]).abs().max()), 0)
        global_out = ContentCalibratedCandidates(model.calibration, False).correct(x, f)[0]
        self.assertLessEqual(float((out-global_out).abs().max()), .050001)
        self.assertTrue(used.any())
        self.assertGreater(float((out-global_out).abs().max()), 0)

    def test_sparse_out_of_support_fallback_and_serialization(self):
        cal = self.fit()
        model = ContentCalibratedCandidates(json.loads(json.dumps(cal)))
        raw = torch.rand(1, 3, 8, 8)
        f = torch.full((1, 2, 8, 8), 5.)
        out, used = model.correct(raw, f)
        base = ContentCalibratedCandidates(cal, False).correct(raw, f)[0]
        self.assertFalse(used.any()); self.assertTrue(torch.equal(out, base))
        for t in cal['global']:
            self.assertTrue(np.all(np.diff(t['y']) >= 0))

    def test_inference_signature_and_fusion_independence(self):
        self.assertEqual(list(inspect.signature(ContentCalibratedCandidates.forward).parameters), ['self', 'image'])
        model = ContentCalibratedCandidates(self.fit())
        rgb = torch.rand(1, 3, 32, 32)
        f, c = model(rgb)
        p1 = FixedCandidateFusion([1, 0, 0])(c)
        p2 = FixedCandidateFusion([0, 1, 0])(c)
        self.assertTrue(torch.equal(c, model(rgb)[1]))
        self.assertTrue(torch.equal(p1, c[:, :1])); self.assertTrue(torch.equal(p2, c[:, 1:2]))
        gate = StudentWeightGate(base_channels=8)
        p, w = gate(f, c.detach())
        loss = (p-torch.rand_like(p)).square().mean(); loss.backward()
        self.assertTrue(any(q.grad is not None and q.grad.abs().sum() > 0 for q in gate.parameters()))
        self.assertTrue(torch.allclose(p, (w*c).sum(1, keepdim=True)))
        self.assertFalse(any(True for _ in model.parameters()))
        ref, _ = ContentDirectReference(base_channels=8)(f.detach())
        self.assertEqual(ref.shape, p.shape)

    def test_physics_square_completion_zero_baseline_and_domain_guard(self):
        y = np.array([.2, .6, .9]); b = np.array([[1., 0., 0.], [.3, .01, .2], [.5, .1, .01]])
        s = np.array([1, 0, 0]); q = b + (s-b)*y[:, None]; valid = np.ones_like(q, bool)
        z, w = response_objective(y, b, q, valid, 3.)
        self.assertTrue(np.allclose(z, y[:, None])); self.assertTrue(np.isfinite(w).all())
        self.assertTrue(np.all(w >= 1)); self.assertTrue(np.array_equal(w[0], np.ones(3)))
        q += .1
        z, w = response_objective(y, b, q, valid, 3.)
        p1, p2 = np.full_like(q, .3), np.full_like(q, .7)
        def original(p): return (p-y[:, None])**2 + 3*(q-(b+(s-b)*p))**2
        self.assertTrue(np.allclose(original(p1)-original(p2), w*((p1-z)**2-(p2-z)**2)))
        z0, w0 = response_objective(y, b, q, np.zeros_like(valid), 3.)
        self.assertTrue(np.allclose(z0, y[:, None])); self.assertTrue(np.all(w0 == 1))
        with self.assertRaises(ValueError): self.fit(physical_strength=1)
        physical = {'verified_protocol': 'wrong_display_domain'}
        with self.assertRaises(ValueError): self.fit(physical_strength=1, physical=physical)

    def test_no_envelope_only_success_and_oracle(self):
        c = np.array([[[.2, .4]], [[.3, .5]], [[.4, .6]]]); y = np.array([[.1, .7]])
        d = candidate_diagnostics(c, y)
        self.assertAlmostEqual(d['offline_oracle_mae'], .1)
        self.assertAlmostEqual(d['offline_oracle_mse'], .01)
        self.assertEqual(d['outside_envelope_fraction'], 1.)
        wider = np.array([[[0., 0.]], [[.3, .5]], [[1., 1.]]])
        self.assertLess(candidate_diagnostics(wider, y)['outside_envelope_fraction'], 1.)
        self.assertGreater(np.mean(np.abs(wider-y)), np.mean(np.abs(c-y)))

    def test_context_is_observed_only_no_per_image_scaling(self):
        a = torch.full((1, 3, 8, 8), .2); b = torch.full_like(a, .8)
        self.assertGreater(float(content_features(b, 3)[:, 0].mean()-content_features(a, 3)[:, 0].mean()), .59)
        with self.assertRaises(ValueError): content_features(torch.full_like(a, float('nan')))
        model = ContentCalibratedCandidates(self.fit())
        p = model(a)[1]
        batch = model(torch.cat([a, b], 0))[1]
        self.assertTrue(torch.allclose(p, batch[:1]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
