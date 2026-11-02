"""Verify the exported trained corrector's input/dependency contracts on CPU."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from src.models.haze_density.dynamic_prior import FrozenCandidates
from src.models.haze_density.learned_correction import RestrictedCorrector, observed_context


def main():
    p = argparse.ArgumentParser()
    for name in ['weights', 'calibration', 'image', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(); assert not args.output.exists()
    torch.set_num_threads(1)
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    assert state['mode'] == 'corrector16'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(args.calibration) == state['calibration_sha256']
    m = RestrictedCorrector(16, state['config']['rho'], state['config']['shift']).eval()
    m.load_state_dict(state['model'])
    ext = FrozenCandidates(json.loads(args.calibration.read_text(encoding='utf-8'))).eval()
    image = np.asarray(Image.open(args.image).convert('RGB'), np.float32)[:64, :64]/255
    rgb = torch.from_numpy(image.transpose(2, 0, 1).copy())[None]
    with torch.no_grad():
        f, _ = ext(rgb); f = torch.cat([f, observed_context(rgb)], 1)
        c, aux = m(f)
        altered = f.clone(); altered[:, 3] = (altered[:, 3]+.1).clamp(max=1)
        changed, other_aux = m(altered)
        isolated = torch.equal(c[:, 1:], changed[:, 1:])
        own_response = float((changed[:, 0]-c[:, 0]).abs().mean())
        assert isolated and own_response > 0
        assert torch.equal(aux['gain'], other_aux['gain'])
        assert (changed[:, 0] >= c[:, 0]-1e-7).all()
        # Other batch members cannot supply hidden scene or target information.
        batch, _ = m(torch.cat([f, f.flip(-1)], 0))
        batch_delta = float((batch[:1]-c).abs().max())
        assert batch_delta < 2e-5, batch_delta
        assert torch.isfinite(c).all() and c.min() >= 0 and c.max() <= 1
        weights = torch.tensor(state['fusion_weights'])
        assert (weights >= 0).all() and abs(float(weights.sum())-1) < 1e-6
    result = {'status': 'PASS', 'weights_sha256': sha(args.weights), 'calibration_sha256': sha(args.calibration),
        'image_sha256': sha(args.image), 'audit_script_sha256': sha(Path(__file__)),
        'forward_signature': str(inspect.signature(m.forward)), 'own_candidate_mean_response': own_response,
        'unrelated_candidates_exactly_unchanged': isolated, 'batch_member_max_difference': batch_delta,
        'fusion_weights_separate_frozen_constants': state['fusion_weights'],
        'scope': 'Actual trained model dependency contracts, not proof of causal necessity or natural-image accuracy'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
