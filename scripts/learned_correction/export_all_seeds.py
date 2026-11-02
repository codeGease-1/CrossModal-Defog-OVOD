"""CPU-only export of every frozen seed; no test-based seed selection."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    read = lambda name: json.loads((args.run/name).read_text())
    protocol, frozen, fits = read('protocol.json'), read('frozen.json'), read('fusion_weights.json')
    records = []
    for name, expected in frozen['checkpoints'].items():
        checkpoint = args.run/name/'best.pth'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
        s = torch.load(checkpoint, map_location='cpu', weights_only=False)
        output = args.output/(name+'_inference.pth')
        torch.save({'model': s['model'], 'mode': s['mode'], 'seed': s['seed'],
            'config': protocol['config'], 'calibration_sha256': protocol['calibration_sha256'],
            'checkpoint_sha256': expected, 'selected_by': 'All registered seeds; no confirmation selection',
            'fusion_weights': fits[name]['weights'] if name.startswith('corrector') else None}, output)
        records.append({'name': name, 'path': output.name, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'bytes': output.stat().st_size, 'checkpoint_epoch_selected_on_development': s['epoch']})
    (args.output/'manifest.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__': main()
