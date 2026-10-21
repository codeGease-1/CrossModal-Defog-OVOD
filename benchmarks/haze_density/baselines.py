"""Adapters around frozen project methods; no training imports."""
from pathlib import Path
import torch
from src.models.haze_density import HazeDensityNet
from src.models.haze_density.physical_prior import dark_channel, PhysicalPriorModule, PRIOR_VERSION
from .common import NotAvailable, sha256_file


def resolve_device(name):
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU/CUDA are supported by this benchmark')
    if device.type == 'cuda' and (not torch.cuda.is_available() or
                                 (device.index is not None and device.index >= torch.cuda.device_count())):
        raise NotAvailable(f'Requested GPU unavailable: {name}')
    return device


class RawDarkChannel(torch.nn.Module):
    def __init__(self, window_size=15):
        super().__init__()
        self.window_size = window_size

    def forward(self, image):
        return dark_channel(image, self.window_size)


def build_method(name, device, config, architecture_only=False):
    device = resolve_device(device)
    params = config.get('prior', {'window_size': 15, 'guided_radius': 15, 'guided_eps': .01})
    info = {'method': name, 'direction': 'higher_response_means_more_haze_by_design',
            'output_meaning': 'relative response, not physical transmission', 'device': str(device)}
    if name == 'dcp':
        model = RawDarkChannel(params['window_size'])
        info['definition'] = 'raw local RGB minimum; no per-image normalization, no A estimation'
    elif name == 'prior':
        model = PhysicalPriorModule(**params)
        info.update({'prior_version': PRIOR_VERSION, 'parameters': params})
    elif name == 'student':
        checkpoint_path = config.get('checkpoint')
        if architecture_only:
            model = HazeDensityNet(base_channels=32, use_sigmoid=True)
            info.update({'weights': 'random initialization', 'accuracy_status': 'NOT EVALUATED',
                         'purpose': 'architecture cost only', 'method': 'student_architecture_only'})
        else:
            if not checkpoint_path or not Path(checkpoint_path).is_file():
                raise NotAvailable('No trained V1 checkpoint; student NOT EVALUATED')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            saved = checkpoint.get('config', {})
            if saved.get('prior_version') != PRIOR_VERSION or checkpoint.get('epoch', 0) < 1:
                raise NotAvailable('Checkpoint has missing/older prior version or no completed training epoch; NOT EVALUATED')
            if not saved.get('use_sigmoid', True):
                raise ValueError('This benchmark requires the documented [0,1] student output')
            model = HazeDensityNet(base_channels=saved.get('base_channels', 32), use_sigmoid=True)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Checkpoint contains nonfinite weights')
            info.update({'checkpoint_sha256': sha256_file(checkpoint_path),
                         'checkpoint_config': saved, 'epoch': checkpoint['epoch'],
                         'training_overlap_status': 'NOT VERIFIED: confirm source-scene isolation from training data'})
    else:
        raise ValueError(f'Unknown method: {name}')
    info['learned_parameters'] = sum(p.numel() for p in model.parameters())
    return model.to(device).eval(), info
