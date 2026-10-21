"""Auxiliary PNG reconstruction hypotheses, not automatic truth certification."""
from pathlib import Path
import numpy as np
from PIL import Image
from .common import write_json, sha256_file, NotAvailable
from .synthetic import srgb_to_linear


def audit_auxiliary(clean, hazy, transmission, airlight, output):
    arrays, files = {}, {}
    for name, path in (('clean', clean), ('hazy', hazy), ('transmission', transmission), ('airlight', airlight)):
        path = Path(path)
        if not path.is_file():
            raise NotAvailable(f'Auxiliary audit file missing: {path}')
        with Image.open(path) as image:
            array = np.asarray(image)
            files[name] = {'path': str(path.resolve()), 'sha256': sha256_file(path), 'mode': image.mode,
                           'dtype': str(array.dtype), 'shape': list(array.shape),
                           'min': float(array.min()), 'max': float(array.max())}
        if array.dtype != np.uint8:
            raise ValueError('Automatic /255 hypothesis only supports uint8; verify other encodings explicitly')
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError('Unexpected auxiliary channel shape')
        arrays[name] = array.astype(np.float64) / 255
    j, i, t, a = [arrays[k] for k in ('clean', 'hazy', 'transmission', 'airlight')]
    if any(x.shape != j.shape for x in (i, t, a)):
        raise ValueError('Auxiliary files are not aligned in shape')
    display_error = j * t + a * (1-t) - i
    linear_error = srgb_to_linear(j) * t + srgb_to_linear(a) * (1-t) - srgb_to_linear(i)
    result = {'status': 'PASS', 'semantic_truth_status': 'UNVERIFIED', 'files': files,
              'hypothesis': 'uint8/255, matched files, airpng is A and transpng is t; not assumed true',
              'transmission_channel_max_spread': float(np.ptp(t, axis=2).max()),
              'airlight_spatial_std_rgb': a.std(axis=(0, 1)).tolist(),
              'display_rgb_reconstruction_mae': float(np.abs(display_error).mean()),
              'display_rgb_reconstruction_max_error': float(np.abs(display_error).max()),
              'linear_srgb_reconstruction_mae': float(np.abs(linear_error).mean()),
              'interpretation': 'Reconstruction agreement alone cannot establish parameter semantics, calibration or independent physical truth. These maps are not used in benchmark truth.'}
    write_json(output, result)
    return result
