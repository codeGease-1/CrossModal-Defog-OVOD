"""Independent atmospheric image formation and scene-disjoint provenance."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image
from .common import read_json, write_json, sha256_file, fresh_directory, environment, NotAvailable


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0) ** (1/2.4) - .055).astype(np.float32)


def synthesize(clear_rgb, t, airlight, brightness=1., domain='linear_srgb'):
    j, t, a = (np.asarray(x, dtype=np.float32) for x in (clear_rgb, t, airlight))
    if j.ndim != 3 or j.shape[-1] != 3 or t.shape != j.shape[:2] or a.shape != (3,):
        raise ValueError('Expected clear H,W,3, transmission H,W, and RGB airlight')
    if any(not np.isfinite(x).all() or x.min() < 0 or x.max() > 1 for x in (j, t, a)):
        raise ValueError('Image formation inputs must be finite in [0,1]')
    if not np.isfinite(brightness) or brightness <= 0 or domain not in ('linear_srgb', 'display_rgb'):
        raise ValueError('Invalid brightness or synthesis domain')
    radiance = srgb_to_linear(j) if domain == 'linear_srgb' else j
    unclipped = radiance * brightness
    radiance = np.clip(unclipped, 0, 1)
    hazy = radiance * t[..., None] + a * (1 - t[..., None])
    clean_display = linear_to_srgb(radiance) if domain == 'linear_srgb' else radiance
    hazy_display = linear_to_srgb(hazy) if domain == 'linear_srgb' else hazy
    return np.clip(clean_display, 0, 1), np.clip(hazy_display, 0, 1), float((unclipped > 1).mean())


def transmission_map(kind, size, seed, uniform_t=None):
    if size < 4:
        raise ValueError('size must be >=4')
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    rng = np.random.default_rng(seed)
    if kind == 'uniform':
        if uniform_t is None or not 0 <= uniform_t <= 1:
            raise ValueError('uniform t must be in [0,1]')
        t = np.full((size, size), uniform_t, dtype=np.float32)
        params = {'t': uniform_t}
    elif kind == 'gradient':
        t = .95 - .6 * x
        params = {'axis': 'left_to_right', 't_min': .35, 't_max': .95}
    elif kind == 'blob':
        cx, cy = rng.uniform(.35, .65, 2)
        sigma = .2
        t = .95 - .6 * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
        params = {'center': [float(cx), float(cy)], 'sigma': sigma, 'baseline_t': .95, 'amplitude': .6}
    elif kind == 'regions':
        values = [.9, .75, .6, .4]
        rng.shuffle(values)
        region = (x >= .5).astype(int) + 2 * (y >= .5).astype(int)
        t = np.asarray(values, dtype=np.float32)[region]
        params = {'quadrant_t': values}
    else:
        raise ValueError(f'Unknown transmission pattern: {kind}')
    return t.astype(np.float32), params


def load_scenes(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise NotAvailable(f'Scene manifest missing: {path}')
    data = read_json(path)
    if data.get('data_kind') not in ('real_clear_remote_sensing', 'implementation_fixture'):
        raise ValueError('Manifest must state real_clear_remote_sensing or implementation_fixture')
    if not data.get('scenes'):
        raise NotAvailable('No clear scenes in manifest')
    ids, groups, contents = set(), {}, {}
    for scene in data['scenes']:
        for key in ('scene_id', 'group_id', 'path', 'split', 'land_cover', 'source'):
            if not isinstance(scene.get(key), str) or not scene[key]:
                raise ValueError(f'Missing scene field: {key}')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', scene['scene_id']) or scene['scene_id'] in ids:
            raise ValueError('Scene IDs must be unique safe filename components')
        ids.add(scene['scene_id'])
        if scene['split'] not in ('train', 'calibration', 'test'):
            raise ValueError('split must be train/calibration/test')
        if groups.setdefault(scene['group_id'], scene['split']) != scene['split']:
            raise ValueError('Scene-group leakage across splits')
        source = (path.parent / scene['path']).resolve()
        if not source.is_file():
            raise NotAvailable(f'Clear image missing: {source}')
        with Image.open(source) as image:
            if image.mode != 'RGB':
                raise ValueError(f'Expected display RGB; verify sensor/bit-depth conversion first: {source}')
            pixels = np.asarray(image)
        pixel_hash = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        if 'roi' in scene:
            roi = scene['roi']
            if (len(roi) != 4 or any(not isinstance(v, int) for v in roi)
                    or not 0 <= roi[0] < roi[2] <= pixels.shape[1]
                    or not 0 <= roi[1] < roi[3] <= pixels.shape[0]):
                raise ValueError('ROI must be an in-bounds [left,top,right,bottom] pixel box')
        identity = (scene['group_id'], scene['split'])
        if pixel_hash in contents and contents[pixel_hash] != identity:
            raise ValueError('Identical decoded scene content assigned to different groups/splits')
        contents[pixel_hash] = identity
        digest = sha256_file(source)
        if scene.get('source_sha256') and scene['source_sha256'] != digest:
            raise ValueError(f'Source checksum mismatch: {source}')
        scene.update({'resolved_path': str(source), 'source_sha256': digest,
                      'pixel_sha256': pixel_hash, 'native_shape': list(pixels.shape)})
    return data


def generate(manifest_path, output, config, profile='minimal'):
    sources = load_scenes(manifest_path)
    output = fresh_directory(output)
    size, seed = int(config['image_size']), int(config['seed'])
    if size < 16 or size % 2 or profile not in ('minimal', 'full'):
        raise ValueError('Use even image_size >=16 and a minimal/full profile')
    levels = config['uniform_t']
    if len(set(levels)) != len(levels) or 1.0 not in levels:
        raise ValueError('uniform_t must contain unique levels including clear t=1')
    airlights = config['airlights'][:1] if profile == 'minimal' else config['airlights']
    brightnesses = [1.] if profile == 'minimal' else config['brightness']
    patterns = [] if profile == 'minimal' else config['spatial_patterns']
    records = []
    for scene in sources['scenes']:
        with Image.open(scene['resolved_path']) as image:
            if 'roi' in scene:
                image = image.crop(tuple(scene['roi']))
            image = image.resize((size, size), Image.Resampling.BILINEAR)
            original = np.asarray(image, dtype=np.float32) / 255
        for ai, a in enumerate(airlights):
            for bi, brightness in enumerate(brightnesses):
                for kind, t_value in [('uniform', t) for t in levels] + [(kind, None) for kind in patterns]:
                    key = f'{scene["scene_id"]}__a{ai}_b{bi}__{kind}' + (f'_t{t_value:g}' if t_value is not None else '')
                    case_seed = int(hashlib.sha256(f'{seed}:{scene["group_id"]}:{kind}'.encode()).hexdigest()[:8], 16)
                    t, params = transmission_map(kind, size, case_seed, t_value)
                    clean, hazy, clipping = synthesize(original, t, a, brightness, config['domain'])
                    case_path = output / 'cases' / f'{key}.npz'
                    case_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(case_path, original_rgb=original, clear_rgb=clean,
                                        hazy_rgb=hazy, transmission=t, airlight=np.asarray(a, dtype=np.float32))
                    records.append({'case_id': key, 'scene_id': scene['scene_id'], 'group_id': scene['group_id'],
                                    'split': scene['split'], 'land_cover': scene['land_cover'],
                                    'pattern': kind, 'uniform_t': t_value, 'airlight': a,
                                    'brightness': brightness, 'domain': config['domain'],
                                    'case_seed': case_seed, 'pattern_parameters': params,
                                    'brightness_clipped_fraction': clipping,
                                    'path': str(case_path.relative_to(output)).replace('\\', '/'),
                                    'sha256': sha256_file(case_path)})
    result = {'status': 'PASS', 'data_kind': sources['data_kind'], 'profile': profile,
              'source_manifest_sha256': sha256_file(manifest_path), 'sources': sources['scenes'],
              'config': config, 'cases': records,
              'truth_definition': 'transmission t; degradation strength 1-t, not extinction beta',
              'environment': environment(seed, config)}
    write_json(output / 'dataset.json', result)
    return result


def discover(root, output, limit=8, seed=42):
    if limit < 1:
        raise ValueError('limit must be positive')
    root = Path(root)
    candidates = sorted(root.glob('RSHaze_*/test/cleanpng/*.png'))
    if not candidates:
        raise NotAvailable(f'No official test cleanpng images under {root}')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    scenes, seen = [], set()
    for index in order:
        path = candidates[index]
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
        digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        scenes.append({'scene_id': 'scene_' + digest[:12],
                       'group_id': 'rshaze_test_geographic_group_unverified',
                       'path': str(path.resolve()), 'split': 'test', 'land_cover': 'unknown',
                       'label_source': 'requires manual annotation; geographic independence unverified',
                       'source': 'RSHaze+ official test cleanpng',
                       'source_sha256': sha256_file(path)})
        if len(scenes) >= limit:
            break
    write_json(output, {'schema_version': 1, 'data_kind': 'real_clear_remote_sensing',
                        'selection_seed': seed, 'scenes': scenes})
    return scenes
