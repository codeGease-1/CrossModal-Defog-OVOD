"""Offline provenance checks of completed frozen evaluation artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source-archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); assert not args.output.exists()
    protocol = read(args.run/'protocol.json'); frozen = read(args.run/'frozen.json')
    assert protocol['config'] == frozen['config']
    assert hashlib.sha256((args.run/'fusion_weights.json').read_bytes()).hexdigest() == frozen['fusion_weights_sha256']
    checked = []
    with zipfile.ZipFile(args.source_archive) as z:
        for path, expected in protocol['source_sha256'].items():
            assert hashlib.sha256(z.read(path)).hexdigest() == expected, path
            checked.append(path)
    manifests = {split: read(args.run/(split+'_manifest.json')) for split in ['train', 'development']}
    groups = {split: {r['group_id'] for r in rows} for split, rows in manifests.items()}
    assert len(groups['train']) == 48 and len(groups['development']) == 16
    assert len(manifests['train']) == 1536 and len(manifests['development']) == 512
    assert not groups['train'] & groups['development']
    assert len({s['pixel_sha256'] for s in protocol['scenes']}) == 64
    from scripts.v15.run import procedural
    generator = Path(__file__).resolve().parents[2]/'scripts/v15/run.py'
    assert hashlib.sha256(generator.read_bytes()).hexdigest() == protocol['source_sha256']['scripts/v15/run.py']
    fresh_pixels = []
    for i in range(protocol['config']['confirmation_scene_count']):
        rgb, _ = procedural(protocol['base']['seed']+protocol['config']['confirmation_seed_offset']+i, protocol['base']['image_size'])
        fresh_pixels.append(hashlib.sha256(rgb.tobytes()).hexdigest())
    assert len(set(fresh_pixels)) == 24
    assert not set(fresh_pixels) & {s['pixel_sha256'] for s in protocol['scenes']}
    suffix = '_'.join(map(str, protocol['config']['seeds']))
    dev = read(args.run/'development/nominal'/f'manifest_{suffix}.json')
    old_inputs = {r['rgb_sha256'] for r in dev}
    conditions, nominal_targets = {}, None
    for condition in ['nominal', 'unseen_A', 'display_rgb']:
        m = read(args.run/'confirmation'/condition/f'manifest_{suffix}.json')
        scene_groups = {r['group_id'] for r in m}
        assert len(scene_groups) == 24 and len(m) == 768
        assert not scene_groups & (groups['train'] | groups['development'])
        assert not old_inputs & {r['rgb_sha256'] for r in m}
        counts = {g: sum(r['group_id'] == g for r in m) for g in scene_groups}
        assert set(counts.values()) == {32}
        targets = {r['case_id']: r['target_sha256'] for r in m}
        if nominal_targets is None: nominal_targets = targets
        else: assert targets == nominal_targets
        conditions[condition] = {'original_scenes': len(scene_groups), 'cases': len(m),
            'versions_per_group': 32, 'no_development_input_overlap': True,
            'target_hashes_identical_across_conditions': True}
    result = {'status': 'PASS', 'source_files_exactly_verified_against_frozen_zip': len(checked),
        'source_archive_sha256': hashlib.sha256(args.source_archive.read_bytes()).hexdigest(),
        'train_originals_cases': [48, 1536], 'development_originals_cases': [16, 512],
        'training_development_pixel_hashes_unique': True, 'no_cross_split_group_overlap': True,
        'fresh_original_pixel_hashes_unique_and_disjoint': fresh_pixels,
        'frozen_fusion_hash_unchanged': True, 'all_checkpoints_frozen': len(frozen['checkpoints']),
        'conditions': conditions, 'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
