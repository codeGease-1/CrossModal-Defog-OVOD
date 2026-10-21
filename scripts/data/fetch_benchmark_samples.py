"""Explicit, bounded HTTP-range extraction of a few official RSHaze+ examples.

Never falls back to downloading the full archive. No extraction of remote paths.
Run manually only; the benchmark itself never downloads data.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import urllib.request
import zipfile


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, budget=24 * 1024 * 1024):
        self.url, self.size, self.budget = url, size, budget
        self.position = self.downloaded = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = (0 if whence == 0 else self.position if whence == 1 else self.size) + offset
        if self.position < 0:
            raise ValueError('negative seek')
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b''
        if self.downloaded + size > self.budget:
            raise RuntimeError('Download budget exceeded; full archive download forbidden')
        end = self.position + size - 1
        # A unique query prevents intermediaries from reusing a different byte range.
        url = self.url + f'&range_start={self.position}&range_end={end}'
        request = urllib.request.Request(url, headers={'Range': f'bytes={self.position}-{end}',
                                                     'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=30) as response:
            expected = f'bytes {self.position}-{end}/{self.size}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise RuntimeError('Server did not honor the requested range; refusing full download')
            payload = response.read(size + 1)
        self.downloaded += len(payload)
        if len(payload) != size:
            raise RuntimeError(f'Unexpected range length: requested {size}, received {len(payload)}, offset {self.position}')
        self.position += size
        return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='experiments/haze_benchmark/source_samples')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.count <= 12:
        parser.error('count must be 1..12')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty; preserve previous sample provenance')
    output.mkdir(parents=True, exist_ok=True)
    url = 'https://zenodo.org/records/13837162/files/RSHaze%2B.zip?download=1'
    reader = RangeReader(url, 7380255872)
    scenes, sources = [], []
    with zipfile.ZipFile(reader) as archive:
        members = sorted(x.filename for x in archive.infolist()
                         if '/RSHaze_G/test/cleanpng/' in '/' + x.filename
                         and x.filename.lower().endswith('.png'))
        if len(members) < args.count:
            raise RuntimeError(f'Only {len(members)} eligible clean images found')
        selected = sorted(random.Random(args.seed).sample(members, args.count))
        for index, member in enumerate(selected):
            entry = archive.getinfo(member)
            if entry.file_size > 8 * 1024 * 1024:
                raise ValueError('Unexpectedly large sample')
            payload = archive.read(member)
            name = f'G_test_{Path(member).name}'
            (output / name).write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            scene = {'scene_id': Path(name).stem, 'group_id': 'rshaze_test_geographic_group_unverified',
                     'path': name, 'split': 'test', 'land_cover': 'unknown',
                     'label_source': 'not annotated', 'source': url, 'archive_member': member,
                     'source_sha256': digest}
            scenes.append(scene)
            sources.append({'member': member, 'path': name, 'sha256': digest})
            print('extracted', member, len(payload), flush=True)
            if index == 0:
                for kind in ('synhazypng', 'transpng', 'airpng'):
                    auxiliary = member.replace('/cleanpng/', '/' + kind + '/')
                    if auxiliary in archive.namelist():
                        data = archive.read(auxiliary)
                        target = output / 'auxiliary' / kind / Path(member).name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
                        sources.append({'member': auxiliary, 'path': str(target.relative_to(output)),
                                        'sha256': hashlib.sha256(data).hexdigest()})
    manifest = {'schema_version': 1, 'data_kind': 'real_clear_remote_sensing',
                'selection_seed': args.seed, 'source_record': 'https://zenodo.org/records/13837162',
                'scenes': scenes}
    (output / 'scenes.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'download_provenance.json').write_text(json.dumps(
        {'url': url, 'bytes_read': reader.downloaded, 'budget_bytes': reader.budget,
         'archive_size': reader.size, 'members': sources}, indent=2), encoding='utf-8')
    print('PASS: extracted bytes', reader.downloaded, 'manifest', output / 'scenes.json')


if __name__ == '__main__':
    main()
