"""Run through python3 stdin on V100. No tracked file or old artifact writes."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

root = Path('/mnt/CrossModal-Defog-OVOD').resolve()
assert root.is_dir()
def git(*args):
    return subprocess.check_output(['git', '-C', str(root), *args])
dest = root/'experiments/semantic_calibration_20261101/cloud_prechange'
dest.mkdir(parents=True, exist_ok=False)
status = git('status', '--porcelain=v1', '-uall').decode()
tracked = git('diff', '--name-only', 'HEAD').decode().splitlines()
untracked = git('ls-files', '--others', '--exclude-standard').decode().splitlines()
# Archive committed state, then back up every tracked change and untracked file.
(dest/'HEAD.tar').write_bytes(git('archive', 'HEAD'))
(dest/'working.patch').write_bytes(git('diff', '--binary', 'HEAD'))
with tarfile.open(dest/'uncommitted.tar.gz', 'w:gz') as archive:
    for name in sorted(set(tracked+untracked)):
        path = (root/name).resolve()
        assert path.is_relative_to(root)
        if path.is_file():
            archive.add(path, arcname=name, recursive=False)
protected = []
for path in sorted((root/'experiments').rglob('*')):
    if not path.is_file() or path.is_relative_to(dest):
        continue
    if path.name == 'latest.pth' or 'cache' in str(path.relative_to(root)).lower():
        item = {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size}
        if path.name == 'latest.pth':
            item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        protected.append(item)
result = {'root': str(root), 'head': git('rev-parse', 'HEAD').decode().strip(),
          'branch': git('branch', '--show-current').decode().strip(),
          'status': status, 'tracked_changes': tracked, 'untracked': untracked,
          'protected': protected, 'synchronization': 'No update needed if verified GitHub head matches; no checkout, reset, clean, pull or artifact deletion performed'}
(dest/'audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
