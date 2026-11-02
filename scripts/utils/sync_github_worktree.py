"""Fetch a pinned GitHub branch into a NEW /mnt worktree; preserve old files."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def synchronize(repository, destination, branch, expected_commit, apply=False):
    repo, dest, mount = repository.resolve(), destination.resolve(), Path('/mnt').resolve()
    for path in [repo, dest]:
        if path == mount or not path.is_relative_to(mount):
            raise ValueError('Repository and destination must resolve strictly inside /mnt')
    if dest.is_relative_to(repo) or repo.is_relative_to(dest):
        raise ValueError('Use a separate sibling worktree, not an overlapping directory')
    if dest.exists():
        raise FileExistsError('Existing destination protected: '+str(dest))
    if not re.fullmatch(r'[0-9a-fA-F]{40}', expected_commit):
        raise ValueError('Supply the full 40-character commit from the prepared local commit')
    expected_commit = expected_commit.lower()
    git(repo, 'check-ref-format', '--branch', branch)
    before_head = git(repo, 'rev-parse', 'HEAD')
    before_status = git(repo, 'status', '--porcelain=v1', '-uall')
    remote = git(repo, 'ls-remote', '--heads', 'origin', 'refs/heads/'+branch).split()
    if not remote or remote[0] != expected_commit:
        raise RuntimeError('Remote branch does not equal the expected commit; push first or recheck the commit')
    receipt = {'mode': 'apply' if apply else 'preview', 'repository': str(repo),
        'new_worktree': str(dest), 'branch': branch, 'expected_commit': expected_commit,
        'original_head': before_head, 'original_worktree_status': before_status,
        'policy': 'No old checkout, reset, clean, file removal or experiment synchronization'}
    if apply:
        git(repo, 'fetch', '--no-tags', 'origin', branch)
        fetched = git(repo, 'rev-parse', 'FETCH_HEAD')
        if fetched != expected_commit:
            raise RuntimeError('Remote advanced during fetch; no worktree was created')
        git(repo, 'worktree', 'add', '--detach', str(dest), fetched)
        assert git(dest, 'rev-parse', 'HEAD') == expected_commit
        assert git(repo, 'rev-parse', 'HEAD') == before_head
        assert git(repo, 'status', '--porcelain=v1', '-uall') == before_status
        receipt['original_head_and_status_unchanged'] = True
        (dest/'.git-sync-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, default=Path('/mnt/CrossModal-Defog-OVOD'))
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--branch', default='v1-nocolab')
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--apply', action='store_true', help='Default is read-only preview')
    a = p.parse_args()
    print(json.dumps(synchronize(a.repository, a.destination, a.branch, a.expected_commit, a.apply), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
