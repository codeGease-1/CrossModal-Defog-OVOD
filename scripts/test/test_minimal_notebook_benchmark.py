"""Check the Colab workflow hands its existing experiment paths to the pilot."""
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'docs/雾密度网络_最简实验流程.ipynb'
STANDALONE = ROOT / 'docs/haze_density_benchmark_colab.ipynb'
TAG = 'stage2-independent-benchmark'


class NotebookBenchmarkIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.integrated = [cell for cell in self.notebook['cells']
                           if TAG in cell.get('metadata', {}).get('tags', [])]

    def test_embedded_code_matches_standalone_snapshot(self):
        self.assertEqual([cell['cell_type'] for cell in self.integrated],
                         ['markdown', 'code', 'code', 'code'])
        self.assertEqual(''.join(self.notebook['cells'][-1]['source']).count('- **独立 Benchmark**：'), 1)
        setup = ''.join(self.integrated[1]['source'])
        tree = ast.parse(setup)
        encoded = next(ast.literal_eval(node.value.args[0]) for node in tree.body
                       if isinstance(node, ast.Assign) and
                       any(isinstance(target, ast.Name) and target.id == 'payload'
                           for target in node.targets))
        digest = hashlib.sha256(base64.b64decode(encoded)).hexdigest()
        expected = json.loads(STANDALONE.read_text(encoding='utf-8'))['metadata']['bundle_sha256']
        self.assertEqual(digest, expected)
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))

    def test_runner_reuses_data_and_only_eligible_checkpoint(self):
        code = ''.join(self.integrated[2]['source'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, run, checkpoint = root / 'RSHaze+', root / 'experiment', root / 'experiment/checkpoints'
            data.mkdir()
            checkpoint.mkdir(parents=True)
            calls = []
            ns = {'DATASET_ROOT': data, 'RUN_DIR': run, 'CHECKPOINT_DIR': checkpoint,
                  'PRIOR_VERSION': 'spatial_minmax_valid_boundary_v3', 'BENCHMARK_ROOT': root / 'bundle',
                  'datetime': datetime, 'timezone': timezone, 'sys': sys, 'torch': torch,
                  'run_command': lambda cmd, cwd: calls.append((cmd, cwd))}
            exec(code, ns)
            command, cwd = calls[-1]
            self.assertEqual(cwd, root / 'bundle')
            self.assertEqual(command[command.index('--data-root') + 1], str(data))
            self.assertTrue(Path(command[command.index('--output') + 1]).is_relative_to(run / 'benchmarks'))
            self.assertNotIn('--checkpoint', command)
            self.assertIn('--fetch-small-sample', command)
            torch.save({'config': {'prior_version': 'older'}, 'epoch': 2}, checkpoint / 'best.pth')
            exec(code, ns)
            self.assertNotIn('--checkpoint', calls[-1][0])
            torch.save({'config': {'prior_version': ns['PRIOR_VERSION']}, 'epoch': 2}, checkpoint / 'best.pth')
            exec(code, ns)
            self.assertEqual(calls[-1][0][calls[-1][0].index('--checkpoint') + 1], str(checkpoint / 'best.pth'))


if __name__ == '__main__':
    unittest.main()
