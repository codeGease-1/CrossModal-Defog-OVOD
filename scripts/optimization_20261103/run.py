"""Resumable driver around frozen pilot source and its isolated export repair."""
import sys,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
O=ROOT/'experiments/optimization_20261103'
def run(script):subprocess.run([sys.executable,'scripts/optimization_20261103/'+script],cwd=ROOT,check=True)
def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['all','latency','airlight','approximate'],default='all');args=a.parse_args()
    if args.stage in ['all','latency'] and not (O/'latency_run03/complete.json').exists():run('latency.py')
    if args.stage in ['all','airlight','approximate']:
        p=O/'airlight_pilot'
        if not (p/'COMPLETE.json').exists():
            result=subprocess.run([sys.executable,'scripts/optimization_20261103/airlight_pilot.py'],cwd=ROOT)
            if not (p/'COMPLETE.json').exists():
                done=len(list(p.glob('*_seed*/complete.json')))==9
                evaluated=all((p/'evaluation'/c/'summary.json').exists() for c in ['nominal','unseen_A','display_rgb'])
                if done and evaluated:run('finish_export.py')
                else:result.check_returncode();raise RuntimeError('Training or evaluation incomplete; inspect protected outputs')
        if not (O/'POSTPROCESS_COMPLETE.json').exists():run('postprocess.py')
    if args.stage in ['all','approximate'] and not (O/'approximate_quality/COMPLETE.json').exists():run('approximate_quality.py')
    print('Requested stages complete; existing results retained.')
if __name__=='__main__':main()
