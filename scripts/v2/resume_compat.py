"""Compatibility launcher for frozen V2 trainers' CUDA RNG restore.

Keep frozen experimental source byte-for-byte unchanged. torch.load(...,
map_location='cuda') also moves ByteTensor RNG state to CUDA; normalize only
RNG state back to CPU in the returned checkpoint. Model/optimizer tensors
and fresh-training computations are unaffected. Scoped to this process.
"""
import argparse,contextlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from unittest.mock import patch


@contextlib.contextmanager
def checkpoint_rng_compat():
    original_load=torch.load
    def compatible_load(*args,**kwargs):
        state=original_load(*args,**kwargs)
        if isinstance(state,dict) and state.get('cuda_rng') is not None:
            state['cuda_rng']=[v.cpu() for v in state['cuda_rng']]
        return state
    with patch.object(torch,'load',compatible_load):yield


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--target',required=True,choices=['v2','investigation','rgb']);args,extra=p.parse_known_args()
    if args.target=='v2':from scripts.v2.run import main as entry
    elif args.target=='investigation':from scripts.v2.investigate import main as entry
    else:from scripts.v2.rgb_control import main as entry
    sys.argv=[sys.argv[0]]+extra
    with checkpoint_rng_compat():entry()

if __name__=='__main__':main()
