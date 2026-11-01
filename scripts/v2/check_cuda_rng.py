"""Check whether CUDA-mapped RNG tensors need CPU conversion on resume."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts.v15.run import read,write
from scripts.v2.investigate import charge

def main():
    cfg=read('configs/v2_investigation.json');out=Path(cfg['output_directory']);start=time.monotonic()
    states=torch.cuda.get_rng_state_all();mapped=[s.cuda() for s in states];result={}
    try:torch.cuda.set_rng_state_all(mapped);result['cuda_mapped_rng_restore']='PASS'
    except Exception as error:result['cuda_mapped_rng_restore']=str(error)
    torch.cuda.set_rng_state_all([s.cpu() for s in mapped]);result['cpu_converted_rng_restore']='PASS'
    write(out/'cuda_rng_restore_check.json',result)
    # Do not concurrently edit the shared training ledger; this subsecond check
    # is reported separately and added by finalization.
    result['elapsed_seconds']=time.monotonic()-start;write(out/'cuda_rng_restore_check.json',result);print(result)

if __name__=='__main__':main()
