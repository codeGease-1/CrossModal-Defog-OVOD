"""Exercise actual CUDA training resume with the frozen-source compatibility fix."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import contextlib,io,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from unittest.mock import patch
import torch
from scripts.v2 import investigate as run
from scripts.v2.resume_compat import checkpoint_rng_compat

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.monotonic();cfg=run.read('configs/v2_investigation.json');out=Path(cfg['output_directory']);p=run.read(out/'protocol.json')
    cfg=dict(cfg,seeds=[42],epochs=2,batch_size=2,student_base_channels=8)
    torch.manual_seed(912);x=torch.rand(4,9,32,32);y=torch.rand(4,1,32,32);y[0]=0
    def cache(*args):return x,y,[]
    differences={}
    with tempfile.TemporaryDirectory(prefix='v2_cuda_resume_') as tmp:
        roots=[Path(tmp)/v for v in ('straight','resumed')]
        for root in roots:
            root.mkdir();run.write(root/'budget.json',{'limit_seconds':3000,'new_gpu_wall_seconds':0,'stages':[]})
        original_write=run.write
        def interrupt(path,value):
            original_write(path,value)
            if Path(path)==roots[1]/'temperature_seed42/history.json' and len(value)==1:raise InterruptedError('intentional fixture interruption')
        with contextlib.redirect_stdout(io.StringIO()),patch.object(run,'cache_split',cache),checkpoint_rng_compat():
            run.train(roots[0],cfg,p)
            try:
                with patch.object(run,'write',interrupt):run.train(roots[1],cfg,p)
            except InterruptedError:pass
            else:raise AssertionError('fixture must interrupt')
            run.train(roots[1],cfg,p)
        for mode in cfg['candidates']:
            a=torch.load(roots[0]/f'{mode}_seed42/latest.pth',weights_only=False,map_location='cpu')
            b=torch.load(roots[1]/f'{mode}_seed42/latest.pth',weights_only=False,map_location='cpu')
            diff=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
            assert diff==0 and a['step']==b['step']==4,(mode,diff)
            assert torch.equal(a['shuffle_rng'],b['shuffle_rng']);differences[mode]=diff
    run.write(out/'resume_fixture.json',{'device':'cuda','actual_trainer':True,'status':'PASS',
        'compatibility_launcher':'scripts/v2/resume_compat.py','parameter_max_abs_difference':differences,
        'interrupted_at':'temperature epoch1 saved checkpoint','deterministic_algorithms':True,'elapsed_seconds':time.monotonic()-start,
        'limit':'Tiny fixture, not simulated power failure or corrupt filesystem'})
    print('CUDA interrupted/resumed training equals uninterrupted: PASS',differences)

if __name__=='__main__':main()
