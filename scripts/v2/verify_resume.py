"""Exercise the real trainer's interrupted checkpoint path on a tiny CPU fixture."""
import contextlib,copy,io,json,tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from unittest.mock import patch
import torch
from scripts.v2 import run

def main():
    torch.set_num_threads(1);out=Path('experiments/v2_20261101');cfg=copy.deepcopy(run.read(out/'protocol.json')['config'])
    cfg.update(seeds=[42],epochs=2,batch_size=2,training_budget_seconds=3600)
    torch.manual_seed(900);x=torch.rand(4,9,16,16);y=torch.rand(4,1,16,16);y[0]=0
    def cache(*args):return x,y,[]
    with tempfile.TemporaryDirectory(prefix='v2_resume_') as tmp:
        roots=[Path(tmp)/name for name in ('straight','resumed')]
        for root in roots:
            root.mkdir();run.write(root/'calibration.json',run.read(out/'calibration.json'))
            run.write(root/'oracle_development.json',{'pilot_only':False})
            run.write(root/'budget.json',{'training_seconds':0,'calibration_seconds':0})
        original_write=run.write;interrupted=False
        def write_then_interrupt(path,data):
            original_write(path,data)
            if Path(path)==roots[1]/'global_seed42/history.json' and len(data)==1:
                raise InterruptedError('Deliberate fixture interruption after epoch checkpoint')
        with contextlib.redirect_stdout(io.StringIO()),patch.object(run,'cache_split',cache):
            run.train(roots[0],cfg,{},'cpu')
            try:
                with patch.object(run,'write',write_then_interrupt):run.train(roots[1],cfg,{},'cpu')
            except InterruptedError:interrupted=True
            assert interrupted and not (roots[1]/'training_complete.json').exists()
            run.train(roots[1],cfg,{},'cpu')
        differences={}
        for mode in ('global','dynamic'):
            a=torch.load(roots[0]/f'{mode}_seed42/latest.pth',weights_only=False)
            b=torch.load(roots[1]/f'{mode}_seed42/latest.pth',weights_only=False)
            diff=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
            assert diff==0 and a['step']==b['step']==4
            assert torch.equal(a['shuffle_rng'],b['shuffle_rng'])
            differences[mode]=diff
        run.write(out/'resume_fixture.json',{'device':'cpu','actual_trainer':True,'interrupted_after':'global epoch1 checkpoint',
            'uninterrupted_vs_resumed_parameter_max_abs':differences,'status':'PASS',
            'limitation':'Tiny deterministic CPU fixture; does not emulate hardware failure or CUDA kernel nondeterminism'})
    print('Real trainer interrupted/resumed equals uninterrupted: PASS')
if __name__=='__main__':main()
