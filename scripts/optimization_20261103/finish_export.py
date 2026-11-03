"""Finish export after Path/string helper error; do not change frozen training."""
import sys,json,time,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from scripts.optimization_20261103.airlight_pilot import init,read_package,OUT,OLD
def main():
    assert len(list(OUT.glob('*_seed*/complete.json')))==9
    for c in ['nominal','unseen_A','display_rgb']:assert (OUT/'evaluation'/c/'summary.json').exists()
    calpath=Path(json.loads((OLD/'protocol.json').read_text())['config']['calibration'])
    manifest=[];dest=OUT/'portable_weights';dest.mkdir(exist_ok=True)
    for seed in [42,43,44]:
        path=OUT/f'joint_dynamic_seed{seed}/best.pth';net,_,_=init('joint_dynamic',seed);net.load_state_dict(read_package(path)['model'])
        file=dest/f'joint_dynamic_seed{seed}.pth';assert not file.exists()
        torch.save(dict(model={k:v.cpu() for k,v in net.state_dict().items()},calibration=json.loads(calpath.read_text()),mode='joint_dynamic',seed=seed,source_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),file)
        manifest.append(dict(path=str(file.relative_to(OUT)),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),bytes=file.stat().st_size))
        del net
    for name,obj in [('portable_manifest.json',manifest),('EXPORT_REPAIR.json',{'reason':'Original training helper read() expected Path but export passed calibration path string; no training/evaluation failure','repair_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'old_training_source_unchanged':True}),('COMPLETE.json',{'all_nine_runs_complete':True,'new_confirmation_scenes':0,'recovered_export':True,'total_epoch_seconds':sum(sum(h['seconds'] for h in json.loads(p.read_text())) for p in OUT.glob('*_seed*/history.json'))})]:
        with (OUT/name).open('x') as f:json.dump(obj,f,indent=2)
    (OUT/'status.json').write_text(json.dumps({'stage':'complete','export_repaired':True}))
    print('EXPORT REPAIR COMPLETE')
if __name__=='__main__':main()
