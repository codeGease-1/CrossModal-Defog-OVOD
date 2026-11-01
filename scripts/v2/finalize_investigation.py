"""Freeze final environment, old-source integrity and complete result hashes."""
from pathlib import Path
import hashlib,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.v15.run import read,write,digest

def main():
    root=Path(__file__).resolve().parents[2];cfg=read(root/'configs/v2_investigation.json');out=root/cfg['output_directory'];p=read(out/'protocol.json');old=root/cfg['previous_directory']
    for s,sha in p['source_sha256'].items():assert digest(root/s)==sha,s
    assert digest(old/'frozen.json')==p['previous_frozen_sha256']
    assert digest(old/'calibration.json')==p['candidate_calibration_sha256']
    for name,sha in read(out/'frozen.json')['checkpoints'].items():assert digest(out/name)==sha
    for name,sha in read(out/'rgb_control/frozen.json')['checkpoint_sha256'].items():assert digest(out/name)==sha
    for name,sha in read(old/'frozen.json')['checkpoint_sha256'].items():assert digest(old/name)==sha
    for script,plan in [('rgb_control.py','rgb_control/plan.json'),('spatial_gate_controls.py','spatial_controls/plan.json')]:
        assert digest(root/'scripts/v2'/script)==read(out/plan)['source_sha256']
    b=read(out/'budget.json')
    if not b.get('auxiliary_checks_added',False):
        seconds=sum(read(out/name)['elapsed_seconds'] for name in ('cuda_rng_restore_check.json','resume_fixture.json'))
        b['new_gpu_wall_seconds']+=seconds;b['stages'].append({'stage':'auxiliary_cuda_checks_conservative_overlap_count','seconds':seconds})
        b['auxiliary_checks_added']=True
        b['accounting']='Conservative GPU-stage wall time including preprocessing/evaluation. Brief CUDA resume fixtures overlapped training and are additionally counted; no concurrent model searches or inference benchmarks.'
        write(out/'budget.json',b)
    assert b['new_gpu_wall_seconds']<=b['limit_seconds']
    previous=read(old/'budget.json');cumulative=b['new_gpu_wall_seconds']+previous['training_seconds']+previous['calibration_seconds']
    assert cumulative<3600
    oldscenes=read(old/'scene_manifest.json');fresh=read(out/'confirmation_manifest.json')
    assert not ({s['group_id'] for s in oldscenes}&{s['group_id'] for s in fresh})
    assert not ({s['original_pixel_sha256'] for s in oldscenes}&{s['original_pixel_sha256'] for s in fresh})
    assert not (set(read(old/'v15_display_compat/frozen.json')['confirmation_seeds'])&{s['seed'] for s in fresh})
    (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    scripts=list((root/'scripts/v2').glob('*investig*'))+[root/'scripts/v2/rgb_control.py',root/'scripts/v2/evaluate_rgb_control.py',root/'scripts/v2/summarize_rgb_comparison.py',root/'scripts/v2/test_student_gate.py',root/'scripts/v2/resume_compat.py',root/'scripts/v2/verify_student_resume.py',root/'scripts/v2/check_cuda_rng.py',root/'scripts/v2/inspect_development.py',root/'scripts/v2/predict_weight_gate.py',root/'scripts/v2/spatial_gate_controls.py',root/'scripts/v2/inference_smoke.py',root/'configs/v2_investigation.json']
    write(out/'completion.json',{'old_sources_calibration_checkpoints_unchanged':True,'all_new_best_checkpoints_verified':True,
        'followup_gpu_wall_seconds':b['new_gpu_wall_seconds'],'previous_fit_seconds':previous['training_seconds']+previous['calibration_seconds'],
        'conservative_combined_seconds':cumulative,'previous_v15_fit_not_included_here':28.0,
        'source_sha256':{str(s.relative_to(root)):digest(s) for s in scripts if s.is_file()},
        'limitations':['procedural training','same generator-family confirmation','real geography unverified','secondary spectral target differs from scalar t'],
        'status':'All declared follow-up experiments evaluated; see report for adoption decision'})
    files=[q for q in out.rglob('*') if q.is_file() and q.name!='artifact_sha256.json']
    write(out/'artifact_sha256.json',{str(q.relative_to(out)):digest(q) for q in sorted(files)})
    print('Frozen integrity OK;',len(files),'artifact hashes;',round(cumulative,1),'combined seconds')

if __name__=='__main__':main()
