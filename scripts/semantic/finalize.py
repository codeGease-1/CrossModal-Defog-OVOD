"""Seal local CPU evidence and prepare an additive cloud delivery archive."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.v15.run import read,write,digest
import numpy as np
import torch


def main():
    out=ROOT/'experiments/semantic_calibration_20261101'
    initial=read(out/'safety_snapshot.json')
    changed=[name for name,sha in initial['sha256'].items() if digest(ROOT/name)!=sha]
    assert changed==['README.md'],changed
    assert read(out/'semantic_probe.json')['proxy_stability_screen_passed'] is False
    assert not read(out/'conditional_calibration.json')['classes']
    assert read(out/'inference_contract.json')['status'].startswith('PASS')
    assert all(r['n_cases']==24 for r in read(out/'real_reference_metrics.json')['summary'].values())
    sources=[Path('README.md'),Path('.gitignore'),Path('requirements-semantic.txt'),Path('configs/semantic_calibration.json'),
             Path('src/models/haze_density/semantic_prior.py'),Path('docs/语义辅助校正阶段诊断报告.md'),
             *[p.relative_to(ROOT) for p in (ROOT/'scripts/semantic').glob('*.py')]]
    manifest={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'sources':{p.as_posix():digest(ROOT/p) for p in sources},
              'historical_code_and_configs_unchanged':True,'expected_historical_document_changes':changed}
    write(out/'final_source_manifest.json',manifest)
    with zipfile.ZipFile(out/'final_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(ROOT/p,p.as_posix())
    write(out/'environment_final.json',{'python':sys.version,'platform':platform.platform(),
          'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','transformers','safetensors','psutil']},
          'numpy_path':np.__file__,'numpy_blas':np.__config__.CONFIG.get('Build Dependencies',{}).get('blas',{}),
          'device':'cpu','torch_threads':4,'gpu_used':False,
          'repair':'Same-version NumPy official wheel inside .venv replaces inherited Anaconda MKL; no system package changes or KMP_DUPLICATE_LIB_OK'})
    (out/'pip_freeze_final.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    write(out/'completion.json',{'implementation':'PASS; 5 contract tests including single CPU fixture gradient update',
          'semantic_stability':'FAIL for tested frozen semantic source',
          'semantic_gain':'Not supported; final class maps empty and C equals pooled calibration',
          'dynamic_independent_gain':'Not evaluated; D not trained',
          'same_information_regression_advantage':'No efficacy result; full CPU D cost higher',
          'gpu_used_seconds':0,'new_neural_training':False,'fresh_confirmation_accessed':False,
          'recovery':{'calibration':'complete, frozen JSON','features':'48 cached cases, no need to recompute',
                      'C_D_candidates':'48 frozen files with SHA256, no truth/clear input',
                      'neural_resume':'No experiment training started; no best/latest created'},
          'scope':'CPU development feasibility; no dehazing/detection/resource changes/commit/push',
          'source_manifest_sha256':digest(out/'final_source_manifest.json')})
    transient={'final_artifacts_sha256.json','cloud_delivery.json','delivery_manifest.json','finalize.log','finalize_final.log'}
    files=[p for p in out.rglob('*') if p.is_file() and p.suffix not in ('.zip','.tgz')]
    write(out/'final_artifacts_sha256.json',{p.relative_to(out).as_posix():digest(p) for p in files
          if p.name not in transient})
    delivery=out/'cpu_delivery.tgz'
    with tarfile.open(delivery,'w:gz') as archive:
        for p in sources:archive.add(ROOT/p,arcname=p.as_posix())
        for path in sorted(out.rglob('*')):
            if not path.is_file() or path==delivery or path.name=='prechange_source_backup.zip':continue
            rel=path.relative_to(out)
            if any(part in ('diagnostic_cache','probe_cache') for part in rel.parts):continue
            if path.name.startswith('cpu_delivery') or path.name in transient-{'final_artifacts_sha256.json'}:continue
            archive.add(path,arcname='evidence/'+rel.as_posix())
    write(out/'delivery_manifest.json',{'archive':delivery.name,'bytes':delivery.stat().st_size,'sha256':digest(delivery),
          'included':'all new sources/config/report, evidence, frozen semantic model and shared C/D candidate cache',
          'kept_local_only':'prechange source backup, diagnostic/probe caches; also retained on local disk',
          'cloud_policy':'extract only into new cpu_delivery directory; never replace repository or historical results'})
    print(json.dumps(read(out/'delivery_manifest.json'),indent=2))


if __name__=='__main__':main()
