"""Seal complete logs only after producers have exited; no model changes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.v15.run import read,write,digest,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='experiments/v15_20261101');a=p.parse_args();out=Path(a.output)
    frozen=read(out/'frozen_selection.json')
    assert digest(out/'calibration.json')==frozen['calibration_sha256']
    for rel,sha in read(out/'protocol.json')['source_sha256'].items():
        assert digest(ROOT/rel)==sha, rel
    write(out/'completion.json',{'implementation':'PASS','controlled_scalar_attenuation':'PRELIMINARY SUPPORT',
          'original_spectral_pairs':'DOES NOT MEET LOCAL NONINFERIORITY; not a validated replacement',
          'selected_on_development':frozen['selected'],'test_used_for_tuning':False,'new_global_head':False,
          'next_stage_started':False,'fit_budget':read(out/'budget.json'),
          'new_source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'scripts/v15').glob('*')) if p.is_file()}})
    manifest={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*'))
              if p.is_file() and p.name!='artifacts_sha256.json'}
    write(out/'artifacts_sha256.json',manifest)
    print('SEALED',len(manifest),'artifacts; calibration',frozen['calibration_sha256'])


if __name__=='__main__':main()
