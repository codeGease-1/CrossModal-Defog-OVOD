"""Offline publication integrity and MAE recomputation; no GPU or datasets."""
from pathlib import Path
import hashlib,json,csv,statistics
ROOT=Path(__file__).resolve().parents[2]
A=ROOT/'experiments/optimization_delivery_20261103'
def main():
    manifest=json.loads((A/'manifest.json').read_text(encoding='utf-8'))
    for name,a in manifest['files'].items():
        p=ROOT/name;assert p.is_file() and p.stat().st_size==a['bytes'],name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256'],name
    s=json.loads((A/'airlight_pilot/summary.json').read_text());n=0
    for cond in ['nominal','unseen_A','display_rgb']:
        with (A/'airlight_pilot/evaluation'/cond/'per_case.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        for mode in ['joint_dynamic','rgb16','direct_same16']:
            for stage in ['before','after']:
                vals=[float(a['pixel_mae']) for a in rows if a['mode']==mode and a['stage']==stage]
                assert len(vals)==1536
                assert abs(statistics.mean(vals)-s[cond][mode+'_'+stage]['pixel_mae'])<1e-12;n+=1
    summary=json.loads((A/'approximate_quality/summary.json').read_text())
    with (A/'approximate_quality/per_case.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    count=0
    for mode in ['joint_dynamic','rgb16','direct_same16']:
        for cond in ['nominal','unseen_A','display_rgb']:
            for name in ['exact','approximate']:
                vals=[float(a['pixel_mae']) for a in rows if a['method']==mode and a['condition']==cond and a['execution']==name]
                assert len(vals)==512
                assert abs(statistics.mean(vals)-summary[mode+'_'+cond][name]['pixel_mae'])<1e-12;count+=1
    print('PASS:',len(manifest['files']),'published hashes;',n,'pilot and',count,'approximation MAE comparisons; no GPU/data/weights.')
if __name__=='__main__':main()
