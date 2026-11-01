"""Check frozen artifacts and preserve source/environment hashes after producers stop."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'experiments/v2_20261101'
    protocol=json.loads((out/'protocol.json').read_text());frozen=json.loads((out/'frozen.json').read_text())
    for name,value in protocol['source_sha256'].items():assert sha(ROOT/name)==value,name
    assert sha(ROOT/protocol['config']['v15_directory']/'calibration.json')==protocol['v15_calibration_sha256']
    assert sha(out/'calibration.json')==frozen['calibration_sha256']
    for name,value in frozen['checkpoint_sha256'].items():assert sha(out/name)==value,name
    (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    sources=list((ROOT/'scripts/v2').glob('*.py'))+[ROOT/'scripts/v2/reproduce.sh',ROOT/'configs/v2_dynamic.json',ROOT/'src/models/haze_density/dynamic_prior.py']
    result={'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'v15_unchanged':True,
        'v2_calibration_unchanged':True,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'status':'implemented and evaluated; dynamic weighting not recommended for main pipeline',
        'limits':['unverified real geographic groups','procedural training domain','original spectral endpoint is secondary']}
    (out/'completion.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    files=[p for p in out.rglob('*') if p.is_file() and not any('cache' in q for q in p.relative_to(out).parts) and p.name!='artifact_sha256.json']
    (out/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in sorted(files)},indent=2))
    print('Verified frozen sources and calibration;',len(files),'artifacts hashed.')
if __name__=='__main__':main()
