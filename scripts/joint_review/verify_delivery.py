"""Verify curated bytes and recompute both rounds' three-condition MAE; standard library only."""
from pathlib import Path
import csv,hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[2]
A=ROOT/"experiments/candidate_correction_delivery_20261102"
def main():
    manifest=json.loads((A/"manifest.json").read_text(encoding="utf-8"))
    for name,a in manifest["files"].items():
        p=ROOT/name
        assert p.is_file() and p.stat().st_size==a["bytes"],name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==a["sha256"],name
    count=0
    for round_name in ["matched_fusion","joint_correction"]:
        for condition in ["nominal","unseen_A","display_rgb"]:
            d=A/round_name/"development"/condition
            summary=json.loads((d/"summary.json").read_text(encoding="utf-8"))
            with (d/"per_case.csv").open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
            for method,s in summary.items():
                rr=[r for r in rows if r["method"]==method]
                assert len(rr)==1536,(round_name,condition,method,len(rr))
                assert abs(statistics.mean(float(r["pixel_mae"]) for r in rr)-s["pixel_mae"])<1e-12
                count+=1
    print("PASS:",len(manifest["files"]),"SHA256 checks;",count,"method/condition MAE recomputations; no GPU or weights.")
if __name__=="__main__":main()
