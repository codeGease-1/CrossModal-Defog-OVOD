"""Watch only: Ctrl+C stops this viewer, never the worker."""
import json,time,argparse,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/optimization_20261103'
def main():
    a=argparse.ArgumentParser();a.add_argument('--watch',action='store_true');args=a.parse_args()
    while True:
        now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
        vals={}
        for name in ['latency_run03','airlight_pilot']:
            p=OUT/name/'status.json';vals[name]=json.loads(p.read_text()) if p.exists() else {'stage':'not_started'}
        print(now,json.dumps(vals,ensure_ascii=False),flush=True)
        if not args.watch:return
        if vals['airlight_pilot']['stage']=='complete':return
        if vals['airlight_pilot']['stage']=='failed':raise SystemExit(1)
        time.sleep(15)
if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Viewer stopped; worker continues.')
