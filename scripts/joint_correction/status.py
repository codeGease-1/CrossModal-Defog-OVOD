"""Watch persistent experiment state; Ctrl+C stops this viewer only."""
import argparse,json,time,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=float,default=10.)
    a=p.parse_args();out=ROOT/'experiments/joint_correction_20261102'
    while True:
        state=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {'stage':'not_started'}
        now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+8')
        runs=[]
        for mode in ['joint_fixed','joint_dynamic']:
            for seed in [42,43,44]:
                d=out/f'{mode}_seed{seed}'
                if (d/'history.json').exists():
                    h=json.loads((d/'history.json').read_text());r=h[-1]
                    runs.append(f"{mode}/{seed}: {r['epoch']}/18 devMAE={r['development_mae']:.6f}"+(' DONE' if (d/'complete.json').exists() else ''))
        print('\n'+now+' stage='+state['stage']+' complete='+str(len(list(out.glob('*_seed*/complete.json'))))+'/6',flush=True)
        for line in runs:print(line,flush=True)
        if state['stage']=='failed':
            print('ERROR:',state.get('error'),'Inspect experiments/joint_correction_20261102/run.log',flush=True);return
        if state['stage']=='complete':print('ALL DONE: docs/joint_correction_results_20261102.md\a',flush=True);return
        if not a.watch:return
        time.sleep(max(1.,a.interval))


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Viewer closed; background training continues.')
