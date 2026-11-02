"""Standalone scientific figures from fixed evaluation and timing artifacts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((args.evidence/'formal_amp/protocol.json').read_text())['config']
    suffix = '_'.join(map(str, cfg['seeds']))
    labels = ['Original fixed', 'Corrected fixed', 'Direct same inputs', 'RGB32']
    modes = ['original_fixed', 'corrector16', 'direct_same16', 'rgb32']
    colors = ['#8b949e', '#146c94', '#e1a340', '#953553']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    rng = np.random.default_rng(20261102)
    for ax, condition in zip(axes, ['nominal', 'unseen_A', 'display_rgb']):
        with (args.evidence/'formal_amp/confirmation'/condition/f'per_image_{suffix}.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        for i, mode in enumerate(modes):
            grouped = {}
            for r in rows:
                if r['method'] != mode and not r['method'].startswith(mode+'_seed'): continue
                grouped.setdefault(r['group_id'], []).append(float(r['pixel_mae']))
            x = np.array([np.mean(v) for v in grouped.values()])
            ci = np.quantile(x[rng.integers(0, len(x), (10000, len(x)))].mean(1), [.025, .975])
            ax.bar(i, x.mean(), color=colors[i], yerr=[[x.mean()-ci[0]], [ci[1]-x.mean()]], capsize=3)
        ax.set(title=condition, ylabel='Pixel MAE', xticks=range(4), xticklabels=labels)
        ax.tick_params(axis='x', rotation=35); ax.grid(axis='y', alpha=.18)
    fig.suptitle('Fresh 24-scene holdout; fixed 3-seed mean, scene bootstrap 95% CI')
    for ext in ['png', 'svg']: fig.savefig(args.output/('holdout_mae.'+ext), dpi=180)
    plt.close(fig)
    costs = json.loads((args.evidence/'cost_assets/full_inference_cost.json').read_text())
    fig, ax = plt.subplots(figsize=(7, 3.7), layout='constrained')
    for j, size in enumerate([256, 512]):
        cc = {r['mode']: r for r in costs if r['size'] == size}
        ax.bar(np.arange(3)+(j-.5)*.35, [cc[m]['p50_ms'] for m in modes[1:]], width=.35, label=f'{size} x {size}')
    ax.set(xticks=range(3), xticklabels=labels[1:], ylabel='Complete RGB tensor to H median (ms)', title='V100, batch 1, same AMP policy')
    ax.legend(); ax.grid(axis='y', alpha=.18)
    for ext in ['png', 'svg']: fig.savefig(args.output/('complete_cost.'+ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__': main()
