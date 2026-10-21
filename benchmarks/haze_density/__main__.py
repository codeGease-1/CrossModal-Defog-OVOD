"""Run independent experiment stages from the project root."""
import argparse
from pathlib import Path
import sys
import yaml
from .common import seed_runtime, NotAvailable, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/haze_benchmark.yaml')
    commands = parser.add_subparsers(dest='command', required=True)
    discover = commands.add_parser('discover')
    discover.add_argument('--root', required=True)
    discover.add_argument('--output', required=True)
    discover.add_argument('--limit', type=int, default=8)
    generate = commands.add_parser('generate')
    generate.add_argument('--scenes', required=True)
    generate.add_argument('--output', required=True)
    generate.add_argument('--profile', choices=['minimal', 'full'], default='minimal')
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--dataset', required=True)
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--split', choices=['train', 'calibration', 'test'], default='test')
    evaluate.add_argument('--checkpoint')
    report = commands.add_parser('report')
    report.add_argument('--output', required=True, help='existing evaluation directory')
    report.add_argument('--max-figures', type=int, default=12)
    performance = commands.add_parser('performance')
    performance.add_argument('--output', required=True)
    performance.add_argument('--devices', nargs='+')
    performance.add_argument('--checkpoint')
    audit = commands.add_parser('audit-auxiliary')
    audit.add_argument('--clean', required=True)
    audit.add_argument('--hazy', required=True)
    audit.add_argument('--transmission', required=True)
    audit.add_argument('--airlight', required=True)
    audit.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    # Plot-only process must not load PyTorch/OpenMP on Windows Conda builds.
    # It reuses saved arrays and has no randomized calculations.
    if args.command != 'report':
        seed_runtime(config['seed'], config.get('threads', 1))
    if getattr(args, 'checkpoint', None):
        config['checkpoint'] = args.checkpoint
    try:
        if args.command == 'discover':
            from .synthetic import discover
            discover(args.root, args.output, args.limit, config['seed'])
        elif args.command == 'generate':
            from .synthetic import generate
            generate(args.scenes, args.output, config, args.profile)
        elif args.command == 'evaluate':
            from .evaluate import evaluate
            result = evaluate(args.dataset, args.output, config, args.device, args.split)
            if result['status'] != 'PASS':
                print(result['status'], 'see results.json')
                return 1 if result['status'] == 'FAIL' else 2
        elif args.command == 'report':
            from .report import render_report
            print(render_report(args.output, args.max_figures))
        elif args.command == 'performance':
            from .performance import benchmark_performance
            result = benchmark_performance(args.output, config, args.devices)
            if result['status'] != 'PASS':
                print(result['status'], 'see performance.json')
                return 1 if result['status'] == 'FAIL' else 2
        elif args.command == 'audit-auxiliary':
            from .audit import audit_auxiliary
            audit_auxiliary(args.clean, args.hazy, args.transmission, args.airlight, args.output)
    except NotAvailable as error:
        print('NOT RUN:', error)
        return 2
    except (ValueError, OSError, RuntimeError) as error:
        print('FAIL:', error)
        return 1
    print('PASS: command completed; consult method-level NOT RUN / N/A statuses')
    return 0


if __name__ == '__main__':
    sys.exit(main())
