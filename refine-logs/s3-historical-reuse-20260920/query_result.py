"""Find recorded configurations from a final accuracy; no training or GPU use."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--value', type=float, required=True)
    p.add_argument('--metric', choices=['faa_macro', 'aaa_macro', 'sample_final'], default='faa_macro')
    p.add_argument('--tolerance', type=float, default=.005)
    p.add_argument('--dataset', default='imageneta')
    p.add_argument('--seed', type=int)
    p.add_argument('--log-contains', default='')
    p.add_argument('--include-incomplete', action='store_true')
    args = p.parse_args()
    if args.tolerance < 0:
        p.error('tolerance must be nonnegative')
    rows = json.loads((HERE / 'reverse_index.json').read_text())['runs']
    found = []
    for r in rows:
        if r.get('dataset') != args.dataset or args.log_contains not in r['log_path']:
            continue
        if args.seed is not None and int(r['seed']) != args.seed:
            continue
        if not args.include_incomplete and r['status'] not in ['complete_verified', 'complete_log_only']:
            continue
        value = r.get(args.metric)
        if value is None or abs(value - args.value) > args.tolerance:
            continue
        found.append({k: r.get(k) for k in ['id', 'log_path', 'log_sha256', 'status', 'seed',
                     'tasks_completed', 'epochs', 'faa_macro', 'aaa_macro', 'sample_final',
                     'metrics_path', 'metrics_sha256', 'config_path', 'config_path_exists',
                     'runtime_branch_signals', 'config']})
    print(json.dumps({'metric': args.metric, 'value': args.value, 'tolerance': args.tolerance,
                      'matches': len(found), 'runs': found,
                      'scope': 'Matching a number retrieves evidence, not a claim that configurations are comparable. FAA is task macro; sample_final is different.'},
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
