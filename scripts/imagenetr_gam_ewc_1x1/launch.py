#!/usr/bin/env python3
"""Print the paired 1x1 experiment by default; --execute starts foreground jobs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--gpus', help='Two distinct physical GPU indices, EWC then GAM-EWC.')
    p.add_argument('--seeds', default='3,4,5')
    p.add_argument('--phase', choices=['task0', 'task1', 'both', 'evaluate', 'all'], default='all')
    p.add_argument('--config', type=Path, default=HERE / 'config.json')
    p.add_argument('--source', type=Path, default=Path('/data/140-1/users/liying/CopyFlatCV'))
    p.add_argument('--output-root', type=Path,
                   default=HERE.parents[1] / 'refine-logs/imagenetr_gam_ewc_1x1/runs')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs', type=int)
    p.add_argument('--fisher-samples', type=int, default=256)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        p.error('Seeds must be distinct nonnegative integers.')
    gpus = args.gpus.split(',') if args.gpus else ['GPU_EWC', 'GPU_GAM']
    if args.gpus and (len(gpus) != 2 or len(set(gpus)) != 2 or not all(g.isdigit() for g in gpus)):
        p.error('--gpus requires two distinct physical indices.')
    if args.execute and not args.gpus:
        p.error('--execute requires --gpus; no GPU is implicitly selected.')
    config = json.loads(args.config.read_text())
    jobs = []
    for seed in seeds:
        for method, gpu in zip(('ewc', 'gam_ewc'), gpus):
            run_dir = args.output_root.resolve() / method / f'seed{seed}'
            commands = []
            if args.phase != 'evaluate':
                stage = args.phase if args.phase in ('task0', 'task1') else 'both'
                command = [sys.executable, str(HERE / 'train.py'), '--config', str(args.config.resolve()),
                           '--source', str(args.source.resolve()), '--output-root', str(args.output_root.resolve()),
                           '--method', method, '--stage', stage, '--seed', str(seed),
                           '--gpu', gpu, '--workers', str(args.workers)]
                if args.epochs is not None:
                    command += ['--epochs', str(args.epochs)]
                commands.append(command)
            if args.phase in ('evaluate', 'all'):
                commands.append([sys.executable, str(HERE / 'evaluate.py'), '--run-dir', str(run_dir),
                                 '--source', str(args.source.resolve()), '--data-root', config['data_root'],
                                 '--device', 'cuda', '--fisher-samples', str(args.fisher_samples),
                                 '--workers', str(args.workers)])
            jobs.append({'seed': seed, 'method': method, 'gpu': gpu, 'commands': commands})
    for job in jobs:
        print(f"# {job['method']} seed{job['seed']}; GPU {job['gpu']}")
        for cmd in job['commands']:
            print(shlex.join(cmd))
    if not args.execute:
        print('DRY RUN: no training, evaluation, or GPU reservation was started.')
        return
    # No waiting queue and no process preemption. Check once, fail if occupied.
    gpu_uuids = {}
    for gpu in gpus:
        value = subprocess.check_output(['nvidia-smi', '-i', gpu,
            '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True).strip().split(',')
        if len(value) != 2 or float(value[1]) > 500:
            raise RuntimeError(f'GPU {gpu} is not idle: {value}')
        gpu_uuids[gpu] = value[0].strip()
    from concurrent.futures import ThreadPoolExecutor
    log_root = args.output_root.resolve() / 'launch_logs'
    log_root.mkdir(parents=True, exist_ok=True)

    def run_job(job):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuids[job['gpu']],
                   PYTHONHASHSEED=str(job['seed']))
        for idx, command in enumerate(job['commands']):
            log = log_root / f"{args.phase}_{job['method']}_seed{job['seed']}_{idx}.log"
            print(f'Starting {job["method"]} seed{job["seed"]}: {log}', flush=True)
            with log.open('x') as handle:
                subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    for seed in seeds:
        pair = [j for j in jobs if j['seed'] == seed]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_job, job) for job in pair]
            for future in futures:
                future.result()
    if args.phase in ('evaluate', 'all'):
        subprocess.run([sys.executable, str(HERE / 'summarize.py'), '--output-root',
                        str(args.output_root.resolve()), '--seeds', args.seeds], check=True)


if __name__ == '__main__':
    main()
