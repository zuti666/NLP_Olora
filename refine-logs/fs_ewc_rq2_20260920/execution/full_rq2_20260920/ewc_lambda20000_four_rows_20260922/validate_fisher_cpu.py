#!/usr/bin/env python3
"""Replay all six real parent Fisher input streams on CPU (never starts CUDA)."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE / 'cpu_validation'


def one_parent(pair):
    method, seed = pair
    name = f'fisher_{method}_seed{seed}'
    command = [sys.executable, str(HERE / 'prepare_fisher.py'), '--anchor-method',
               method, '--seed', str(seed), '--validate-only', '--result-json',
               str(OUT / f'{name}.json')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with (OUT / f'{name}.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=600)
    if result.returncode:
        raise RuntimeError(f'{name} failed: {OUT / (name + ".log")}')
    receipt = json.loads((OUT / f'{name}.json').read_text())
    assert receipt['status'] == 'cpu_validated'
    assert receipt['cuda_initialized'] is False
    assert receipt['prepared_payload_written'] is False
    assert receipt['exact_inputs_receipt']['matched'] is True
    print(f'{name}: real CPU replay passed', flush=True)
    return receipt


def main():
    OUT.mkdir(exist_ok=True)
    pairs = [(method, seed) for seed in range(3) for method in ('sgd', 'gam')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(one_parent, pairs))
    for seed in range(3):
        left, right = [r for r in receipts if r['seed'] == seed]
        assert left['exact_inputs_receipt'] == right['exact_inputs_receipt']
    assert len({r['exact_inputs_receipt']['generator_seed'] for r in receipts}) == 3
    assert len({r['exact_inputs_receipt']['indices_sha256'] for r in receipts}) == 3
    summary = {'status': 'passed', 'parents': len(receipts), 'cuda_initialized': False,
               'same_seed_sgd_gam_inputs_identical': True,
               'distinct_seed_generators_and_sample_orders': True,
               'receipts': [str(OUT / f'fisher_{m}_seed{s}.json') for m, s in pairs]}
    (OUT / 'FISHER_CPU_REPLAY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
