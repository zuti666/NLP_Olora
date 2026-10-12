#!/usr/bin/env python3
"""Strict CPU-only report for all six new runs plus fifteen reusable cells."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent
OLD = FORMAL / 'six_row_imageneta_20260921'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension-root', type=Path, default=HERE)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    extension = args.extension_root.resolve()
    output = args.output_dir.resolve() if args.output_dir else extension / 'final_report'
    if extension != HERE:
        parser.error('This report is pinned to its adjacent extension training directory.')
    if output == OLD or output.is_relative_to(OLD) or output == FORMAL / 'final_report':
        parser.error('Use the new extension report directory; original snapshots must be preserved.')
    expected = [extension / 'training' / f'{arm}_seed{seed}' / 'COMPLETE.json'
                for arm in ('shaping_only', 'task2_shaping') for seed in range(3)]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError('All six formal completion receipts are required; missing: ' + ', '.join(missing))
    # Smoke outputs under extension/smoke are deliberately not searched.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    command = [sys.executable, str(OLD / 'build_table.py'), '--extension-root', str(extension),
               '--output-dir', str(output), '--require-extension-complete']
    subprocess.run(command, env=env, check=True)
    data = json.loads((output / 'table_data.json').read_text())
    ledger = json.loads((output / 'VALIDITY_LEDGER.json').read_text())
    assert data['validated_main_cells'] == data['expected_main_cells'] == 21
    assert data['missing_main_cells'] == ledger['missing_main_cells'] == 0
    assert ledger['invalid_observed_cells'] == 0 and len(data['extension_validation']) == 6
    assert all(r['status'] == 'complete' for r in data['individual'])
    assert [r['arm'] for r in data['aggregates']] == [
        'vanilla', 'shaping_only', 'ewclora', 'fs_ewclora',
        'task1_shaping_no_ewc', 'task1_shaping', 'task2_shaping']
    payload = {'status': 'complete', 'scope': 'Seven-row ImageNet-A table, three seeds, CPU report',
               'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'validated_main_cells': 21, 'expected_main_cells': 21, 'missing_main_cells': 0,
               'validated_extension_runs': 6, 'gpu_evaluation_performed': False,
               'reporter_sha256': sha256(__file__), 'table_builder_sha256': sha256(OLD / 'build_table.py'),
               'outputs_sha256': {name: sha256(output / name) for name in [
                   'TABLE_IMAGENETA.md', 'table_imageneta.tex', 'table_data.json',
                   'individual.csv', 'paired_contrasts.csv', 'VALIDITY_LEDGER.json']}}
    temporary = output / 'COMPLETE.json.tmp'
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(output / 'COMPLETE.json')
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    main()
