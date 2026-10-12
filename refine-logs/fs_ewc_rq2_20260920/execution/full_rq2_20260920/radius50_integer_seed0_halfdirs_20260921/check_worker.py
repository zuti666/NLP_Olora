#!/usr/bin/env python3
"""CPU-only protocol, historical replay, and fault-rejection checks."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import diagnostic_radius50 as worker


def main():
    frozen_files = [worker.BASE / 'diagnostic_full.py', worker.EXECUTION / 'rq2_common.py',
                    worker.EXECUTION / 'diagnostic.py', worker.HELPER_PATH,
                    worker.BASE / 'radius50_extension_20260921/diagnostic_radius50.py',
                    worker.BASE / 'radius50_extension_20260921/check_worker.py']
    frozen_files += sorted((worker.BASE / 'radius50_extension_20260921/configs').glob('*.json'))
    before_hashes = {str(path): worker.sha(path) for path in frozen_files}
    evidence = []
    for seed in [0]:
        for method in ('sgd', 'gam'):
            for job in ('dense', 'sparse_train'):
                command = [sys.executable, str(worker.HERE / 'diagnostic_radius50.py'),
                           '--config', str(worker.HERE / f'configs/{job}_seed{seed}.json'),
                           '--method', method, '--job', job, '--validate-only']
                completed = subprocess.run(command, check=True, text=True, capture_output=True)
                result = json.loads(completed.stdout)
                worker.require(result['status'] == 'passed', 'CPU preflight did not pass')
                worker.require(result['expected_states'] == (1001 if job == 'dense' else 321), 'Incorrect integer-grid expected count')
                evidence.append(result)
    replay = []
    for seed in [0]:
        for method in ('sgd', 'gam'):
            for job in ('dense', 'sparse_train'):
                directory = worker.BASE / f'formal_results/seed{seed}/{method}/{job}'
                reference = {'meta': worker.read(directory / 'manifest.json'),
                             'rows': [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]}
                summary = worker.read(directory / 'summary.json')
                radii = [item['radius'] for item in summary['radius_summary']]
                check = worker.validate_finished(summary, reference['meta'], reference['rows'], reference, job, radii, direction_count=100 if job == 'dense' else 16)
                replay.append(dict(seed=seed, method=method, job=job, status='passed', states=check['measured_states']))
    # Replay the completed coarse extension as well as the formal originals.
    coarse_replay = []
    for seed in [0]:
        for method in ('sgd', 'gam'):
            for job in ('dense', 'sparse_train'):
                directory = worker.BASE / f'radius50_extension_20260921/results/seed{seed}/{method}/{job}'
                formal_directory = worker.BASE / f'formal_results/seed{seed}/{method}/{job}'
                reference = {'meta': worker.read(formal_directory / 'manifest.json'),
                             'rows': [json.loads(line) for line in (formal_directory / 'states.jsonl').read_text().splitlines()]}
                meta = worker.read(directory / 'manifest.json')
                summary = worker.read(directory / 'summary.json')
                rows = [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]
                radii = [item['radius'] for item in summary['radius_summary']]
                check = worker.validate_finished(summary, meta, rows, reference, job, radii, direction_count=100 if job == 'dense' else 16)
                coarse_replay.append(dict(seed=seed, method=method, job=job, status='passed', states=check['measured_states']))
    # Validate the production 50-direction setting on selected REAL raw rows.
    # These derived summaries are temporary in-memory checks, not measurements.
    prefix_replay = []
    for source in ('formal_results', 'radius50_extension_20260921/results'):
        for method in ('sgd', 'gam'):
            directory = worker.BASE / source / f'seed0/{method}/dense'
            formal_directory = worker.BASE / f'formal_results/seed0/{method}/dense'
            reference = {'meta': worker.read(formal_directory / 'manifest.json'),
                         'rows': [json.loads(line) for line in (formal_directory / 'states.jsonl').read_text().splitlines()]}
            meta = worker.read(directory / 'manifest.json')
            summary = worker.read(directory / 'summary.json')
            all_rows = [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]
            rows = [row for row in all_rows if row['state_id'] == 'r0' or row['point']['direction']['index'] < 50]
            by_id = {row['state_id']: row for row in rows}
            origin = by_id['r0']
            summary['direction_indices'] = list(range(50))
            summary['expected_states'] = summary['states_completed'] = len(rows)
            for item in summary['radius_summary']:
                radius = item['radius']
                selected = [origin] if not radius else [by_id[f'r{radius:.12g}_u{i:03d}'] for i in range(50)]
                item['directions'] = len(selected)
                for split in ('train', 'test'):
                    signed = [row['metrics'][split]['loss'] - origin['metrics'][split]['loss'] for row in selected]
                    positive = [max(value, 0.) for value in signed]
                    accuracy = sum(row['metrics'][split]['accuracy'] for row in selected) / len(selected)
                    item[split].update(D_each=signed, D_mean=sum(signed)/len(signed), G=sum(positive)/len(positive),
                        Q95=worker.quantile(positive, .95), sampled_max=max(positive),
                        absolute_loss_mean=sum(row['metrics'][split]['loss'] for row in selected)/len(selected),
                        accuracy_mean=accuracy, accuracy_drop_mean_pp=origin['metrics'][split]['accuracy']-accuracy)
            radii = [item['radius'] for item in summary['radius_summary']]
            for item in summary['tolerance']:
                radius = 0
                for row in summary['radius_summary']:
                    if row[item['split']][item['metric']] > item['epsilon']:
                        break
                    radius = row['radius']
                item['prefix_radius'] = radius
                item['right_censored'] = radius == max(radii)
            check = worker.validate_finished(summary, meta, rows, reference, 'dense', radii)
            prefix_replay.append(dict(source=source, seed=0, method=method, job='dense',
                status='passed', states=check['measured_states'], selected_direction_indices=list(range(50)),
                derived_summary_only=True, stored_as_measurement=False))
    # Use real completed records with single-field faults; no synthetic results
    # are published as measurements and no GPU/model is imported.
    directory = worker.BASE / 'formal_results/seed0/sgd/sparse_train'
    reference = {'meta': worker.read(directory / 'manifest.json'),
                 'rows': [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]}
    original_summary = worker.read(directory / 'summary.json')
    radii = [item['radius'] for item in original_summary['radius_summary']]
    faults = []
    for name in ('nonfinite', 'direction_mismatch', 'normalized_fisher', 'summary_mismatch', 'incomplete_states'):
        rows = copy.deepcopy(reference['rows'])
        summary = copy.deepcopy(original_summary)
        if name == 'nonfinite':
            rows[1]['metrics']['train']['loss'] = float('nan')
        elif name == 'direction_mismatch':
            rows[1]['point']['direction']['sha256'] = '0' * 64
        elif name == 'normalized_fisher':
            rows[1]['fisher_by_scope']['subset']['normalized'] = True
        elif name == 'summary_mismatch':
            summary['radius_summary'][1]['train']['G'] += .1
        elif name == 'incomplete_states':
            rows.pop()
        try:
            worker.validate_finished(summary, reference['meta'], rows, reference, 'sparse_train', radii)
        except ValueError as error:
            faults.append(dict(fault=name, rejected=True, reason=str(error)))
        else:
            raise AssertionError(f'Injected fault was not rejected: {name}')
    config = worker.read(worker.HERE / 'configs/dense_seed0.json')
    config['dense_directions'] = 16
    try:
        worker.validate_configuration(config, worker.read(worker.BASE / 'configs/seed0.json'), 'dense')
    except ValueError as error:
        faults.append(dict(fault='changed_direction_count', rejected=True, reason=str(error)))
    else:
        raise AssertionError('Changed direction count was not rejected')
    after_hashes = {str(path): worker.sha(path) for path in frozen_files}
    worker.require(before_hashes == after_hashes, 'Original frozen sources changed during verification')
    receipt = dict(status='passed', checked_utc=datetime.now(timezone.utc).isoformat(),
                   gpu_used=False, wrapper_sha256=worker.sha(worker.HERE / 'diagnostic_radius50.py'),
                   checks_sha256=worker.sha(__file__), cli_preflight_jobs=evidence,
                   historical_validator_replays=replay, completed_coarse_extension_replays=coarse_replay,
                   real_row_prefix_replays=prefix_replay,
                   fault_rejections=faults, frozen_source_hashes=after_hashes, original_sources_unchanged=True)
    destination = worker.HERE / 'worker_cpu_check.json'
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='passed', cpu_cli_jobs=len(evidence), historical_replays=len(replay), coarse_extension_replays=len(coarse_replay), prefix_replays=len(prefix_replay),
                          faults_rejected=len(faults), receipt=str(destination)), allow_nan=False))


if __name__ == '__main__':
    main()
