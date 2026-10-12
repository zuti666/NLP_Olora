#!/usr/bin/env python3
"""Validate actual P-family updates and pair the smoke conditions within each seed."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from contract import FAMILY, check_job, read_json, select_job, sha256
from scientific_validate import finite_tree, validate_job


def validate(suite, job):
    _, resolved = check_job(suite, job)
    receipt = validate_job(resolved)
    out = Path(job['output_dir'])
    summary = read_json(out / 'final_summary.json')
    effective = read_json(out / 'effective_config.json')
    effective_seed = effective['seed']
    if effective_seed not in (job['seed'], [job['seed']]):
        raise AssertionError('Actual effective configuration seed differs')
    for key, expected in dict(ewc_lambda=0., ewc_skip_task_fisher=True,
                              in_task_weight_merge=False, task_endpoint_weight_merge=False,
                              approximate_fisher_in_task=False, in_task_path_normalize_snapshots=False,
                              lora_rank=10, gam_grad_norm_rho=1., gam_grad_rho=.05,
                              init_epoch=50, epochs=50, memory_size=0, memory_per_class=0).items():
        if effective.get(key) != expected:
            raise AssertionError('Actual effective configuration differs: ' + key)
    identities = read_json(out / 'task_identities.json')
    after = read_json(out / 'after_task_checks.json')
    if len(identities) != job['expected_tasks'] or len(after) != job['expected_tasks']:
        raise AssertionError('Missing task identity/after-task purity receipt')
    for task, (identity, purity) in enumerate(zip(identities, after)):
        if identity['seed'] != job['seed'] or identity['task'] != task:
            raise AssertionError('Actual task identity seed/order differs')
        if identity['class_slice'] != [20 * task, 20 * (task + 1)]:
            raise AssertionError('Actual task identity class range differs')
        if not identity['old_fc_rows_inherited_exact']:
            raise AssertionError('Classifier expansion changed existing rows')
        if purity['task'] != task or not all(purity[k] for k in
                  ('parameters_exact', 'parameter_references_exact', 'fisher_empty')):
            raise AssertionError('After-task work changed parameters/Fisher')
    rows = [json.loads(line) for line in (out / 'step_trace.jsonl').read_text().splitlines()]
    for row in rows:
        if row['identity']['data_seed'] != job['seed'] * 100000 + row['task'] * 1000 + row['epoch']:
            raise AssertionError('Actual batch/augmentation seed differs')
        if row['forward_rng_rule'] != 'R0 replay for every probe; final RNG=R1 from g0':
            raise AssertionError('Forward RNG replay contract changed')
    if summary['scheduler_T_max'] != 50 or summary['smoke'] != resolved['smoke']:
        raise AssertionError('Scheduler horizon/phase mismatch')
    receipt.update(passed=True, family=FAMILY, job_id=job['id'], seed=job['seed'],
                   condition=job['condition'], phase=job['phase'],
                   final_metrics=summary['final_metrics'], output_dir=str(out),
                   task_identities_sha256=sha256(out / 'task_identities.json'),
                   after_task_checks_sha256=sha256(out / 'after_task_checks.json'))
    finite_tree(receipt)
    return receipt


def validate_smokes(suite):
    jobs = [j for j in suite['jobs'] if j['family'] == FAMILY and j['phase'] == 'smoke']
    expected = {(seed, 'P%02d' % i) for seed in (0, 1) for i in range(1, 17)}
    if len(jobs) != 32 or {(j['seed'], j['condition']) for j in jobs} != expected:
        raise AssertionError('All sixteen conditions at both seeds require independent short GPU runs')
    baselines, proofs, maximum_error = {}, {}, 0.
    for job in jobs:
        checked = validate(suite, job)
        output = Path(job['output_dir'])
        summary = read_json(output / 'final_summary.json')
        rows = [json.loads(line) for line in (output / 'step_trace.jsonl').read_text().splitlines()]
        initial = summary['initial_identity']
        identity = dict(batch_identities=[r['identity'] for r in rows],
                        coordinates=[(r['task'], r['epoch'], r['batch']) for r in rows],
                        nominal_lrs=[r['nominal_lr'] for r in rows],
                        task0_initial_parameter_sha256=initial['task_initial_parameter_sha256'],
                        initial_fc_rows=initial['new_fc_rows_sha256'],
                        frozen_origin_sha256=initial['frozen_origin_sha256'])
        previous = baselines.setdefault(job['seed'], identity)
        if previous != identity:
            raise AssertionError('Same-seed smoke initialization/data pairing failed: ' + job['id'])
        for row in rows:
            distance = row['probe_displacements']['phi2']['parameter_norm']
            if row['norms']['q'] > 1e-10 and not math.isclose(distance, 1., rel_tol=1e-5, abs_tol=1e-6):
                raise AssertionError('Actual outer point is not one unit from the base')
            if row['step_mode'] == 'matched':
                maximum_error = max(maximum_error, row['actual_match_relative_error'])
        proofs[job['id']] = checked
    if baselines[0]['task0_initial_parameter_sha256'] == baselines[1]['task0_initial_parameter_sha256']:
        raise AssertionError('Independent seeds unexpectedly have identical initialization')
    return dict(status='passed', passed=True, checked_at=datetime.now(timezone.utc).isoformat(),
                smoke_conditions=32, seeds=[0, 1], actual_updates=64,
                paired_initial_parameters_and_batch_identities_within_each_seed=True,
                seed_initializations_differ=True, actual_outer_radius=1., rho=.05,
                no_g3_no_projection=True, matching_relative_error_max=maximum_error,
                tolerance=1e-3, scope='execution/configuration/pairing; no cross-GPU bitwise trajectory claim',
                proofs=proofs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--job')
    choice.add_argument('--smokes', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    suite = read_json(args.suite)
    report = validate_smokes(suite) if args.smokes else validate(suite, select_job(suite, args.job))
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
