#!/usr/bin/env python3
"""Collect verified observations and explicit conditional skips without imputation."""
import csv
import io
from pathlib import Path

from common import atomic_json, atomic_text, now
from conditional_gate import read, sha, require, evaluate, validated_record, verify_skipped, paired_inputs

HERE = Path(__file__).resolve().parent
METRICS = ('AAA_task_macro', 'FAA_task_macro', 'LA', 'BWT', 'forgetting',
           'AAA_sample_weighted', 'FAA_sample_weighted')
FIELDS = ('job_id', 'phase', 'variant', 'rule', 'scale', 'seed', 'radius',
          'lambda_corr', 'first_response', 'second_response', 'attention_backend',
          'completed_tasks', *METRICS, 'max_match_relative_error',
          'metrics_source', 'summary_sha256', 'receipt_sha256')


def collect(root=HERE):
    root = Path(root)
    suite = read(root / 'suite.json')
    rows, diagnostics, skipped, pending, completed_smokes = [], [], [], [], []
    for job in suite['jobs']:
        directory = root / 'runtime/jobs' / job['id']
        skip_path = directory / 'skipped.json'
        checked = validated_record(job, root)
        if checked is None:
            if skip_path.exists():
                receipt = read(skip_path)
                gate_result = verify_skipped(job, receipt, suite, root)
                skipped.append({'job_id': job['id'], 'phase': job['phase'], 'gate': job.get('gate'),
                                'skipped_receipt_sha256': sha(skip_path), 'gate_result': gate_result,
                                'status': 'skipped_by_gate'})
            else:
                pending.append(job['id'])
            continue
        if job['smoke']:
            completed_smokes.append(job['id'])
            continue
        summary = checked['summary']
        if job['kind'] == 'curvature':
            diagnostics.append({'job_id': job['id'], 'task': job['task'], 'rule': job['rule'],
                                'scale': job['scale'], 'radius': job['radius'],
                                'hvp_all_available': summary['hvp_all_available'],
                                'observed_batches': len(summary['batches']), **checked['input']})
            continue
        require(job['kind'] == 'horizon', 'unknown completed job kind')
        metrics = summary['final_metrics']
        rows.append({'job_id': job['id'], **{k: job[k] for k in
                    ('phase', 'variant', 'rule', 'scale', 'seed', 'radius', 'lambda_corr',
                     'first_response', 'second_response', 'attention_backend')},
                     'completed_tasks': summary['completed_tasks'],
                     **{k: metrics[k] for k in METRICS},
                     'max_match_relative_error': summary['max_match_relative_error'],
                     'metrics_source': str(Path(job['output_dir']) / 'task_matrix.json'),
                     'summary_sha256': checked['input']['summary_sha256'],
                     'receipt_sha256': checked['input']['receipt_sha256']})
    gates = {}
    for job in suite['jobs']:
        gate = job.get('gate')
        if gate is not None and gate not in gates:
            gates[gate] = evaluate(job, suite, root)
    completed_ids = {r['job_id'] for r in rows}
    pairing = paired_inputs([j for j in suite['jobs'] if j['id'] in completed_ids]) if len(rows) >= 2 else None
    directory = root / 'results'
    directory.mkdir(exist_ok=True)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    # Always replace even an empty table, so a removed/absent result cannot leave
    # an old CSV looking current. Corrupt present receipts fail above, closed.
    atomic_text(directory / 'horizon_metrics.csv', stream.getvalue())
    atomic_json(directory / 'skipped_jobs.json', {'updated_at': now(), 'jobs': skipped})
    report = {'updated_at': now(), 'suite_sha256': sha(root / 'suite.json'),
              'frozen_gates_sha256': sha(root / 'FROZEN_GATES.json'),
              'registered_jobs': len(suite['jobs']), 'validated_full_horizons': len(rows),
              'validated_diagnostics': diagnostics, 'validated_smokes': completed_smokes,
              'skipped_count': len(skipped), 'missing_completion_ids': pending,
              'terminal_count': len(rows) + len(diagnostics) + len(completed_smokes) + len(skipped),
              'all_jobs_terminal': not pending, 'gate_decisions': gates,
              'completed_horizon_pairing': pairing,
              'absent_results_policy': 'Omit absent/skipped numeric results; never substitute zeros.',
              'interpretation': 'Seed2 exploratory; a conditional skip is not proof of no effect.',
              'additional_seeds_or_phases_auto_launch': False}
    atomic_json(directory / 'progress_summary.json', report)
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(collect(), indent=2, allow_nan=False))
