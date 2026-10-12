"""Collect accepted formal results; keep partial progress separate from metrics.

The old cl10 history-fusion result is source-attributed and hash-verified. It
does not assert that a new full CUDA trajectory has been rerun or matched.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from common import HERE, atomic_json, atomic_text, now, read_json, sha256

PLAN = Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-plan-20261009T090226Z')
OLD = Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-history-response-execution-20261008T175403Z')
OLD_ID = 'formal_history_fused_k10_s2'
OLD_OUTPUT = OLD / 'outputs' / OLD_ID
REUSE_PIN = HERE / 'gates/REUSED_RESULT_PIN.json'
METHOD_LABELS = {
    'sgd': 'SGD', 'current_g2': 'Current-response relocated gradient',
    'history_fusion': 'History-response direct fusion',
    'history_g2': 'History-response relocated gradient (final-gradient cache)',
}
CSV_FIELDS = ('method', 'layout', 'seed', 'status', 'new_or_reused',
              'final_test_accuracy', 'FAA', 'AAA', 'BWT', 'LA', 'forgetting',
              'completed_steps', 'expected_steps', 'gpu', 'run', 'source',
              'output_dir', 'completed_at_utc', 'receipt_sha256', 'error')


def independent_metrics(matrix):
    n = len(matrix)
    if n != 10 or any(len(row) != i + 1 for i, row in enumerate(matrix)):
        raise AssertionError('Reused result is not a complete 10-task matrix')
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 100
           for row in matrix for v in row):
        raise AssertionError('Invalid reused accuracy matrix')
    return dict(FAA=math.fsum(matrix[-1]) / n,
                AAA=math.fsum(math.fsum(row) / len(row) for row in matrix) / n,
                BWT=math.fsum(matrix[-1][i] - matrix[i][i] for i in range(n - 1)) / (n - 1),
                LA=math.fsum(matrix[i][i] for i in range(n)) / n,
                forgetting=math.fsum(max(matrix[k][i] for k in range(i, n)) - matrix[-1][i]
                                     for i in range(n - 1)) / (n - 1))


def freeze_reuse(python):
    """One-time CPU-only audit; refuses to replace a previously frozen pin."""
    if REUSE_PIN.exists():
        raise FileExistsError('Reuse source pin is immutable: ' + str(REUSE_PIN))
    suite = read_json(OLD / 'suite.json')
    old_jobs = {j['id']: j for j in suite['jobs']}
    job = old_jobs[OLD_ID]
    state = read_json(OLD / 'runtime/queue_state.json')
    receipt = read_json(OLD / 'runtime/jobs' / OLD_ID / 'completed.json')
    if state['status'] != 'completed' or state['jobs'][OLD_ID]['status'] != 'completed':
        raise AssertionError('Old formal result was not accepted as completed')
    if receipt['job_id'] != OLD_ID or receipt['exit']['returncode'] != 0:
        raise AssertionError('Old exit/acceptance receipt failed')
    files = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise AssertionError('Existing source hash mismatch: ' + str(path))
        files[str(path)] = actual

    old_pins = read_json(OLD / 'RUN_PINS.json')['files']
    for relative, digest in old_pins.items():
        pin(OLD / relative, digest)
    source = read_json(OLD / 'source_manifest.json')['files']
    for relative, digest in source.items():
        pin(Path(suite['source_root']) / relative, digest)
    for path in (OLD / 'RUN_PINS.json', OLD / 'runtime/queue_state.json',
                 OLD / 'runtime/jobs' / OLD_ID / 'completed.json',
                 OLD / 'runtime/jobs' / OLD_ID / 'exit.json',
                 OLD / 'runtime/jobs' / OLD_ID / 'cuda_witness.json',
                 PLAN / 'gates/PRIOR_RESULTS_REUSE_AUDIT.json',
                 PLAN / 'gates/INDEPENDENT_FOUR_ORACLE.json',
                 PLAN / 'gates/INDEPENDENT_PROTOCOL_CHECK.json',
                 PLAN / 'trainer/response_step.py', PLAN / 'trainer/response_train.py',
                 PLAN / 'trainer/layout_lifecycle.py'):
        pin(path)
    audit = read_json(PLAN / 'gates/PRIOR_RESULTS_REUSE_AUDIT.json')
    oracle = read_json(PLAN / 'gates/INDEPENDENT_FOUR_ORACLE.json')
    protocol = read_json(PLAN / 'gates/INDEPENDENT_PROTOCOL_CHECK.json')
    if audit['status'] != 'passed' or not audit['matching_existing_hrf_result_reuse_verified']:
        raise AssertionError('Existing reuse audit failed')
    if (oracle['status'] != 'passed'
            or not oracle['checks']['old_hrf_equivalence_24steps_deterministic_and_dropout']
            or oracle['new_core_sha256'] != sha256(HERE / 'trainer/response_step.py')):
        raise AssertionError('CPU algorithm equivalence proof does not match new core')
    if protocol['status'] != 'passed' or not all(protocol[k] for k in (
            'cl10_prior_partition_metadata_and_order_exact',
            'cl10_seen_evaluation_groups_10stages_exact',
            'cl10_old_task_loss_new_explicit_ce_10stages_bitwise_equal')):
        raise AssertionError('Native data/loss/evaluation equivalence proof failed')
    for name in ('response_step.py', 'response_train.py', 'layout_lifecycle.py'):
        if sha256(HERE / 'trainer' / name) != sha256(PLAN / 'trainer' / name):
            raise AssertionError('New algorithm/trainer/layout differs from accepted implementation')
        pin(HERE / 'trainer' / name)

    # The original validator executes in its own cwd/module namespace, on CPU.
    code = ('import json; from pathlib import Path; from validate_job import validate_job, HERE; '
            'suite=json.loads((HERE/"suite.json").read_text()); '
            'job=next(j for j in suite["jobs"] if j["id"]=="' + OLD_ID + '"); '
            'print(json.dumps(validate_job(job)))')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(OLD))
    checked = json.loads(subprocess.check_output([python, '-B', '-c', code], cwd=OLD,
                                                env=env, text=True, timeout=180))
    for key, value in checked.items():
        if receipt.get(key) != value:
            raise AssertionError('Original acceptance receipt differs: ' + key)
    summary = read_json(OLD_OUTPUT / 'final_summary.json')
    if (summary['smoke'] or summary['seed'] != 2 or summary['completed_tasks'] != 10
            or summary['completed_steps'] != 2550 or summary['completed_epochs'] != 500):
        raise AssertionError('Old HRF does not have the full requested budget')
    for name in ('final_summary.json', 'step_trace.jsonl', 'task_matrix.json',
                 'effective_config.json', 'task_checkpoints.json', 'tasks.json',
                 'task_identities.json', 'after_task_checks.json', 'metrics.json'):
        pin(OLD_OUTPUT / name)
    pin(OLD_OUTPUT / 'final_summary.json', receipt['summary_sha256'])
    pin(OLD_OUTPUT / 'step_trace.jsonl', receipt['trace_sha256'])
    pin(OLD_OUTPUT / 'task_matrix.json', receipt['matrix_sha256'])
    native_dirs = set()
    for task, dirname in enumerate(summary['checkpoint_dirs']):
        d = Path(dirname)
        manifest = read_json(d / 'manifest.json')
        if manifest['task'] != task:
            raise AssertionError('Old task checkpoint mislabelled')
        pin(d / 'manifest.json')
        for name, digest in manifest['files_sha256'].items():
            pin(d / name, digest)
        native_dirs.add(Path(manifest['native_checkpoint_dir']))
    native_files = 0
    for d in native_dirs:
        for path in sorted(d.iterdir()):
            if path.is_file():
                pin(path)
                native_files += 1
    metrics = independent_metrics(read_json(OLD_OUTPUT / 'task_matrix.json')['matrix'])
    aliases = dict(FAA='FAA_task_macro', AAA='AAA_task_macro', BWT='BWT', LA='LA', forgetting='forgetting')
    for k, v in metrics.items():
        if not math.isclose(v, summary['final_metrics'][aliases[k]], abs_tol=1e-9, rel_tol=0):
            raise AssertionError('Reused metric recomputation differs: ' + k)
    report = dict(status='passed', audited_at_utc=now(), cpu_only=True, gpu_used=False,
                  old_campaign=str(OLD), old_run=OLD_ID, source_output=str(OLD_OUTPUT),
                  original_validator_acceptance=checked, metrics=metrics, files=files,
                  source_files_verified=len(source), immutable_old_pins_verified=len(old_pins),
                  audit_task_checkpoints_verified=len(summary['checkpoint_dirs']),
                  native_task_exports_verified=10, native_export_files_verified=native_files,
                  completed_at_utc=receipt['completed_at'],
                  reuse_basis='Existing accepted seed2 cl10 result plus CPU algorithm and native protocol equivalence',
                  limitations=['No new full CUDA training trajectory is claimed for reused HRF.',
                               'The prior 20-task-checkpoint audit covered two old 10-task runs; this pin revalidates the 10 HRF checkpoints.',
                               'The dynamic radius is eta*global_norm(g0), not a fixed GAM radius.',
                               'No multiseed statistics or cross-layout aggregation.'])
    atomic_json(REUSE_PIN, report)
    atomic_text(HERE / 'gates/REUSED_RESULT_AUDIT.md',
                '# Reused result provenance\n\n'
                'The existing seed2, 10-task history-response fusion result was verified on CPU. '
                f'The audit checked {len(source)} source files, {len(old_pins)} original execution pins, '
                'the original completion/exit receipts, 2,550 update records, the accuracy matrix, '
                '10 serialized task checkpoints and 10 native task exports. The original validator '
                'checked the cached gradient/displacement tensors.\n\n'
                f'FAA = {metrics["FAA"]:.3f}%, AAA = {metrics["AAA"]:.3f}%, BWT = {metrics["BWT"]:.3f} pp.\n\n'
                'Reuse is supported by the prior CPU bitwise algorithm and native data/loss/evaluation '
                'equivalence checks. It does not claim a new full CUDA trajectory. Every collection '
                'rechecks all frozen source/output/checkpoint hashes.\n')
    return report


def verify_reused_result():
    pin = read_json(REUSE_PIN)
    if pin['status'] != 'passed' or pin['old_run'] != OLD_ID:
        raise AssertionError('Invalid reused result source pin')
    for path, digest in pin['files'].items():
        if sha256(path) != digest:
            raise AssertionError('Frozen reused result changed: ' + path)
    summary = read_json(OLD_OUTPUT / 'final_summary.json')
    metrics = independent_metrics(read_json(OLD_OUTPUT / 'task_matrix.json')['matrix'])
    for k, v in metrics.items():
        if not math.isclose(v, pin['metrics'][k], abs_tol=1e-9, rel_tol=0):
            raise AssertionError('Frozen reused metrics changed: ' + k)
    return pin, summary, metrics


def external_seed2_ready():
    verify_reused_result()
    return True


def partial_progress(output):
    """Never place partial epoch/test accuracy into final-result columns."""
    path = output / 'step_trace.jsonl'
    progress = dict(completed_step_records=0, last_observed_task=None, last_observed_epoch=None)
    if path.is_file():
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 65536))
            raw = f.read()
        # A training writer can be between write and newline; ignore the trailing fragment.
        lines = raw.split(b'\n')[:-1]
        for line in reversed(lines):
            try:
                last = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            progress.update(completed_step_records=last['step_index'] + 1,
                            last_observed_task=last['task'], last_observed_epoch=last['epoch'])
            break
    return progress


def new_row(job, record):
    out = Path(job['output_dir'])
    row = dict(method=job['method'], layout=job['layout'], seed=2, status=record['status'],
               new_or_reused='new_formal', final_test_accuracy=None, FAA=None, AAA=None,
               BWT=None, LA=None, forgetting=None, expected_steps=job['expected_steps'],
               completed_steps=0, gpu=record.get('gpu'), run=job['id'],
               source=read_json(HERE / 'suite.json')['source_root'], output_dir=str(out),
               completed_at_utc=record.get('completed_at'), receipt_sha256=None, error=record.get('error'))
    row['progress'] = partial_progress(out)
    row['completed_steps'] = row['progress']['completed_step_records']
    if record['status'] != 'completed':
        return row
    receipt_path = HERE / 'runtime/jobs' / job['id'] / 'completed.json'
    receipt = read_json(receipt_path)
    if receipt['job_id'] != job['id'] or receipt['exit']['returncode'] != 0:
        raise AssertionError('Missing successful acceptance/exit receipt: ' + job['id'])
    if sha256(job['config']) != job['config_sha256']:
        raise AssertionError('Completed input config changed: ' + job['id'])
    from validate_job import validate_job
    checked = validate_job(job)
    if not checked['full_formal_training'] or not checked['checkpoint_tensor_cache_verified']:
        raise AssertionError('Formal training/checkpoint cache not accepted: ' + job['id'])
    for key, value in checked.items():
        if receipt.get(key) != value:
            raise AssertionError('Acceptance receipt changed: ' + job['id'] + ':' + key)
    if record['summary_sha256'] != checked['summary_sha256']:
        raise AssertionError('Completed queue summary differs from acceptance')
    summary = read_json(out / 'final_summary.json')
    m = summary['final_metrics']
    row.update(completed_steps=checked['updates'], receipt_sha256=sha256(receipt_path))
    if job['layout'] == 'single200':
        row['final_test_accuracy'] = m['final_test_accuracy']
    else:
        row.update(FAA=m['FAA_task_macro'], AAA=m['AAA_task_macro'], BWT=m['BWT'],
                   LA=m['LA'], forgetting=m['forgetting'])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze-reuse', action='store_true')
    ap.add_argument('--python', default='/home-local/liying/.conda/envs/Pilot_new_local/bin/python')
    a = ap.parse_args()
    if a.freeze_reuse:
        pin = freeze_reuse(a.python)
        print(json.dumps(dict(status='passed', pinned_files=len(pin['files']), metrics=pin['metrics'])))
        return
    suite = read_json(HERE / 'suite.json')
    state_path = HERE / 'runtime/queue_state.json'
    state = read_json(state_path) if state_path.is_file() else dict(status='registered_not_started', jobs={})
    rows, errors = [], []
    for job in suite['jobs']:
        if job['kind'] != 'training' or job['smoke']:
            continue
        record = state['jobs'].get(job['id'], dict(status='pending'))
        try:
            rows.append(new_row(job, record))
        except Exception as exc:
            row = new_row(job, dict(status='integrity_failed', error=repr(exc), gpu=record.get('gpu')))
            rows.append(row)
            errors.append(dict(run=job['id'], error=repr(exc)))
    try:
        pin, summary, metrics = verify_reused_result()
        rows.append(dict(method='history_fusion', layout='cl10', seed=2, status='completed',
                         new_or_reused='reused_existing', final_test_accuracy=None,
                         **metrics, completed_steps=2550, expected_steps=2550, gpu=None,
                         run=OLD_ID, source=summary['source'], output_dir=str(OLD_OUTPUT),
                         completed_at_utc=pin['completed_at_utc'],
                         receipt_sha256=sha256(OLD / 'runtime/jobs' / OLD_ID / 'completed.json'), error=None))
    except Exception as exc:
        errors.append(dict(run=OLD_ID, error=repr(exc)))
        rows.append(dict(method='history_fusion', layout='cl10', seed=2, status='integrity_failed',
                         new_or_reused='reused_existing', run=OLD_ID, error=repr(exc)))
    report = dict(updated_at_utc=now(), queue_status=state['status'], seed=2,
                  registered_new_formal_training=7, reused_existing_formal_training=1,
                  accepted_new_formal_training=sum(r['status'] == 'completed' and r['new_or_reused'] == 'new_formal' for r in rows),
                  result_units='accuracy %, BWT/forgetting percentage points',
                  single_task_selection='fixed final epoch 50, not best test epoch',
                  aggregation='No averaging across single200 and cl10; single seed only',
                  rows=rows, integrity_errors=errors)
    atomic_json(HERE / 'RESULTS.json', report)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(HERE / 'RESULTS.csv', output.getvalue())
    lines = ['# ImageNet-A response-method validation', '',
             f'Updated (UTC): {report["updated_at_utc"]}. Queue: {state["status"]}.', '',
             'Seven new formal runs are registered. The cl10 history-fusion row reuses the '
             'verified existing result; it is not a new run. All rows use seed 2 and rank 10. '
             'The predictor/relocation radius is dynamic, eta × global gradient norm; '
             'the old fixed GAM radii are inactive. EWC/Fisher, weight fusion and projection are disabled.', '',
             '## All 200 classes in one task', '',
             'Final test accuracy is measured at epoch 50. BWT is undefined; per-epoch accuracy '
             'and partially completed training are not final results.', '',
             '| Method | Status | Final test accuracy (%) | Updates | GPU | Run |',
             '|---|---|---:|---:|---:|---|']
    def fmt(value):
        return '—' if value is None else f'{value:.3f}'
    for r in rows:
        if r['layout'] == 'single200':
            lines.append(f'| {METHOD_LABELS[r["method"]]} | {r["status"]} | {fmt(r.get("final_test_accuracy"))} | '
                         f'{r.get("completed_steps", 0)}/{r.get("expected_steps", 2350)} | '
                         f'{r.get("gpu") if r.get("gpu") is not None else "—"} | {r["run"]} |')
    lines += ['', '## Ten tasks of 20 classes', '',
              'FAA and AAA use equal task weights. BWT uses final accuracy minus the accuracy '
              'immediately after learning each earlier task. Partial runs have empty metric columns.', '',
              '| Method | Status | FAA (%) | AAA (%) | BWT (pp) | Origin | Updates | Run |',
              '|---|---|---:|---:|---:|---|---:|---|']
    for r in rows:
        if r['layout'] == 'cl10':
            lines.append(f'| {METHOD_LABELS[r["method"]]} | {r["status"]} | {fmt(r.get("FAA"))} | '
                         f'{fmt(r.get("AAA"))} | {fmt(r.get("BWT"))} | {r["new_or_reused"]} | '
                         f'{r.get("completed_steps", 0)}/{r.get("expected_steps", 2550)} | {r["run"]} |')
    lines += ['', 'The two task layouts have different task/head and learning-rate schedules; '
              'do not interpret their difference as a controlled algorithm effect. '
              'Results are single-seed measurements. No performance benefit is claimed for pending runs.', '',
              'Provenance: [reused result pin](gates/REUSED_RESULT_PIN.json); '
              '[reuse audit](gates/REUSED_RESULT_AUDIT.md); '
              '[queue state](runtime/queue_state.json).']
    if errors:
        lines += ['', '**Integrity verification failed. The affected final metrics are withheld.**', '']
        lines += [f'- `{e["run"]}`: {e["error"]}' for e in errors]
    atomic_text(HERE / 'RESULTS.md', '\n'.join(lines) + '\n')
    if errors:
        raise RuntimeError('Result integrity checks failed: ' + repr(errors))
    print(json.dumps(dict(status='passed', accepted_new=report['accepted_new_formal_training'], reused=1)))


if __name__ == '__main__':
    main()
