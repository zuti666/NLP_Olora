"""Independent, CPU-only read-only audit of the October 8 HRF campaign.

Writes only below this new campaign's gates/results directories. The old
validator is imported in its original module namespace; its HERE therefore
continues to point to the old campaign rather than this new one.
"""
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
OLD = Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-history-response-execution-20261008T175403Z')
NEW = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(OLD))
old_validator = importlib.import_module('validate_job')
assert old_validator.HERE == OLD

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def independent_metrics(matrix):
    n = len(matrix)
    assert n == 10 and all(len(row) == t + 1 for t, row in enumerate(matrix))
    assert all(math.isfinite(v) and 0 <= v <= 100 for row in matrix for v in row)
    stage_means = [math.fsum(row) / len(row) for row in matrix]
    return {
        'AAA': math.fsum(stage_means) / n,
        'FAA': math.fsum(matrix[-1]) / n,
        'BWT': math.fsum(matrix[-1][i] - matrix[i][i] for i in range(n - 1)) / (n - 1),
        'LA': math.fsum(matrix[i][i] for i in range(n)) / n,
        'forgetting': math.fsum(max(matrix[t][i] for t in range(i, n)) - matrix[-1][i] for i in range(n - 1)) / (n - 1),
    }

suite = json.loads((OLD / 'suite.json').read_text())
state = json.loads((OLD / 'runtime/queue_state.json').read_text())
pins = json.loads((OLD / 'RUN_PINS.json').read_text())['files']
source_manifest = json.loads((OLD / 'source_manifest.json').read_text())['files']
pin_failures = [p for p, digest in pins.items() if sha(OLD / p) != digest]
source_root = Path(suite['source_root'])
source_failures = [p for p, digest in source_manifest.items() if sha(source_root / p) != digest]
assert not pin_failures and not source_failures
assert state['status'] == 'completed'
reuse_gates = {}
for gate_name in ('INDEPENDENT_FOUR_ORACLE.json', 'INDEPENDENT_PROTOCOL_CHECK.json'):
    p = NEW / 'gates' / gate_name
    if p.is_file():
        reuse_gates[gate_name] = json.loads(p.read_text())
core_gate = reuse_gates.get('INDEPENDENT_FOUR_ORACLE.json', {})
protocol_gate = reuse_gates.get('INDEPENDENT_PROTOCOL_CHECK.json', {})
reuse_verified = (core_gate.get('status') == 'passed'
                  and core_gate.get('new_core_sha256') == sha(NEW / 'trainer/response_step.py')
                  and core_gate.get('checks', {}).get('old_hrf_equivalence_24steps_deterministic_and_dropout') is True
                  and protocol_gate.get('status') == 'passed'
                  and protocol_gate.get('cl10_prior_partition_metadata_and_order_exact') is True
                  and protocol_gate.get('cl10_seen_evaluation_groups_10stages_exact') is True
                  and protocol_gate.get('cl10_old_task_loss_new_explicit_ce_10stages_bitwise_equal') is True)
rows = []
receipts = {}
for job in suite['jobs']:
    if job['phase'] != 'formal':
        continue
    job_id = job['id']
    assert state['jobs'][job_id]['status'] == 'completed'
    exit_receipt = json.loads((OLD / 'runtime/jobs' / job_id / 'exit.json').read_text())
    assert exit_receipt['returncode'] == 0
    accepted = old_validator.validate_job(job)
    out = Path(job['output_dir'])
    summary = json.loads((out / 'final_summary.json').read_text())
    matrix = json.loads((out / 'task_matrix.json').read_text())['matrix']
    metrics = independent_metrics(matrix)
    keys = {'AAA': 'AAA_task_macro', 'FAA': 'FAA_task_macro', 'BWT': 'BWT', 'LA': 'LA', 'forgetting': 'forgetting'}
    for key, val in metrics.items():
        assert math.isclose(val, summary['final_metrics'][keys[key]], rel_tol=0, abs_tol=1e-9)
    traces = [json.loads(line) for line in (out / 'step_trace.jsonl').read_text().splitlines()]
    history_rows = [r for r in traces if r['history_used']]
    assert len(traces) == 2550
    if job['direction'] == 'history_fused':
        assert len(history_rows) == 2540
        for prev, cur in zip(traces, traces[1:]):
            if cur['history_used']:
                assert cur['task'] == prev['task']
                assert cur['history_cache_raw_gradient_sha256'] == prev['raw_gradient_sha256']
                assert cur['history_cache_actual_displacement_sha256'] == prev['actual_step_displacement_sha256']
        # Evidence that cache selection is not silently the fused update vector.
        assert any(r['raw_gradient_sha256'] != r['U_sha256'] for r in history_rows)
    else:
        assert not history_rows
    row = dict(job=job_id, method='history_fusion' if job['direction'] == 'history_fused' else 'future_mirror_control',
               seed=2, task_protocol='10x20', epochs_per_task=50, total_epochs=500,
               steps=2550, **metrics, history_used_steps=len(history_rows),
               completed_at_utc=state['jobs'][job_id]['completed_at'],
               source_output=str(out), reuse_status=('verified_sourced_existing_cl10_result_not_newly_run' if reuse_verified else 'conditional_on_new_core_equivalence_and_protocol_identity') if job['direction'] == 'history_fused' else 'auxiliary_control_not_one_of_four_requested_methods')
    rows.append(row)
    receipts[job_id] = dict(old_validator_acceptance=accepted, independent_metrics=metrics,
                            exit_receipt=exit_receipt, queue_completed_at=state['jobs'][job_id]['completed_at'],
                            cache_raw_gradient_chain_verified=job['direction'] == 'history_fused')

report = dict(status='passed', audited_at_utc=datetime.now(timezone.utc).isoformat(),
              old_campaign=str(OLD), old_module_here=str(old_validator.HERE),
              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_used=False,
              source_files_verified=len(source_manifest), immutable_pinned_files_verified=len(pins),
              source_failures=source_failures, pin_failures=pin_failures,
              complete_formal_runs=len(rows), receipts=receipts,
              matching_existing_hrf_result_reuse_verified=reuse_verified,
              reuse_gate_files_sha256={name:sha(NEW/'gates'/name) for name in reuse_gates},
              restrictions=['Results are seed2, not multiseed means.',
                            'Dynamic predictor radius eta*global_norm(g0), not fixed GAM r=1.',
                            'HRF is a sourced existing result; no new full CUDA training trajectory is claimed.',
                            'Future Mirror is an auxiliary no-history control, not current_g2/history_g2/SGD.',
                            'No prior 1x200 result is supplied by this 10x20 campaign.'])
(NEW / 'gates/PRIOR_RESULTS_REUSE_AUDIT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
with (NEW / 'results/PRIOR_RESULTS.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({'status':'passed', 'source_files_verified':len(source_manifest), 'pinned_files_verified':len(pins), 'rows':rows}, indent=2))
