#!/usr/bin/env python3
"""CPU-only read-only audit of the frozen completed G2 run."""
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path('/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2')
OUT = Path(__file__).with_name('INDEPENDENT_INTEGRITY_AUDIT.json')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text())

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def finite_all(value):
    if isinstance(value, dict):
        return all(finite_all(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_all(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)

queue = load('_independent_g2_queue', ROOT/'source/scripts_llama/run_g2_orders_queue.py')
manifest = queue.verify_manifest(ROOT)
runner = load('_independent_g2_runner', ROOT/'source/scripts_llama/run_g2_llama.py')
# Native verify_stage writes its deterministic receipt as its final action.
# Suppress only that write in this audit process to preserve the original run.
runner.write_json = lambda *_args, **_kwargs: None
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()

state_path = ROOT/'runtime/queue_state.json'
state = read(state_path)
assert state['status'] == 'completed'
assert state['formal_completed'] == 24 and state['smoke_completed'] == 12
assert not state.get('failure') and state.get('child_pid') is None
observed = {row['key']: row for row in state['completed_stages']}
assert len(observed) == len(state['completed_stages']) == 36
expected_keys = []
ledger = []
failures = []
for job in manifest['jobs']:
    for prepared in job['plan']['stages']:
        kind = 'smoke' if prepared['smoke'] else 'formal'
        key = f"{job['id']}:s{prepared['seed']}:{kind}:{prepared['stage']}"
        expected_keys.append(key)
        spec_path = ROOT/'specs'/(key.replace(':', '_')+'.json')
        spec = read(spec_path)
        item = {'key': key, 'kind': kind, 'model_size': spec['model_size'],
                'order': spec['order'], 'task': spec['task'], 'stage': spec['stage'],
                'seed': spec['seed'], 'expected_updates': spec['expected_updates'],
                'expected_saved_rank': spec['expected_saved_rank'], 'spec_sha256': sha(spec_path),
                'output_dir': spec['output_dir']}
        try:
            assert spec == prepared, 'Written stage spec differs from pinned plan'
            obs = observed[key]
            output = Path(spec['output_dir'])
            receipt_path = output/'g2_script_verification.json'
            before = sha(receipt_path)
            verified = runner.verify_stage(spec)
            assert sha(receipt_path) == before, 'Read-only receipt changed'
            stored = read(receipt_path)
            assert verified == stored == obs['receipt'], 'Native revalidation differs from stored receipt/state'
            traces = [json.loads(s) for s in (output/'g2_step_trace.jsonl').read_text().splitlines() if s.strip()]
            assert finite_all(traces), 'Nonfinite numeric step diagnostics'
            train = read(output/'train_results.json')
            trainer = read(output/'trainer_state.json')
            assert finite_all(train) and finite_all(trainer), 'Nonfinite train metric/state'
            assert isinstance(train.get('train_loss'), (int, float)) and math.isfinite(train['train_loss'])
            assert finite_all(stored['prediction_results']), 'Nonfinite evaluation metric'
            normal = sum(row['g2_final_direction'] == 'g2' for row in traces)
            fallback = sum(row['g2_final_direction'] == 'g0_fallback' for row in traces)
            assert normal + fallback == spec['expected_updates']
            log_path = Path(manifest['artifact_root'])/'logs'/(spec_path.stem+'.log')
            log = log_path.read_text()
            assert '***** predict metrics *****' in log and '***** train metrics *****' in log
            assert 'Traceback (most recent call last)' not in log
            item.update(status='PASS', native_verification='PASS', receipt_unchanged=True,
                receipt_sha256=before, trace_sha256=stored['g2_trace_sha256'],
                adapter_sha256=stored['adapter_sha256'], prediction_sha256=stored['prediction_sha256'],
                runtime_audit_sha256=stored['g2_runtime_audit_sha256'],
                train_results_sha256=sha(output/'train_results.json'),
                trainer_state_sha256=sha(output/'trainer_state.json'),
                log_sha256=sha(log_path), log_final_train_and_predict_metrics=True,
                observed_updates=len(traces), normal_three_read_steps=normal, fallback_steps=fallback,
                actual_optimizer=stored['actual_base_optimizer'],
                actual_optimizer_param_groups=stored['actual_base_optimizer_param_groups'],
                all_recorded_numeric_diagnostics_finite=True, train_loss=train['train_loss'],
                train_runtime_seconds=train['train_runtime'],
                log_history_training_loss_count=sum('loss' in row for row in trainer['log_history']),
                adapter_finite_fp32=stored['adapter_finite_fp32'],
                historical_prefix_exact=stored['historical_prefix_exact'],
                saved_rank=stored['saved_rank'], prediction_samples=stored['prediction_samples'],
                prediction_task_counts=stored['prediction_task_counts'],
                completed_at=obs['completed_at'],
                exit_provenance='Frozen run_child raises on any nonzero return code; accepted completion is appended only after run_child returns and native verify_stage passes. No separate per-stage exit-code file exists.')
        except Exception as exc:
            item.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            failures.append({'key': key, 'error': item['error']})
        ledger.append(item)
        print(json.dumps({'key': key, 'status': item['status'], 'fallback_steps': item.get('fallback_steps')}, ensure_ascii=False), flush=True)

assert set(expected_keys) == set(observed) and len(expected_keys) == 36
assert not torch.cuda.is_initialized()
result = {
    'schema': 'IndependentCompletedG2IntegrityAudit_v1',
    'audited_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
    'status': 'PASS' if not failures else 'FAIL',
    'scope': 'CPU-only independent reread of all 12 smoke and 24 formal stages; no GPU calls, training, signals or frozen artifact/source writes',
    'queue_root': str(ROOT), 'completed_at': state['completed_at'],
    'manifest_sha256': sha(ROOT/'manifest.json'), 'queue_state_sha256': sha(state_path),
    'source_file_count': len(manifest['source_sha256']), 'input_file_count': len(manifest['input_sha256']),
    'source_and_input_pin_verification': 'PASS',
    'source_sha256': manifest['source_sha256'], 'input_sha256': manifest['input_sha256'],
    'expected_smoke_stages': 12, 'expected_formal_stages': 24,
    'passed_smoke_stages': sum(row['status']=='PASS' and row['kind']=='smoke' for row in ledger),
    'passed_formal_stages': sum(row['status']=='PASS' and row['kind']=='formal' for row in ledger),
    'observed_formal_updates': sum(row.get('observed_updates',0) for row in ledger if row['kind']=='formal'),
    'formal_normal_three_read_steps': sum(row.get('normal_three_read_steps',0) for row in ledger if row['kind']=='formal'),
    'formal_fallback_steps': sum(row.get('fallback_steps',0) for row in ledger if row['kind']=='formal'),
    'smoke_fallback_steps': sum(row.get('fallback_steps',0) for row in ledger if row['kind']=='smoke'),
    'limitations': ['Only seed 0 ran; no multi-seed statistical claim.',
        'Per-stage exit success is evidenced by frozen checked controller flow plus unmodified accepted receipts, final logs and complete outputs; no standalone OS waitpid exit-code file was retained.',
        'Predictions predict_loss=0 is a generation evaluator placeholder; actual finite training and probe losses were audited.'],
    'failures': failures, 'stage_ledger': ledger,
    'cuda_initialized': torch.cuda.is_initialized(),
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ('status','passed_smoke_stages','passed_formal_stages','observed_formal_updates','formal_fallback_steps','smoke_fallback_steps','completed_at')}), flush=True)
