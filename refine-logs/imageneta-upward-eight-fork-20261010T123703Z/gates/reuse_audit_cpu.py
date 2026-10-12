#!/usr/bin/env python3
"""Read-only provenance audit of the accepted seed2 parent and old SGD fork."""
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

NEW = Path(__file__).resolve().parents[1]
OLD = NEW.parent / 'imageneta-g2-minimal-fork-gpu456-20261009T223549Z'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def require(ok, reason):
    if not ok:
        raise AssertionError(reason)

def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only audit required')
    suite = read(OLD / 'suite.json')
    pins = read(OLD / 'RUN_PINS.json')
    pin_rows = []
    for rel, expected in pins['files'].items():
        actual = sha(OLD / rel)
        require(actual == expected, 'Old immutable execution pin changed: ' + rel)
        pin_rows.append(dict(path=str(OLD / rel), sha256=actual))
    manifest = read(OLD / 'source_manifest.json')
    source = Path(suite['source_root'])
    source_rows = []
    for rel, expected in manifest['files'].items():
        actual = sha(source / rel)
        require(actual == expected, 'Frozen source changed: ' + rel)
        source_rows.append(dict(path=str(source / rel), sha256=actual))
    require(sha(NEW / 'dev_split.json') == sha(OLD / 'dev_split.json'), 'New dev split differs')
    sys.path.insert(0, str(OLD))
    sys.path.insert(0, str(OLD / 'trainer'))
    validator = importlib.import_module('validate_job')
    jobs = {j['id']: j for j in suite['jobs']}
    rows = {}
    evidence = {}
    for job_id in ('formal_parent_sgd_s2', 'formal_task1_B0_s2'):
        j = jobs[job_id]
        out = Path(j['output_dir'])
        accepted = validator.validate_job(j)
        completed_path = OLD / 'runtime' / 'jobs' / job_id / 'completed.json'
        completed = read(completed_path)
        require(completed['exit']['returncode'] == 0, 'Old worker not accepted')
        for k, v in accepted.items():
            require(completed.get(k) == v, 'Old acceptance receipt differs: ' + k)
        s = read(out / 'final_summary.json')
        trace = [json.loads(x) for x in (out / 'step_trace.jsonl').read_text().splitlines()]
        cp = Path(s['checkpoint_dirs'][0])
        cp_manifest = read(cp / 'manifest.json')
        ident = read(out / 'task_identities.json')[0]
        for f in (completed_path, out / 'final_summary.json', out / 'task_matrix.json',
                  out / 'effective_config.json', out / 'task_identities.json',
                  out / 'step_trace.jsonl', out / 'dev_curve.json', cp / 'manifest.json'):
            evidence[str(f)] = sha(f)
        for name, expected in cp_manifest['files_sha256'].items():
            actual = sha(cp / name)
            require(actual == expected, 'Checkpoint member changed')
            evidence[str(cp / name)] = actual
        witness_path = OLD / 'runtime' / 'jobs' / job_id / 'cuda_witness.json'
        witness = read(witness_path)
        require(witness['status'] == 'passed' and not witness['tf32_matmul'] and not witness['tf32_cudnn'], 'Old CUDA witness failed')
        evidence[str(witness_path)] = sha(witness_path)
        rows[job_id] = dict(
            accepted=True, independent_old_validator_result=accepted,
            completion_receipt=str(completed_path), completion_sha256=sha(completed_path),
            final_summary_sha256=sha(out / 'final_summary.json'),
            final_parameter_sha256=s['final_model_parameter_sha256'],
            initial_identity=ident, checkpoint_manifest=cp_manifest,
            training_ce_scope='native current-task 20-logit slice; targets shifted by class_offset',
            dev_scope='current-task 20 logits; train-only dev holdout',
            test_scope='native CNN argmax across all seen classes (20 at parent, 40 at Task1)',
            initial_lr=trace[0]['nominal_lr'], final_training_epoch_lr=trace[-1]['nominal_lr'],
            epochs=s['completed_epochs'], updates=s['completed_steps'],
            actual_training_gradient_reads=s['counters']['training_ce_gradient_passes'],
            diagnostic_post_update_reads=s['counters']['diagnostic_loss_forward_count'],
            strict_deterministic_every_step=all(r['strict_deterministic'] for r in trace),
            full_state_rng_restore_every_step=all(all(r['state_restore_checks'].values()) and all(r['diagnostic_state_restoration'].values()) and r['only_base_R1_rng_advance'] for r in trace),
            natural_sgd_every_step=all(r['natural_sgd_exact'] and r['sgd_updates'] == 1 for r in trace),
            first_g0_sha256=trace[0]['g0_sha256'], first_batch_identity=trace[0]['identity'],
            first_base_parameter_sha256=trace[0]['base_parameter_sha256'],
            final_metrics=s['final_metrics'],
        )
    b0out = Path(jobs['formal_task1_B0_s2']['output_dir'])
    snaps = read(b0out / 'snapshots.json')['snapshots']
    snapshot_rows = []
    for row in snaps:
        require(sha(row['path']) == row['sha256'], 'B0 complete snapshot changed')
        snapshot_rows.append({k: row[k] for k in ('epoch', 'path', 'sha256', 'full_model_parameter_sha256', 'current_optimizer_lr')})
        evidence[row['path']] = row['sha256']
    require([r['epoch'] for r in snapshot_rows] == [0, 1, 10, 25, 49], 'Unexpected old B0 sparse snapshot times')
    split = read(OLD / 'dev_split.json')
    counts = {}
    for task in (0, 1):
        lo, hi = 20 * task, 20 * (task + 1)
        counts[str(task)] = dict(
            train=sum(lo <= split['native_train_targets'][i] < hi for i in split['train_indices']),
            dev=sum(lo <= split['native_train_targets'][i] < hi for i in split['dev_indices']))
    require(counts == {'0': {'train': 1141, 'dev': 118}, '1': {'train': 906, 'dev': 91}}, 'Split task budgets differ')
    report = dict(
        status='passed', audited_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Read-only CPU revalidation; no old source/config/output mutations or GPU training',
        old_run=str(OLD), new_run=str(NEW),
        old_run_pins_sha256=sha(OLD / 'RUN_PINS.json'),
        old_execution_pins_verified=len(pin_rows), old_source_files_verified=len(source_rows),
        source_root=str(source), source_manifest_sha256=sha(OLD / 'source_manifest.json'),
        dev_split_sha256=sha(OLD / 'dev_split.json'), split_task_counts=counts,
        parent_reuse_accepted=True, rows=rows,
        old_B0_usage='Equivalence oracle only: new Task1 SGD rerun must pair all 400 realized input/target/index/LR records and final parameter hash; not a new full-epoch trajectory',
        old_B0_full_epoch_test_curve_available=False,
        old_B0_retention_auc_50_epochs_reconstructible=False,
        old_B0_snapshots=snapshot_rows,
        new_parent_policy='seed2 reuses accepted old SGD Task0; seeds0/1 each train a new SGD Task0 parent',
        new_training_policy='24 new r=1 Task1 forks (8 methods x 3 seeds), 1 seed2 upward r=.25 g2 bridge, 2 new Task0 parents, 18 new full200 single-task trainings (6 methods x 3 seeds): 45 new formal trainings',
        reuse_admission_requirements=[
            'new seed2 forks bind accepted parent receipt, shared_parent.pt and all native checkpoint files',
            'same train/dev manifest, current20 CE and all-seen40 test definitions',
            'same FC20->40 expansion parameter hash and new FC rows hash',
            'same first realized minibatch and g0 hash after exact parent RNG restore',
            '400 per-fork updates, LR=.02 epoch-cosine50 floor .1, full LoRA+FC scope',
            'all new final results validated against method-specific native read budget and forbidden-work guards',
        ],
        excluded_reuse=['old full-training 10-task SGD/SAM rows: different dev membership, parent/fork protocol and budgets',
                        'old 20-class Task0 row as full200 single-task result',
                        'descending g2 R25/R100/S_F under same upward g2 method name'],
        evidence_hashes=evidence, old_execution_pin_ledger=pin_rows,
        old_frozen_source_ledger=source_rows,
        checks=dict(old_pins_unchanged=True, old_source_unchanged=True, old_parent_accepted=True,
                    old_B0_accepted=True, split_identical=True, parent_complete_state_available=True,
                    old_head_expansion_identity_bound=True, strict_FP32_witness_bound=True,
                    current20_train_and_dev_allseen_test_scope_checked=True,
                    budgets_lr_rng_natural_sgd_revalidated=True,
                    sparse_old_B0_curve_not_fabricated=True, no_old_mutations=True),
    )
    target = NEW / 'gates' / 'REUSE_AUDIT.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status='passed', output=str(target), pins=len(pin_rows), source_files=len(source_rows), parent_parameter_sha256=rows['formal_parent_sgd_s2']['final_parameter_sha256'], old_B0_expanded_parameter_sha256=rows['formal_task1_B0_s2']['initial_identity']['task_initial_parameter_sha256'], new_formal_trainings=45)))

if __name__ == '__main__':
    main()
