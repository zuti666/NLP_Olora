#!/usr/bin/env python3
"""Wait for the ten authorized Task0 runs and launch only their offline diagnostics.

Legacy failures remain intact. The recovered SAM arm uses its new output and
receipt. This program never creates or starts Task1/Task2 training.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--original-suite', required=True)
    ap.add_argument('--radii-suite', required=True)
    ap.add_argument('--sam-suite', required=True)
    ap.add_argument('--runtime', required=True)
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()
    original_path, radii_path, sam_path = map(Path, (
        args.original_suite, args.radii_suite, args.sam_suite))
    original, radii = read(original_path), read(radii_path)
    root = Path(original['campaign'])
    sys.path.insert(0, str(root))
    from common import atomic_json, now, sha256, verify_source, proc_identity, assert_host
    runtime = Path(args.runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    inputs = {str(p): sha256(p) for p in (original_path, radii_path)}
    source_manifest_sha = sha256(original['source_manifest'])
    source_gam_sha = sha256(Path(original['source_root']) / 'optimer_PerturabtionType' / 'gam.py')

    def finite(value):
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return True

    def validate_suite(suite, path):
        assert_host(suite)
        if (suite['source_root'] != original['source_root'] or
                suite['source_manifest'] != original['source_manifest'] or
                sha256(suite['source_manifest']) != source_manifest_sha):
            raise RuntimeError('Recovered source identity changed')
        verify_source(suite)
        for pathstr, expected in suite['file_pins'].items():
            if sha256(pathstr) != expected:
                raise RuntimeError('Pinned file changed: ' + pathstr)
        if Path(suite['campaign']) != root:
            raise RuntimeError('Campaign changed')
        if path is not None and str(path) in inputs and sha256(path) != inputs[str(path)]:
            raise RuntimeError('Suite changed: ' + str(path))

    validate_suite(original, original_path)
    validate_suite(radii, radii_path)
    if {j['label'] for j in original['jobs']} != {
            's', 'g', 'mean', 'sum', 'ce', 'true_sam',
            'g_r0', 'g_r01', 'g_r025', 'g_r05'}:
        raise RuntimeError('Unexpected original ten-arm matrix')
    if args.validate_only:
        print(json.dumps({'status': 'passed', 'sam_suite_may_appear_later': str(sam_path),
                          'ten_task0_arms_only': True, 'next_training_stages_queued': False}))
        return
    lock = open(runtime / 'controller.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (runtime / 'controller.json').exists():
        raise RuntimeError('Existing reconciliation runtime; no blind restart')
    atomic_json(runtime / 'controller.json', {
        **proc_identity(os.getpid()), 'time': now(), 'input_suites': inputs,
        'sam_suite_expected': str(sam_path), 'script_sha256': sha256(__file__)})
    try:
        while True:
            for suite in (radii,):
                sp = Path(suite['runtime']) / 'state.json'
                ep = Path(suite['runtime']) / 'exception.json'
                if ep.exists() or (sp.exists() and read(sp).get('failed')):
                    raise RuntimeError('Recovery queue failed: ' + suite['runtime'])
            if not sam_path.exists():
                smoke_runtime = sam_path.parent / 'runtime' / 'smoke14'
                sp = smoke_runtime / 'state.json'
                if (sp.exists() and read(sp).get('failed')) or (smoke_runtime / 'exception.json').exists():
                    raise RuntimeError('TrueSAM v2 GPU preflight failed; not finalizing')
                atomic_json(runtime / 'state.json', {
                    'time': now(), 'status': 'waiting_for_sam_preflight',
                    'next_training_stages_queued': False})
                time.sleep(20)
                continue
            sam = read(sam_path)
            inputs.setdefault(str(sam_path), sha256(sam_path))
            validate_suite(sam, sam_path)
            if len(sam['jobs']) != 1:
                raise RuntimeError('Recovered SAM suite is not a single arm')
            sp = Path(sam['runtime']) / 'state.json'
            ep = Path(sam['runtime']) / 'exception.json'
            if ep.exists() or (sp.exists() and read(sp).get('failed')):
                raise RuntimeError('Recovered SAM queue failed')
            actual_jobs = []
            waiting = []
            for job in original['jobs']:
                label = job['label']
                chosen, suite = job, original
                if label == 'true_sam':
                    chosen, suite = sam['jobs'][0], sam
                elif label.startswith('g_r'):
                    chosen = next(j for j in radii['jobs'] if j['id'] == job['id'])
                    suite = radii
                receipt_path = Path(suite['runtime']) / 'jobs' / chosen['id'] / 'completed.json'
                if not receipt_path.exists():
                    waiting.append(job['id'])
                actual_jobs.append({**job, 'config': chosen['config'],
                                    'protocol_config': job['config'], 'output': chosen['output'],
                                    'receipt': str(receipt_path),
                                    'started': str(receipt_path.with_name('started.json')),
                                    'runtime_job_id': chosen['id'],
                                    'runtime_suite_sha256': sha256(sam_path if label == 'true_sam' else radii_path if label.startswith('g_r') else original_path),
                                    'trainer_sha256': sha256(suite['script'])})
            atomic_json(runtime / 'state.json', {
                'time': now(), 'status': 'waiting_for_training' if waiting else 'verifying',
                'pending': waiting, 'completed': 10 - len(waiting),
                'next_training_stages_queued': False})
            if not waiting:
                break
            time.sleep(20)

        validate_suite(original, original_path)
        validate_suite(radii, radii_path)
        validate_suite(sam, sam_path)
        rows, identities, orders, batches, receipts, pairing_details = [], [], [], [], [], []
        for job in actual_jobs:
            out = Path(job['output'])
            summary = read(out / 'final_summary.json')
            epochs = read(out / 'metrics.json')['epochs']
            receipt = read(job['receipt'])
            started = read(job['started'])
            cfg = read(job['config'])
            if cfg != read(job['protocol_config']):
                raise RuntimeError('Recovery changed original configuration: ' + job['id'])
            if (started['job']['config'] != job['config'] or
                    started['job']['output'] != str(out) or
                    started['job']['id'] != job['runtime_job_id'] or
                    started.get('id', started['job']['id']) != job['runtime_job_id'] or
                    receipt.get('id') != job['runtime_job_id']):
                raise RuntimeError('Started configuration/output binding mismatch')
            if 'script_sha256' in started:
                if started['script_sha256'] != job['trainer_sha256'] or started['config_sha256'] != sha256(job['config']):
                    raise RuntimeError('Started script/config hash mismatch')
            elif (started.get('trainer_v2_sha256') != job['trainer_sha256'] or
                  started.get('suite_sha256') != job['runtime_suite_sha256'] or
                  started.get('source') != original['source_root']):
                raise RuntimeError('Recovered SAM started suite/trainer/source mismatch')
            if (receipt.get('returncode') != 0 or receipt.get('errors') or
                    receipt.get('summary') != summary or receipt.get('output') != str(out)):
                raise RuntimeError('Completion receipt mismatch: ' + job['id'])
            if (summary['status'] != 'completed' or summary['smoke'] or
                    summary['completed_epochs'] != 50 or len(epochs) != 50 or
                    summary['seed'] != 2 or summary['trained_tasks'] != [0] or
                    summary['trainer_sha256'] != job['trainer_sha256'] or
                    summary['source_gam_sha256'] != source_gam_sha or
                    summary['r'] != cfg['gam_grad_norm_rho'] or
                    summary['direction'] != cfg['stage1_direction']):
                raise RuntimeError('Incomplete or mismatched Task0 arm: ' + job['id'])
            if ([e['epoch'] for e in epochs] != list(range(1, 51)) or
                    any(e['task'] != 0 for e in epochs) or summary['final'] != epochs[-1]):
                raise RuntimeError('Epoch sequence/task/final summary mismatch')
            geometry = read(out / 'geometry.json')['records']
            if (not finite(epochs + geometry) or len(geometry) != 50 or
                    [g['epoch'] for g in geometry] != list(range(1, 51)) or
                    any(g['task'] != 0 or g['batch'] != 0 for g in geometry) or
                    not summary['finite_check']['passed'] or
                    not all(summary['checkpoint_checks'].values()) or not summary['task_check']['passed']):
                raise RuntimeError('Finite/geometry/checkpoint self-audit failed')
            for g in geometry:
                if not g['base_sgd_audit']['passed'] or g['selected_gradient_relative_error'] > 1e-4:
                    raise RuntimeError('Selected-gradient or actual-SGD audit failed')
                if summary['direction'] == 'true_sam':
                    selfsam = g['self_sam_audit']
                    if (not selfsam['probe_parameter_check']['all_tensors_exact'] or
                            not selfsam['saved_clean_parameters_exact'] or
                            not selfsam['actual_probe_gradient_check']['passed'] or
                            not g['base_sgd_audit']['sam_saved_copy_restored_exactly']):
                        raise RuntimeError('Native SAM own probe/backward/restore gate failed')
            if cfg['ewc_lambda'] != 0 or not cfg['ewc_skip_task_fisher'] or cfg['in_task_weight_merge'] or cfg['task_endpoint_weight_merge']:
                raise RuntimeError('Unexpected EWC/Fisher/merge configuration')
            if summary['checkpoint_epochs'] != [0, 10, 20, 30, 40, 50]:
                raise RuntimeError('Missing checkpoint schedule')
            for e in epochs:
                if e['train_samples'] != 1259 or e['eval_samples'] != 307:
                    raise RuntimeError('Changed sample count')
                for key in ('eval_accuracy', 'eval_ce', 'train_accuracy', 'train_ce'):
                    if not math.isfinite(e[key]):
                        raise RuntimeError('Nonfinite epoch metric')
                expected_lr = .004 + (.04 - .004) * (1 + math.cos(math.pi * (e['epoch'] - 1) / 50)) / 2
                if len(e['lr_used']) != 1 or not math.isclose(e['lr_used'][0], expected_lr, rel_tol=1e-12, abs_tol=1e-12):
                    raise RuntimeError('Changed 50-epoch cosine LR schedule')
            count = summary['counters']
            expected_passes = 2 if summary['direction'] == 'true_sam' else 5
            if (count['training_batches'] != 500 or count['training_samples'] != 62950 or
                    count['training_ce_gradient_passes'] != 500 * expected_passes):
                raise RuntimeError('Training budget mismatch')
            identities.append(summary['initial_identity']['initial_model_sha256'])
            orders.append([e['sampler_order_sha256'] for e in epochs])
            batches.append([e['first_batch']['input_sha256'] for e in epochs])
            pairing_details.append([
                {k: e[k] for k in ('eval_per_class_counts', 'lr_used', 'lr_after_scheduler')} |
                {'first_target': e['first_batch']['target_sha256'],
                 'first_indices': e['first_batch']['indices'], 'data_seed': e['first_batch']['data_seed']}
                for e in epochs])
            f = summary['final']
            rows.append({'id': job['id'], 'direction': summary['direction'],
                         'r': summary['r'], 'seed': 2, 'epoch': 50,
                         'a11_sample_accuracy': f['eval_accuracy'], 'eval_ce': f['eval_ce'],
                         'train_ce': f['train_ce'], 'train_accuracy': f['train_accuracy'],
                         'elapsed_seconds': summary['elapsed_seconds'], 'output': str(out)})
            receipts.append({'job': job['id'], 'receipt': job['receipt'],
                             'receipt_sha256': sha256(job['receipt']),
                             'summary_sha256': sha256(out / 'final_summary.json'),
                             'metrics_sha256': sha256(out / 'metrics.json'),
                             'geometry_sha256': sha256(out / 'geometry.json'),
                             'started_sha256': sha256(job['started']),
                             'config_sha256': sha256(job['config']),
                             'trainer_sha256': job['trainer_sha256']})
        checks = {'same_initial_model': len(set(identities)) == 1,
                  'same_epoch_sample_orders': all(x == orders[0] for x in orders),
                  'same_epoch_first_augmented_batch': all(x == batches[0] for x in batches),
                  'same_targets_indices_counts_and_lr': all(x == pairing_details[0] for x in pairing_details)}
        if not all(checks.values()):
            raise RuntimeError('Cross-arm pairing check failed: ' + repr(checks))
        resultdir = root / 'results_recovery_20261001'
        resultdir.mkdir(exist_ok=False)
        with open(resultdir / 'task0_results.csv', 'x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        atomic_json(resultdir / 'task0_results.json', {
            'status': 'completed', 'time': now(), 'rows': rows, 'pairing_checks': checks,
            'primary_metric': 'fixed epoch50 Task0 sample accuracy a11',
            'selection_status': 'pending_joint_accuracy_and_matched_diagnostic_review',
            'next_training_stages': 'not_queued'})
        atomic_json(resultdir / 'TRAINING_COMPLETION_AUDIT.json', {
            'time': now(), 'checks': checks, 'inputs': inputs, 'receipts': receipts,
            'source_manifest_sha256': source_manifest_sha, 'source_gam_sha256': source_gam_sha,
            'legacy_sam_failure_preserved': True})
        reconciled = dict(original, jobs=actual_jobs, final_command=[])
        atomic_json(resultdir / 'suite_reconciled.json', reconciled)

        diag = root / 'diagnostics'
        ds = {k: original[k] for k in ('python', 'execution_hostname', 'allowed_gpus',
                                      'source_root', 'source_manifest', 'environment')}
        ds.update(campaign=str(root), script=str(diag / 'paired_gam_diagnostic.py'),
                  runtime=str(diag / 'runtime' / 'formal_completion_recovery_20261001'),
                  max_parallel=2, summarize=True, file_pins={}, jobs=[])
        for name in ('paired_gam_diagnostic.py', 'run_queue.py', 'summarize_diagnostics.py'):
            ds['file_pins'][str(diag / name)] = sha256(diag / name)
        for label, epoch in (('s', 0), ('s', 50), ('g', 50), ('sum', 50), ('mean', 50)):
            job = next(j for j in actual_jobs if j['label'] == label)
            cp = Path(job['output']) / 'checkpoints' / ('epoch_%03d' % epoch)
            ds['file_pins'][job['config']] = sha256(job['config'])
            for name in ('lora_w_a_0.pt', 'lora_w_b_0.pt', 'fc_state_0.pt', 'lora_meta_0.json'):
                ds['file_pins'][str(cp / name)] = sha256(cp / name)
            output = diag / 'results' / (label + '_epoch' + str(epoch) + '.json')
            if output.exists():
                raise RuntimeError('Existing diagnostic output, no blind relaunch: ' + str(output))
            ds['jobs'].append({'id': label + '_epoch' + str(epoch), 'task': 0,
                               'max_batches': 4, 'config': job['config'],
                               'checkpoint_dir': str(cp), 'output': str(output)})
        dsp = diag / 'suite_completion_recovery_20261001.json'
        if dsp.exists() or (diag / 'suite.json').exists():
            raise RuntimeError('Existing diagnostic registration; do not duplicate')
        atomic_json(dsp, ds)
        command = [original['python'], str(diag / 'run_queue.py'), '--suite', str(dsp)]
        env = dict(os.environ, **original['environment'])
        subprocess.run(command + ['--validate-only'], check=True, env=env)
        atomic_json(runtime / 'state.json', {
            'time': now(), 'status': 'running_offline_diagnostics', 'completed_training': 10,
            'diagnostic_suite': str(dsp), 'next_training_stages_queued': False})
        subprocess.run(command, check=True, env=env)
        atomic_json(runtime / 'state.json', {
            'time': now(), 'status': 'completed', 'completed_training': 10,
            'diagnostic_suite': str(dsp), 'next_training_stages_queued': False})
    except BaseException as exc:
        atomic_json(runtime / 'exception.json', {'time': now(), 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
