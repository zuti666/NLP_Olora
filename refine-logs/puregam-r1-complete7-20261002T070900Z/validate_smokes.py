"""Release full runs only after native GPU execution and direction proofs pass."""
from __future__ import annotations
import json
import math
from common import (HERE, atomic_json, inspect_metrics, now, read_json, resolve,
                    sha256, verify_run_pins, verify_reuse)


def validate_job_audit(suite, job, *, require_receipt=True):
    run = HERE / 'runtime/jobs' / job['id']
    summary = read_json(run / 'direction_audit_summary.json')
    audit_path = run / 'direction_audit.jsonl'
    if summary['status'] != 'completed' or summary['selector'] != job['direction_selector']:
        raise RuntimeError('Direction audit incomplete or wrong selector')
    if summary['legacy_direction_sha256'] != sha256(HERE / 'legacy_direction.py'):
        raise RuntimeError('Direction implementation changed')
    if summary['audit_sha256'] != sha256(audit_path):
        raise RuntimeError('Direction trace changed after summary')
    if summary['steps'] != job['expected_steps'] or summary['samples'] != job['expected_samples']:
        raise RuntimeError('Incomplete native training step/sample budget')
    if summary['task_steps'] != job['expected_task_steps']:
        raise RuntimeError('Incorrect per-task training budget')
    if (summary['native_gradient_closures'] != 4*job['expected_steps']
            or summary['native_ce_forward_passes'] != 5*job['expected_steps']
            or summary['actual_sgd_steps'] != job['expected_steps']
            or summary['extra_training_forwards'] != 0
            or summary['RNG_reseeds_or_restores'] != 0
            or not summary['factory_and_bound_methods_restored']):
        raise RuntimeError('Native execution/RNG contract differs')
    count = 0
    with audit_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['step_index'] != count or row['selector'] != job['direction_selector']:
                raise RuntimeError('Trace order/selector invalid')
            if (row['closure_count'], row['forward_count'], row['sgd_count'], row['native_decomposition_called']) != (4,5,1,1):
                raise RuntimeError('Native four probes or SGD count invalid')
            for key in ['selected_exact','four_native_gradient_caches_exact','projector_exact',
                        'fisher_cache_empty','delta_cache_empty','selection_rng_unchanged',
                        'projector_rng_unchanged','sgd_exact','sgd_rng_unchanged']:
                if row.get(key) is not True:
                    raise RuntimeError('Direction proof failed: ' + key)
            for left,right in [('selected_expected','selected_actual'),('projected_expected','projected_actual')]:
                if row[left]['sha256'] != row[right]['sha256']:
                    raise RuntimeError('Gradient identity mismatch')
            if row['sgd_expected_parameter_sha256'] != row['sgd_actual_parameter_sha256']:
                raise RuntimeError('Actual update differs from native SGD')
            count += 1
    if count != job['expected_steps']:
        raise RuntimeError('Incorrect audit row count')
    _, config, output = __import__('common').verify_config(job)
    files = sorted(output.rglob('*.pt'))
    if len(files) != 5*job['expected_tasks']:
        raise RuntimeError('Task-end model checkpoint set incomplete')
    checkpoint_hashes = {str(path.relative_to(output)): sha256(path) for path in files}
    log = (run/'console.log').read_text(errors='replace')
    skips = log.count('ewc_skip_task_fisher=true: skipped task-end')
    if skips != job['expected_tasks'] or 'has_fisher=True' in log or 'source=post_weight_merge checkpoints=' in log:
        raise RuntimeError('Unexpected Fisher computation/history')
    if not all([config['ewc_skip_task_fisher'], config['ewc_lambda']==0,
                not config['in_task_weight_merge'], not config['task_endpoint_weight_merge'],
                not config['approximate_fisher_in_task']]):
        raise RuntimeError('Pure GAM configuration violated')
    result = {'audit_sha256':sha256(audit_path),
              'audit_summary_sha256':sha256(run/'direction_audit_summary.json'),
              'steps':count, 'fisher_skips':skips, 'checkpoint_hashes':checkpoint_hashes}
    if require_receipt:
        receipt = read_json(run/'completed.json')
        if receipt['direction_proof'] != result:
            raise RuntimeError('Completion receipt does not bind the current execution/checkpoints')
    return result


def _compare(left, right, reports, name):
    import torch
    if isinstance(left, torch.nn.Module):
        if type(left)!=type(right):raise RuntimeError('Control module types differ: '+name)
        _compare(left.state_dict(),right.state_dict(),reports,name+'/state_dict')
    elif isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError('Control checkpoint tensor structure differs: '+name)
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise RuntimeError('Nonfinite control checkpoint: '+name)
        exact = torch.equal(left,right)
        maximum = float((left.double()-right.double()).abs().max()) if left.numel() else 0.
        scale = float(left.double().abs().max()) if left.numel() else 0.
        if maximum > 1e-7 + 1e-6*scale:
            raise RuntimeError('Control parameter deviation exceeds predeclared tolerance: '+name)
        reports.append({'name':name,'bitwise_equal':exact,'max_absolute_difference':maximum})
    elif isinstance(left,dict):
        if not isinstance(right,dict) or set(left)!=set(right):raise RuntimeError('Control keys differ')
        for key in left:_compare(left[key],right[key],reports,name+'/'+str(key))
    elif isinstance(left,(list,tuple)):
        if type(left)!=type(right) or len(left)!=len(right):raise RuntimeError('Control lists differ')
        for i,(a,b) in enumerate(zip(left,right)):_compare(a,b,reports,name+'/'+str(i))
    elif left != right:
        raise RuntimeError('Control non-tensor value differs: '+name)


def validate_smokes(suite):
    verify_run_pins(suite);verify_reuse(suite)
    jobs={job['id']:job for job in suite['jobs']}
    proofs={}
    for jid in suite['smoke_jobs']:
        result=inspect_metrics(suite,jobs[jid])
        receipt=read_json(HERE/'runtime/jobs'/jid/'completed.json')
        if receipt['result']['metrics_sha256']!=result['metrics_sha256']:
            raise RuntimeError('Smoke raw metrics changed')
        proofs[jid]=validate_job_audit(suite,jobs[jid])
    control_ids=['smoke_native_control_r1_k10_s2','smoke_g1_control_r1_k10_s2']
    controls=[jobs[j] for j in control_ids]
    results=[inspect_metrics(suite,j) for j in controls]
    if results[0]['cnn_matrix']!=results[1]['cnn_matrix'] or results[0]['cnn_top1_curve']!=results[1]['cnn_top1_curve']:
        raise RuntimeError('Native g1 control and hooked g1 evaluation differs')
    paths=[HERE/'runtime/jobs'/j['id']/'direction_audit.jsonl' for j in controls]
    paired=0
    with paths[0].open() as left,paths[1].open() as right:
        for a,b in zip(left,right):
            a,b=json.loads(a),json.loads(b)
            for key in ['input_sha256','target_sha256','rng_before_batch_sha256','rng_after_batch_sha256',
                        'task','task_step_index','class_offset','batch_samples','parameter_names','lr']:
                if a[key]!=b[key]:raise RuntimeError('Control data/RNG sequence differs: '+key)
            paired+=1
    if paired!=controls[0]['expected_steps']:raise RuntimeError('Control trace incomplete')
    import torch
    reports=[]
    output_paths=[resolve(__import__('common').verify_config(j)[1]['outputs_root']) for j in controls]
    checkpoints=[{p.name:p for p in root.rglob('*.pt')} for root in output_paths]
    if set(checkpoints[0])!=set(checkpoints[1]):raise RuntimeError('Control checkpoint names differ')
    for name in checkpoints[0]:
        values=[torch.load(c[name],map_location='cpu',weights_only=False) for c in checkpoints]
        _compare(values[0],values[1],reports,name)
    proof={'passed':True,'validated_at':now(),'suite_sha256':sha256(HERE/'suite.json'),
           'run_pins_sha256':sha256(HERE/'RUN_PINS.json'),'smoke_jobs':proofs,
           'paired_native_g1_batches':paired,'same_input_target_rng':True,
           'control_cnn_matrix_equal':True,'control_parameter_atol':1e-7,'control_parameter_rtol':1e-6,
           'control_checkpoint_tensors':reports,
           'control_all_tensors_bitwise_equal':all(r['bitwise_equal'] for r in reports)}
    atomic_json(HERE/'GPU_SMOKE_PROOF.json',proof)
    return proof
