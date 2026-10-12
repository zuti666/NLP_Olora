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
    for module in ['residual_selection','effective_step']:
        if summary[module+'_sha256'] != sha256(HERE/(module+'.py')):
            raise RuntimeError('Residual/step implementation changed: '+module)
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
            validate_new_step_record(row, job)
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


def validate_new_step_record(row, job):
    if row['schema_version']!=2 or row.get('component_scale')!='natural' or row.get('final_norm_matching') is not False:
        raise RuntimeError('Unexpected matching/scaling intervention')
    if row['lr_nominal'] != row['step_scale'] or row['scale_over_nominal_lr']!=1. or any(v!=row['lr_nominal'] for v in row['lr']):
        raise RuntimeError('Actual SGD learning rate is not the nominal LR')
    if (row['reference_counterfactual_state_unchanged'] is not True
            or row['effective_step_measurement_only'] is not True
            or row['effective_geometry_scope']!='unscaled_LoRA_BA_and_all_seen_FC'
            or row['reference_definition']!='same_candidate_state_same_batch_native_projected_g1_after_native_unperturb_actual_FP32_SGD'):
        raise RuntimeError('Effective step observation changed state or reference point')
    for key in ['actual_effective_delta','reference_effective_delta']:
        values=row[key]
        if any(not math.isfinite(values[k]) or values[k]<0 for k in ['joint_norm','lora_norm','fc_norm','parameter_norm']):
            raise RuntimeError('Nonfinite effective update geometry')
        if not math.isclose(values['joint_norm']**2,values['lora_norm']**2+values['fc_norm']**2,rel_tol=1e-10,abs_tol=1e-20):
            raise RuntimeError('Effective step components disagree')
    if not math.isclose(row['parameter_update_l2'],row['actual_effective_delta']['parameter_norm'],rel_tol=1e-10,abs_tol=1e-20):
        raise RuntimeError('Actual parameter norm alias disagrees')
    ref=row['reference_effective_delta']['joint_norm']
    if row['reference_target_norm']!=ref:raise RuntimeError('Wrong effective reference norm')
    ratio=row['actual_to_reference_norm_ratio']
    if ref==0:
        if ratio is not None:raise RuntimeError('Undefined zero-reference ratio must be null')
    elif ratio is None or not math.isclose(ratio,row['actual_effective_delta']['joint_norm']/ref,rel_tol=1e-12):
        raise RuntimeError('Effective ratio disagrees with measured norms')
    if job['direction_selector'] not in ['c21_base','c21_perp']:return
    a=row['c21_audit'];n=a['norms'];q=a['norm_g1_sq'];dot=a['dot_g1_c21']
    def finite_values(value):
        if isinstance(value,float) and not math.isfinite(value):raise RuntimeError('Nonfinite residual field')
        if isinstance(value,dict):
            for item in value.values():finite_values(item)
        if isinstance(value,list):
            for item in value:finite_values(item)
    finite_values(a)
    if (a['rule']!=job['direction_selector'] or a['projection_scope']!='all_trainable_LoRA_and_seen_FC'
            or a['reduction_dtype']!='float64' or a['selected_spec_exact'] is not True
            or a['final_norm_matching'] is not False or a['tensor_count']!=len(row['parameter_names'])):
        raise RuntimeError('Wrong global residual selector/projection scope')
    formula='g0+(g2-g1)' if job['direction_selector']=='c21_base' else 'g1+((g2-g1)-a*g1)'
    if a['formula']!=formula or q<0 or any(v<0 for v in n.values()):raise RuntimeError('Wrong residual formula/norms')
    if not math.isclose(q,n['g1']**2,rel_tol=1e-12,abs_tol=1e-20):raise RuntimeError('Invalid g1 squared norm')
    expected=dot/q if q else 0.
    if not math.isclose(a['coefficient_a'],expected,rel_tol=1e-12,abs_tol=1e-15):raise RuntimeError('Wrong global projection coefficient')
    ratio=n['perp']/n['c21'] if n['c21'] else 0.
    denom=n['g1']*n['c21'];orth=abs(a['dot_g1_perp'])/denom if denom else 0.
    if (not math.isclose(a['perp_over_c21'],ratio,rel_tol=1e-12,abs_tol=1e-15)
            or not math.isclose(a['orthogonality_relative_error'],orth,rel_tol=1e-12,abs_tol=1e-15)
            or a['orthogonality_tolerance']!=1e-5 or orth>1e-5 or a['orthogonality_passed'] is not True):
        raise RuntimeError('Residual orthogonality check failed')
    perpendicular_sq=max(0.,n['c21']**2-dot*dot/q) if q else n['c21']**2
    if abs(n['perp']**2-perpendicular_sq)>1e-5*max(n['c21']**2,1e-30):raise RuntimeError('Projected residual norm inconsistent')


def validate_smokes(suite):
    """Fresh GPU identities/own-spec checks, without cross-GPU bitwise claims."""
    verify_run_pins(suite);verify_reuse(suite)
    jobs={job['id']:job for job in suite['jobs']}
    proofs={}; traces={}
    for jid in suite['smoke_jobs']:
        job=jobs[jid];result=inspect_metrics(suite,job)
        receipt=read_json(HERE/'runtime/jobs'/jid/'completed.json')
        if receipt['result']['metrics_sha256']!=result['metrics_sha256']:
            raise RuntimeError('Smoke raw matrix changed')
        proofs[jid]=validate_job_audit(suite,job)
        traces[jid]=[json.loads(line) for line in (HERE/'runtime/jobs'/jid/'direction_audit.jsonl').read_text().splitlines()]
    control=traces['smoke_native_control_r1_k10_s2']
    for jid,rows in traces.items():
        if len(rows)!=len(control):raise RuntimeError('Smoke batch counts differ')
        for a,b in zip(rows,control):
            for key in ['input_sha256','target_sha256','rng_before_batch_sha256','rng_after_batch_sha256',
                        'task','task_step_index','class_offset','batch_samples','parameter_names','lr']:
                if a[key]!=b[key]:raise RuntimeError('Smoke native data/RNG pairing differs: '+key)
        if rows[0]['probe_parameter_sha256'][0]!=control[0]['probe_parameter_sha256'][0]:
            raise RuntimeError('Smoke initial training parameters differ')
    document={'passed':True,'scope':'same-run exact formula/projector/SGD plus native data/RNG identity; cross-GPU trajectories not asserted',
              'validated_at':now(),'suite_sha256':sha256(HERE/'suite.json'),
              'run_pins_sha256':sha256(HERE/'RUN_PINS.json'),'smoke_jobs':proofs,
              'actual_verified_gpu_steps':sum(len(v) for v in traces.values()),
              'paired_batches':len(control),'same_initial_parameters':True,'same_input_target_rng':True,
              'allowed_gpus':suite['allowed_gpus']}
    atomic_json(HERE/'GPU_SMOKE_PROOF.json',document)
    return document
