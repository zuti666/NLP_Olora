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
    for module in ['residual_selection','effective_step','step_reference']:
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
    if summary['expected_radius'] != job['outer_radius']:
        raise RuntimeError('Training probe radius differs')
    reference_rows = None
    reference_identity = None
    matching_errors = []
    if job['direction_selector'] == 'gmatch':
        reference_job = next(j for j in suite['jobs'] if j['id'] == job['reference_job'])
        if reference_job['arm'] != 'S' or reference_job['phase'] != job['phase']:
            raise RuntimeError('Wrong reference arm or phase')
        validate_job_audit(suite, reference_job)
        refdir = HERE / 'runtime/jobs' / reference_job['id']
        receipt = read_json(refdir/'completed.json')
        if (receipt['config_sha256'] != reference_job['config_sha256']
                or receipt['suite_sha256'] != sha256(HERE/'suite.json')
                or receipt['run_pins_sha256'] != sha256(HERE/'RUN_PINS.json')):
            raise RuntimeError('Reference is not a completed frozen S from this campaign')
        reference_identity = {'reference_job_id':reference_job['id'], 'reference_phase':job['phase'],
            'reference_trace_sha256':sha256(refdir/'direction_audit.jsonl'),
            'reference_summary_sha256':sha256(refdir/'direction_audit_summary.json'),
            'reference_receipt_sha256':sha256(refdir/'completed.json'),
            'reference_config_sha256':sha256(refdir/'config.json')}
        reference_rows = [json.loads(line) for line in (refdir/'direction_audit.jsonl').read_text().splitlines()]
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
            if reference_rows is not None:
                ref = reference_rows[count]
                for key in ['step_index','task','task_step_index','seen_classes','class_offset','batch_samples',
                            'input_sha256','target_sha256','rng_before_batch_sha256','rng_after_batch_sha256',
                            'parameter_names','parameter_shapes','lr_nominal']:
                    if row[key] != ref[key]:
                        raise RuntimeError('Gmatch and independent S batch identity differs: '+key)
                if row['reference_effective_delta'] != ref['actual_effective_delta']:
                    raise RuntimeError('Gmatch target was not taken from independent S actual update')
                for key,value in reference_identity.items():
                    if row[key] != value: raise RuntimeError('Independent S evidence changed: '+key)
                matching_errors.append(row['actual_match_relative_error'])
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
    if reference_rows is not None:
        result.update(independent_reference=reference_identity, matched_steps=len(matching_errors),
                      matching_relative_error_max=max(matching_errors),
                      matching_relative_error_mean=sum(matching_errors)/len(matching_errors))
    if require_receipt:
        receipt = read_json(run/'completed.json')
        if receipt['direction_proof'] != result:
            raise RuntimeError('Completion receipt does not bind the current execution/checkpoints')
    return result


def validate_new_step_record(row, job):
    matched_arm = job['direction_selector'] == 'gmatch'
    if row['schema_version'] != 3 or row['final_norm_matching'] is not matched_arm:
        raise RuntimeError('Unexpected update matching intervention')
    if row['component_scale'] != ('independent_S_matched' if matched_arm else 'natural'):
        raise RuntimeError('Unexpected component scale')
    nominal,scale = row['lr_nominal'],row['step_scale']
    if not all(math.isfinite(v) and v>0 for v in [nominal,scale]) or any(v!=nominal for v in row['lr']):
        raise RuntimeError('Nonfinite/inconsistent SGD scalar')
    if not math.isclose(row['scale_over_nominal_lr'],scale/nominal,rel_tol=1e-12):
        raise RuntimeError('SGD scalar ratio disagrees')
    if not matched_arm and (scale!=nominal or row['scale_over_nominal_lr']!=1.):
        raise RuntimeError('Natural arm changed its nominal LR')
    task=row['task'];epochs=50 if job['phase']=='formal' else 1
    batches=job['expected_task_steps'][str(task)]//epochs
    epoch=row['task_step_index']//batches
    peak=(.04 if task==0 else .02)*job['lr_multiplier']
    expected_lr=peak*.1+(peak-peak*.1)*(1+math.cos(math.pi*epoch/epochs))/2
    if not math.isclose(nominal,expected_lr,rel_tol=1e-12,abs_tol=1e-15):
        raise RuntimeError('Nominal cosine schedule does not preserve the declared multiplier')
    if (row['reference_counterfactual_state_unchanged'] is not True
            or row['nominal_learning_rates_restored'] is not True
            or row['effective_step_measurement_only'] is not (not matched_arm)
            or row['effective_geometry_scope']!='unscaled_LoRA_BA_and_all_seen_FC'
            or row['same_state_g1_reference_definition']!='same_candidate_state_same_batch_native_projected_g1_after_native_unperturb_actual_FP32_SGD'):
        raise RuntimeError('Effective geometry point/scope or optimizer restoration differs')
    def geometry(values):
        if any(not math.isfinite(values[k]) or values[k]<0 for k in ['joint_norm','lora_norm','fc_norm','parameter_norm']):
            raise RuntimeError('Nonfinite effective update geometry')
        if not math.isclose(values['joint_norm']**2,values['lora_norm']**2+values['fc_norm']**2,rel_tol=1e-10,abs_tol=1e-20):
            raise RuntimeError('Effective step components disagree')
    geometry(row['actual_effective_delta']);geometry(row['same_state_g1_effective_delta'])
    if not math.isclose(row['parameter_update_l2'],row['actual_effective_delta']['parameter_norm'],rel_tol=1e-10,abs_tol=1e-20):
        raise RuntimeError('Actual parameter norm alias disagrees')
    if row['matching_tolerance'] != 1e-3:
        raise RuntimeError('Matching tolerance changed')
    if not matched_arm:
        if row['reference_kind']!='none' or any(row.get(k) is not None for k in
                ['reference_definition','reference_effective_delta','reference_target_norm','actual_to_reference_norm_ratio',
                 'actual_match_relative_error','matched','matching_solver','reference_job_id']):
            raise RuntimeError('Natural arm falsely reports an independent target')
        return
    if (row['reference_kind']!='independent_completed_S_trace'
            or row['reference_definition']!='independent_completed_S_actual_FP32_SGD_at_aligned_native_batch'
            or row['reference_job_id']!=job['reference_job'] or row['reference_phase']!=job['phase']
            or row['reference_batch_identity_exact'] is not True
            or row['reference_step_index']!=row['step_index']
            or row['reference_task_step_index']!=row['task_step_index']):
        raise RuntimeError('Wrong independent S reference identity')
    geometry(row['reference_effective_delta'])
    target=row['reference_target_norm'];actual=row['actual_effective_delta']['joint_norm']
    if not math.isfinite(target) or target<=0 or target!=row['reference_effective_delta']['joint_norm']:
        raise RuntimeError('Invalid independent S target')
    error=abs(actual-target)/target
    if (row['matched'] is not True or error>1e-3
            or not math.isclose(row['actual_match_relative_error'],error,rel_tol=1e-12,abs_tol=1e-15)
            or not math.isclose(row['actual_to_reference_norm_ratio'],actual/target,rel_tol=1e-12)):
        raise RuntimeError('Actual Gmatch effective step misses independent S target')
    solver=row['matching_solver']
    geometry(solver['fp32_predicted_effective_delta'])
    pred=abs(solver['fp32_predicted_effective_delta']['joint_norm']-target)/target
    if (solver['matching_tolerance']!=1e-3 or solver['no_live_trial_updates'] is not True
            or solver['fp32_candidate_evaluations']<1 or pred>1e-3
            or not math.isclose(solver['fp32_predicted_relative_error'],pred,rel_tol=1e-12,abs_tol=1e-15)
            or not all(math.isclose(solver['fp32_predicted_effective_delta'][k],row['actual_effective_delta'][k],
                                    rel_tol=1e-12,abs_tol=1e-20)
                       for k in ['joint_norm','lora_norm','fc_norm','parameter_norm'])):
        raise RuntimeError('FP32 precheck disagrees with real update')


def validate_smokes(suite):
    """Fresh native-budget proofs on GPU1, including independent-S matching."""
    verify_run_pins(suite);verify_reuse(suite)
    jobs={j['id']:j for j in suite['jobs']};proofs={};traces={}
    for jid in suite['smoke_jobs']:
        job=jobs[jid];result=inspect_metrics(suite,job)
        receipt=read_json(HERE/'runtime/jobs'/jid/'completed.json')
        if receipt['result']['metrics_sha256']!=result['metrics_sha256']:
            raise RuntimeError('Smoke raw matrix changed')
        proofs[jid]=validate_job_audit(suite,job)
        traces[jid]=[json.loads(line) for line in (HERE/'runtime/jobs'/jid/'direction_audit.jsonl').read_text().splitlines()]
    control=traces['smoke_S_r1_k10_s2']
    for jid,rows in traces.items():
        if len(rows)!=len(control):raise RuntimeError('Smoke batch counts differ')
        for a,b in zip(rows,control):
            for key in ['input_sha256','target_sha256','rng_before_batch_sha256','rng_after_batch_sha256',
                        'task','task_step_index','class_offset','seen_classes','batch_samples',
                        'parameter_names','parameter_shapes']:
                if a[key]!=b[key]:raise RuntimeError('Smoke native batch/RNG pairing differs: '+key)
        if rows[0]['probe_parameter_sha256'][0]!=control[0]['probe_parameter_sha256'][0]:
            raise RuntimeError('Smoke initial model parameters differ')
    summed=traces['smoke_Sum_r1_k10_s2'][0]['actual_effective_delta']['joint_norm']
    averaged=traces['smoke_Mean2_r1_k10_s2'][0]['actual_effective_delta']['joint_norm']
    first_norm_difference=abs(summed-averaged)/summed
    if not math.isfinite(first_norm_difference) or first_norm_difference>1e-3:
        raise RuntimeError('Initial Sum/Mean2 FP32 effective norm consistency failed')
    document={'passed':True,'scope':'own exact formula/projector/SGD, native data/RNG pairing, independent-S effective length matching; no full cross-run trajectory equality claim',
              'validated_at':now(),'suite_sha256':sha256(HERE/'suite.json'),
              'run_pins_sha256':sha256(HERE/'RUN_PINS.json'),'smoke_jobs':proofs,
              'actual_verified_gpu_steps':sum(len(v) for v in traces.values()),
              'paired_batches':len(control),'same_initial_parameters':True,'same_input_target_rng':True,
              'initial_Sum_Mean2_norm_relative_difference':first_norm_difference,
              'allowed_gpus':suite['allowed_gpus']}
    atomic_json(HERE/'GPU_SMOKE_PROOF.json',document)
    return document
