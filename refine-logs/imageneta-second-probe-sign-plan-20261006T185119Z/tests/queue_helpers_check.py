"""CPU-only helper verification; all synthetic artifacts live in temporary directories."""
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-plan-20261006T185119Z')
sys.path.insert(0, str(ROOT))
import collect_results
import common
import job_worker
import run_queue
import validate_job

files = ['common.py','build_suite.py','job_worker.py','validate_job.py','smoke_pairing.py','run_queue.py','collect_results.py']
checks = {}
for name in files:
    compile((ROOT/name).read_text(), str(ROOT/name), 'exec')
checks['all_helper_scripts_compile'] = True
suite = common.load_suite()
assert len(suite['jobs']) == 29
for job in suite['jobs']:
    if job['kind'] == 'training':
        common.verify_config(job)
checks['suite29_and_configs28'] = True
common.verify_cpu_gate(suite)
checks['pinned_cpu_gate_passed'] = True
for item in suite['reused_results']:
    assert validate_job.validate_reuse(suite,item)['updates'] == 2550
checks['original_validator_and_receipt_both_reuses'] = True
assert len(job_worker.verify_environment(suite)) == 14
checks['actual_environment_versions'] = True
responses = ['0, GPU-a, 0, 0\n2, GPU-b, 100, 0\n3, GPU-c, 0, 0\n','GPU-b, 999\n']
with patch('common.subprocess.check_output', side_effect=responses):
    telemetry = common.query_gpus()
    assert not common.gpu_is_idle(telemetry[2]) and common.gpu_is_idle(telemetry[3])
    assert telemetry[2]['compute_pids'] == [999]
with patch('common.subprocess.check_output', side_effect=RuntimeError('missing telemetry')):
    try: common.query_gpus()
    except RuntimeError: pass
    else: raise AssertionError('Missing telemetry accepted')
checks['compute_pid_block_and_telemetry_fail_closed'] = True
assert common.process_matches(common.proc_identity(os.getpid()))
bad = common.proc_identity(os.getpid()); bad['proc_starttime']='0'
assert not common.process_matches(bad)
checks['pid_start_time_identity'] = True
for sign in (-1,1):
    for norm in (0.,1e-10,1.):
        expected = .05*norm/(norm+1e-12)
        record = dict(rho=.05,eps=1e-12,norms={'g2':norm},second_probe_parameter_audit=dict(
            passed=True,requested_sign=sign,target_radius=.05,expected_parameter_length=expected,
            parameter_length=expected,length_error=0.,absolute_tolerance=max(1e-8,.05*1e-4),
            cosine_tolerance=1e-4,degenerate=expected<=1e-24,
            alignment_cosine=None if norm==0 else float(sign)))
        validate_job.validate_second_probe(record,sign)
        bad=copy.deepcopy(record);bad['second_probe_parameter_audit']['parameter_length']=.1
        try: validate_job.validate_second_probe(bad,sign)
        except AssertionError: pass
        else: raise AssertionError('Wrong displacement accepted')
checks['zero_epsilon_and_both_probe_signs'] = True
assert validate_job.metrics([[80],[65,90]])['AAA_task_macro']==78.75
checks['macro_metric_recomputation'] = True
with tempfile.TemporaryDirectory() as tmp:
    old=Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g1-plus-g3-minus-g2-20261005T170708Z/outputs/smoke_g1_d32_r025_k10_s2')
    here=Path(tmp);out=here/'outputs/x';out.mkdir(parents=True);(here/'trainer').mkdir()
    (here/'trainer/four_probe_train.py').write_text('synthetic tracer')
    summary=json.loads((old/'final_summary.json').read_text())
    matrix=json.loads((old/'task_matrix.json').read_text())
    effective=json.loads((old/'effective_config.json').read_text())
    trace=[json.loads(line) for line in (old/'step_trace.jsonl').read_text().splitlines()]
    job=copy.deepcopy(next(j for j in suite['jobs'] if j['smoke'] and j['radius']==.25 and j['variant']=='N5'))
    job.update(output_dir=str(out),config='synthetic',config_sha256='fixture')
    factors=dict(first_probe_sign=1,second_probe_sign=-1,second_difference_sign=-1,
                 update_base='g0',second_difference_weight=1.)
    summary.update(factors);summary['update_direction']=job['direction']
    summary['input_config_sha256']='fixture'
    summary['trainer_sha256']=common.sha256(here/'trainer/four_probe_train.py')
    summary['checks'].update(second_probe_geometry=True,second_probe_sign=True,
                            second_residual_formula=True,isolated_diagnostics_restored=True)
    summary['counters'].update(training_g3_positive_gradient_passes=0,training_g3_negative_gradient_passes=2,
        diagnostic_snapshots=2,diagnostic_gradient_calls=4,diagnostic_forward_count=4,diagnostic_sgd_updates=0,
        diagnostic_g3_positive_gradient_calls=2,diagnostic_g3_negative_gradient_calls=2)
    effective.update(factors)
    for row in trace:
        norm=row['norms']['g2'];expected=.05*norm/(norm+row['eps'])
        row.update(factors);row.update(update_direction=job['direction'],g3_positive_gradient_calls=0,
            g3_negative_gradient_calls=1,second_residual_formula_exact=True,
            second_probe_parameter_audit=dict(passed=True,requested_sign=-1,target_radius=.05,
                alignment_cosine=-1.,parameter_length=expected,expected_parameter_length=expected,
                absolute_tolerance=max(1e-8,.05*1e-4),cosine_tolerance=1e-4,degenerate=False,length_error=0.))
    diagnostic=[dict(task=index,diagnostic_gradient_calls=2,diagnostic_forward_count=2,
        diagnostic_sgd_updates=0,selected_side_exact=True,state_restoration={'rng_exact':True},
        fisher_calls=0,weight_merge_calls=0,gradient_projection_calls=0,hvp_calls=0,create_graph_calls=0)
        for index in range(2)]
    for name,value in [('task_matrix.json',matrix),('effective_config.json',effective)]:
        (out/name).write_text(json.dumps(value))
    (out/'step_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in trace))
    (out/'side_diagnostics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in diagnostic))
    summary.update(step_trace_sha256=common.sha256(out/'step_trace.jsonl'),
        task_matrix_sha256=common.sha256(out/'task_matrix.json'),
        side_diagnostics_sha256=common.sha256(out/'side_diagnostics.jsonl'))
    (out/'final_summary.json').write_text(json.dumps(summary))
    with patch.object(validate_job,'HERE',here),patch.object(validate_job,'verify_config',return_value={}):
        assert validate_job.validate_job(job)['updates']==2
        summary['second_probe_sign']=1;(out/'final_summary.json').write_text(json.dumps(summary))
        try: validate_job.validate_job(job)
        except AssertionError: pass
        else: raise AssertionError('Wrong sign accepted')
checks['complete_signed_smoke_fixture_and_wrong_sign_rejection'] = True
with tempfile.TemporaryDirectory() as tmp:
    here=Path(tmp);(here/'suite.json').write_text((ROOT/'suite.json').read_text())
    with patch.object(collect_results,'HERE',here):collect_results.main()
    result=json.loads((here/'results.json').read_text())
    assert len(result['rows'])==16 and result['accepted_conditions']==2
    assert result['new_formal_accepted']==0 and result['reused_accepted']==2
    assert all(row['AAA'] is None for row in result['rows'] if not row['accepted'])
checks['collector16_complete_cells_and_partial_exclusion'] = True
report=dict(status='passed',cpu_only=True,checks=checks,
            files={name:common.sha256(ROOT/name) for name in files},suite_sha256=common.sha256(ROOT/'suite.json'),
            scope='CPU helper tests and old receipt revalidation only; no GPU/queue launch or campaign artifact mutation.')
path=Path('/tmp/second_probe_queue_helpers_validation_20261006.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],checks=len(checks),report=str(path))))
