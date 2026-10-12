"""Read-only GPU-smoke acceptance audit; writes only this final gate report.

Run in the actual host PID namespace; CPU-only result parsing and checkpoint
tensor loads. No GPU allocations, queue mutation, training launch or stop.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from common import sha256,bad_loss_lines
from validate_job import validate_job

suite=json.loads((ROOT/'suite.json').read_text())
state=json.loads((ROOT/'runtime/queue_state.json').read_text())
assert socket.gethostname()==suite['execution_hostname']
assert os.readlink('/proc/self/ns/pid')==suite['execution_pid_namespace']
failed={k:v for k,v in state['jobs'].items() if v['status'] in ('failed','blocked','cancelled')}
if failed:raise AssertionError('Failed jobs: '+repr(failed))
if state['status']!='completed' or any(v['status']!='completed' for v in state['jobs'].values()):
    print(json.dumps({'status':'pending','counts':state.get('counts'),'no_errors':True}));sys.exit(3)
training=[j for j in suite['jobs'] if j['kind']=='training']
assert len(training)==8 and len(suite['jobs'])==9 and all(j['smoke'] for j in training)
assert not any(j.get('phase')=='formal' for j in suite['jobs'])
pins=json.loads((ROOT/'RUN_PINS.json').read_text())['files']
for name,digest in pins.items():assert sha256(ROOT/name)==digest,name
source_manifest=json.loads(Path(suite['source_manifest']).read_text())['files']
for name,digest in source_manifest.items():assert sha256(Path(suite['source_root'])/name)==digest,name

evidence=[];total_updates=total_gradients=total_diagnostics=total_history=0
for job in suite['jobs']:
    job_id=job['id'];entry=state['jobs'][job_id]
    assert entry['gpu'] in (2,3,4,5,6)
    accepted=validate_job(job)
    runtime=ROOT/'runtime/jobs'/job_id
    exit_receipt=json.loads((runtime/'exit.json').read_text())
    assert exit_receipt['returncode']==0
    assert not bad_loss_lines(runtime/'console.log')
    if job['kind']!='training':
        evidence.append(dict(job=job_id,acceptance=accepted,exit_receipt=exit_receipt));continue
    witness=json.loads((runtime/'cuda_witness.json').read_text())
    assert witness['status']=='passed' and witness['cuda_visible_devices']==entry['gpu_uuid'] and witness['seed']==2
    assert witness['cuda']=='12.1' and witness['tf32_matmul'] is False and witness['tf32_cudnn'] is False
    out=Path(job['output_dir'])
    summary=json.loads((out/'final_summary.json').read_text())
    trace=[json.loads(x) for x in (out/'step_trace.jsonl').read_text().splitlines()]
    receipt=summary['dataset_partition_receipt']
    assert receipt['nb_classes']==200 and receipt['full_train_samples']==5981 and receipt['full_test_samples']==1519
    expected_reads={'sgd':1,'current_g2':3,'history_fusion':2,'history_g2':3}[job['method']]
    expected_updates=2 if job['layout']=='single200' else 4
    assert len(trace)==expected_updates
    assert all(r['ordinary_gradient_calls']==expected_reads and not r['fallback'] for r in trace)
    assert all(r['actual_model_forward_count']==expected_reads+1 and r['sgd_updates']==1 and r['diagnostic_loss_forward_count']==1 for r in trace)
    assert all(math.isfinite(r['loss0']) and math.isfinite(r['post_update_ce']) for r in trace)
    if job['layout']=='single200':
        assert receipt['full_task_increments']==[200]
        assert all(r['class_slice']==[0,200] and r['fc_shape'][0]==200 and r['seen_classes']==200 for r in trace)
    else:
        assert receipt['full_task_increments']==[20]*10
        assert {tuple(r['class_slice']) for r in trace}=={(0,20),(20,40)}
        assert [r['fc_shape'][0] for r in trace]==[20,20,40,40]
        assert not trace[2]['history_used'] and trace[2]['previous_batch_identity'] is None
    for index,r in enumerate(trace):
        assert r['history_source']==job['history_source']
        if r['history_used']:
            prev=trace[index-1]
            assert prev['task']==r['task']
            expected_hash=prev['g0_sha256'] if job['history_source']=='base' else prev['final_gradient_sha256']
            assert r['history_incoming_vector_sha256']==expected_hash
            assert r['history_cache_actual_displacement_sha256']==prev['actual_step_displacement_sha256']
            if job['method']=='history_g2':
                assert prev['final_gradient_sha256']==prev['g2_sha256']
                assert prev['final_gradient_sha256']!=prev['g0_sha256']
                assert r['history_incoming_vector_norm']==prev['gradient_norms']['final']
    g=sum(r['ordinary_gradient_calls'] for r in trace);d=sum(r['diagnostic_loss_forward_count'] for r in trace)
    total_updates+=len(trace);total_gradients+=g;total_diagnostics+=d;total_history+=sum(r['history_used'] for r in trace)
    evidence.append(dict(job=job_id,method=job['method'],layout=job['layout'],gpu=entry['gpu'],gpu_uuid=entry['gpu_uuid'],
                         completed_at_utc=entry['completed_at'],updates=len(trace),gradient_reads=g,diagnostic_forwards=d,
                         history_used_steps=sum(r['history_used'] for r in trace),
                         head_widths=[r['fc_shape'][0] for r in trace],acceptance=accepted,
                         witness_sha256=sha256(runtime/'cuda_witness.json'),exit_receipt=exit_receipt))
assert (total_updates,total_gradients,total_diagnostics,total_history)==(24,54,24,6)
pair=json.loads((ROOT/'gates/smoke_pairing/smoke_pairing.json').read_text())
assert pair['status']=='passed' and pair['smoke_count']==8 and all(pair['checks'].values())
snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
report=dict(status='passed',audited_at_utc=datetime.now(timezone.utc).isoformat(),actual_host=socket.gethostname(),
            pid_namespace=os.readlink('/proc/self/ns/pid'),cpu_only_parsing=True,new_gpu_work_started_by_auditor=False,
            source_files_verified=len(source_manifest),execution_pins_verified=len(pins),
            registered_formal_training=0,completed_formal_training=0,completed_smokes=8,completed_pairing_gate=1,
            totals=dict(actual_sgd_updates=24,new_gradient_reads=54,passive_diagnostic_forwards=24,total_training_and_diagnostic_forwards=78,history_used_steps=6),
            pairing_checks=pair['checks'],pairing_report_sha256=sha256(ROOT/'gates/smoke_pairing/smoke_pairing.json'),
            forbidden_work_zero=True,all_losses_finite=True,task_history_reset_verified=True,
            full200_single_task_real_receipt_head_ce_verified=True,history_base_vs_full_final_cache_verified=True,
            evidence=evidence,post_completion_gpu_snapshot=snapshot,post_completion_compute_apps=apps,
            limitation='Bounded CUDA implementation checks only (single200 two updates, cl10 four updates each); no new 50-epoch method result or learning-performance conclusion.')
(ROOT/'gates/GPU_IMPLEMENTATION_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'passed','smokes':8,'pairing':1,'totals':report['totals'],'registered_formal_training':0},indent=2))
