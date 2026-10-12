"""Register four new conditions, GPU smokes, and a pairing barrier; no training launch."""
import argparse
import csv
import os
from pathlib import Path
from common import HERE, PROTOCOL, atomic_json, expected_budget, now, read_json, sha256, verify_cpu_gate

OLD = Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-deterministic-20261007T071400Z')
PLAN = Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-center-and-joint-backward-plan-20261007T134842Z')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execution-hostname', default='cudahpc40.cvc.uab.es')
    ap.add_argument('--execution-pid-namespace', default='pid:[4026531836]')
    args = ap.parse_args()
    if (HERE/'runtime/queue_state.json').exists():
        raise RuntimeError('Refuse to rebuild a started suite')
    old = read_json(OLD/'suite.json')
    cpu_gate = dict(path=str(HERE/'trainer/CPU_GATE.json'),sha256=sha256(HERE/'trainer/CPU_GATE.json'))
    verify_cpu_gate(dict(cpu_gate=cpu_gate))
    jobs, settings = [], []
    for smoke in (True,False):
        for variant in ('CPLUS','CMINUS','BF','BB'):
            config = read_json(PLAN/'configs'/f'{variant}.draft.json')
            condition = f'{variant}_r1_k10_s2'
            jobid = ('smoke_' if smoke else 'formal_')+condition
            output = HERE/'outputs'/jobid
            config.update(prefix=jobid,outputs_root=str(output/'native'),component_rule=config['update_direction'],registered=True,plan_only=False,
                          automatic_start=True,plan_status='authorized_queued_pending_gpu_gates',
                          protocol=PROTOCOL,proposed_protocol=PROTOCOL,
                          required_runner=str(HERE/'trainer/four_probe_train.py'),
                          optimizer_tag_override='center_joint_backward_explicit_sgd_v3',
                          compute_g3_probe=True,second_probe_side_diagnostics=False,
                          diagnostic_protocol='training_context_only_no_extra_gradient_v3')
            cfgpath=HERE/'configs'/f'{jobid}.json'
            budget=expected_budget(config)
            steps=2 if smoke else 2550
            config.update(expected_sgd_updates=steps,expected_training_ce=steps*budget['train_ce_per_update'])
            atomic_json(cfgpath,config)
            command=[old['python'],str(HERE/'trainer/four_probe_train.py'),
                     '--source',old['source_root'],'--config',str(cfgpath),'--output-dir',str(output)]
            if smoke:
                command+=['--max-tasks','2','--max-epochs','1','--max-batches','1']
            factors={k:config[k] for k in ('first_location_estimator','second_residual_estimator','residual_coefficient','update_base')}
            job=dict(id=jobid,condition_id=condition,variant=variant,kind='training',smoke=smoke,
                     radius=1.,rho=.05,seed=2,rank=10,branch='P' if config['first_location_estimator']=='forward' else 'C',
                     location_family='F' if config['first_location_estimator']=='forward' else 'B',
                     direction=config['update_direction'],formula=config['update_direction'],**factors,
                     source_root=old['source_root'],config=str(cfgpath),config_sha256=sha256(cfgpath),
                     output_dir=str(output),command=command,depends_on=[] if smoke else ['smoke_pairing_gate'],
                     expected_tasks=2 if smoke else 10,expected_epochs_per_task=1 if smoke else 50,
                     expected_steps=steps,expected_training_ce=steps*budget['train_ce_per_update'],
                     expected_diagnostic_gradient_calls=0,training_ce_per_update=budget['train_ce_per_update'])
            jobs.append(job)
            if not smoke:
                settings.append({k:job[k] for k in ('id','variant','branch','radius','rho','seed','rank',
                                                   'first_location_estimator','second_residual_estimator',
                                                   'residual_coefficient','update_base','formula','expected_training_ce','config')})
    smokes=[j['id'] for j in jobs if j['smoke']]
    jobs.insert(4,dict(id='smoke_pairing_gate',kind='pairing',smoke=False,
                      output_dir=str(HERE/'gates/smoke_pairing'),report_file='smoke_pairing.json',
                      command=[old['python'],str(HERE/'smoke_pairing.py')],depends_on=smokes))
    references=[]
    for label,variant in [('FF','B0'),('FB','N5')]:
        job=next(j for j in old['jobs'] if j['kind']=='training' and not j['smoke'] and j['variant']==variant and j['radius']==1.)
        out=Path(job['output_dir'])
        paths=dict(summary=str(out/'final_summary.json'),matrix=str(out/'task_matrix.json'),
                   receipt=str(OLD/'runtime/jobs'/job['id']/'completed.json'),config=job['config'],
                   trace=str(out/'step_trace.jsonl'))
        references.append(dict(variant=label,job_id=job['id'],output_dir=job['output_dir'],
                               source_campaign=str(OLD),source_protocol=old['protocol'],
                               evidence_paths=paths,evidence_sha256={k:sha256(v) for k,v in paths.items()}))
    environment=dict(read_json(HERE/'environment_spec.json')['env'])
    environment.update({k:'/tmp/'+HERE.name for k in ('TMPDIR','TMP','TEMP')})
    suite={k:v for k,v in old.items() if k not in ('jobs','cpu_gate','reused_results','upstream_campaign')}
    suite.update(schema_version=3,run_id=HERE.name,run_dir=str(HERE),generated_at=now(),
                 execution_hostname=args.execution_hostname,execution_pid_namespace=args.execution_pid_namespace,
                 source_manifest=str(HERE/'source_manifest.json'),source_manifest_sha256=sha256(HERE/'source_manifest.json'),
                 environment_spec=str(HERE/'environment_spec.json'),environment_spec_sha256=sha256(HERE/'environment_spec.json'),
                 environment=environment,protocol=PROTOCOL,cpu_gate=cpu_gate,
                 diagnostic_protocol='training_context_only_no_extra_gradient_v3',
                 diagnostic_gradients_per_task=0,diagnostic_accounting=dict(per_task_new_gradient_calls=0,per_task_new_forward_calls=0),
                 reused_results=references,jobs=jobs,
                 upstream_campaign=dict(suite_path=str(OLD/'suite.json'),suite_sha256=sha256(OLD/'suite.json'),
                                        state_path=str(OLD/'runtime/queue_state.json'),required_formal_count=16,
                                        policy='wait_for_successful_existing_campaign_then_idle_gpu'))
    atomic_json(HERE/'suite.json',suite)
    with (HERE/'EXPERIMENT_SETTINGS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(settings[0]));writer.writeheader();writer.writerows(settings)
    print('Registered 9 jobs: 4 GPU smokes +1 CPU pairing +4 formal; 2 reference baselines; upstream FIFO')

if __name__=='__main__':
    main()
