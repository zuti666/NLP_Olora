#!/usr/bin/env python3
"""Materialize an isolated manifest; --freeze is an explicit final CPU-only step."""
import argparse
import copy
import json
from pathlib import Path
from common import HERE as H, atomic_json, load_suite, sha256, verify_config, verify_source

OLD = H.parent / 'g012-stage2-revalidation-20261003/ordinary_stage2'
PRIOR = H.parent / 'g012-stage2-revalidation-20261003/MATH_GRADIENT_ACCEPTANCE.json'
PARENT = H.parent / 'g2-central-location-execution-20261003T102355Z'
FORMAL = [f'G2-CD0{i}_O_{b}_natural_s2' for i,b in enumerate(('P','C','CD'),1)]
PAIRING = 'G2-CD_smoke_pairing'
DIAGNOSTIC = 'G2-CD_location_diagnostics'


def materialize():
    if (H/'suite.json').exists() or (H/'runtime/queue_state.json').exists() or (H/'RUN_PINS.json').exists():
        raise RuntimeError('Refusing to overwrite an already materialized, frozen or started campaign')
    old=json.loads((OLD/'suite.json').read_text())
    raw=json.loads((OLD/'configs/S2-B01_Sec_math_natural_s2.json').read_text())
    env=copy.deepcopy(old['environment'])
    for k in ('TMPDIR','TMP','TEMP'):env[k]='/tmp/'+H.name
    suite=dict(schema_version=2,run_id=H.name,run_dir=str(H),
        execution_hostname=old['execution_hostname'],python=old['python'],source_root=old['source_root'],
        source_manifest=str(H/'source_manifest.json'),source_manifest_sha256=sha256(H/'source_manifest.json'),
        allowed_gpus=[0,1],max_parallel=2,poll_seconds=15,
        gpu_lock_dir='/tmp/olora_gamfs_gpu_locks',gpu_free_threshold_mib=500,
        gpu_free_util_below_percent=10,expected_tasks=10,environment=env,
        ordinary_gradient_receipt=str(PRIOR),ordinary_gradient_receipt_sha256=sha256(PRIOR),
        no_external_notifications=True,stage2_backend='math',
        scope='P/C/CD g2 natural seed2 only; four ordinary CE probes per update; isolated replay gates',
        failure_policy='halt new launches on any failure; drain active jobs; no retry, preemption or stale override',
        parent_run=str(PARENT),jobs=[])
    smokes=['smoke_'+j for j in FORMAL]
    for smoke in (True,False):
        for ident,branch in zip(FORMAL,('P','C','CD')):
            if smoke:ident='smoke_'+ident
            output=H/'outputs'/ident
            cfg=copy.deepcopy(raw)
            cfg.update(prefix=ident,outputs_root=str(output/'native'),probe_branch=branch,
                update_direction='g2',step_mode='natural',attention_backend='math',
                component_rule='u0',component_scale='natural',in_task_path_normalize_snapshots=False)
            path=H/'configs'/(ident+'.json');atomic_json(path,cfg)
            command=[suite['python'],str(H/'trainer/four_probe_train.py'),'--source',suite['source_root'],
                '--config',str(path),'--output-dir',str(output)]
            if smoke:command+=['--max-tasks','2','--max-epochs','1','--max-batches','1']
            suite['jobs'].append(dict(id=ident,kind='training',phase='smoke' if smoke else 'formal',
                branch=branch,direction='g2',mode='natural',backend='math',smoke=smoke,gpu_required=True,
                command=command,config=str(path),config_sha256=sha256(path),output_dir=str(output),
                depends_on=[] if smoke else smokes+[PAIRING],expected_tasks=2 if smoke else 10,
                expected_epochs_per_task=1 if smoke else 50,expected_steps=2 if smoke else 2550,
                expected_training_ce=8 if smoke else 10200,expected_isolated_replay_ce=16 if smoke else 80))
    pairing=dict(id=PAIRING,kind='pairing',phase='smoke_pairing',smoke=False,gpu_required=False,
        output_dir=str(H/'outputs'/PAIRING),report_file='smoke_pairing_report.json',depends_on=smokes,
        command=[suite['python'],str(H/'verify_pairing.py'),'--run-dir',str(H),
            '--output-dir',str(H/'outputs'/PAIRING)])
    suite['jobs'].insert(3,pairing)
    output=H/'outputs'/DIAGNOSTIC
    suite['jobs'].append(dict(id=DIAGNOSTIC,kind='diagnostic',phase='posthoc',smoke=False,gpu_required=True,
        backend='math',expected_tasks=3,expected_ordinary_ce=108,expected_state_batches=18,
        report_file='g2_location_report.json',output_dir=str(output),depends_on=FORMAL,
        command=[suite['python'],str(H/'trainer/g2_location_diagnostics.py'),'--source',suite['source_root'],
            '--run-dir',str(H),'--output-dir',str(output)]))
    atomic_json(H/'suite.json',suite)
    load_suite(H/'suite.json')
    print(json.dumps(dict(status='materialized',smokes=3,formal=3,cpu_pairing=1,gpu_posthoc=1,launched=False)))


def freeze():
    if (H/'runtime/queue_state.json').exists():
        raise RuntimeError('Cannot repin a started queue')
    if (H/'RUN_PINS.json').exists():
        raise RuntimeError('Already frozen; do not silently replace immutable execution pins')
    suite=load_suite(H/'suite.json');verify_source(suite)
    for job in suite['jobs']:
        if not Path(job['command'][1]).is_file():raise FileNotFoundError(job['command'][1])
        if job['kind']=='training':verify_config(job)
    for name in ('gates/INPUT_ASSETS.json','trainer/G2_CD_CPU_GATE.json',
                 'gates/G2_LOCATION_CPU_GATE.json','gates/TRAINER_ENTRYPOINTS.json'):
        if json.loads((H/name).read_text())['status']!='passed':
            raise RuntimeError('Preflight did not pass: '+name)
    cpu=json.loads((H/'trainer/G2_CD_CPU_GATE.json').read_text())
    for field,name in (('direction_step_sha256','direction_step.py'),('trainer_sha256','four_probe_train.py'),('tests_sha256','test_g2_cd_cpu.py')):
        if cpu[field]!=sha256(H/'trainer'/name):
            raise RuntimeError('CPU gate predates final code: '+name)
    offline=json.loads((H/'gates/G2_LOCATION_CPU_GATE.json').read_text())
    if not offline['cpu_only'] or offline['gpu_acceptance'] or offline['tests_run']!=9 or offline['errors'] or offline['failures']:
        raise RuntimeError('Incomplete offline CPU gate')
    for field,name in (('implementation_sha256','g2_location_diagnostics.py'),('tests_sha256','test_g2_location_cpu.py')):
        if offline[field]!=sha256(H/'trainer'/name):
            raise RuntimeError('Offline CPU gate predates final code: '+name)
    entry=json.loads((H/'gates/TRAINER_ENTRYPOINTS.json').read_text())
    if entry['trainer_sha256']!=sha256(H/'trainer/four_probe_train.py') or len(entry['jobs'])!=6 or any(row['returncode'] for row in entry['jobs']):
        raise RuntimeError('Actual trainer entrypoint validation is incomplete')
    # All engine, lifecycle, diagnostics, CPU evidence and local input/environment
    # receipts are pinned together. Runtime/results/output artifacts are excluded.
    names={'suite.json','source_manifest.json','RUN.md','GPU_ROUTING_CHANGE.json'}
    names.update(str(p.relative_to(H)) for p in H.glob('*.py'))
    names.update(str(p.relative_to(H)) for p in (H/'configs').glob('*.json'))
    names.update(str(p.relative_to(H)) for p in (H/'trainer').glob('*.py'))
    names.update(str(p.relative_to(H)) for p in (H/'trainer').glob('*.json'))
    names.update(str(p.relative_to(H)) for p in (H/'gates').glob('*.json') if p.name!='DOC_CHECK.json')
    for name in ('environment_spec.json','ENVIRONMENT.md','EXPERIMENT_PLAN.md'):
        if (H/name).is_file():names.add(name)
    atomic_json(H/'RUN_PINS.json',dict(schema_version=2,run_id=suite['run_id'],
        files={name:sha256(H/name) for name in sorted(names)},
        external_files={suite['ordinary_gradient_receipt']:sha256(suite['ordinary_gradient_receipt'])},
        source_files_count=len(json.loads((H/'source_manifest.json').read_text())['files']),
        immutable_after_launch=True,launched=False))
    print(json.dumps(dict(status='frozen',files=len(names),launched=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze',action='store_true')
    action.add_argument('--materialize',action='store_true')
    args=parser.parse_args();freeze() if args.freeze else materialize()
