#!/usr/bin/env python3
"""Materialize four uniform three-probe conditions for seeds 0 and 1; freezing is a separate explicit CPU-only action."""
import argparse
import copy
import json
from pathlib import Path
from common import HERE as H, atomic_json, load_suite, sha256, verify_config, verify_source, verify_preflight, DIRECTIONS, PAIRING, JOB_SEEDS, SEEDS, PARENT_RUN

PARENT = PARENT_RUN


def materialize():
    if any((H/p).exists() for p in ('suite.json','runtime/queue_state.json','RUN_PINS.json')):
        raise RuntimeError('Refusing to overwrite an already materialized, frozen or started campaign')
    old=json.loads((PARENT/'suite.json').read_text())
    env=copy.deepcopy(old['environment'])
    for k in ('TMPDIR','TMP','TEMP'):env[k]='/tmp/oe_s01_20261012T104100Z'
    suite=dict(schema_version=4,run_id=H.name,run_dir=str(H),
        execution_hostname=old['execution_hostname'],python=old['python'],source_root=old['source_root'],
        source_manifest=str(H/'source_manifest.json'),source_manifest_sha256=sha256(H/'source_manifest.json'),
        allowed_gpus=[4,5,6],max_parallel=3,poll_seconds=15,
        gpu_lock_dir='/tmp/olora_gamfs_gpu_locks',gpu_free_threshold_mib=500,
        gpu_free_util_below_percent=10,expected_tasks=10,environment=env,
        ordinary_gradient_receipt=old['ordinary_gradient_receipt'],
        ordinary_gradient_receipt_sha256=old['ordinary_gradient_receipt_sha256'],
        no_external_notifications=True,stage2_backend='math',method_name='odd_even_local_v1',
        outer_radius=None,seeds=list(SEEDS),historical_seed=2,scope='Four conditions x supplemental seeds 0/1; eight smokes, CPU four-way pairing separately within each seed, eight formal runs; 3 CE/update and 6 isolated replay CE/task',
        failure_policy='halt new launches on any failure; drain active jobs; no retry, preemption or stale override',
        parent_run=str(PARENT),jobs=[])
    formal=[ident for ident,_ in DIRECTIONS];smokes=['smoke_'+ident for ident in formal]
    for smoke in (True,False):
        if not smoke:
            output=H/'outputs'/PAIRING
            suite['jobs'].append(dict(id=PAIRING,kind='pairing',phase='pairing',smoke=False,gpu_required=False,
                depends_on=smokes,expected_tasks=2,report_file='smoke_pairing_report.json',output_dir=str(output),
                command=[suite['python'],str(H/'verify_pairing.py'),'--run-dir',str(H),'--output-dir',str(output)]))
        for formal_id,direction in DIRECTIONS:
            ident=('smoke_' if smoke else '')+formal_id
            output=H/'outputs'/ident
            seed=JOB_SEEDS[formal_id]
            parent_config=PARENT/'configs'/(formal_id.rsplit('_s',1)[0]+'_s2.json')
            cfg=json.loads(parent_config.read_text())
            cfg.update(seed=[seed],prefix=ident,outputs_root=str(output/'native'))
            path=H/'configs'/(ident+'.json');atomic_json(path,cfg)
            command=[suite['python'],str(H/'trainer/odd_even_train.py'),'--source',suite['source_root'],
                     '--config',str(path),'--output-dir',str(output)]
            if smoke:command+=['--max-tasks','2','--max-epochs','1','--max-batches','1']
            deps=[] if smoke else [PAIRING]
            suite['jobs'].append(dict(id=ident,kind='training',phase='smoke' if smoke else 'formal',
                branch='local_symmetric',direction=direction,seed=seed,
                parent_config=str(parent_config),parent_config_sha256=sha256(parent_config),mode='natural',backend='math',smoke=smoke,gpu_required=True,
                command=command,config=str(path),config_sha256=sha256(path),output_dir=str(output),
                depends_on=deps,expected_tasks=2 if smoke else 10,
                expected_epochs_per_task=1 if smoke else 50,expected_steps=2 if smoke else 2550,
                expected_training_ce=6 if smoke else 7650,expected_isolated_replay_ce=12 if smoke else 60))
    atomic_json(H/'suite.json',suite)
    load_suite(H/'suite.json')
    for job in suite['jobs']:
        if job['kind']=='training':verify_config(job)
    print(json.dumps(dict(status='materialized',smokes=8,pairing=1,formal=8,allowed_gpus=[4,5,6],max_parallel=3,launched=False)))


def freeze():
    if (H/'runtime/queue_state.json').exists():raise RuntimeError('Cannot repin a started queue')
    if (H/'RUN_PINS.json').exists():raise RuntimeError('Already frozen; never replace execution pins')
    suite=load_suite(H/'suite.json');verify_source(suite)
    for job in suite['jobs']:
        if not Path(job['command'][1]).is_file():raise FileNotFoundError(job['command'][1])
        if job['kind']=='training':verify_config(job)
    verify_preflight(suite)
    names={'suite.json','source_manifest.json','environment_spec.json','RUN.md','ENVIRONMENT.md','EXPERIMENT_PLAN.md'}
    names.update(str(p.relative_to(H)) for p in H.glob('*.py'))
    names.update(str(p.relative_to(H)) for p in H.glob('*.json') if p.name not in ('RUN_PINS.json','results.json','LAUNCH.json','LAUNCH_VERIFIED.json'))
    names.update(str(p.relative_to(H)) for p in (H/'configs').glob('*.json'))
    names.update(str(p.relative_to(H)) for p in (H/'trainer').glob('*.py'))
    names.update(str(p.relative_to(H)) for p in (H/'trainer').glob('*.json'))
    names.update(str(p.relative_to(H)) for p in (H/'gates').glob('*.json') if p.name!='DOC_CHECK.json')
    names.update(str(p.relative_to(H)) for p in (H/'gates').glob('*.py'))
    atomic_json(H/'RUN_PINS.json',dict(schema_version=3,run_id=suite['run_id'],
        files={name:sha256(H/name) for name in sorted(names)},
        external_files={suite['ordinary_gradient_receipt']:sha256(suite['ordinary_gradient_receipt']),
            **{job['parent_config']:job['parent_config_sha256'] for job in suite['jobs'] if job['kind']=='training'}},
        source_files_count=190,immutable_after_launch=True,launched=False))
    print(json.dumps(dict(status='frozen',files=len(names),launched=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze',action='store_true')
    action.add_argument('--materialize',action='store_true')
    args=parser.parse_args();freeze() if args.freeze else materialize()
