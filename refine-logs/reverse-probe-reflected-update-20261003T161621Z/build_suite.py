#!/usr/bin/env python3
"""Materialize two isolated jobs; freezing is a separate explicit CPU-only action."""
import argparse
import copy
import json
from pathlib import Path
from common import HERE as H, atomic_json, load_suite, sha256, verify_config, verify_source, verify_preflight

PARENT = H.parent/'g2-central-location-gpu01-20261003T105925Z'
FORMAL = 'RB01_Backward_reflected_natural_s2'


def materialize():
    if any((H/p).exists() for p in ('suite.json','runtime/queue_state.json','RUN_PINS.json')):
        raise RuntimeError('Refusing to overwrite an already materialized, frozen or started campaign')
    old=json.loads((PARENT/'suite.json').read_text())
    raw=json.loads((PARENT/'configs/G2-CD01_O_P_natural_s2.json').read_text())
    env=copy.deepcopy(old['environment'])
    for k in ('TMPDIR','TMP','TEMP'):env[k]='/tmp/'+H.name
    suite=dict(schema_version=3,run_id=H.name,run_dir=str(H),
        execution_hostname=old['execution_hostname'],python=old['python'],source_root=old['source_root'],
        source_manifest=str(H/'source_manifest.json'),source_manifest_sha256=sha256(H/'source_manifest.json'),
        allowed_gpus=[0],max_parallel=1,poll_seconds=15,
        gpu_lock_dir='/tmp/olora_gamfs_gpu_locks',gpu_free_threshold_mib=500,
        gpu_free_util_below_percent=10,expected_tasks=10,environment=env,
        ordinary_gradient_receipt=old['ordinary_gradient_receipt'],
        ordinary_gradient_receipt_sha256=old['ordinary_gradient_receipt_sha256'],
        no_external_notifications=True,stage2_backend='math',method_name='reverse_probe_reflected_v1',
        outer_radius=None,scope='One backward-local reflected natural seed2 smoke and one formal run only; 2 ordinary CE per update; 4 isolated replay CE per task',
        failure_policy='halt new launches on any failure; drain active jobs; no retry, preemption or stale override',
        parent_run=str(PARENT),jobs=[])
    for smoke in (True,False):
        ident=('smoke_' if smoke else '')+FORMAL
        output=H/'outputs'/ident
        cfg=copy.deepcopy(raw)
        cfg.update(prefix=ident,outputs_root=str(output/'native'),probe_branch='backward_local',
            update_direction='g0_plus_g0_minus_g1minus',method_name='reverse_probe_reflected_v1',
            gam_grad_rho=.05,gam_grad_norm_rho=0.,step_mode='natural',attention_backend='math',
            component_rule='u0',component_scale='natural',in_task_path_normalize_snapshots=False)
        path=H/'configs'/(ident+'.json');atomic_json(path,cfg)
        command=[suite['python'],str(H/'trainer/reflected_train.py'),'--source',suite['source_root'],
                 '--config',str(path),'--output-dir',str(output)]
        if smoke:command+=['--max-tasks','2','--max-epochs','1','--max-batches','1']
        suite['jobs'].append(dict(id=ident,kind='training',phase='smoke' if smoke else 'formal',
            branch='backward_local',direction='g0_plus_g0_minus_g1minus',mode='natural',backend='math',smoke=smoke,gpu_required=True,
            command=command,config=str(path),config_sha256=sha256(path),output_dir=str(output),
            depends_on=[] if smoke else ['smoke_'+FORMAL],expected_tasks=2 if smoke else 10,
            expected_epochs_per_task=1 if smoke else 50,expected_steps=2 if smoke else 2550,
            expected_training_ce=4 if smoke else 5100,expected_isolated_replay_ce=8 if smoke else 40))
    atomic_json(H/'suite.json',suite)
    load_suite(H/'suite.json')
    for job in suite['jobs']:verify_config(job)
    print(json.dumps(dict(status='materialized',smokes=1,formal=1,allowed_gpus=[0],max_parallel=1,launched=False)))


def freeze():
    if (H/'runtime/queue_state.json').exists():raise RuntimeError('Cannot repin a started queue')
    if (H/'RUN_PINS.json').exists():raise RuntimeError('Already frozen; never replace execution pins')
    suite=load_suite(H/'suite.json');verify_source(suite)
    for job in suite['jobs']:
        if not Path(job['command'][1]).is_file():raise FileNotFoundError(job['command'][1])
        verify_config(job)
    verify_preflight(suite)
    names={'suite.json','source_manifest.json','environment_spec.json','RUN.md','ENVIRONMENT.md','EXPERIMENT_PLAN.md'}
    names.update(str(p.relative_to(H)) for p in H.glob('*.py'))
    names.update(str(p.relative_to(H)) for p in H.glob('*.json') if p.name not in ('RUN_PINS.json','results.json','LAUNCH.json','LAUNCH_VERIFIED.json'))
    names.update(str(p.relative_to(H)) for p in (H/'configs').glob('*.json'))
    names.update(str(p.relative_to(H)) for p in (H/'trainer').glob('*.py'))
    names.update(str(p.relative_to(H)) for p in (H/'trainer').glob('*.json'))
    names.update(str(p.relative_to(H)) for p in (H/'gates').glob('*.json') if p.name!='DOC_CHECK.json')
    atomic_json(H/'RUN_PINS.json',dict(schema_version=3,run_id=suite['run_id'],
        files={name:sha256(H/name) for name in sorted(names)},
        external_files={suite['ordinary_gradient_receipt']:sha256(suite['ordinary_gradient_receipt'])},
        source_files_count=190,immutable_after_launch=True,launched=False))
    print(json.dumps(dict(status='frozen',files=len(names),launched=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze',action='store_true')
    action.add_argument('--materialize',action='store_true')
    args=parser.parse_args();freeze() if args.freeze else materialize()
