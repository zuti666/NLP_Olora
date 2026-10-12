"""Generate the first-round frozen two-arm experiment suite before launch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import os

H=Path(__file__).resolve().parent
PY='/home-local/liying/.conda/envs/Pilot_new_local/bin/python'
PRIOR=H.parent/'imageneta-r1-g12-mean-d32-multiseed-20261006T151850Z'
LATEST=H.parent/'imageneta-g12-midpoint-interaction-execution-20261008T140723Z'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
    if a.freeze:
        files=[p for p in H.glob('*.py')]+[p for p in (H/'trainer').glob('*.py')]+list((H/'configs').glob('*.json'))
        files += [H/'suite.json',H/'source_manifest.json',H/'environment_spec.json',H/'RUN.md',H/'gates/CPU_GATE.json',H/'gates/TEMP_DIRECTORY_AND_HOST_GATE.json']
        if not all(p.is_file() for p in files):raise FileNotFoundError('Missing freeze prerequisite')
        write(H/'RUN_PINS.json',dict(files={str(p.relative_to(H)):sha(p) for p in sorted(files)},
                                   description='Immutable execution files; results and progress are separate'))
        print('Frozen',len(files),'execution artifacts');return
    if (H/'RUN_PINS.json').exists():raise RuntimeError('Suite is frozen; never overwrite an active campaign')
    base=json.loads((PRIOR/'configs/formal_sum12_r1_k10_s0.json').read_text())
    old_suite=json.loads((LATEST/'suite.json').read_text())
    source=old_suite['source_root']
    jobs=[]
    variants=[('history_fused',.5),('future_only',0.)]
    for smoke in (True,False):
        for direction,w in variants:
            jid=('smoke_' if smoke else 'formal_')+direction+'_k10_s2'
            out=H/'outputs'/jid
            cfg=copy.deepcopy(base)
            cfg.update(seed=[2],prefix=jid,outputs_root=str(out/'native'),update_direction=direction,
                       component_rule=direction,history_weight=w,response_lambda_mode='sgd_prediction_distance',
                       history_reset='task',history_min_distance=1e-12,strict_deterministic=True,
                       post_update_loss=True,probe_gradient_count=2,compute_g3_probe=False,g3_used_in_final=False,
                       gam_radii_active=False,reference_outer_radius=1.,reference_inner_radius=.05,
                       in_task_path_normalize_snapshots=False,optimizer_tag_override=direction+'_two_gradient_v1')
            config=H/'configs'/(jid+'.json');write(config,cfg)
            command=[PY,str(H/'trainer/history_train.py'),'--source',source,'--config',str(config),'--output-dir',str(out)]
            if smoke:command+=['--max-tasks','2','--max-epochs','2','--max-batches','2']
            jobs.append(dict(id=jid,kind='training',phase='smoke' if smoke else 'formal',
                             direction=direction,history_weight=w,seed=2,smoke=smoke,
                             command=command,output_dir=str(out),config=str(config),config_sha256=sha(config),
                             expected_tasks=2 if smoke else 10,expected_epochs_per_task=2 if smoke else 50,
                             expected_steps=8 if smoke else 2550,smoke_batches_per_epoch=2 if smoke else None,
                             depends_on=[] if smoke else ['smoke_pairing']))
    pair=dict(id='smoke_pairing',kind='pairing',phase='gate',smoke=False,
              command=[PY,str(H/'smoke_pairing.py')],output_dir=str(H/'gates/smoke_pairing'),
              report_file='smoke_pairing.json',expected_tasks=2,
              depends_on=[j['id'] for j in jobs if j['smoke']])
    jobs.insert(2,pair)
    env=copy.deepcopy(old_suite['environment'])
    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['PYTHONDONTWRITEBYTECODE']='1'
    for k in ('TMPDIR','TMP','TEMP'):env[k]='/tmp/olora_hrf_'+H.name.rsplit('-',1)[-1]
    suite=dict(run_id=H.name,run_dir=str(H),protocol='cached_history_sgd_prediction_direct_response_exact_restore_rng_replay_v1',
               execution_hostname='cudahpc40.cvc.uab.es',execution_pid_namespace='pid:[4026531836]',
               python=PY,source_root=source,source_manifest=str(H/'source_manifest.json'),
               source_manifest_sha256=sha(H/'source_manifest.json'),allowed_gpus=[2,3,4,5,6],
               max_parallel=2,poll_seconds=15,gpu_lock_dir=old_suite['gpu_lock_dir'],
               expected_tasks=10,environment=env,jobs=jobs)
    write(H/'suite.json',suite)
    write(H/'ENVIRONMENT_REUSE.json',dict(status='warm_reuse',canonical_spec_sha256='4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8',
          unchanged=sha(H/'environment_spec.json')==sha(LATEST/'environment_spec.json'),
          ledger=str(H.parent.parent/'.aris/compute/local.md'),no_package_changes=True,
          python=PY,kernel_witness='Every allocated worker before model run'))
    print(json.dumps(dict(configs=4,jobs=5,smokes=2,formal=2,seed=2,max_parallel=2)))

if __name__=='__main__':main()
