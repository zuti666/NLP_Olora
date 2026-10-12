#!/usr/bin/env python3
"""Register a new campaign without mutating any earlier experiment."""
import copy,hashlib,json,os
from pathlib import Path
H=Path(__file__).resolve().parent
B=H.parent/'imageneta-g2-minimal-fork-gpu456-20261009T223549Z'
STORAGE=Path('/data/115-2/users/liying/cache/tmp')/H.name
O=STORAGE/'outputs';O.mkdir(parents=True,exist_ok=True)
if not (H/'outputs').exists(): (H/'outputs').symlink_to(O,target_is_directory=True)
P='/home-local/liying/.conda/envs/Pilot_new_local/bin/python'
base=json.loads((B/'suite.json').read_text());source=base['source_root']
template=json.loads((B/'configs/formal_parent_sgd_s2.json').read_text())
D=('g0','g1','g2_minus_s','g2','g2_plus_s','g0_plus_d','g1_plus_d','g3')
S=('g0','g1','g2_minus_s','g2','g1_plus_d','g3')
READS=dict(zip(D,(1,2,3,3,3,4,4,4)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
jobs=[]
def parent_paths(seed,smoke=False):
    jid=('smoke' if smoke else 'formal')+'_parent_sgd_s'+str(seed)
    out=O/jid if smoke or seed!=2 else B/'outputs'/jid
    receipt=H/'runtime/jobs'/jid/'completed.json' if smoke or seed!=2 else B/'runtime/jobs'/jid/'completed.json'
    return dict(study_parent_output_dir=str(out),study_parent_checkpoint=str(out/'checkpoints/task_000'),study_parent_receipt=str(receipt))
def train(jid,direction,seed,smoke=False,role='branch',single=False,r=1.,deps=()):
    cfg=copy.deepcopy(template);out=O/jid
    cfg.update(outputs_root=str(out/'native'),prefix=jid,seed=[seed],upward_direction=direction,
        gam_grad_rho=.05,gam_grad_norm_rho=r,gam_radii_active=True,
        history_weight=0.,history_source=None,study_role=role,
        study_condition_id='parent_sgd' if role=='parent' else 'upward_'+direction,
        study_dev_split_path=str(H/'dev_split.json'),study_parent_output_dir=None,
        study_parent_checkpoint=None,study_parent_receipt=None)
    for k in ('study_inner_mode','study_inner_radius','study_outer_mode','study_outer_radius'):cfg.pop(k,None)
    if role=='branch' and not single:cfg.update(parent_paths(seed,smoke))
    if single:
        cfg.update(response_layout='single200',init_cls=200,increment=200,nb_tasks=1,total_sessions=1,
            max_train_tasks=1,single_task_full_200_classes=True)
        runner='upward_single_train.py';task=0;train_n=5981;dev_n=0;batches=47
    else:
        runner='upward_fork_train.py';task=0 if role=='parent' else 1
        train_n=1141 if task==0 else 906;dev_n=118 if task==0 else 91;batches=9 if task==0 else 8
    config=H/'configs'/(jid+'.json');save(config,cfg)
    command=[P,str(H/'trainer'/runner),'--source',source,'--config',str(config),'--output-dir',str(out)]
    if smoke:command+=['--max-epochs','1','--max-batches','2']
    jobs.append(dict(id=jid,kind='training',phase='single200' if single else role,smoke=smoke,
        method=direction,upward_direction=direction,study_role=role,study_condition_id=cfg['study_condition_id'],
        task_id=task,seed=seed,layout='single200' if single else 'cl10',r=r,rho=.05,
        command=command,output_dir=str(out),config=str(config),config_sha256=sha(config),
        expected_tasks=1 if single else task+1,expected_epochs_per_task=1 if smoke else 50,
        expected_batches_per_epoch=2 if smoke else batches,expected_steps=2 if smoke else 50*batches,
        expected_gradient_calls=READS[direction],expected_train_samples=train_n,expected_dev_samples=dev_n,
        depends_on=list(deps)))
train('smoke_parent_sgd_s2','g0',2,True,'parent')
for d in D:train('smoke_task1_'+d+'_s2',d,2,True,deps=['smoke_parent_sgd_s2'])
train('smoke_task1_g2_r025_s2','g2',2,True,r=.25,deps=['smoke_parent_sgd_s2'])
for d in S:train('smoke_single200_'+d+'_s2',d,2,True,single=True,deps=['smoke_parent_sgd_s2'])
smoketrains=[j['id'] for j in jobs]
jobs.append(dict(id='upward_smoke_pairing',kind='pairing',phase='gate',smoke=True,
    command=[P,str(H/'smoke_pairing.py'),'--suite',str(H/'suite.json'),'--output-dir',str(O/'upward_smoke_pairing')],
    output_dir=str(O/'upward_smoke_pairing'),report_file='pairing_report.json',depends_on=smoketrains))
jobs.append(dict(id='smoke_local_upward_s2',kind='local_probe_audit',phase='gate',smoke=True,seed=2,
    command=[P,str(H/'trainer/upward_local_audit.py'),'--source',source,'--suite',str(H/'suite.json'),'--output-dir',str(O/'smoke_local_upward_s2'),'--seed','2','--smoke'],
    output_dir=str(O/'smoke_local_upward_s2'),depends_on=['upward_smoke_pairing']))
gate=['smoke_local_upward_s2']
# Highest priority: old accepted seed2 parent, eight Task1 rules and radius bridge.
for d in D:train('formal_task1_'+d+'_s2',d,2,deps=gate)
train('formal_task1_g2_r025_s2','g2',2,r=.25,deps=gate)
for seed in (0,1):
    parent='formal_parent_sgd_s'+str(seed);train(parent,'g0',seed,role='parent',deps=gate)
    for d in D:train('formal_task1_'+d+'_s'+str(seed),d,seed,deps=gate+[parent])
for seed in (2,0,1):
    for d in S:train('formal_single200_'+d+'_s'+str(seed),d,seed,single=True,deps=gate)
for seed in (2,0,1):
    refs={d:str(O/('formal_task1_'+d+'_s'+str(seed))) for d in ('g0','g1','g2')}
    jobs.append(dict(id='formal_local_upward_s'+str(seed),kind='local_probe_audit',phase='offline',smoke=False,seed=seed,
        command=[P,str(H/'trainer/upward_local_audit.py'),'--source',source,'--suite',str(H/'suite.json'),'--output-dir',str(O/('formal_local_upward_s'+str(seed))),'--seed',str(seed)],
        output_dir=str(O/('formal_local_upward_s'+str(seed))),depends_on=['formal_task1_'+d+'_s'+str(seed) for d in refs]))
suite={k:copy.deepcopy(v) for k,v in base.items() if k not in ('jobs','budget')}
suite.update(run_id=H.name,run_dir=str(H),artifact_root=str(STORAGE),outputs_root=str(O),
    purpose='upward_eight_complete_parent_forks_and_single200_controls',
    protocol='upward_eight_shared_sgd_parent_exact_restore_native_skip_v1',
    allowed_gpus=[2,3,4,5,6],max_parallel=5,
    source_manifest=str(H/'source_manifest.json'),study_dev_split_path=str(H/'dev_split.json'),
    dev_split_path=str(H/'dev_split.json'),environment_spec=str(H/'environment_spec.json'),
    implementation_acceptance_file=str(H/'gates/CPU_IMPLEMENTATION_ACCEPTED.json'),
    fresh_doc_check=str(H/'gates/FRESH_DOC_CHECK.json'),jobs=jobs,
    upward_audit_refs_by_seed={str(s):{d:str(O/('formal_task1_'+d+'_s'+str(s))) for d in ('g0','g1','g2')} for s in (2,0,1)},
    upward_audit_smoke_refs_by_seed={'2':{d:str(O/('smoke_task1_'+d+'_s2')) for d in ('g0','g1','g2')}},
    reused_seed2_parent=parent_paths(2),old_seed2_sgd_task1_reference=str(B/'outputs/formal_task1_B0_s2'),
    old_descending_bridge_references={d:str(B/'outputs'/('formal_task1_'+d+'_s2')) for d in ('R25','S_F','R100')},
    budget=dict(formal_training_jobs=45,formal_parent_jobs=2,formal_task1_forks=25,
        formal_single200_jobs=18,formal_offline_jobs=3,formal_sgd_updates=53200,
        formal_gradient_reads=150750,smoke_training_jobs=16,smoke_sgd_updates=32,
        smoke_pairing_jobs=1,smoke_offline_jobs=1,
        single200_samples=5981,single200_batches_per_epoch=47,
        task_counts=base['budget']['task_counts'],seeds=[2,0,1],
        task2_status='deferred pending preregistered evidence review'))
tmp='/tmp/olora_up8_20261010T123703Z'
for k in ('TMPDIR','TMP','TEMP'):suite['environment'][k]=tmp
save(H/'suite.json',suite)
print(json.dumps(dict(jobs=len(jobs),budget=suite['budget'],outputs_root=str(O))))
