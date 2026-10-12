#!/usr/bin/env python3
"""Register only the user-authorized six Task0 formation policies."""
import copy,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
B=H.parent/'imageneta-upward-eight-fork-20261010T123703Z'
P='/home-local/liying/.conda/envs/Pilot_new_local/bin/python'
O=Path('/data/115-2/users/liying/cache/tmp')/H.name/'outputs'
O.mkdir(parents=True,exist_ok=True)
CONDITIONS=['sgd','sam','gam_residual','sf_g2','sf_g2_then_sgd','sgd_then_sf_g2']
FORMAL_READS=dict(zip(CONDITIONS,[340,520,880,700,520,520]))
SMOKE_READS=dict(zip(CONDITIONS,[40,60,100,80,60,60]))
PHASES={'sgd':['sgd','sgd'],'sam':['sam','sam'],'gam_residual':['gam_residual','gam_residual'],'sf_g2':['sf_g2','sf_g2'],'sf_g2_then_sgd':['sf_g2','sgd'],'sgd_then_sf_g2':['sgd','sf_g2']}
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['dev_split.json','source_manifest.json','environment_spec.json']:
    (H/name).write_bytes((B/name).read_bytes())
template=json.loads((B/'configs/formal_parent_sgd_s0.json').read_text())
jobs=[]
def register(condition,seed,smoke=False,deps=()):
    jid=('smoke' if smoke else 'formal')+'_'+condition+'_s'+str(seed)
    out=O/jid;cfg=copy.deepcopy(template)
    for k in list(cfg):
        if k.startswith('upward_') or k.startswith('study_parent_') or k.startswith('study_inner_') or k.startswith('study_outer_'):cfg.pop(k,None)
    cfg.update(prefix=jid,outputs_root=str(out/'native'),seed=[seed],init_epoch=20,epochs=20,
      max_train_tasks=2,formation_condition_id=condition,formation_task0_schedule=PHASES[condition],
      formation_task1_method='sgd',formation_switch_epoch=11,study_role='formation',
      study_condition_id=condition,study_dev_split_path=str(H/'dev_split.json'),
      response_method='sgd',response_layout='cl10',gam_grad_rho=.05,gam_grad_norm_rho=1.,
      study_inner_mode='sgd_step',study_outer_mode='fixed',study_outer_radius=.25,
      history_weight=0.,history_source=None,compute_g3_probe=(condition=='gam_residual'),g3_used_in_final=(condition=='gam_residual'),
      gam_grad_beta_3=1. if condition=='gam_residual' else 0.,gam_radii_active=(condition in ('sam','gam_residual')),
      formation_schedule_authoritative=True,
      formation_protocol='task0_formation_20ep_task1_sgd_v1',formation_smoke=smoke,
      probe_gradient_count=None,update_direction='formation_schedule',optimizer_tag_override='formation20_native_sgd')
    c=H/'configs'/(jid+'.json');save(c,cfg)
    command=[P,str(H/'trainer/formation_train.py'),'--source',template_source,'--config',str(c),'--output-dir',str(out)]
    if smoke:command+=['--max-batches','1']
    jobs.append(dict(id=jid,kind='training',smoke=smoke,seed=seed,condition=condition,
       formation_condition_id=condition,config=str(c),config_sha256=sha(c),output_dir=str(out),
       command=command,expected_tasks=2,expected_epochs_per_task=20,
       expected_steps=40 if smoke else 340,expected_gradient_reads=(SMOKE_READS if smoke else FORMAL_READS)[condition],
       expected_task_steps=[20,20] if smoke else [180,160],expected_batches_per_epoch=[1,1] if smoke else [9,8],
       expected_train_samples=[1141,906],expected_dev_samples=[118,91],depends_on=list(deps)))
base=json.loads((B/'suite.json').read_text());template_source=base['source_root']
for c in CONDITIONS:register(c,2,True)
smoke_ids=[j['id'] for j in jobs]
pair_id='formation_smoke_pairing';pairout=O/pair_id
jobs.append(dict(id=pair_id,kind='pairing',smoke=False,seed=2,expected_tasks=2,
    output_dir=str(pairout),report_file='pairing_report.json',depends_on=smoke_ids,
    command=[P,str(H/'smoke_pairing.py'),'--suite',str(H/'suite.json'),'--output-dir',str(pairout)]))
for seed in [2,0,1]:
    for c in CONDITIONS:register(c,seed,deps=smoke_ids+[pair_id])
keys=['execution_hostname','execution_pid_namespace','python','source_root','source_manifest_sha256','poll_seconds','gpu_lock_dir','environment','environment_fingerprint']
suite={k:copy.deepcopy(base[k]) for k in keys}
suite.update(run_id=H.name,run_dir=str(H),purpose='old_knowledge_formation_and_phase_under_common_task1_sgd',
    protocol='task0_formation_20ep_task1_sgd_v1',source_manifest=str(H/'source_manifest.json'),
    allowed_gpus=[4,5,6],max_parallel=3,expected_tasks=2,artifact_root=str(O.parent),outputs_root=str(O),
    study_dev_split_path=str(H/'dev_split.json'),dev_split_path=str(H/'dev_split.json'),study_dev_split_sha256=sha(H/'dev_split.json'),
    environment_spec=str(H/'environment_spec.json'),
    implementation_acceptance_file=str(H/'gates/CPU_IMPLEMENTATION_ACCEPTED.json'),
    independent_implementation_review_file=str(H/'gates/INDEPENDENT_IMPLEMENTATION_REVIEW.json'),
    fresh_doc_check=str(H/'gates/FRESH_DOC_CHECK.json'),jobs=jobs,
    budget=dict(formal_training_jobs=18,formal_sgd_updates=6120,formal_gradient_reads=10440,
      smoke_training_jobs=6,smoke_sgd_updates=240,smoke_gradient_reads=400,pairing_jobs=1,
      seeds=[0,1,2],epochs_per_task=20,train_tasks=2,conditions=CONDITIONS,
      task_counts={'0':dict(train=1141,dev=118,test=307,batches_per_epoch=9),'1':dict(train=906,dev=91,test=254,batches_per_epoch=8)}))
for k in ['TMPDIR','TMP','TEMP']:suite['environment'][k]='/tmp/olora_form20_20261010T205804Z'
save(H/'suite.json',suite)
save(H/'protocol.json',dict(protocol=suite['protocol'],seeds=[0,1,2],conditions=CONDITIONS,
    task0_phases=PHASES,task1='sgd',epochs=[20,20],switch_epoch=11,lr=[.04,.02],eta_min_ratio=.1,
    sam_rho=.05,residual_gam='g0+(g3-g2)',residual_gam_outer_r=1.,sf_inner='theta-eta*g0',sf_outer_r=.25,
    old_drop='A_old_start_after_head_expansion_joint40-A_old_end_joint40',
    data_split_sha256=sha(H/'dev_split.json'),historical_50epoch_scores_pooled=False,
    resource=dict(gpus=[4,5,6],max_parallel=3,idle_only=True,preemption=False),
    inherited_environment_spec_is_reference=True,active_workload_defined_here=True))
print(json.dumps(dict(jobs=len(jobs),budget=suite['budget'],output_root=str(O))))
