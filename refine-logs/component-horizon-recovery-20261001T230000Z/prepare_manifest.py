#!/usr/bin/env python3
"""Paired ten-task decomposition cohort, held behind the complete formation study."""
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
DEPLOY=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-component-horizon-recovery-20261001T230000Z')
PARENT=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-formation-run-20261001T180712Z')
SOURCE='/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source'
PYTHON='/home-local/liying/.conda/envs/Pilot_new_local/bin/python'
RULES=['u0','us','ug','uo','ufull']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

template=json.loads((PARENT/'configs/formation_s_s2.json').read_text())
for key in list(template):
    if key.startswith(('formation_','stage1_','step_reference')):template.pop(key)
template.update(seed=[2],init_epoch=50,epochs=50,init_lr=.04,lrate=.02,cosine_min_lr_ratio=.1,
    gam_grad_rho=.05,gam_grad_norm_rho=.25,gam_grad_beta_1=1.,gam_grad_beta_2=1.,gam_grad_beta_3=0.,
    gam_grad_gamma=0.,max_train_tasks=10,mechanism_eval=False,mechanism_grad_stats=False,
    train_num_workers=8,eval_num_workers=4,component_probe_tasks=[0,4,9],component_metrics_every_epoch=True)
sys.path.insert(0,SOURCE)
import numpy as np
from utils.data_manager import DataManager
dm=DataManager('imageneta',False,2,20,20,template)
inventory={'dataset':'imageneta','class_shuffle':False,'class_order':list(map(int,dm._class_order)),'tasks':[]}
for task in range(10):
    lo,hi=dm.get_task_class_range(task)
    item={'task':task,'class_range':[lo,hi]}
    for split in ('train','test'):
        data,labels,_=dm.get_dataset(np.arange(lo,hi),split,'test',ret_data=True)
        item[split+'_n']=len(labels)
        item[split+'_labels_sha256']=hashlib.sha256(np.asarray(labels,dtype=np.int64).tobytes()).hexdigest()
        item[split+'_canonical_ids_sha256']=hashlib.sha256(json.dumps(list(map(str,data)),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    item['batches_per_epoch']=(item['train_n']+127)//128
    inventory['tasks'].append(item)
inventory['formal_steps']=50*sum(x['batches_per_epoch'] for x in inventory['tasks'])
inventory['formal_samples']=50*sum(x['train_n'] for x in inventory['tasks'])
save(HERE/'data_inventory.json',inventory)
jobs=[];smokeids=[]
def trainjob(rule,scale,smoke=False):
    jid=f'{"smoke" if smoke else "screen"}_{rule}_{scale}_r025_s2'
    cfg=copy.deepcopy(template);out=DEPLOY/'outputs'/jid
    cfg.update(component_rule=rule,component_scale=scale,prefix=jid+'_recovery_20261001T230000Z',outputs_root=str(out))
    p=HERE/'configs'/f'{jid}.json';save(p,cfg)
    cmd=[PYTHON,str(DEPLOY/'trainer/component_horizon.py'),'--source',SOURCE,'--config',str(DEPLOY/'configs'/p.name),'--output-dir',str(out)]
    if smoke:cmd+=['--max-tasks','2','--max-epochs','1','--max-batches','1']
    jobs.append({'id':jid,'phase':'smoke' if smoke else 'screen','kind':'horizon','rule':rule,'scale':scale,
        'seed':2,'radius':.25,'smoke':smoke,'config':str(DEPLOY/'configs'/p.name),'config_sha256':sha(p),
        'output_dir':str(out),'command':cmd,'depends_on':[] if smoke else list(smokeids),
        'expected_tasks':2 if smoke else 10,'expected_epochs_per_task':1 if smoke else 50,
        'expected_steps':2 if smoke else inventory['formal_steps']})
    return jid
for rule in RULES:
    for scale in ('natural','matched'):smokeids.append(trainjob(rule,scale,True))
for rule in RULES:trainjob(rule,'matched')
for rule in ('u0','ug','uo','ufull'):trainjob(rule,'natural')
for rule in ('us','ufull'):
    ref=f'screen_{rule}_matched_r025_s2'
    for task in (0,4,9):
        jid=f'curvature_{rule}_t{task}_r025_s2';out=DEPLOY/'outputs'/jid
        config=DEPLOY/'configs'/f'{ref}.json'
        jobs.append({'id':jid,'phase':'curvature','kind':'curvature','rule':rule,'scale':'matched',
            'seed':2,'radius':.25,'smoke':False,'task':task,'expected_tasks':1,'max_batches':3 if task==9 else 4,
            'config':str(config),'config_sha256':sha(HERE/'configs'/f'{ref}.json'),'output_dir':str(out),
            'command':[PYTHON,str(DEPLOY/'diagnostics/curvature_audit.py'),'--source',SOURCE,'--config',str(config),
                '--checkpoint-dir',str(DEPLOY/'outputs'/ref/'checkpoints'/f'task_{task:03d}'),
                '--task',str(task),'--output',str(out/'curvature_audit.json'),'--rho','.05','--radius','.25',
                '--epsilons','.05,.025,.0125','--max-batches','3' if task==9 else '4','--batch-size','128','--hvp-backend','auto','--device','cuda:0'],
            'depends_on':[ref]})
save(HERE/'suite.json',{'schema_version':1,'run_id':DEPLOY.name,'run_dir':str(DEPLOY),
    'execution_hostname':'cudahpc40.cvc.uab.es','python':PYTHON,'source_root':SOURCE,
    'source_manifest':str(DEPLOY/'source_manifest.json'),'source_manifest_sha256':sha(HERE/'source_manifest.json'),
    'allowed_gpus':list(range(7)),'max_parallel':7,'expected_tasks':10,'poll_seconds':15,
    'gpu_lock_dir':'/tmp/olora_gamfs_gpu_locks','gpu_free_threshold_mib':500,'gpu_free_max_util_percent':4,
    'environment':{**json.loads((PARENT/'suite.json').read_text())['environment'],
        'TMPDIR':'/tmp/component-horizon-recovery-20261001T230000Z','TMP':'/tmp/component-horizon-recovery-20261001T230000Z','TEMP':'/tmp/component-horizon-recovery-20261001T230000Z'},
    'jobs':jobs,'parent_gate':str(DEPLOY/'handoff_guard.json'),'auto_phase2_or_phase3':False,
    'no_external_notifications':True,'failure_policy':'halt launches; preserve artifacts; allow active processes to finish'})
save(HERE/'FROZEN_GATES.json',{'schema_version':1,'status':'frozen_before_new_screen_training',
    'primary_metric':'AAA_task_macro','secondary':['FAA_task_macro','LA','BWT','forgetting'],
    'seed':2,'exploratory_only':True,'matched_target':'projected native g1 counterfactual actual FP32 SGD from candidate own post-unperturb base',
    'target_is_independent_S_trace':False,'matching_relative_tolerance':.001,
    'US_identity':{'required':True,'c_equals_nominal_eta_exact':True},
    'eligible_candidate':{'AAA_gain_min_pp':1.,'FAA_gain_min_pp':-1.,'must_be_validated':True},
    'approx_symbol':'within +/-1 pp for AAA and FAA is exploratory proximity, not statistical equivalence',
    'X_selection':{'eligible':['ug','uo','ufull'],'ranking':['largest AAA gain','largest FAA gain','fixed tie order uo,ug,ufull'],
        'no_eligible':'no candidate chosen; examine magnitude/schedule; do not claim proof of zero direction effect'},
    'phase2':{'auto_launch':False,'seeds':[0,1,2],'rules':['SS','XS','SX','XX'],'independent_retraining':True},
    'phase3':{'auto_launch':False,'seeds':[0,1,2],'formation_dominates':['SSS','XSS'],
        'continuation_dominates':['SSS','SXS','SSX'],'both':['SSS','XSS','SXS','SSX','XXX']},
    'radius1_robustness':{'auto_launch':False,'after':'choose component X; same protocol recheck S and X'},
    'Hessian_interpretation':'requires available same-point verified HVP, FD step convergence and observed relative error; performance alone cannot prove Hessian mechanism',
    'reuse':'legacy natural g3 r.25 is historical background; fresh natural ufull required for same-cohort scale comparison'})
print(json.dumps({'jobs':len(jobs),'smokes':10,'fresh_ten_task_runs':9,'curvature_audits':6,'formal_steps_per_run':inventory['formal_steps']}))
