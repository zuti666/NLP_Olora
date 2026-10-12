#!/usr/bin/env python3
"""Build immutable per-job configs and strict-success dependency manifest."""
import copy
import hashlib
import json
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEPLOY=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-formation-run-20261001T180712Z')
OLD=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-stage1-task0-run-20260930T213643Z')
PYTHON='/home-local/liying/.conda/envs/Pilot_new_local/bin/python'
SOURCE='/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source'
ARMS=['s','sum','mean_lr2','s_lr15','s_lr2','g_raw','g_r025','g_match']
RHO={'g_r025':.25}
MULT={'s_lr15':1.5,'s_lr2':2.,'mean_lr2':2.}
BETA={'s':(1.,0.),'s_lr15':(1.,0.),'s_lr2':(1.,0.),'sum':(1.,1.),'mean_lr2':(.5,.5),'g_raw':(0.,1.),'g_r025':(0.,1.),'g_match':(0.,1.)}

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

template=json.loads((OLD/'configs/task0_s_k10_s2.json').read_text())
jobs=[]
smokes=[]
def formation(arm,seed,smoke=False):
    group='smoke' if smoke else 'formation'
    jid=f'{group}_{arm}_s{seed}'
    output=DEPLOY/'outputs'/jid
    cfg=copy.deepcopy(template)
    multiplier=MULT.get(arm,1.)
    b1,b3=BETA[arm]
    cfg.update(prefix=jid,outputs_root=str(output),seed=[seed],stage1_sampler_seed=seed,
        formation_arm=arm,stage1_direction={'g_raw':'g','g_r025':'g','g_match':'g','mean_lr2':'mean'}.get(arm,'s' if arm.startswith('s') and arm!='sum' else arm),
        init_lr=.04*multiplier,lrate=.02*multiplier,init_epoch=50,epochs=50,cosine_min_lr_ratio=.1,
        gam_grad_beta_1=b1,gam_grad_beta_2=1.,gam_grad_beta_3=b3,gam_grad_gamma=0.,
        gam_grad_rho=.05,gam_grad_norm_rho=RHO.get(arm,1.),
        train_num_workers=8,eval_num_workers=4,
        ewc_lambda=0.,ewc_skip_task_fisher=True,approximate_fisher_in_task=False,
        in_task_weight_merge=False,task_endpoint_weight_merge=False,lambda_flat=1.)
    deps=[] if smoke else list(smokes)
    if arm=='g_match':
        ref=f'{group}_s_s{seed}'
        cfg['step_reference_dir']=str(DEPLOY/'outputs'/ref)
        deps.append(ref)
    configpath=HERE/'configs'/f'{jid}.json'
    write(configpath,cfg)
    command=[PYTHON,str(DEPLOY/'trainer/formation_task0.py'),'--source',SOURCE,'--config',str(DEPLOY/'configs'/configpath.name),'--output-dir',str(output)]
    if smoke: command+=['--max-epochs','1','--max-batches','1']
    jobs.append({'id':jid,'phase':group,'kind':'formation','arm':arm,'seed':seed,'smoke':smoke,
        'config':str(DEPLOY/'configs'/configpath.name),'config_sha256':digest(configpath),'output_dir':str(output),
        'command':command,'depends_on':deps,'expected_epochs':1 if smoke else 50,'expected_steps':1 if smoke else 500})
    return jid

for arm in ARMS: smokes.append(formation(arm,2,True))
for seed in (0,1): smokes.append(formation('s',seed,True))

def future(arm,seed,epoch,rule='s',smoke=False):
    formation_id=f'{"smoke" if smoke else "formation"}_{arm}_s{seed}'
    jid=f'{"smoke_future" if smoke else "future"}_{arm}_e{epoch:02d}_{rule}_s{seed}'
    output=DEPLOY/'outputs'/jid
    command=[PYTHON,str(DEPLOY/'trainer/future_probe.py'),'--source',SOURCE,
        '--config',str(DEPLOY/'configs'/f'{formation_id}.json'),
        '--checkpoint-dir',str(DEPLOY/'outputs'/formation_id/'checkpoints'/f'epoch_{epoch:03d}'),
        '--output-dir',str(output),'--future-rule',rule,'--steps','1' if smoke else '100',
        '--milestones','1' if smoke else '1,10,50,100','--formation-arm',arm,'--formation-epoch',str(epoch),'--seed',str(seed),
        '--selection-split',str(DEPLOY/'selection_split.json')]
    deps=[formation_id]
    if smoke: command+=['--smoke']
    if rule=='g_match':
        ref=f'{"smoke_future" if smoke else "future"}_s_e{epoch if smoke else 50:02d}_s_s{seed}'
        deps.append(ref)
        command+=['--reference-dir',str(DEPLOY/'outputs'/ref)]
    jobs.append({'id':jid,'phase':'smoke' if smoke else 'future_s' if rule=='s' else 'switch',
        'kind':'future','arm':arm,'seed':seed,'smoke':smoke,'formation_epoch':epoch,'future_rule':rule,
        'config':str(DEPLOY/'configs'/f'{formation_id}.json'),
        'config_sha256':digest(HERE/'configs'/f'{formation_id}.json'),'output_dir':str(output),
        'command':command,'depends_on':deps,'expected_steps':1 if smoke else 100})
    return jid

smokes.append(future('s',2,1,'s',True))
smokes.append(future('s',2,1,'g_match',True))
# All twelve short tests must succeed before ANY formal training is launched.
for seed in (0,1,2):
    for arm in ARMS: formation(arm,seed,False)
for seed in (0,1,2):
    for arm in ('s','g_raw','g_match','g_r025'):
        for epoch in (0,10,20,30,40,50): future(arm,seed,epoch)
    for arm in ('s_lr15','s_lr2'): future(arm,seed,50)
for seed in (0,1,2):
    for arm in ('s','g_match'): future(arm,seed,50,'g_match')
formation_jobs=[j['id'] for j in jobs if j['phase']=='formation']
for job in jobs:
    if not job['smoke'] and job['kind']=='future':
        job['depends_on'].extend(x for x in formation_jobs if x not in job['depends_on'])

shutil.copyfile(OLD/'common.py',HERE/'common.py')
shutil.copyfile(OLD/'source_manifest.json',HERE/'source_manifest.json')
suite={'schema_version':1,'run_id':DEPLOY.name,'execution_hostname':'cudahpc40.cvc.uab.es',
    'run_dir':str(DEPLOY),'python':PYTHON,'source_root':SOURCE,'source_manifest':str(DEPLOY/'source_manifest.json'),
    'source_manifest_sha256':digest(HERE/'source_manifest.json'),'expected_tasks':1,
    'allowed_gpus':list(range(7)),'max_parallel':7,'gpu_free_threshold_mib':500,'gpu_free_max_util_percent':4,
    'gpu_lock_dir':'/tmp/olora_gamfs_gpu_locks','poll_seconds':15,'failure_policy':'halt new launches, preserve outputs; active jobs finish',
    'environment':{'PYTHONHASHSEED':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','FLATNESS_CV_DATA_ROOT':'/data/140-1/datasets',
        'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':'/home-local/liying/.cache/huggingface',
        'TORCH_HOME':'/home-local/liying/.cache/torch','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
        'TMPDIR':'/tmp/formation-20261001T180712Z','TMP':'/tmp/formation-20261001T180712Z','TEMP':'/tmp/formation-20261001T180712Z'},
    'jobs':jobs,'auto_phase2_or_phase3':False,'no_external_notifications':True}
write(HERE/'suite.json',suite)
write(HERE/'conditional_plan.json',{'phase2':{'enabled':False,'when':'frozen Phase1 pilot science gates pass; implement and review before launch',
    'seeds':[0,1,2],'two_task_independent_training':['SS','GS','SG','GG'],'G':'g_match'},
    'phase3':{'enabled':False,'when':'Phase2 formation effect supported',
    'seeds':[0,1,2],'three_task_independent_training':['SSS','GSS','SGS','SSG','GGG']},
    'optional_more_seeds':{'enabled':False,'seeds':[3,4]},'initial_formal_formation':24,'initial_future_S':78,'initial_short_switch_Gmatch':6})
gates=json.loads((HERE/'review/FALSIFIABLE_GATES.json').read_text())
gates.update(status='frozen_before_new_training',frozen_by='root',
    zero_target_requires_no_update=False,zero_target_policy='invalid target: fail closed; never count as matched',
    selection_split_sha256=digest(HERE/'selection_split.json'))
gates['suggested_phase1_science_gate']['status']='frozen pilot decision rule; not a significance test'
write(HERE/'FROZEN_GATES.json',gates)
print(json.dumps({'jobs':len(jobs),'smokes':len(smokes),'formation':24,'future_s':78,'switch':6,'deployment':str(DEPLOY)}))
