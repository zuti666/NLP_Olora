#!/usr/bin/env python3
"""Register frozen seed2 continuation experiments, with explicit branches."""
import copy
import json
from pathlib import Path
from common import sha256,exclusive_json

HERE=Path(__file__).resolve().parent
PARENT=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-component-horizon-recovery-20261001T230000Z')
DEPLOY=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-fdhvp-followup-20261001T223005Z')
old=json.loads((PARENT/'suite.json').read_text())
template=json.loads((PARENT/'configs/screen_ug_natural_r025_s2.json').read_text())
template.update(gam_grad_norm_rho=1.,component_first_response='fd',component_second_response='fd',
    component_lambda_corr=1.,component_attention_backend='native')
inventory=json.loads((HERE/'data_inventory.json').read_text())
jobs=[]
def horizon(jid,rule,scale,group,*,smoke=False,phase='screen',first='fd',second='fd',coefficient=1.,backend='native',gate=None,deps=()):
    cfg=copy.deepcopy(template);out=DEPLOY/'outputs'/jid
    cfg.update(prefix=jid,outputs_root=str(out),component_rule=rule,component_scale=scale,
        component_first_response=first,component_second_response=second,component_lambda_corr=coefficient,
        component_attention_backend=backend)
    path=HERE/'configs'/f'{jid}.json';exclusive_json(path,cfg)
    cmd=[old['python'],str(DEPLOY/'trainer/component_horizon.py'),'--source',old['source_root'],
        '--config',str(DEPLOY/'configs'/path.name),'--output-dir',str(out)]
    if smoke:cmd+=['--max-tasks','2','--max-epochs','1','--max-batches','1']
    jobs.append({'id':jid,'kind':'horizon','phase':'smoke' if smoke else phase,'variant':first+'_'+second,
        'rule':rule,'scale':scale,'first_response':first,'second_response':second,'lambda_corr':coefficient,
        'attention_backend':backend,'seed':2,'radius':1.,'smoke':smoke,'smoke_group':group,'gate':gate,
        'config':str(DEPLOY/'configs'/path.name),'config_sha256':sha256(path),'output_dir':str(out),
        'command':cmd,'depends_on':list(deps),'expected_tasks':2 if smoke else 10,
        'expected_epochs_per_task':1 if smoke else 50,'expected_steps':2 if smoke else inventory['formal_steps']})
    return jid
base_smokes=[]
for rule in ('u0','us','ug','uo','ufull'):
    for scale in ('natural','matched'):
        base_smokes.append(horizon(f'smoke_r1_{rule}_{scale}_s2',rule,scale,'base',smoke=True))
core=[]
for rule in ('u0','us','ug','uo','ufull'):
    core.append(horizon(f'screen_r1_{rule}_natural_s2',rule,'natural','base',deps=base_smokes))
diagnostics=[]
ug='screen_r1_ug_natural_s2'
for task in (0,4,9):
    jid=f'curvature_ug_r1_t{task}_s2';out=DEPLOY/'outputs'/jid;config=DEPLOY/'configs'/f'{ug}.json'
    diagnostics.append(jid)
    jobs.append({'id':jid,'phase':'curvature','kind':'curvature','rule':'ug','scale':'natural','seed':2,'radius':1.,
        'smoke':False,'gate':None,'smoke_group':None,'task':task,'expected_tasks':1,'max_batches':3 if task==9 else 4,
        'config':str(config),'config_sha256':sha256(HERE/'configs'/f'{ug}.json'),'output_dir':str(out),
        'command':[old['python'],str(DEPLOY/'diagnostics/curvature_audit.py'),'--source',old['source_root'],
            '--config',str(config),'--checkpoint-dir',str(DEPLOY/'outputs'/ug/'checkpoints'/f'task_{task:03d}'),
            '--task',str(task),'--output',str(out/'curvature_audit.json'),'--rho','.05','--radius','1',
            '--epsilons','.05,.025,.0125','--max-batches','3' if task==9 else '4','--batch-size','128','--hvp-backend','auto','--device','cuda:0'],
        'depends_on':[ug]})
second_smokes=[]
for second in ('fd','hvp'):
    for scale in ('natural','matched'):
        second_smokes.append(horizon(f'smoke_second_{second}_{scale}_s2','ug',scale,'second_hvp',smoke=True,
            second=second,backend='math',gate='correction_signal_with_hvp',deps=core+diagnostics))
second_runs=[]
for second in ('fd','hvp'):
    for scale in ('natural','matched'):
        second_runs.append(horizon(f'second_{second}_{scale}_s2','ug',scale,'second_hvp',phase='second_hvp',
            second=second,backend='math',gate='correction_signal_with_hvp',deps=second_smokes))
first_smokes=[]
for second in ('fd','hvp'):
    for scale in ('natural','matched'):
        first_smokes.append(horizon(f'smoke_first_hvp_second_{second}_{scale}_s2','ug',scale,'first_hvp',smoke=True,
            first='hvp',second=second,backend='math',gate='second_hvp_effect',deps=second_runs))
for second in ('fd','hvp'):
    for scale in ('natural','matched'):
        horizon(f'first_hvp_second_{second}_{scale}_s2','ug',scale,'first_hvp',phase='first_hvp',
            first='hvp',second=second,backend='math',gate='second_hvp_effect',deps=first_smokes)
coefficient_smokes=[]
for rule in ('us','ug'):
    for label,value in (('05',.5),('20',2.)):
        coefficient_smokes.append(horizon(f'smoke_coeff_{rule}_l{label}_s2',rule,'natural','lambda',smoke=True,
            coefficient=value,gate='no_correction_signal',deps=core))
for rule in ('us','ug'):
    for label,value in (('05',.5),('20',2.)):
        horizon(f'coeff_{rule}_l{label}_s2',rule,'natural','lambda',phase='coefficient',
            coefficient=value,gate='no_correction_signal',deps=coefficient_smokes)
suite={**old,'run_id':DEPLOY.name,'run_dir':str(DEPLOY),'source_manifest':str(DEPLOY/'source_manifest.json'),
    'source_manifest_sha256':sha256(HERE/'source_manifest.json'),'parent_gate':str(DEPLOY/'handoff_guard.json'),
    'jobs':jobs,'smoke_groups':{'base':base_smokes,'second_hvp':second_smokes,'first_hvp':first_smokes,'lambda':coefficient_smokes},
    'environment':{**old['environment'],**{k:'/tmp/fdhvp-followup-20261001T223005Z' for k in ('TMPDIR','TMP','TEMP')}},
    'auto_phase2_or_phase3':False,'conditional_branches':True,'no_external_notifications':True,
    'failure_policy':'halt new launches on any failure; preserve active workers and all artifacts'}
exclusive_json(HERE/'suite.json',suite)
signal={'baseline_id':core[0],'candidate_id':ug,'min_AAA_gain_pp':1.,'min_FAA_gain_pp':-1.}
gates={'schema_version':1,'status':'frozen_before_followup_training','seed':2,'exploratory_only':True,
    'definitions':{'correction_signal_with_hvp':{**signal,'kind':'correction_signal_with_hvp','diagnostic_ids':diagnostics},
        'no_correction_signal':{**signal,'kind':'no_correction_signal'},
        'second_hvp_effect':{'kind':'second_hvp_effect','min_abs_AAA_difference_pp':1.,
            'comparisons':[{'fd_id':f'second_fd_{scale}_s2','hvp_id':f'second_hvp_{scale}_s2'} for scale in ('natural','matched')]}},
    'matching_relative_tolerance':.001,'US_smoke_relative_parameter_tolerance':.0001,
    'native_radius1_core_unconditional':True,'HVP_alignment_is_not_a_release_gate':True,
    'technical_gate':'actual detached-direction HVP finite, same-point CE/gradient agreement, all state restored, exact selected gradient/projector/SGD; unavailable stops HVP training, never falls back',
    'lambda_definition':'correction coefficient, not EWC lambda or lambda_flat',
    'conditional_skip_is_not_proof_of_no_effect':True,
    'later_unregistered':{'additional_seeds':'retain single seed2 unless later explicitly expanded',
        'two_tasks':'after direction selection: independent SS/XS/SX/XX',
        'three_tasks':'after two-task effect: select persistence comparison'}}
exclusive_json(HERE/'FROZEN_GATES.json',gates)
guard={'parent_dir':str(PARENT),'parent_suite_sha256':sha256(PARENT/'suite.json'),
    'parent_run_pins_sha256':sha256(PARENT/'RUN_PINS.json'),'expected_parent_jobs':len(old['jobs']),
    'execution_pid_namespace':'pid:[4026531836]','release_policy':'all parent jobs validate successfully and parent handoff/controller exited'}
exclusive_json(HERE/'handoff_guard.json',guard)
print(json.dumps({'jobs':len(jobs),'smokes':sum(j['smoke'] for j in jobs),'formal_trainings_registered':sum(j['kind']=='horizon' and not j['smoke'] for j in jobs),'curvature':len(diagnostics),'seed':2}))
