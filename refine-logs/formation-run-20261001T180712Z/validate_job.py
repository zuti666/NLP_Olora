"""Fail-closed completion checks; raw, finite metrics plus immutable receipts."""
import hashlib
import json
import math
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def require(ok,message):
    if not ok: raise ValueError(message)
def finite(value):
    if isinstance(value,float): require(math.isfinite(value),'nonfinite observed result')
    elif isinstance(value,dict):
        for item in value.values(): finite(item)
    elif isinstance(value,list):
        for item in value: finite(item)
def closed_config(cfg):
    require(cfg['lambda_flat']==1. and cfg['ewc_lambda']==0. and cfg['gam_grad_gamma']==0.,'wrong loss/direction flags')
    require(cfg['ewc_skip_task_fisher'] is True,'Fisher not skipped')
    for key in ('approximate_fisher_in_task','in_task_weight_merge','task_endpoint_weight_merge',
                'compute_full_fisher','posterior_state_save','as1_delta_w_penalty','use_sgv','gam_use_sgv'):
        require(cfg.get(key,False) is False,'enabled forbidden component '+key)

def validate_job(job):
    out=Path(job['output_dir'])
    summary=read(out/'final_summary.json')
    finite(summary)
    require(summary['status']=='completed','missing successful runner completion')
    require(summary['seed']==job['seed'] and summary['formation_arm']==job['arm'],'result identity mismatch')
    require(summary['smoke']==job['smoke'],'smoke/formal mismatch')
    cfg=read(out/'effective_config.json');closed_config(cfg)
    require(cfg['seed']==job['seed'] and cfg['batch_size']==128 and cfg['lora_rank']==10,'wrong seed/batch/rank')
    require(cfg['init_momentum']==0 and cfg['momentum']==0 and cfg['init_weight_decay']==0 and cfg['weight_decay']==0,'unexpected SGD modifiers')
    require(cfg['gam_grad_rho']==.05 and cfg['gam_adaptive'] is False and cfg['gam_grad_beta_2']==1.,'wrong native GAM probes')
    rows=[json.loads(line) for line in (out/'step_trace.jsonl').read_text().splitlines()]
    require(len(rows)==job['expected_steps'],'wrong trace length')
    finite(rows)
    require(summary['step_trace_sha256']==sha(out/'step_trace.jsonl'),'trace hash mismatch')
    if job['kind']=='formation':
        multipliers={'s_lr15':1.5,'s_lr2':2.,'mean_lr2':2.}
        beta={'s':(1.,0.),'s_lr15':(1.,0.),'s_lr2':(1.,0.),'sum':(1.,1.),'mean_lr2':(.5,.5),'g_raw':(0.,1.),'g_r025':(0.,1.),'g_match':(0.,1.)}
        multiplier=multipliers.get(job['arm'],1.)
        require(cfg['formation_arm']==job['arm'] and cfg['init_lr']==.04*multiplier and cfg['lrate']==.02*multiplier,'incorrect nominal LR control')
        require(cfg['epochs']==50 and cfg['init_epoch']==job['expected_epochs'] and cfg['cosine_min_lr_ratio']==.1 and summary['scheduler_T_max']==50,'wrong cosine schedule')
        require((cfg['gam_grad_beta_1'],cfg['gam_grad_beta_3'])==beta[job['arm']] and cfg['gam_grad_norm_rho']==(.25 if job['arm']=='g_r025' else 1.),'wrong selected formation direction/radius')
        require(summary['task']==0 and summary['trained_tasks']==[0],'not isolated Task0')
        require(summary['completed_epochs']==job['expected_epochs'],'wrong epoch count')
        require(summary['counters']['training_batches']==job['expected_steps'],'batch counter mismatch')
        require(summary['counters']['training_ce_gradient_passes']==5*job['expected_steps'],'closure pass mismatch')
        for key in ('finite_check','task_check','step_checks'): require(summary[key]['passed'] is True,'failed '+key)
        require(all(summary['checkpoint_checks'].values()),'missing checkpoint')
        grid=[0,1] if job['smoke'] else [0,10,20,30,40,50]
        require(summary['checkpoint_epochs']==grid and set(summary['checkpoint_checks'])=={str(e) for e in grid},'incomplete declared checkpoint grid')
        require(summary['trainer_sha256']==sha(HERE/'trainer/formation_task0.py') and summary['effective_step_sha256']==sha(HERE/'trainer/effective_step.py') and summary['base_trainer_sha256']==sha(HERE/'trainer/task0_trainer_base_v2.py'),'runner source mismatch')
        for i,row in enumerate(rows):
            require(row['step_index']==i and row['seed']==job['seed'] and row['task']==0,'trace sequence mismatch')
            require(row['epoch']==(1 if job['smoke'] else i//10+1) and row['batch']==(0 if job['smoke'] else i%10),'wrong epoch/batch sequence')
            require(row['self_spec_passed'] and row['closure_count']==4,'unverified GAM step')
            for key in ('decomposition_spec_exact','projector_spec_exact','base_sgd_exact'):
                require(row[key] is True,'failed own-step verification '+key)
            require(row['training_ce_gradient_passes']==5 and row['actual_effective_delta']['joint_norm']>0,'invalid actual update')
            if job['arm']=='g_match': require(row['matched'] is True and row['actual_match_relative_error']<=1e-3,'unmatched effective step')
        require(summary['final']['eval_samples']==307,'wrong Task0 test set')
        manifests=[]
        for epoch in summary['checkpoint_epochs']:
            cp=out/'checkpoints'/f'epoch_{epoch:03d}'
            require(all((cp/name).is_file() for name in ('lora_w_a_0.pt','lora_w_b_0.pt','lora_meta_0.json','fc_state_0.pt','training_state.pt','manifest.json')),'missing native checkpoint file')
            m=read(cp/'manifest.json')
            require(m['epoch']==epoch and m['seed']==job['seed'] and m['formation_arm']==job['arm'],'checkpoint identity mismatch')
            manifests.append({'epoch':epoch,'files_sha256':{p.name:sha(p) for p in cp.iterdir() if p.is_file()}})
    else:
        require(cfg['epochs']==50 and cfg['init_epoch']==50 and cfg['init_lr']==.04 and cfg['lrate']==.02,'wrong future learning schedule')
        require((cfg['gam_grad_beta_1'],cfg['gam_grad_beta_3'])==((1.,0.) if job['future_rule']=='s' else (0.,1.)) and cfg['gam_grad_norm_rho']==1.,'wrong future gradient/radius')
        require(summary['task']==1 and summary['completed_steps']==job['expected_steps'],'wrong future count/task')
        require(summary['formation_epoch']==job['formation_epoch'] and summary['future_rule']==job['future_rule'],'wrong future origin/rule')
        require(summary['checkpoint_unchanged'] and summary['source_unchanged'],'future mutated frozen input')
        require(summary['identity_sha256']==sha(out/'identity.json') and summary['evaluations_sha256']==sha(out/'evaluations.json'),'future output hash mismatch')
        evaluations=read(out/'evaluations.json');finite(evaluations)
        milestones={0,1} if job['smoke'] else {0,1,10,50,100}
        require(set(evaluations['milestones'])=={str(k) for k in milestones},'missing raw evaluation milestone')
        require(summary['final_evaluation']==evaluations['milestones'][str(job['expected_steps'])],'final evaluation differs from raw milestone')
        for value in evaluations['milestones'].values():
            require(set(value)=={'task0','task1'},'missing evaluation task')
            for task,counts in (('task0',{'full':307,'diagnostic':102,'reporting':205}),('task1',{'full':254,'diagnostic':84,'reporting':170})):
                for subset,count in counts.items():
                    for space in ('local','all_seen'):
                        require(value[task][subset][space]['samples']==count,'wrong evaluation sample count')
        identity=read(out/'identity.json')
        require(identity['seed']==job['seed'] and identity['formation_arm']==job['arm'] and identity['formation_epoch']==job['formation_epoch'] and identity['future_rule']==job['future_rule'],'wrong future identity')
        require(identity['config_sha256']==job['config_sha256'] and identity['selection_split_sha256']==sha(HERE/'selection_split.json'),'future input pins differ')
        require(identity['trainer_sha256']==sha(HERE/'trainer/future_probe.py') and identity['effective_step_sha256']==sha(HERE/'trainer/effective_step.py'),'future script pins differ')
        require(identity['expansion']['old_rows_exact'] is True and identity['expansion']['backbone_exact'] is True,'changed original rows on expansion')
        require(identity['lr_schedule']=={'start':.02,'eta_min':.002,'T_max':50,'unit':'completed native epochs'},'wrong future schedule identity')
        require({p.name:sha(p) for p in Path(identity['checkpoint_dir']).iterdir() if p.is_file()}==identity['checkpoint_files_sha256'],'original checkpoint changed after fork')
        require(identity['task1_train_samples']==997 and identity['task1_batches_per_epoch']==8,'incorrect future data budget')
        require(identity['training_old_data'] is False,'old examples used to train')
        require(summary['counters']['fisher_estimation_passes']==0 and summary['counters']['weight_merge_calls']==0,'unclean future train')
        require(set(summary['milestones_completed'])==({0,1} if job['smoke'] else {0,1,10,50,100}),'missing future milestone')
        for i,row in enumerate(rows,1):
            require(row['step_index']==i and row['task']==1 and row['seed']==job['seed'],'future trace sequence mismatch')
            require(row['epoch']==(i-1)//8+1 and row['batch']==(i-1)%8,'wrong future data step sequence')
            expected_lr=.002+.5*(.02-.002)*(1+math.cos(math.pi*((i-1)//8)/50))
            require(math.isclose(row['lr_nominal'],expected_lr,rel_tol=1e-12,abs_tol=1e-14),'wrong future per-epoch cosine LR')
            require(row['closure_count']==4,'future missing native probes')
            for key in ('self_spec_passed','decomposition_spec_exact','projector_spec_exact','base_sgd_exact'):
                require(row[key] is True,'failed future update verification '+key)
            require(row['training_ce_gradient_passes']==5 and row['training_forward_count']==5 and row['fisher_gradient_passes']==0 and row['weight_merge_calls']==0,'wrong future execution counters')
            if job['future_rule']=='g_match': require(row['actual_match_relative_error']<=1e-3,'future effective norm mismatch')
        manifests=[]
    require(sha(job['config'])==job['config_sha256'],'job config changed')
    return {'config_sha256':job['config_sha256'],'summary_sha256':sha(out/'final_summary.json'),'step_trace_sha256':sha(out/'step_trace.jsonl'),
            'identity_sha256':sha(out/'identity.json'),'evaluations_sha256':sha(out/'evaluations.json') if job['kind']=='future' else None,'checkpoints':manifests,
            'summary':summary}

def receipt_matches(receipt,actual):
    return all(receipt.get(k)==actual.get(k) for k in ('config_sha256','summary_sha256','step_trace_sha256','identity_sha256','evaluations_sha256','checkpoints'))
