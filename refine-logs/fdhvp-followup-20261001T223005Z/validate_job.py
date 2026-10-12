"""Strict native lifecycle, own-state step and curvature completion validation."""
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from common import inspect_metrics
HERE=Path(__file__).resolve().parent
REFERENCE='same_candidate_state_same_batch_native_projected_g1_after_native_unperturb_actual_FP32_SGD'
RESPONSE_BINDINGS={
    'component_first_response':'first_response',
    'component_second_response':'second_response',
    'component_lambda_corr':'lambda_corr',
    'component_attention_backend':'attention_backend',
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)
def finite(value):
    if isinstance(value,float):require(math.isfinite(value),'nonfinite observed value')
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,list):
        for v in value:finite(v)
def response_protocol(value,job,label):
    """Bind the response intervention, not just the five outer rule names."""
    for config_key,job_key in RESPONSE_BINDINGS.items():
        require(config_key in value and job_key in job and value[config_key]==job[job_key],label+' response identity mismatch: '+config_key)
    first,second=job['first_response'],job['second_response']
    coefficient=job['lambda_corr'];backend=job['attention_backend']
    require(first in ('fd','hvp') and second in ('fd','hvp'),'invalid response family')
    require(type(coefficient) in (int,float) and math.isfinite(coefficient) and coefficient>0,'invalid response coefficient')
    require(backend in ('native','math'),'invalid attention backend')
    require((first==second=='fd') or (job['rule']=='ug' and backend=='math'),'HVP training requires UG and common math backend')
    require(coefficient==1. or job['rule'] in ('us','ug'),'coefficient is undefined for this component')
    return int(first=='hvp')+int(second=='hvp')
def validate_response(row,job):
    a=row['response_audit'];n=int(job['first_response']=='hvp')+int(job['second_response']=='hvp')
    require(a['first_response']==job['first_response'] and a['second_response']==job['second_response'] and a['lambda_corr']==job['lambda_corr'] and a['backend']==job['attention_backend'],'step response protocol differs')
    require(a['rho']==.05 and a['r']==job['radius'],'response radii differ')
    backend_flags={'flash':False,'math':True,'mem_efficient':False,'cudnn':False}
    require(set(a['backend_flags'])==set(backend_flags) and all(type(v) is bool for v in a['backend_flags'].values()),'missing actual attention backend state')
    if job['attention_backend']=='math':require(a['backend_flags']==backend_flags,'math FD/HVP control used an incompatible attention backend')
    require(a['native_closure_count']==4 and a['native_ce_gradient_passes']==5,'native gradient budget changed')
    require(all(a[k]==n for k in ('hvp_forward_count','hvp_first_derivative_passes','hvp_second_derivative_passes')),'HVP derivative execution count differs')
    require(a['fallback_used'] is False,'HVP update silently fell back to FD')
    points={'first_inner_exact','outer_point_exact','second_inner_exact'}
    require(set(a['native_point_checks'])==points and all(a['native_point_checks'][k] is True for k in points),'response evaluated at wrong actual native point')
    expected={key for key in ('first','second') if job[key+'_response']=='hvp'}
    require(set(a['hvp_reports'])==expected,'HVP response missing or unexpectedly executed')
    for key,h in a['hvp_reports'].items():
        require(h['status']=='available' and h['backend']=='math' and h['fallback_used'] is False,'HVP training requires available math derivative without fallback')
        require(h['backend_flags']==backend_flags,'HVP autograd did not actually enforce math attention')
        require(h['stage']==('original' if key=='first' else 'outer') and h['native_closure_index']==(0 if key=='first' else 2),'HVP taken at wrong closure point')
        require(h['direction_detached'] is True and h['rho_multiplication_exact'] is True and h['rho']==.05,'HVP direction/rho specification invalid')
        require(h['direction_matches_native_unit_exact'] is True and 0<=h['direction_norm']<=1.00001,'HVP direction is not the local native normalized gradient')
        require(h['hvp_forward_count']==h['first_derivative_passes']==h['second_derivative_passes']==1,'HVP was not differentiated exactly once')
        require(h['hvp_norm']>=0 and h['rho_hvp_norm']>=0 and math.isclose(h['rho_hvp_norm'],.05*h['hvp_norm'],rel_tol=2e-6,abs_tol=1e-12),'HVP correction omitted or altered rho factor')
        agreement=h['same_point_gradient']
        require(agreement['relative_l2']<=2e-5 if agreement['relative_l2'] is not None else agreement['max_abs']<=1e-10,'HVP first derivative differs from actual closure gradient')
        require(abs(h['same_point_ce']-h['native_baseline_ce'])<=1e-5*max(1.,abs(h['native_baseline_ce'])),'HVP replay changed CE or current-task slice')
        require(math.isclose(h['same_point_ce_abs_error'],abs(h['same_point_ce']-h['native_baseline_ce']),rel_tol=1e-12,abs_tol=1e-12),'same-point CE difference misreported')
        keys={'parameters_exact','frozen_versions_exact','buffers_exact','modes_exact','gradients_exact','rng_exact'}
        require(set(h['entry_state_restore'])==keys and all(h['entry_state_restore'][k] is True for k in keys),'HVP replay failed to restore entry state')
    require(a['correction_norm']>=0 and a['g0_norm']>=0,'invalid response norm')
    if job['second_response']=='hvp':require(math.isclose(a['correction_norm'],a['hvp_reports']['second']['rho_hvp_norm'],rel_tol=1e-12,abs_tol=1e-12),'selected correction differs from rho HVP')
    require(a['us_lambda1_identity_control'] is (job['rule']=='us' and job['lambda_corr']==1.),'wrong US identity scope')
    formula=('g0 + lambda_corr * rho * H(theta2) * stopgrad(N_eps(g2))' if job['rule']=='ug' and job['second_response']=='hvp'
        else 'g0 + lambda_corr * (g3 - g2)' if job['rule']=='ug' else 'g1 exactly' if job['rule']=='us' and job['lambda_corr']==1.
        else 'g0 + lambda_corr * (g1 - g0)' if job['rule']=='us' else job['rule'])
    require(a['selected_formula']==formula,'selected SGD formula mislabeled')
    flow=row['gradient_flow'];hashes=('selected_expected_sha256','selected_actual_sha256','projected_expected_sha256','projected_actual_sha256','actual_sgd_gradient_sha256')
    require(all(re.fullmatch('[0-9a-f]{64}',flow[k]) for k in hashes),'missing gradient-flow identity')
    require(flow['selected_to_projector_to_sgd_exact'] is True and flow['selected_expected_sha256']==flow['selected_actual_sha256'] and flow['projected_expected_sha256']==flow['projected_actual_sha256']==flow['actual_sgd_gradient_sha256'],'selected response did not reach actual SGD')
    return n
def closed(cfg,job):
    require(cfg['dataset']=='imageneta' and cfg['class_shuffle'] is False and cfg['lora_rank']==10 and cfg['batch_size']==128,'wrong data/rank/batch')
    require(cfg['lambda_flat']==1. and cfg['ewc_lambda']==0. and cfg['ewc_skip_task_fisher'] is True,'wrong no-Fisher loss')
    for key in ('approximate_fisher_in_task','in_task_weight_merge','task_endpoint_weight_merge','compute_full_fisher','posterior_state_save','as1_delta_w_penalty','use_sgv','gam_use_sgv','ewc_normalize_fisher','ewc_normalize_accumulated_fisher'):
        require(cfg.get(key,False) is False,'forbidden module '+key)
    require(cfg['gam_grad_rho']==.05 and cfg['gam_grad_norm_rho']==job['radius'] and cfg['gam_grad_gamma']==0.,'wrong probe radii/gamma')
    require(cfg['init_lr']==.04 and cfg['lrate']==.02 and cfg['epochs']==50 and cfg['init_epoch']==50 and cfg['cosine_min_lr_ratio']==.1,'wrong native full schedule')
    require(all(cfg[key]==0 for key in ('init_momentum','momentum','init_weight_decay','weight_decay')),'unexpected SGD modifiers')
    require(cfg['seed']==job['seed'] and cfg['component_rule']==job['rule'] and cfg['component_scale']==job['scale'],'wrong effective identity')
    require(cfg['model_name']=='ewclora_normfisher_gam_intask_damp' and cfg['optimizer_type']=='gam' and cfg['gam_base_optimizer']=='sgd' and cfg['optimizer']=='sgd','wrong model/optimizer')
    require((cfg['gam_grad_beta_1'],cfg['gam_grad_beta_2'],cfg['gam_grad_beta_3'])==(1.,1.,0.) and cfg['gam_adaptive'] is False and cfg['sgv_lambda']==0.,'wrong native probe construction')
    response_protocol(cfg,job,'effective config')

def validate_job(job):
    out=Path(job['output_dir']);config_sha=sha(job['config'])
    require(config_sha==job['config_sha256'],'input config changed')
    if job['kind']=='curvature':
        p=out/'curvature_audit.json';s=read(p);finite(s)
        require(s['status']=='completed' and s['task']==job['task'] and s['seed']==job['seed'],'invalid curvature completion')
        require(s['script_sha256']==sha(HERE/'diagnostics/curvature_audit.py') and s['config_sha256']==config_sha,'curvature input pins mismatch')
        require(s['checkpoint_unchanged'] and s['source_unchanged'],'curvature mutated source/checkpoint')
        require({p.name:sha(p) for p in Path(s['checkpoint_dir']).iterdir() if p.is_file()}==s['checkpoint_files_sha256'],'curvature checkpoint changed')
        require(s['checkpoint_manifest']['component_rule']==job['rule'] and s['checkpoint_manifest']['component_scale']==job['scale'],'wrong curvature origin')
        suite=read(HERE/'suite.json');upstream=next(j for j in suite['jobs'] if j['id']==job['depends_on'][0])
        cp=Path(upstream['output_dir'])/'checkpoints'/f"task_{job['task']:03d}"
        require(Path(s['checkpoint_dir'])==cp,'curvature checkpoint differs from declared upstream')
        parent_receipt=read(HERE/'runtime/jobs'/upstream['id']/'completed.json')
        require(parent_receipt['job_id']==upstream['id'] and parent_receipt['exit']['returncode']==0 and receipt_matches(parent_receipt,validate_job(upstream)),'curvature parent not validated')
        expected_source={k:v for k,v in read(HERE/'source_manifest.json')['files'].items() if k.endswith('.py')}
        require(s['source_python_sha256']==expected_source,'curvature source pins differ')
        require(all(sha(Path(suite['source_root'])/k)==v for k,v in expected_source.items()),'curvature frozen source changed')
        require(len(s['batches'])==job['max_batches'],'wrong diagnostic batch count')
        dataset=read(HERE/'data_inventory.json')['tasks'][job['task']]
        for i,b in enumerate(s['batches']):
            require(b['batch']==i and b['rho']==.05 and b['r']==job['radius'] and b['epsilons']==[.05,.025,.0125],'wrong HVP/FD setup')
            keys={'parameters_exact','frozen_versions_exact','buffers_exact','modes_exact','gradients_exact','rng_exact'}
            require(set(b['restore'])==keys and all(b['restore'][k] is True for k in keys),'HVP restoration failed')
            native=b['native'];points={'g1_native_perturb_exact','g2_native_roundtrip_outer_exact','g3_native_perturb_exact'}
            require(native['closure_count']==4 and native['training_forward_count']==5 and native['training_ce_gradient_passes']==5,'curvature native probes missing')
            require(set(native['native_point_checks'])==points and all(native['native_point_checks'][k] is True for k in points),'curvature probe points invalid')
            require(b['samples']==min(128,dataset['train_n']-i*128) and len(b['indices'])==b['samples'] and len(set(b['indices']))==b['samples'] and all(type(v) is int and 0<=v<dataset['train_n'] for v in b['indices']),'wrong fixed diagnostic samples')
            require(b['data_seed']==job['seed']*100000+70000+job['task'] and all(re.fullmatch('[0-9a-f]{64}',b[k]) for k in ('input_sha256','target_sha256')),'wrong diagnostic sample identity')
            require(set(b['responses'])=={'original','outer'},'missing curvature response')
            for response in b['responses'].values():
                require(response['hvp']['status'] in ('available','unsupported'),'invalid HVP state')
                require(len(response['finite_difference_scan'])==3,'missing FD convergence grid')
                require([v['epsilon'] for v in response['finite_difference_scan']]==[.05,.025,.0125],'wrong FD grid')
                require(response['hvp']['direction_detached'] is True,'HVP direction contains differentiation graph')
                agreement=response['same_point_native_gradient'];require(agreement['relative_l2']<=2e-5 if agreement['relative_l2'] is not None else agreement['max_abs']<=1e-10,'diagnostic replay disagrees with native gradient')
                if response['hvp']['status']=='unsupported':
                    require(response['hvp']['attempts'] and all(a['unsupported_double_backward'] is True for a in response['hvp']['attempts']),'HVP absence not explained')
                    require(response['native_finite_response_vs_hvp'] is None and all(v['forward_vs_hvp'] is None and v['central_vs_hvp'] is None for v in response['finite_difference_scan']),'invented unsupported HVP metric')
                else:
                    require(response['native_finite_response_vs_hvp'] is not None and response['hvp']['backend'] in ('native','math'),'missing available HVP comparison')
                    hvp_agreement=response['hvp']['same_point_gradient']
                    require(hvp_agreement['relative_l2']<=2e-5 if hvp_agreement['relative_l2'] is not None else hvp_agreement['max_abs']<=1e-10,'HVP backend disagrees with native gradient')
                    require(abs(response['hvp']['same_point_ce']-response['native_baseline_ce'])<=1e-5*max(1.,abs(response['native_baseline_ce'])),'HVP backend loss differs from native')
        ids=[v for b in s['batches'] for v in b['indices']];require(len(ids)==len(set(ids)),'duplicate fixed diagnostic images')
        require(s['hvp_all_available']==all(r['hvp']['status']=='available' for b in s['batches'] for r in b['responses'].values()),'HVP availability conflicts with records')
        return {'config_sha256':config_sha,'summary_sha256':sha(p),'step_trace_sha256':None,'identity_sha256':None,
            'evaluations_sha256':sha(p),'checkpoints':[],'summary':s}
    s=read(out/'final_summary.json');finite(s)
    response_protocol(read(job['config']),job,'input config')
    per_step_hvp=response_protocol(s,job,'summary')
    response_protocol(s['response_config'],job,'summary nested config')
    require(s['r']==job['radius'] and s['rho']==.05,'summary probe radii differ')
    require(s['status']=='completed' and s['seed']==job['seed'] and s['component_rule']==job['rule'] and s['component_scale']==job['scale'],'wrong horizon identity/completion')
    require(s['smoke']==job['smoke'] and s['completed_tasks']==job['expected_tasks'] and s['task_count']==job['expected_tasks'],'wrong task/smoke count')
    require(s['completed_epochs']==job['expected_tasks']*job['expected_epochs_per_task'] and s['completed_steps']==job['expected_steps'],'wrong epoch/step budget')
    require(s['scheduler_T_max']==50 and s['input_config_sha256']==config_sha,'schedule/input hash mismatch')
    require(s['trainer_sha256']==sha(HERE/'trainer/component_horizon.py') and s['effective_step_sha256']==sha(HERE/'trainer/effective_step.py'),'wrong executable source')
    require(s['hvp_helpers_sha256']==sha(HERE/'trainer/hvp_helpers.py') and s['curvature_primitives_sha256']==sha(HERE/'diagnostics/curvature_audit.py'),'wrong HVP helper implementation')
    source_gam=read(HERE/'source_manifest.json')['files']['optimer_PerturabtionType/gam.py']
    require(s['source_gam_sha256']==source_gam,'wrong frozen GAM implementation')
    require(s['reference_definition']==REFERENCE,'wrong matching definition')
    cfg=read(out/'effective_config.json');closed(cfg,job)
    require(s['effective_config_sha256']==sha(out/'effective_config.json'),'effective config changed')
    for key in ('finite','all_steps_self_spec','fisher_empty_all_tasks','actual_weight_merges_zero','complete_task_lifecycle'):
        require(s['checks'][key] is True,'failed native check '+key)
    ctr=s['counters'];require(ctr['training_batches']==job['expected_steps'] and ctr['training_ce_gradient_passes']==5*job['expected_steps'],'invalid training counter')
    require(all(ctr[k]==per_step_hvp*job['expected_steps'] for k in ('hvp_forward_passes','hvp_first_derivative_passes','hvp_second_derivative_passes')),'HVP summary budget differs from protocol')
    require(ctr['total_training_forward_passes']==(5+per_step_hvp)*job['expected_steps'],'total forward budget differs from native plus HVP')
    require(all(ctr[k]==0 for k in ('fisher_estimation_passes','weight_merge_updates','forbidden_work_attempts')),'forbidden work performed')
    inventory=read(HERE/'data_inventory.json')['tasks'][:job['expected_tasks']]
    details=s['task_details'];require(len(details)==len(inventory),'incomplete task accounting')
    identities=read(out/'task_identities.json');require(s['task_identities_sha256']==sha(out/'task_identities.json') and len(identities)==len(inventory),'incomplete task identities')
    rows=[json.loads(line) for line in (out/'step_trace.jsonl').read_text().splitlines()];finite(rows)
    require(len(rows)==job['expected_steps'] and s['step_trace_sha256']==sha(out/'step_trace.jsonl'),'incomplete trace/hash')
    offset=0
    for task,(actual,expected) in enumerate(zip(details,inventory)):
        batches=1 if job['smoke'] else expected['batches_per_epoch'];epochs=job['expected_epochs_per_task'];steps=batches*epochs
        require(actual['task']==task and actual['train_dataset_samples']==expected['train_n'] and actual['batches_per_epoch']==batches and actual['epochs']==epochs and actual['steps']==steps,'wrong per-task training budget')
        ident=identities[task]
        require(ident['task']==task and ident['seed']==job['seed'] and ident['old_fc_rows_inherited_exact'] is True and ident['fc_shape'][0]==20*(task+1),'wrong head inheritance/identity')
        for index,r in enumerate(rows[offset:offset+steps]):
            require(r['step_index']==offset+index and r['task_step_index']==index and r['task']==task and r['epoch']==index//batches+1 and r['batch']==index%batches,'wrong trace task/epoch/batch sequence')
            require(r['component_rule']==job['rule'] and r['component_scale']==job['scale'] and r['seed']==job['seed'],'wrong trace rule/seed')
            extra=validate_response(r,job)
            require(r['closure_count']==4 and r['training_forward_count']==5+extra and r['native_training_forward_count']==5 and r['training_ce_gradient_passes']==5,'wrong GAM execution count')
            require(all(r[k] is True for k in ('self_spec_passed','native_decomposition_exact','component_spec_exact','projector_spec_exact','base_sgd_exact','reference_counterfactual_state_unchanged')),'failed actual step specification')
            eta0=.04 if task==0 else .02;eta=eta0*(.1+.45*(1+math.cos(math.pi*(r['epoch']-1)/50)))
            require(math.isclose(r['lr_nominal'],eta,rel_tol=1e-12,abs_tol=1e-14),'wrong native cosine LR')
            require(r['seen_classes']==20*(task+1) and r['fc_shape'][0]==20*(task+1) and r['class_slice']==[20*task,20*(task+1)],'wrong full-seen FC or loss slice')
            require(r['reference_target_norm']>0 and r['actual_effective_delta']['joint_norm']>0,'invalid effective norm')
            require(r['reference_definition']==REFERENCE,'wrong reference trajectory semantics')
            require(r['step_scale']>0 and r['actual_match_relative_error']>=0,'invalid scalar/match error')
            actual_error=abs(r['actual_effective_delta']['joint_norm']-r['reference_target_norm'])/r['reference_target_norm']
            require(math.isclose(r['actual_match_relative_error'],actual_error,rel_tol=1e-12,abs_tol=1e-12),'reported norm error not actual')
            identity=r['identity'];require(identity['data_seed']==job['seed']*100000+task*1000+r['epoch'],'wrong epoch data seed')
            ids=identity['indices'];require(len(ids)==r['batch_samples'] and len(ids)==len(set(ids)) and all(type(v) is int and 0<=v<expected['train_n'] for v in ids),'invalid batch sample IDs')
            require(all(re.fullmatch('[0-9a-f]{64}',identity[k]) for k in ('input_sha256','target_sha256','indices_sha256')),'missing batch identity hash')
            require(hashlib.sha256(struct.pack(str(len(ids))+'q',*ids)).hexdigest()==identity['indices_sha256'],'sample IDs disagree with their hash')
            if job['scale']=='matched':require(r['matched'] is True and r['actual_match_relative_error']<=.001,'unmatched candidate step')
            if job['scale']=='natural' or (job['rule']=='us' and job['lambda_corr']==1.):require(r['step_scale']==r['lr_nominal'],'nominal scalar identity violated')
            if job['rule']=='us' and job['lambda_corr']==1.:require(r['us_nominal_reference_after_tensors_exact'] is True,'US no longer native baseline')
        offset+=steps
    require(offset==len(rows),'extra observed steps')
    epochs=read(out/'metrics.json')['epochs'];finite(epochs)
    require(len(epochs)==s['completed_epochs'],'incomplete epoch metrics')
    for index,metric in enumerate(epochs):
        task=index//job['expected_epochs_per_task'];epoch=index%job['expected_epochs_per_task']+1
        selected=[r for r in rows if r['task']==task and r['epoch']==epoch]
        ids=[v for r in selected for v in r['identity']['indices']]
        require(metric['task']==task and metric['epoch']==epoch and metric['samples']==len(ids) and metric['batches']==len(selected),'epoch metrics conflict with steps')
        require(len(ids)==len(set(ids)) and (job['smoke'] or len(ids)==inventory[task]['train_n']),'incomplete/duplicate epoch data coverage')
    after=s['after_task_checks'];require(len(after)==len(inventory),'missing task-end checks')
    require(all(x['task']==i and x['parameters_exact'] and x['parameter_references_exact'] and x['fisher_empty'] for i,x in enumerate(after)),'unclean task boundary')
    mapping=read(out/'task_checkpoints.json');require(s['task_checkpoints_sha256']==sha(out/'task_checkpoints.json') and set(mapping['all_tasks'])=={str(i) for i in range(job['expected_tasks'])},'missing checkpoint map')
    checkpoints=[]
    for task in range(job['expected_tasks']):
        cp=out/'checkpoints'/f'task_{task:03d}';m=read(cp/'manifest.json')
        response_protocol(m,job,'checkpoint')
        response_protocol(m['response_config'],job,'checkpoint nested config')
        require(Path(mapping['all_tasks'][str(task)])==cp and m['task']==task and m['seed']==job['seed'] and m['component_rule']==job['rule'] and m['component_scale']==job['scale'],'checkpoint binding mismatch')
        require(m['fc_shape'][0]==20*(task+1) and m['epoch']==job['expected_epochs_per_task'],'wrong checkpoint head/epoch')
        required={f'lora_w_a_{task}.pt',f'lora_w_b_{task}.pt',f'lora_meta_{task}.json',f'fc_state_{task}.pt'}
        require(set(m['files_sha256'])==required and all((cp/name).is_file() for name in required|{'manifest.json','training_state.pt'}),'incomplete native checkpoint')
        require(m['source_gam_sha256']==source_gam and m['class_range']==[0,20*(task+1)] and m['current_task_class_range']==[20*task,20*(task+1)],'checkpoint source/class range mismatch')
        require(all(sha(cp/name)==expected for name,expected in m['files_sha256'].items()),'native checkpoint file pin changed')
        checkpoints.append({'task':task,'files_sha256':{p.name:sha(p) for p in cp.iterdir() if p.is_file()}})
    matrix=read(out/'task_matrix.json');finite(matrix)
    require(s['task_matrix_sha256']==sha(out/'task_matrix.json') and s['final_metrics']==matrix,'matrix result mismatch')
    require(len(matrix['matrix'])==job['expected_tasks'] and all(len(row)==i+1 for i,row in enumerate(matrix['matrix'])),'wrong matrix shape')
    suite=read(HERE/'suite.json');native=inspect_metrics(suite,job)
    require([row[:i+1] for i,row in enumerate(native['cnn_matrix'])]==matrix['matrix'],'wrapper/native CNN matrix differs')
    require(math.isclose(native['faa_task_macro'],matrix['FAA_task_macro'],abs_tol=1e-9) and math.isclose(native['aaa_task_macro'],matrix['AAA_task_macro'],abs_tol=1e-9),'matrix-derived metric differs')
    mm=matrix['matrix'];n=len(mm);la=sum(mm[i][i] for i in range(n))/n
    bwt=sum(mm[-1][j]-mm[j][j] for j in range(n-1))/(n-1)
    forgetting=sum(max(mm[t][j] for t in range(j,n))-mm[-1][j] for j in range(n-1))/(n-1)
    require(math.isclose(matrix['LA'],la,abs_tol=1e-9) and math.isclose(matrix['BWT'],bwt,abs_tol=1e-9) and math.isclose(matrix['forgetting'],forgetting,abs_tol=1e-9),'acquisition/retention metrics conflict with matrix')
    require(matrix['sample_accuracy_curve']==native['cnn_top1_curve'] and math.isclose(matrix['FAA_sample_weighted'],native['faa_sample_weighted'],abs_tol=1e-9) and math.isclose(matrix['AAA_sample_weighted'],native['aaa_sample_weighted'],abs_tol=1e-9),'sample metric conflicts with native log')
    return {'config_sha256':config_sha,'summary_sha256':sha(out/'final_summary.json'),'step_trace_sha256':sha(out/'step_trace.jsonl'),
        'identity_sha256':sha(out/'task_identities.json'),'evaluations_sha256':sha(out/'task_matrix.json'),'checkpoints':checkpoints,'summary':s}

def receipt_matches(receipt,actual):
    return all(receipt.get(k)==actual.get(k) for k in ('config_sha256','summary_sha256','step_trace_sha256','identity_sha256','evaluations_sha256','checkpoints'))

def validate_paired_inputs(jobs,required_backend=None):
    """Post-completion control: same actual batches/augmentation and nominal LR.

    This checks paired observed inputs, not equality of model states, gradients,
    scalar-matched steps, or resulting trajectories. Pending jobs are excluded
    by the caller; a partial or unvalidated completion is never accepted here.
    """
    require(len(jobs)>=2,'a paired input audit needs at least two completed arms')
    require(len({job['id'] for job in jobs})==len(jobs),'duplicate arm in paired input audit')
    first=jobs[0];traces=[];receipts=[]
    for job in jobs:
        require(job['kind']=='horizon' and all(job[k]==first[k] for k in ('seed','radius','smoke','expected_tasks','expected_epochs_per_task','expected_steps')),'paired jobs use different data/schedule protocols')
        if required_backend is not None:require(job['attention_backend']==required_backend,'paired FD/HVP arms use different attention backends')
        result=validate_job(job);receipts.append({'job_id':job['id'],'step_trace_sha256':result['step_trace_sha256'],'summary_sha256':result['summary_sha256']})
        traces.append([json.loads(line) for line in (Path(job['output_dir'])/'step_trace.jsonl').read_text().splitlines()])
    for rows in traces[1:]:
        require(len(rows)==len(traces[0]),'paired trace lengths differ')
        for reference,candidate in zip(traces[0],rows):
            require(all(reference[k]==candidate[k] for k in ('step_index','task_step_index','task','epoch','batch','seed','lr_nominal','class_slice','seen_classes','batch_samples')),'paired minibatch order or LR differs')
            require(reference['identity']==candidate['identity'],'paired samples or augmentation differ')
    return {'status':'PASS','same_actual_inputs_and_nominal_lr':True,'steps_per_arm':len(traces[0]),'arms':receipts,'required_backend':required_backend,
            'scope':'No claim of equal model states, actual effective steps, or update trajectories.'}
