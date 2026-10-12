"""Strict lifecycle and global c21 residual audit for the two natural-step controls."""
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from common import inspect_metrics
HERE=Path(__file__).resolve().parent
REFERENCE='same_candidate_state_same_batch_native_projected_g1_after_native_unperturb_actual_FP32_SGD'
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

def validate_c21_audit(r,job):
    audit=r.get('c21_audit')
    require(isinstance(audit,dict),'missing global c21 residual audit')
    finite(audit)
    require(audit['schema_version']==1 and audit['rule']==job['rule'],'wrong residual audit identity')
    require(audit['projection_scope']=='all_trainable_LoRA_and_seen_FC' and audit['reduction_dtype']=='float64','projection is not global over trainable LoRA and seen FC')
    require(type(audit['tensor_count']) is int and audit['tensor_count']>0 and type(audit['parameter_count']) is int and audit['parameter_count']>0,'invalid global projection coverage')
    expected='g0+(g2-g1)' if job['rule']=='c21_base' else 'g1+((g2-g1)-a*g1)'
    require(audit['formula']==expected and audit['selected_spec_exact'] is True,'wrong residual update formula/specification')
    norms=audit['norms']
    require(set(norms)=={'g0','g1','g2','g3','c21','perp','selected'} and all(v>=0 for v in norms.values()),'incomplete/nonphysical residual norms')
    q=audit['norm_g1_sq'];dot=audit['dot_g1_c21'];a=audit['coefficient_a']
    require(q>=0 and math.isclose(norms['g1']**2,q,rel_tol=1e-12,abs_tol=1e-20),'g1 squared norm mismatch')
    require(audit['g1_zero'] is (q==0) and audit['c21_zero'] is (norms['c21']==0),'wrong zero-vector flags')
    expected_a=dot/q if q else 0.
    require(math.isclose(a,expected_a,rel_tol=1e-12,abs_tol=1e-15),'coefficient is not global dot(g1,c21)/dot(g1,g1)')
    if q==0:require(dot==0 and audit['dot_g1_perp']==0,'nonzero dot with zero g1')
    ratio=norms['perp']/norms['c21'] if norms['c21'] else 0.
    require(math.isclose(audit['perp_over_c21'],ratio,rel_tol=1e-12,abs_tol=1e-15),'residual ratio conflicts with observed norms')
    denom=norms['g1']*norms['c21']
    error=abs(audit['dot_g1_perp'])/denom if denom else 0.
    require(math.isclose(audit['orthogonality_relative_error'],error,rel_tol=1e-12,abs_tol=1e-15),'orthogonality error conflicts with observed dot/norms')
    require(audit['orthogonality_tolerance']==1e-5 and error<=1e-5 and audit['orthogonality_passed'] is True,'orthogonal residual failed frozen numerical tolerance')
    # This checks the geometry independently of the selector's Boolean claim.
    # Float32 residual storage can round; bound error against the original residual.
    projected_sq=max(0.,norms['c21']**2-dot*dot/q) if q else norms['c21']**2
    require(abs(norms['perp']**2-projected_sq)<=1e-5*max(norms['c21']**2,1e-30),'residual norm is inconsistent with removing the global parallel component')

def validate_job(job):
    require(job['kind']=='horizon' and job['rule'] in ('c21_base','c21_perp','us') and job['scale']=='natural','job outside native c21 A/B scope')
    require(job['seed']==2 and job['radius']==.25,'wrong frozen seed/radius')
    require(job['rule']!='us' or job['smoke'] is True,'US is compatibility smoke only, not a new formal experiment')
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
    require(s['status']=='completed' and s['seed']==job['seed'] and s['component_rule']==job['rule'] and s['component_scale']==job['scale'],'wrong horizon identity/completion')
    require(s['smoke']==job['smoke'] and s['completed_tasks']==job['expected_tasks'] and s['task_count']==job['expected_tasks'],'wrong task/smoke count')
    require(s['completed_epochs']==job['expected_tasks']*job['expected_epochs_per_task'] and s['completed_steps']==job['expected_steps'],'wrong epoch/step budget')
    require(s['scheduler_T_max']==50 and s['input_config_sha256']==config_sha,'schedule/input hash mismatch')
    require(s['trainer_sha256']==sha(HERE/'trainer/component_horizon.py') and s['effective_step_sha256']==sha(HERE/'trainer/effective_step.py'),'wrong executable source')
    source_gam=read(HERE/'source_manifest.json')['files']['optimer_PerturabtionType/gam.py']
    require(s['source_gam_sha256']==source_gam,'wrong frozen GAM implementation')
    require(s['reference_definition']==REFERENCE,'wrong matching definition')
    cfg=read(out/'effective_config.json');closed(cfg,job)
    require(s['effective_config_sha256']==sha(out/'effective_config.json'),'effective config changed')
    for key in ('finite','all_steps_self_spec','fisher_empty_all_tasks','actual_weight_merges_zero','complete_task_lifecycle'):
        require(s['checks'][key] is True,'failed native check '+key)
    ctr=s['counters'];require(ctr['training_batches']==job['expected_steps'] and ctr['training_ce_gradient_passes']==5*job['expected_steps'],'invalid training counter')
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
            require(r['closure_count']==4 and r['training_forward_count']==5 and r['training_ce_gradient_passes']==5,'wrong GAM execution count')
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
            if job['rule'] in ('c21_base','c21_perp'):validate_c21_audit(r,job)
            if job['scale']=='matched':require(r['matched'] is True and r['actual_match_relative_error']<=.001,'unmatched candidate step')
            if job['scale']=='natural' or job['rule']=='us':require(r['step_scale']==r['lr_nominal'],'nominal scalar identity violated')
            if job['rule']=='us':require(r['us_nominal_reference_after_tensors_exact'] is True,'US no longer native baseline')
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
