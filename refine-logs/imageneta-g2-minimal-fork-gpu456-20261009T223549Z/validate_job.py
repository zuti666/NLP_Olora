"""Accept completed own-task budgets, exact shared parent and passive diagnostics."""
import json,math
from pathlib import Path
from common import HERE as H,sha256

FORBIDDEN=('fisher_estimation_passes','weight_merge_updates','projection_calls','hvp_calls','qr_calls','create_graph_calls','gradient_clipping_calls','final_norm_rescale_calls')
def read(p):return json.loads(Path(p).read_text())
def finite(v):
    if isinstance(v,float) and not math.isfinite(v):raise AssertionError('Nonfinite evidence')
    if isinstance(v,dict):
        for x in v.values():finite(x)
    if isinstance(v,(list,tuple)):
        for x in v:finite(x)
def require(ok,message):
    if not ok:raise AssertionError(message)

def validate_offline(j):
    p=Path(j['output_dir'])/'final_summary.json';a=read(p);finite(a)
    require(a['status']=='passed' and a['committed_updates']==0 and not a['model_selection_eligible'] and not a['trained_method_result'],'Offline diagnostic became training/selection')
    require(a['gradient_reads']==j['expected_gradient_reads'] and a['loss_reads']==j['expected_loss_reads'] and a['checked_snapshots']==j['expected_snapshots'] and a['checked_batches']==j['expected_batches'],'Incorrect diagnostic budget')
    require(a['audit_sha256']==sha256(H/'trainer/local_probe_audit.py') and a['dev_split_sha256']==sha256(H/'dev_split.json'),'Diagnostic code/split differs')
    ref=Path(j['reference_output_dir']);require(a['reference_final_summary_sha256']==sha256(ref/'final_summary.json'),'Reference changed')
    require(a['smoke']==j['smoke'] and all(a['checks'].values()),'Diagnostic restoration/purity check failed')
    ids=[i for x in a['data_receipts'] for i in x['raw_train_indices']];split=read(H/'dev_split.json')
    require(len(ids)==len(set(ids)) and set(ids)<=set(split['train_indices']) and not set(ids)&set(split['dev_indices']),'Probe/holdout/dev membership overlap')
    require(all(20<=split['native_train_targets'][i]<40 for i in ids),'Probe read old task labels')
    expected_epochs=[0] if j['smoke'] else [0,1,10,25,49]
    require([s['epoch'] for s in a['snapshots']]==expected_epochs,'Wrong diagnostic states')
    for s in a['snapshots']:
        require(s['sha256']==sha256(s['path']),'Snapshot changed')
        for b in s['batches']:
            require(b['committed_updates']==0 and all(b['checks'].values()) and b['all_probes_share_same_entry_R0'] and b['no_R1_advance'],'Diagnostic state/RNG mutation')
    return dict(summary_sha256=sha256(p),gradient_reads=a['gradient_reads'],committed_updates=0,passive_diagnostic=True,full_formal_training=False)

def validate_job(j):
    out=Path(j['output_dir'])
    if j['kind']=='local_probe_audit':return validate_offline(j)
    if j['kind']=='pairing':
        p=out/j['report_file'];a=read(p)
        require(a['status']=='passed' and a['smoke_count']==7 and all(a['checks'].values()),'Paired GPU model smoke failed')
        return dict(summary_sha256=sha256(p),gate='passed',full_formal_training=False)
    p=out/'final_summary.json';a=read(p);finite(a);cfg=read(out/'effective_config.json');raw=read(j['config'])
    task=j['task_id'];method=j['method'];epochs=j['expected_epochs_per_task'];steps=j['expected_steps'];calls=1 if method=='sgd' else 3
    require(a['status']=='completed' and (a['completed_tasks'],a['epochs_per_task'],a['completed_epochs'],a['completed_steps'])==(task+1,epochs,epochs,steps),'Incomplete own-task budget')
    require(a['trained_task_ids']==[task] and a['training_tasks_this_job']==1,'Branch retrained Task0 or wrong task')
    require((a['response_method'],a['study_role'],a['study_condition_id'],a['seed'],a['history_source'])==(method,j['study_role'],j['study_condition_id'],2,None),'Wrong study method/role/seed')
    require(a['full_partition_tasks']==10 and a['total_dataset_classes']==200 and cfg['init_cls']==20 and cfg['increment']==20,'Changed class partition')
    require(a['smoke']==j['smoke'] and a['input_config_sha256']==j['config_sha256'] and sha256(j['config'])==j['config_sha256'],'Configuration/smoke differs')
    require(a['trainer_sha256']==sha256(H/'trainer/probe_study_train.py') and a['inherited_algorithm_sha256']==sha256(H/'trainer/response_step.py'),'Trainer/core changed')
    require(a['algorithm_sha256']==sha256(H/'trainer'/('response_step.py' if method=='sgd' else 'independent_response_step.py')),'Wrong active algorithm')
    require(a['dev_split_sha256']==sha256(H/'dev_split.json'),'Development split changed')
    require(cfg['ewc_lambda']==0 and cfg['ewc_skip_task_fisher'] and not any(cfg[k] for k in ('approximate_fisher_in_task','in_task_weight_merge','task_endpoint_weight_merge')),'Forbidden Fisher/fusion active')
    for k,v in dict(seed=2,lora_rank=10,init_epoch=50,epochs=50,init_lr=.04,lrate=.02,batch_size=128,cosine_min_lr_ratio=.1,history_weight=0.,momentum=0,weight_decay=0.,attention_backend='native').items():require(cfg[k]==v,'Changed baseline: '+k)
    for k in ('study_inner_mode','study_inner_radius','study_outer_mode','study_outer_radius'):require(cfg[k]==raw[k]==j[k],'Radius registration differs: '+k)
    require(all(a['checks'].values()) and a['history_used_steps']==0 and not a['gam_radii_active'],'Purity/state checks failed')
    receipt=a['dataset_partition_receipt'];require(receipt['full_train_samples']==5981 and receipt['full_test_samples']==1519 and receipt['class_order']==list(range(200)) and receipt['full_task_increments']==[20]*10,'Dataset source/partition changed')
    rows=[json.loads(x) for x in (out/'step_trace.jsonl').read_text().splitlines()];require(len(rows)==steps,'Incomplete step trace')
    coords=[(task,e,b) for e in range(1,epochs+1) for b in range(j['expected_batches_per_epoch'])];require(len(coords)==steps,'Incorrect manifest batch budget')
    for index,(r,coord) in enumerate(zip(rows,coords)):
        finite(r);require((r['task'],r['epoch'],r['batch'])==coord and r['step_index']==index and r['study_condition_id']==j['study_condition_id'] and r['study_role']==j['study_role'],'Wrong step coordinate/condition')
        require(r['method']==method and r['seed']==2 and r['class_slice']==[20*task,20*(task+1)] and r['seen_classes']==20*(task+1) and r['fc_shape'][0]==20*(task+1),'Wrong task/head/CE')
        n=r['ordinary_gradient_calls'];require(1<=n<=calls and (n==calls or r['fallback_reason']),'Incorrect gradient budget/fallback')
        require(r['actual_model_forward_count']==n+1 and r['diagnostic_loss_forward_count']==1 and r['sgd_updates']==1 and r['natural_sgd_exact'],'Wrong actual forwards/SGD update')
        require(all(r['state_restore_checks'].values()) and all(r['diagnostic_state_restoration'].values()) and r['only_base_R1_rng_advance'] and r['strict_deterministic'] and not any(r[k] for k in FORBIDDEN),'State/RNG/forbidden operation failed')
        eta=(.04 if task==0 else .02)*(.1+.9*(1+math.cos(math.pi*(coord[1]-1)/50))/2)
        require(math.isclose(r['nominal_lr'],eta,rel_tol=1e-12,abs_tol=1e-12) and math.isclose(r['h'],eta*r['g_grad_norm'],rel_tol=1e-12,abs_tol=1e-12),'Changed LR schedule or h')
        require(not r['history_used'] and not r['history_available'] and not r['history_cache_valid_after_update'] and r['effective_history_weight']==0,'History entered pure study')
        if method=='sgd':require(n==1 and r['g1_sha256'] is None and r['g2_sha256'] is None and r['U_sha256']==r['g0_sha256'],'SGD contains extra probes')
        else:
            require(r['radius_mode']=='independent_inner_outer_v1' and r['response_probe_variant']==j['study_condition_id'],'Independent radii ignored')
            for prefix in ('inner','outer'):
                mode=j['study_'+prefix+'_mode'];expected=j['study_'+prefix+'_radius'] if mode=='fixed_l2' else r['h']
                require(r[prefix+'_probe_radius_mode']==mode and r[prefix+'_probe_radius_theoretical']==expected,'Wrong actual probe policy')
                if not r['fallback_reason']:require(math.isclose(r[prefix+'_probe_radius_actual'],expected,rel_tol=1e-4,abs_tol=1e-8),'Measured probe radius differs')
            if n==3:require(r['final_gradient_sha256']==r['g2_sha256'] and r['relocation_origin']=='base','Final update is not unscaled base-origin g2')
        require(abs(r['post_update_delta_ce']-(r['post_update_ce']-r['loss0']))<1e-9,'Wrong passive CE delta')
    c=a['counters'];require(c['training_sgd_updates']==steps and c['training_ce_gradient_passes']==sum(r['ordinary_gradient_calls'] for r in rows) and c['diagnostic_loss_forward_count']==steps and c['training_forward_count']==c['training_ce_gradient_passes'],'Total cost counters mismatch')
    require(not any(c[k] for k in ('fisher_estimation_passes','weight_merge_updates','forbidden_work_attempts','training_hvp_passes','training_create_graph_passes','training_global_projection_calls')),'Forbidden total work')
    for name,key in [('step_trace.jsonl','step_trace_sha256'),('task_matrix.json','task_matrix_sha256')]:require(sha256(out/name)==a[key],'Result integrity differs')
    dev=read(out/'dev_curve.json')['epochs'];require(len(dev)==epochs and a['endpoint_dev']==dev[-1] and [d['epoch'] for d in dev]==list(range(1,epochs+1)),'Incomplete passive dev curve')
    split=read(H/'dev_split.json');expected_dev={i for i in split['dev_indices'] if 20*task<=split['native_train_targets'][i]<20*(task+1)}
    for d in dev:
        finite(d);require(d['task']==task and d['dev_samples']==j['expected_dev_samples'] and set(d['dev_raw_train_indices'])==expected_dev and d['passive_state_and_rng_restored'] and not d['test_used_for_selection'],'Wrong dev membership/state')
    m=read(out/'task_matrix.json');matrix=m['matrix'];n=len(matrix);finite(m);require(n==task+1 and all(len(row)==i+1 for i,row in enumerate(matrix)) and all(0<=v<=100 for row in matrix for v in row),'Wrong observed accuracy matrix')
    expected_metrics={'FAA_task_macro':sum(matrix[-1])/n,'AAA_task_macro':sum(sum(row)/len(row) for row in matrix)/n,'LA':sum(matrix[i][i] for i in range(n))/n,'BWT':matrix[1][0]-matrix[0][0] if n==2 else None,'forgetting':max(matrix[0][0],matrix[1][0])-matrix[1][0] if n==2 else None}
    for k,v in expected_metrics.items():require((m[k] is None and a['final_metrics'][k] is None) if v is None else abs(m[k]-v)<1e-9 and abs(a['final_metrics'][k]-v)<1e-9,'Metric recomputation failed: '+k)
    require(len(a['checkpoint_dirs'])==1 and len(a['task_details'])==1,'Own trained task checkpoint count differs')
    details=a['task_details'][0];require(details['task']==task and details['train_dataset_samples']==j['expected_train_samples'] and details['steps']==steps,'Wrong actual training samples/budget')
    cp=Path(a['checkpoint_dirs'][0]);manifest=read(cp/'manifest.json');require(manifest['task']==task and manifest['model_sha256']==a['final_model_parameter_sha256'] and manifest['fc_shape'][0]==20*(task+1),'Checkpoint model/FC/task differs')
    required={'training_state.pt',f'lora_w_a_{task}.pt',f'lora_w_b_{task}.pt',f'lora_meta_{task}.json',f'fc_state_{task}.pt'}
    if task==0:required.add('shared_parent.pt')
    require(required<=set(manifest['files_sha256']),'Incomplete native+complete checkpoint')
    for name,digest in manifest['files_sha256'].items():require(sha256(cp/name)==digest,'Checkpoint file changed')
    import torch
    state=torch.load(cp/'training_state.pt',map_location='cpu',weights_only=False);cache=state['algorithm_state']
    require(state['algorithm']==method and state['previous_batch_identity']==rows[-1]['identity'] and cache['method']==method and cache['real_updates']==steps and cache['history'] is None and cache['history_weight']==0.,'Checkpoint algorithm/update/cache mismatch')
    require(cache['format']==('ResponseStep_v1' if method=='sgd' else 'IndependentResponseStep_v1'),'Wrong checkpoint algorithm schema')
    if task==1:
        pr=a['parent_provenance'];parent=Path(raw['study_parent_output_dir']);pa=read(parent/'final_summary.json');rp=Path(raw['study_parent_receipt'])
        require(pr['parent_summary_sha256']==sha256(parent/'final_summary.json') and pr['parent_completion_receipt_sha256']==sha256(rp) and read(rp)['summary_sha256']==sha256(parent/'final_summary.json'),'Parent acceptance changed')
        require(pr['parent_model_sha256']==pa['final_model_parameter_sha256'] and all(pr[k] for k in ('complete_network_restored','native_checkpoint_loader_used','parameter_identity_verified','parent_test_prefix_inherited_exact')),'Incomplete exact parent restore')
        require(matrix[0]==pa['final_metrics']['matrix'][0],'Parent test prefix differs')
    if j['study_condition_id']=='B0':
        snapshots=read(out/'snapshots.json')['snapshots'];require([s['epoch'] for s in snapshots]==([0,1] if j['smoke'] else [0,1,10,25,49]),'SGD reference snapshots incomplete')
        for snap in snapshots:require(sha256(snap['path'])==snap['sha256'],'Saved snapshot changed')
    return dict(summary_sha256=sha256(p),trace_sha256=a['step_trace_sha256'],updates=steps,tasks=n,trained_task_ids=[task],shared_parent_verified=task==1,metric_recomputation=True,full_formal_training=not j['smoke'])
