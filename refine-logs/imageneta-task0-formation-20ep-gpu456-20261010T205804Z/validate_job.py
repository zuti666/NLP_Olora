"""Fail closed on the independent two-task 20-epoch formation protocol."""
import json,math
from pathlib import Path
from common import HERE as H,sha256
CONDITIONS=('sgd','sam','gam_residual','sf_g2','sf_g2_then_sgd','sgd_then_sf_g2')
READS={'sgd':1,'sam':2,'gam_residual':4,'sf_g2':3}
FORBIDDEN=('fisher_estimation_passes','weight_merge_updates','projection_calls','hvp_calls','qr_calls','create_graph_calls','gradient_clipping_calls','final_norm_rescale_calls')
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise AssertionError(message)
def finite(value):
    if isinstance(value,float):require(math.isfinite(value),'Nonfinite evidence')
    if isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)
def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)
def active(condition,task,epoch):
    if task==1:return 'sgd'
    if condition=='sf_g2_then_sgd':return 'sf_g2' if epoch<=10 else 'sgd'
    if condition=='sgd_then_sf_g2':return 'sgd' if epoch<=10 else 'sf_g2'
    return condition

def validate_job(j):
    out=Path(j['output_dir']);p=out/j.get('report_file','final_summary.json');a=read(p);finite(a)
    if j['kind']=='pairing':
        require(a['status']=='passed' and a['smoke_count']==6 and a.get('checks') and all(a['checks'].values()),'Six-smoke pairing gate failed')
        for q,d in a['evidence_hashes'].items():require(sha256(q)==d,'Pairing evidence changed: '+q)
        return dict(summary_sha256=sha256(p),gate='passed',full_formal_training=False)
    require(j['kind']=='training','Unknown job kind')
    condition=j['formation_condition_id'];seed=j['seed'];smoke=j['smoke'];steps=j['expected_steps'];cfg=read(out/'effective_config.json')
    require(condition in CONDITIONS and seed in (0,1,2),'Unexpected registered identity')
    require(a['status']=='completed' and a['protocol']=='task0_formation_20ep_task1_sgd_v1','Training incomplete/wrong protocol')
    require(a['completed_tasks']==2 and a['completed_epochs']==40 and a['completed_steps']==steps,'Incomplete 2x20-epoch/update budget')
    require(a['formation_condition_id']==condition and a['seed']==seed and a['smoke']==smoke,'Wrong actual identity')
    require(a['input_config_sha256']==j['config_sha256']==sha256(j['config']),'Input configuration changed')
    require(a['trainer_sha256']==sha256(H/'trainer/formation_train.py') and a['algorithm_sha256']==sha256(H/'trainer/formation_step.py'),'Wrong actual trainer/kernel')
    fixed=dict(lora_rank=10,init_epoch=20,epochs=20,init_lr=.04,lrate=.02,batch_size=128,cosine_min_lr_ratio=.1,
        momentum=0,weight_decay=0.,init_momentum=0,init_weight_decay=0.,history_weight=0.,attention_backend='native',
        ewc_lambda=0.,ewc_skip_task_fisher=True,approximate_fisher_in_task=False,in_task_weight_merge=False,
        task_endpoint_weight_merge=False,init_cls=20,increment=20,gam_grad_gamma=0.,formation_condition_id=condition)
    for k,v in fixed.items():require(cfg[k]==v,'Changed common setting: '+k)
    require(a.get('checks') and all(a['checks'].values()) and a['history_used_steps']==0,'Purity/full-state check failed')
    require(a['dev_split_sha256']==sha256(H/'dev_split.json'),'Data split changed')
    rec=a['dataset_partition_receipt'];require(rec['full_train_samples']==5981 and rec['full_test_samples']==1519 and rec['class_order']==list(range(200)) and rec['full_task_increments']==[20]*10,'Data/class order changed')
    trace=out/'step_trace.jsonl';require(sha256(trace)==a['step_trace_sha256'],'Trace changed')
    rows=[json.loads(line) for line in trace.read_text().splitlines() if line.strip()];require(len(rows)==steps,'Incomplete native update trace')
    count=0;gradient_reads=0
    for task in (0,1):
        batches=j['expected_batches_per_epoch'][task];n=j['expected_train_samples'][task]
        for epoch in range(1,21):
            epoch_ids=[]
            for batch in range(batches):
                r=rows[count];finite(r);method=active(condition,task,epoch);calls=READS[method]
                require((r['step_index'],r['task'],r['epoch'],r['batch'],r['seed'])==(count,task,epoch,batch,seed),'Wrong trace coordinate')
                require(r['formation_condition_id']==condition and r['active_method']==method and r['normal_gradient_calls']==calls,'Inactive stage method')
                require(r['ordinary_gradient_calls']==calls and r['actual_model_forward_count']==calls+1 and r['diagnostic_loss_forward_count']==1 and r['native_probe_skipping'],'Wrong native probe/forward budget')
                require(r['sgd_updates']==1 and r['natural_sgd_exact'] and r['base_sgd_max_abs_error']==0.,'Actual natural SGD differs')
                require(r['only_base_R1_rng_advance'] and r['strict_deterministic'] and all(r['state_restore_checks'].values()) and all(r['diagnostic_state_restoration'].values()),'State/RNG not restored')
                require(not any(r[k] for k in FORBIDDEN) and not r['history_used'] and not r['history_cache_valid_after_update'] and not r['fallback'],'Forbidden work/history/fallback entered')
                require(r['class_slice']==[task*20,(task+1)*20] and r['seen_classes']==(task+1)*20 and r['fc_shape'][0]==(task+1)*20,'Wrong loss/head scope')
                eta=(.04 if task==0 else .02)*(.1+.9*(1+math.cos(math.pi*(epoch-1)/20))/2)
                require(close(r['nominal_lr'],eta) and close(r['h'],eta*r['g_grad_norm']),'Continuous cosine/radius reference changed')
                require(r['actual_post_parameter_sha256'] and r['actual_step_displacement_sha256'],'Missing actual commit identity')
                if method=='sgd':require(r['U_sha256']==r['g0_sha256'] and r['g1_sha256'] is None and r['g2_sha256'] is None and r['g3_sha256'] is None,'SGD performed unused probes/wrong update')
                elif method=='sam':require(r['U_sha256']==r['g1_sha256'] and r['rho']==.05 and r['first_probe_sign']==1. and r['g2_sha256'] is None and r['g3_sha256'] is None,'SAM formula/probe budget differs')
                elif method=='gam_residual':require(r['formula']=='g0+(g3-g2)' and r['rho']==.05 and r['r']==1. and r['first_probe_sign']==1. and r['outer_origin']=='base' and r['g3_sha256'] is not None,'Fixed GAM residual formula differs')
                else:
                    require(r['U_sha256']==r['g2_sha256'] and r['sf_outer_radius']==.25 and r['response_probe_variant']=='S_F','S_F g2 update differs')
                    require(r['inner_probe_radius_mode']=='sgd_step' and r['outer_probe_radius_mode']=='fixed_l2' and close(r['inner_probe_radius_theoretical'],eta*r['g_grad_norm']) and r['outer_probe_radius_theoretical']==.25 and r['relocation_origin']=='base','S_F radii/origin differ')
                ids=r['identity']['indices'];require(len(ids)==len(set(ids)) and set(ids)<=set(range(n)),'Input indices escape current TRAIN dataset')
                epoch_ids+=ids;count+=1;gradient_reads+=calls
            require(len(epoch_ids)==len(set(epoch_ids)) and len(epoch_ids)==(128 if smoke else n),'Epoch sample budget/repeat differs')
    require(count==steps and gradient_reads==j['expected_gradient_reads'],'Registered total budget differs')
    counters=a['counters'];require(counters['training_sgd_updates']==steps and counters['training_ce_gradient_passes']==gradient_reads and counters['diagnostic_loss_forward_count']==steps,'Total actual counters differ')
    require(not any(counters[k] for k in ('forbidden_work_attempts','fisher_estimation_passes','weight_merge_updates','training_hvp_passes','training_create_graph_passes','training_global_projection_calls')),'Forbidden total work')
    require(len(a['task_details'])==2,'Task details missing')
    for task,detail in enumerate(a['task_details']):
        require(detail['task']==task and detail['epochs']==20 and detail['steps']==j['expected_task_steps'][task] and detail['train_dataset_samples']==j['expected_train_samples'][task] and detail['batches_per_epoch']==j['expected_batches_per_epoch'][task],'Task-specific actual budget differs')
    require(len(a['switch_receipts'])==1,'Epoch11 continuous-optimizer receipt missing')
    switch=a['switch_receipts'][0]
    for key in ('optimizer_same','scheduler_same','model_parameters_unchanged_at_switch','parameter_references_unchanged_at_switch','rng_unchanged_at_switch'):require(switch[key],'Algorithm switching reset state: '+key)
    require(close(switch['lr_used_epoch10'],.04*(.1+.9*(1+math.cos(math.pi*9/20))/2)) and close(switch['lr_used_epoch11'],.022),'Epoch11 LR schedule restarted')
    require(switch['scheduler_last_epoch_at_epoch11']==10 and switch['scheduler_T_max']==20 and not any(switch[k] for k in ('reset_model','reset_optimizer','reset_scheduler','reset_rng')),'Scheduler/model reset at switch')
    curve=a['retention_curve'];require([r['epoch'] for r in curve]==list(range(21)),'Missing Task1 epoch0..20 retention curve')
    test_counts=read(H/'environment_spec.json')['data']['test_samples_per_task']
    for r in curve:
        require(r['evaluation_state_rng_restored'] and not r['test_used_for_selection'],'Passive evaluation affected training/selection')
        for scope,expected in (('old_test_allseen40',test_counts[0]),('current_test_allseen40',test_counts[1])):
            require(r[scope]['samples']==expected and r[scope]['classification_classes']==40,'Retention test scope/count differs')
    endpoint=a['task0_endpoint'];require(endpoint['samples']==test_counts[0],'Task0 official endpoint incomplete')
    metrics=a['formation_metrics'];old0=curve[0]['old_test_allseen40']['accuracy'];old=curve[-1]['old_test_allseen40']['accuracy'];current=curve[-1]['current_test_allseen40']['accuracy'];t0=endpoint['accuracy'];faa=(old+current)/2
    expected=dict(task0_end20=t0,old_start40=old0,old_final40=old,current_final40=current,old_drop_since_task1_start=old0-old,
        training_BWT=old-old0,expansion_damage=t0-old0,FAA=faa,traditional_BWT=old-t0,stage_AAA=(t0+faa)/2,
        retention_auc=sum(r['old_test_allseen40']['accuracy'] for r in curve[1:])/20)
    for k,v in expected.items():require(close(metrics[k],v),'Expansion/retention/stage metric miscomputed: '+k)
    require(a['task1_baselines']['post_expand']==curve[0] and a['task1_baselines']['task0_end20']==endpoint,'Epoch0 baseline changed')
    matrix_path=out/'task_matrix.json';require(sha256(matrix_path)==a['task_matrix_sha256'],'Native matrix changed')
    matrix=a['final_metrics']['matrix'];require(len(matrix)==2 and len(matrix[0])==1 and len(matrix[1])==2,'Wrong task matrix shape')
    for x,y in zip((matrix[0][0],matrix[1][0],matrix[1][1]),(t0,old,current)):require(abs(x-y)<=.011,'Native and exact retained accuracy differ')
    require(close(a['final_metrics']['FAA_task_macro'],sum(matrix[1])/2) and close(a['final_metrics']['AAA_task_macro'],(matrix[0][0]+sum(matrix[1])/2)/2) and close(a['final_metrics']['BWT'],matrix[1][0]-matrix[0][0]),'Native FAA/AAA/BWT miscomputed')
    checkpoints=a['checkpoint_dirs'];require(len(checkpoints)==2,'Two native checkpoints missing')
    for task,cpstr in enumerate(checkpoints):
        cp=Path(cpstr);manifest=read(cp/'manifest.json');required={f'lora_w_a_{task}.pt',f'lora_w_b_{task}.pt',f'lora_meta_{task}.json',f'fc_state_{task}.pt','training_state.pt'}
        require(required<=set(manifest['files_sha256']) and manifest['task']==task and manifest['epoch']==20,'Checkpoint task/epoch/inventory differs')
        for name,digest in manifest['files_sha256'].items():require(sha256(cp/name)==digest,'Native checkpoint changed: '+name)
    runtime=H/'runtime/jobs'/j['id'];witness=read(runtime/'cuda_witness.json');launch=read(runtime/'launched.json');software=read(H/'environment_spec.json')['software']
    require(witness['status']=='passed' and witness['cuda_visible_devices']==launch['gpu_uuid'] and launch['gpu'] in (4,5,6) and witness['torch']==software['torch_runtime'] and witness['cuda']==software['torch_cuda_compile_version'] and not witness['tf32_matmul'] and not witness['tf32_cudnn'],'Leased CUDA witness/environment mismatch')
    return dict(summary_sha256=sha256(p),trace_sha256=sha256(trace),updates=steps,gradient_reads=gradient_reads,
        metric_recomputation=True,epoch11_continuous_schedule_verified=True,full_formal_training=not smoke)
