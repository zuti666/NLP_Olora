"""Fail closed on budgets, exact state, pure updates and immutable receipts."""
import json,math
from pathlib import Path
from common import HERE as H,sha256
READS=dict(zip(('g0','g1','g2_minus_s','g2','g2_plus_s','g0_plus_d','g1_plus_d','g3'),(1,2,3,3,3,4,4,4)))
FORMULAS=dict(zip(READS,('g0','g1','g2-(g1-g0)','g2','g2+(g1-g0)','g0+(g3-g2)','g1+(g3-g2)','g3')))
FORBIDDEN=('fisher_estimation_passes','weight_merge_updates','projection_calls','hvp_calls','qr_calls','create_graph_calls','gradient_clipping_calls','final_norm_rescale_calls')
def read(p):return json.loads(Path(p).read_text())
def require(ok,msg):
    if not ok:raise AssertionError(msg)
def finite(x):
    if isinstance(x,float):require(math.isfinite(x),'Nonfinite evidence')
    if isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,(list,tuple)):
        for v in x:finite(v)
def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
def validate_offline(j):
    p=Path(j['output_dir'])/'final_summary.json';a=read(p);finite(a);sm=j['smoke']
    require(a['status']=='passed' and a['schema']=='UpwardEightOfflineAudit_v1','Offline incomplete')
    require(a['committed_training_updates']==0 and a['old_task_gradient_reads']==0 and not a['model_selection_eligible'],'Offline became training/selection')
    require(a['no_dev_or_test_examples_used'] and a['old_examples_only_offline_no_grad'] and all(a['checks'].values()),'Offline leakage/state mutation')
    require(a['seed']==j['seed'] and a['smoke']==sm and a['audit_sha256']==sha256(H/'trainer/upward_local_audit.py'),'Offline identity differs')
    require(a['dev_split_sha256']==sha256(H/'dev_split.json'),'Offline split changed')
    expected=dict(checked_snapshots=1 if sm else 9,checked_probe_batches=1 if sm else 18,
        candidate_virtual_updates=16 if sm else 288,benchmark_methods=8 if sm else 72,
        gradient_reads=24 if sm else 396,microbenchmark_gradient_reads=72 if sm else 2160,
        holdout_loss_reads=68 if sm else 1224,microbenchmark_temporary_sgd_updates=24 if sm else 720)
    for k,v in expected.items():require(a[k]==v,'Offline registered budget differs: '+k)
    split=read(H/'dev_split.json');ids=[i for r in a['data_receipts'] for i in r['raw_train_indices']]
    require(len(ids)==len(set(ids)) and set(ids)<=set(split['train_indices']) and not set(ids)&set(split['dev_indices']),'Offline data overlap/dev access')
    suite=read(H/'suite.json');refs=suite['upward_audit_smoke_refs_by_seed' if sm else 'upward_audit_refs_by_seed'][str(j['seed'])]
    expected_states={('g0',1)} if sm else {(d,e) for d in ('g0','g1','g2') for e in (1,10,50)}
    states=[(s['trajectory'],s['epoch']) for s in a['snapshots']]
    require(len(states)==len(set(states)) and set(states)==expected_states,'Missing/repeated diagnostic source state')
    pairs={(d,mode) for d in READS for mode in ('native','nominal_parameter_length_match')}
    for s in a['snapshots']:
        require(Path(s['source_output_dir']).resolve()==Path(refs[s['trajectory']]).resolve(),'Wrong diagnostic reference trajectory')
        require(sha256(s['source_snapshot'])==s['source_snapshot_sha256'] and sha256(Path(s['source_output_dir'])/'final_summary.json')==s['source_summary_sha256'],'Offline source changed')
        require([b['probe_batch_index'] for b in s['batches']]==list(range(1 if sm else 2)),'Wrong diagnostic probe batches')
        for b in s['batches']:
            require(b['gradient_reads']==20 and b['loss_reads']==68 and b['committed_updates']==0 and b['no_R1_advance'] and all(b['checks'].values()),'Diagnostic batch mutation/read budget')
            cells=[(r['method'],r['mode']) for r in b['rows']]
            require(len(cells)==16 and set(cells)==pairs,'Diagnostic candidates missing/repeated')
            require(all(r['no_training_commit'] and r['old_data_only_in_offline_no_grad_loss'] and r['not_used_for_training_or_selection'] for r in b['rows']),'Diagnostic became selection/training')
        bench=s['native_cost_microbenchmark'];require(all(bench['checks'].values()) and bench['committed_training_updates']==0 and bench['no_microbenchmark_update_survives'],'Benchmark mutation')
        for d,count in READS.items():
            for phase,n in (('warmup',1 if sm else 3),('timed',1 if sm else 5),('verification',1 if sm else 2)):
                trials=[r for r in bench['trials'] if r['method']==d and r['phase']==phase]
                require(len(trials)==n and all(r['ordinary_gradient_reads']==count and r['native_skips_unused_probes'] for r in trials),'Native benchmark probe/cost budget')
                if phase=='verification':require(all(r['verification'].get('native_and_offline_U_exact') and r['verification'].get('native_optimizer_delta_exact') for r in trials),'Native/offline direction differs')
    return dict(summary_sha256=sha256(p),passive_diagnostic=True,committed_training_updates=0,full_formal_training=False,**expected)
def validate_job(j):
    out=Path(j['output_dir']);p=out/'final_summary.json'
    if j['kind']=='local_probe_audit':return validate_offline(j)
    if j['kind']=='pairing':
        p=out/j['report_file'];a=read(p);finite(a)
        require(a['status']=='passed' and a['smoke_count']==16 and all(a['checks'].values()),'GPU pairing failed')
        for q,d in a.get('evidence_hashes',{}).items():require(sha256(q)==d,'Pairing evidence changed')
        return dict(summary_sha256=sha256(p),gate='passed',full_formal_training=False)
    a=read(p);finite(a);cfg=read(out/'effective_config.json');raw=read(j['config'])
    d=j['upward_direction'];seed=j['seed'];task=j['task_id'];single=j['layout']=='single200';epochs=j['expected_epochs_per_task'];steps=j['expected_steps'];calls=READS[d]
    require(a['status']=='completed' and a['completed_tasks']==j['expected_tasks'] and a['completed_epochs']==epochs and a['completed_steps']==steps,'Incomplete own-task budget')
    require(a['upward_direction']==d and a['gradient_formula']==FORMULAS[d] and a['seed']==seed and a['smoke']==j['smoke'],'Wrong actual direction/seed/smoke')
    require(a['input_config_sha256']==j['config_sha256']==sha256(j['config']),'Config changed')
    runner='upward_single_train.py' if single else 'upward_fork_train.py'
    require(a['trainer_sha256']==sha256(H/'trainer'/runner) and a['algorithm_sha256']==sha256(H/'trainer/upward_residual_step.py'),'Wrong actual code')
    fixed=dict(seed=seed,lora_rank=10,init_epoch=50,epochs=50,init_lr=.04,lrate=.02,batch_size=128,
        cosine_min_lr_ratio=.1,history_weight=0.,momentum=0,weight_decay=0.,attention_backend='native',ewc_lambda=0.,
        ewc_skip_task_fisher=True,approximate_fisher_in_task=False,in_task_weight_merge=False,task_endpoint_weight_merge=False,
        gam_grad_rho=.05,gam_grad_norm_rho=j['r'])
    for k,v in fixed.items():require(cfg[k]==v,'Changed frozen setting: '+k)
    require(all(a['checks'].values()) and a['history_used_steps']==0,'Purity/restoration check failed')
    rec=a['dataset_partition_receipt']
    require(rec['full_train_samples']==5981 and rec['full_test_samples']==1519 and rec['class_order']==list(range(200)),'Original dataset changed')
    require(rec['full_task_increments']==([200] if single else [20]*10),'Class partition changed')
    require(cfg['init_cls']==(200 if single else 20) and cfg['increment']==(200 if single else 20),'Classifier partition changed')
    detail=a['task_details'];require(len(detail)==1 and detail[0]['task']==task and detail[0]['train_dataset_samples']==j['expected_train_samples'] and detail[0]['batches_per_epoch']==j['expected_batches_per_epoch'] and detail[0]['epochs']==epochs and detail[0]['steps']==steps,'Wrong real dataset/epoch/batch budget')
    if not single:
        require(a['trained_task_ids']==[task] and a['training_tasks_this_job']==1 and a['study_role']==j['study_role'],'Wrong parent/fork task')
        require(a['dev_split_sha256']==sha256(H/'dev_split.json'),'TRAIN-only dev split changed')
        require(len(a['dev_curve'])==epochs if 'dev_curve' in a else len(read(out/'dev_curve.json')['epochs'])==epochs,'Dev curve incomplete')
        require(a['endpoint_dev']['dev_samples']==j['expected_dev_samples'] and not a['endpoint_dev']['test_used_for_selection'],'Wrong dev/selection')
        split=read(H/'dev_split.json');expected_dev={i for i in split['dev_indices'] if task*20<=split['native_train_targets'][i]<(task+1)*20}
        expected_train={i for i in split['train_indices'] if task*20<=split['native_train_targets'][i]<(task+1)*20}
        require(len(expected_train)==j['expected_train_samples'] and len(expected_dev)==j['expected_dev_samples'],'Registered split/count mismatch')
        dev_curve=read(out/'dev_curve.json')['epochs']
        require([r['epoch'] for r in dev_curve]==list(range(1,epochs+1)),'Dev epochs incomplete')
        for r in dev_curve:require(r['dev_samples']==len(expected_dev) and set(r['dev_raw_train_indices'])==expected_dev and r['passive_state_and_rng_restored'] and not r['test_used_for_selection'],'Dev membership/restoration/selection mismatch')
    else:
        require(a['BWT'] is None and a['forgetting'] is None,'Single-task BWT must be null')
        require(a['endpoint_test']['test_samples']==1519 and sum(a['endpoint_test']['test_group_counts'])==1519,'Single test incomplete')
        require(a['endpoint_test']['test_group_counts']==read(H/'gates/DATA_COUNTS.json')['test_counts_original10groups'],'Single200 test groups differ')
        require(len(read(out/'epoch_test_curve.json')['epochs'])==epochs,'Single200 curve incomplete')
    require(sha256(out/'step_trace.jsonl')==a['step_trace_sha256'] and sha256(out/'task_matrix.json')==a['task_matrix_sha256'],'Trace/matrix changed')
    rows=[json.loads(l) for l in (out/'step_trace.jsonl').read_text().splitlines()];require(len(rows)==steps,'Incomplete trace')
    for index,r in enumerate(rows):
        finite(r);epoch=index//j['expected_batches_per_epoch']+1;batch=index%j['expected_batches_per_epoch']
        require((r['step_index'],r['task'],r['epoch'],r['batch'],r['seed'])==(index,task,epoch,batch,seed),'Wrong trace coordinate')
        require(r['direction']==d and r['upward_direction']==d and r['formula']==FORMULAS[d],'Direction not active')
        require(r['ordinary_gradient_calls']==calls and r['actual_model_forward_count']==calls+1 and r['diagnostic_loss_forward_count']==1,'Native skip probes/cost incorrect')
        require(r['sgd_updates']==1 and r['natural_sgd_exact'] and r['base_sgd_max_abs_error']==0.,'Actual SGD commit incorrect')
        require(r['only_base_R1_rng_advance'] and r['strict_deterministic'] and all(r['state_restore_checks'].values()) and all(r['diagnostic_state_restoration'].values()),'Full state/RNG not restored')
        require(not any(r[k] for k in FORBIDDEN) and not r['history_used'] and not r['history_cache_valid_after_update'],'Forbidden work/history entered')
        lo=0 if single else task*20;hi=200 if single else lo+20
        require(r['class_slice']==[lo,hi] and r['seen_classes']==hi and r['fc_shape'][0]==hi,'Wrong loss/head scope')
        eta=(.04 if task==0 else .02)*(.1+.9*(1+math.cos(math.pi*(epoch-1)/50))/2)
        require(close(r['nominal_lr'],eta) and close(r['h'],eta*r['g_grad_norm']),'LR/nominal base length changed')
        require(r['first_probe_sign']==1. and r['outer_origin']=='base' and r['final_update_origin']=='base','Probe sign/origin changed')
        require(r['rho']==.05 and r['r']==j['r'],'Radius inactive')
        for probe,count in (('g1',2),('g2',3),('g3',4)):
            require((r[probe+'_sha256'] is not None)==(calls>=count),'Unused probe not skipped')
        if d in ('g0','g1','g2','g3'):require(r['U_sha256']==r[d+'_sha256'],'Committed wrong gradient')
    counters=a['counters'];require(counters['training_sgd_updates']==steps and counters['training_ce_gradient_passes']==calls*steps and counters['diagnostic_loss_forward_count']==steps,'Counter mismatch')
    for epoch in range(1,epochs+1):
        ids=[i for r in rows if r['epoch']==epoch for i in r['identity']['indices']]
        require(len(ids)==len(set(ids)) and set(ids)<=set(range(j['expected_train_samples'])),'Training indices repeat/escape current TRAIN dataset')
        require(len(ids)==(256 if j['smoke'] else j['expected_train_samples']),'Actual epoch sample budget differs')
    require(not any(counters[k] for k in ('forbidden_work_attempts','fisher_estimation_passes','weight_merge_updates','training_hvp_passes','training_create_graph_passes','training_global_projection_calls')),'Forbidden total work')
    matrix=a['final_metrics']['matrix'];require(len(matrix)==j['expected_tasks'],'Wrong evaluation prefix')
    faa=sum(matrix[-1])/len(matrix[-1]);aaa=sum(sum(x)/len(x) for x in matrix)/len(matrix)
    require(close(faa,a['final_metrics']['FAA_task_macro']) and close(aaa,a['final_metrics']['AAA_task_macro']),'Stage FAA/AAA miscomputed')
    if task==1:
        parent=Path(raw['study_parent_output_dir']);pa=read(parent/'final_summary.json');pr=a['parent_provenance'];rp=Path(raw['study_parent_receipt'])
        require(pr['parent_summary_sha256']==sha256(parent/'final_summary.json') and pr['parent_completion_receipt_sha256']==sha256(rp),'Parent receipt changed')
        require(pa['seed']==seed and pr['parent_model_sha256']==pa['final_model_parameter_sha256'] and matrix[0]==pa['final_metrics']['matrix'][0],'Wrong parent')
        curve=a['retention_curve'];require([r['epoch'] for r in curve]==list(range(1,epochs+1)),'Missing full retention curve')
        counts=read(H/'gates/DATA_COUNTS.json')['test_counts_original10groups']
        for r in curve:require(r['evaluation_state_rng_restored'] and not r['test_used_for_selection'] and r['old_test_allseen40']['samples']==counts[0] and r['current_test_allseen40']['samples']==counts[1] and r['old_test_allseen40']['classification_classes']==40 and r['current_test_allseen40']['classification_classes']==40,'Retention curve sample/head/restoration mismatch')
        require(close(a['retention_auc'],sum(x['old_test_allseen40']['accuracy'] for x in curve)/epochs),'Retention AUC incorrect')
        require(close(a['old_training_delta'],curve[-1]['old_test_allseen40']['accuracy']-a['fork_baselines']['post_expand']['old_test_allseen40']['accuracy']),'Expansion/training damage mixed')
        require(close(a['final_metrics']['BWT'],matrix[-1][0]-matrix[0][0]),'BWT incorrect')
        snaps=read(out/'snapshots.json')['snapshots'];expected=[0,1] if j['smoke'] else [0,1,2,5,10,20,30,40,50]
        require([s['epoch'] for s in snaps]==expected,'Complete snapshots missing')
        for s in snaps:require(sha256(s['path'])==s['sha256'],'Snapshot changed')
        if seed==2 and d=='g0' and not j['smoke']:
            suite=read(H/'suite.json');old=Path(suite['old_seed2_sgd_task1_reference']);oa=read(old/'final_summary.json');oldrows=[json.loads(l) for l in (old/'step_trace.jsonl').read_text().splitlines()]
            require(len(oldrows)==len(rows) and all(r['identity']==o['identity'] and r['nominal_lr']==o['nominal_lr'] for r,o in zip(rows,oldrows)),'Old SGD400 input/LR stream differs')
            require(a['final_model_parameter_sha256']==oa['final_model_parameter_sha256'],'Old accepted SGD endpoint differs')
    for cpstr in a['checkpoint_dirs']:
        cp=Path(cpstr);manifest=read(cp/'manifest.json')
        required={f'lora_w_a_{task}.pt',f'lora_w_b_{task}.pt',f'lora_meta_{task}.json',f'fc_state_{task}.pt','training_state.pt'}
        if j['study_role']=='parent' and not single:required.add('shared_parent.pt')
        require(required<=set(manifest['files_sha256']) and manifest['task']==task and manifest['epoch']==epochs and manifest['model_sha256']==a['final_model_parameter_sha256'],'Missing/incorrect checkpoint inventory or final model')
        for name,digest in manifest['files_sha256'].items():require(sha256(cp/name)==digest,'Checkpoint file changed')
    witness=read(H/'runtime/jobs'/j['id']/'cuda_witness.json');require(witness['status']=='passed' and not witness['tf32_matmul'] and not witness['tf32_cudnn'],'Actual CUDA witness failed')
    return dict(summary_sha256=sha256(p),trace_sha256=a['step_trace_sha256'],updates=steps,gradient_reads=steps*calls,
        metric_recomputation=True,shared_parent_verified=task==1,full_formal_training=not j['smoke'])
