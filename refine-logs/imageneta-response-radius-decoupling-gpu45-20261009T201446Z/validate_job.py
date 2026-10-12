"""Acceptance of complete formal budgets and saved tensor/cache/metric evidence."""
import hashlib
import json
import math
from pathlib import Path
from common import HERE, sha256

READS = {'sgd':1, 'current_g2':3, 'history_fusion':2, 'history_g2':3}
SOURCES = {'sgd':None, 'current_g2':None, 'history_fusion':'base', 'history_g2':'final'}
FORBIDDEN = ('fisher_estimation_passes','weight_merge_updates','projection_calls','hvp_calls',
             'qr_calls','create_graph_calls','gradient_clipping_calls','final_norm_rescale_calls')

def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):raise AssertionError('Nonfinite evidence')
    if isinstance(value,dict):
        for item in value.values():finite_tree(item)
    elif isinstance(value,(list,tuple)):
        for item in value:finite_tree(item)

def tensor_digest(values):
    import torch
    digest=hashlib.sha256()
    for v in values:
        if not torch.is_tensor(v) or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()):
            raise AssertionError('Invalid cached tensor')
        v=v.detach().cpu().contiguous()
        digest.update(str((tuple(v.shape),str(v.dtype))).encode())
        digest.update(v.numpy().tobytes())
    return digest.hexdigest()

def validate_cached_state(state,last,method,task_updates,job):
    import torch
    cache=state.get('algorithm_state');source=SOURCES[method]
    if not isinstance(cache,dict) or cache.get('format')!=('IndependentResponseStep_v1' if job.get('radius_variant')!='original' else 'ResponseStep_v1'):
        raise AssertionError('Missing serialized algorithm state')
    if job.get('radius_variant')!='original':
        expected={k:job[k] for k in ('inner_probe_radius_mode','outer_probe_radius_mode','radius_variant','response_probe_variant')}
        expected['fixed_inner_probe_radius']=job['inner_probe_radius']
        expected['fixed_outer_probe_radius']=job['outer_probe_radius']
        if cache.get('independent_radius_config')!=expected or cache.get('history') is not None:
            raise AssertionError('Wrong independent checkpoint geometry/history')
    if state.get('algorithm')!=method or state.get('history_source')!=source or cache.get('method')!=method or cache.get('cache_source')!=source:
        raise AssertionError('Wrong method/cache source in checkpoint')
    if cache.get('history_weight')!=(.5 if source else 0.) or cache.get('real_updates')!=task_updates:
        raise AssertionError('Wrong weight or number of committed task steps')
    if state.get('previous_batch_identity')!=last['identity']:
        raise AssertionError('Checkpoint does not contain final batch identity')
    layout=cache.get('layout')
    if not isinstance(layout,list) or not layout or len({x['name'] for x in layout})!=len(layout):
        raise AssertionError('Invalid trainable parameter layout')
    history=cache.get('history')
    if not last['history_cache_valid_after_update']:
        if history is not None:raise AssertionError('Disabled/invalid history was retained')
        return
    if source is None or not isinstance(history,dict) or history.get('cache_source')!=source or history.get('valid') is not True or history.get('layout')!=layout:
        raise AssertionError('Missing or mislabelled actual history cache')
    vectors={}
    for key in ('vector','displacement'):
        values=history.get(key)
        if not isinstance(values,(list,tuple)) or len(values)!=len(layout):raise AssertionError('Wrong cached tensor count')
        if any(list(v.shape)!=entry['shape'] or str(v.dtype)!=entry['dtype'] for v,entry in zip(values,layout)):
            raise AssertionError('Cached shape/dtype differs from trainable layout')
        vectors[key]=values
    expected=last['g0_sha256'] if source=='base' else last['final_gradient_sha256']
    if tensor_digest(vectors['vector'])!=expected or expected!=history.get('vector_sha256') or expected!=last['history_cached_vector_sha256']:
        raise AssertionError('Cache is not the complete method-specific submitted/base vector')
    delta_hash=tensor_digest(vectors['displacement'])
    if delta_hash!=history.get('displacement_sha256') or delta_hash!=last['actual_step_displacement_sha256']:
        raise AssertionError('Cache is not the actual committed FP32 displacement')
    norm=math.sqrt(sum(float((v.double()*v.double()).sum()) for v in vectors['displacement']))
    if not math.isclose(norm,history['length'],rel_tol=1e-12,abs_tol=1e-12) or not math.isclose(norm,last['actual_parameter_step_norm'],rel_tol=1e-12,abs_tol=1e-12):
        raise AssertionError('Wrong actual displacement length')

def validate_job(job):
    out=Path(job['output_dir'])
    if job['kind']=='pairing':
        p=out/job['report_file'];r=json.loads(p.read_text())
        if r['status']!='passed' or r['smoke_count']!=4 or not all(r['checks'].values()):raise AssertionError('Pairing failed')
        return dict(summary_sha256=sha256(p),gate='passed',full_formal_training=False)
    p=out/'final_summary.json';a=json.loads(p.read_text());finite_tree(a)
    cfg=json.loads((out/'effective_config.json').read_text())
    method=job['method'];source=SOURCES[method];layout=job['layout'];width=200 if layout=='single200' else 20
    full_tasks=1 if layout=='single200' else 10
    expected=(job['expected_tasks'],job['expected_epochs_per_task'],job['expected_steps'])
    if a['status']!='completed' or (a['completed_tasks'],a['epochs_per_task'],a['completed_steps'])!=expected or a['completed_epochs']!=expected[0]*expected[1]:
        raise AssertionError('Incomplete requested budget')
    if (a['response_method'],a['response_layout'],a['seed'],a['history_source'])!=(method,layout,2,source):
        raise AssertionError('Incorrect experimental condition')
    if (a['full_partition_tasks'],a['total_dataset_classes'],a['init_cls'],a['increment'])!=(full_tasks,200,width,width):
        raise AssertionError('Wrong full class partition')
    if a['r'] is not None or a['rho'] is not None or a['gam_radii_active'] or a['response_radius_mode']!=('sgd_step' if job['radius_variant']=='original' else 'independent_inner_outer_v1') or a['radius_scale']!=1:
        raise AssertionError('Wrong dynamic radius protocol')
    if a['input_config_sha256']!=job['config_sha256'] or a['trainer_sha256']!=sha256(HERE/'trainer/response_train.py') or a['algorithm_sha256']!=sha256(HERE/'trainer'/('response_step.py' if job['radius_variant']=='original' else 'independent_response_step.py')) or a['layout_sha256']!=sha256(HERE/'trainer/layout_lifecycle.py'):
        raise AssertionError('Frozen execution code/config changed')
    if cfg['seed']!=2 or cfg['lora_rank']!=10 or cfg['ewc_lambda']!=0 or not cfg['ewc_skip_task_fisher'] or any(cfg[k] for k in ('approximate_fisher_in_task','in_task_weight_merge','task_endpoint_weight_merge')):
        raise AssertionError('Wrong baseline model/seed or forbidden modules')
    raw=json.loads(Path(job['config']).read_text())
    if a['response_probe_variant']!=job['response_probe_variant'] or a['normal_gradient_calls']!=3:
        raise AssertionError('Wrong registered variant or gradient budget')
    if a['inherited_algorithm_sha256']!=sha256(HERE/'trainer/response_step.py'):
        raise AssertionError('Original SGD/response core changed')
    if cfg['init_epoch']!=50 or cfg['epochs']!=50 or cfg['init_lr']!=.04 or cfg['lrate']!=.02 or cfg['batch_size']!=128 or cfg['cosine_min_lr_ratio']!=.1:
        raise AssertionError('Baseline training hyperparameters changed')
    if job['radius_variant']!='original':
        for k in ('radius_variant','inner_probe_radius_mode','outer_probe_radius_mode','inner_probe_radius','outer_probe_radius'):
            if cfg[k]!=raw[k] or cfg[k]!=job[k]:raise AssertionError('Probe configuration changed: '+k)
        if a['inner_rho']!=raw['inner_probe_radius'] or a['outer_R']!=raw['outer_probe_radius']:
            raise AssertionError('Configured radii missing from summary')
    receipt=a['dataset_partition_receipt'];boundaries=[[i*width,(i+1)*width] for i in range(full_tasks)]
    if receipt['full_train_samples']!=5981 or receipt['full_test_samples']!=1519 or receipt['nb_classes']!=200 or receipt['full_task_increments']!=[width]*full_tasks or receipt['full_task_boundaries']!=boundaries or receipt['class_order']!=list(range(200)):
        raise AssertionError('Incorrect ImageNet-A data/task partition')
    if not all(a['checks'].values()):raise AssertionError('Run-level checks failed')
    rows=[json.loads(x) for x in (out/'step_trace.jsonl').read_text().splitlines()]
    if len(rows)!=expected[2]:raise AssertionError('Incomplete step trace')
    batches=[job['smoke_batches_per_epoch']]*expected[0] if job['smoke'] else ([47] if layout=='single200' else [10,8,8,5,4,3,3,3,4,3])
    coordinates=[(t,e,b) for t in range(expected[0]) for e in range(1,expected[1]+1) for b in range(batches[t])]
    if len(coordinates)!=len(rows):raise AssertionError('Incorrect epoch/batch count')
    previous=None;used=0
    for i,(r,coord) in enumerate(zip(rows,coordinates)):
        finite_tree(r)
        if r['step_index']!=i or (r['task'],r['epoch'],r['batch'])!=coord or r['seed']!=2 or r['method']!=method or r['response_layout']!=layout:
            raise AssertionError('Incorrect step coordinates/method')
        task=coord[0]
        if r['class_slice']!=boundaries[task] or r['seen_classes']!=boundaries[task][1] or r['fc_shape'][0]!=boundaries[task][1]:
            raise AssertionError('Classifier/CE slice does not match task layout')
        calls=r['ordinary_gradient_calls']
        if not 1<=calls<=READS[method] or (calls!=READS[method] and not r['fallback_reason']):raise AssertionError('Incorrect gradient budget')
        if r['actual_model_forward_count']!=calls+1 or r['diagnostic_loss_forward_count']!=1 or r['sgd_updates']!=1 or not r['natural_sgd_exact']:
            raise AssertionError('Incorrect actual forward/SGD accounting')
        if not all(r['state_restore_checks'].values()) or not all(r['diagnostic_state_restoration'].values()) or not r['only_base_R1_rng_advance'] or not r['strict_deterministic'] or any(r[k] for k in FORBIDDEN):
            raise AssertionError('State/RNG/forbidden-work check failed')
        if not math.isclose(r['h'],r['nominal_lr']*r['g_grad_norm'],rel_tol=1e-12,abs_tol=1e-12):raise AssertionError('Wrong prediction radius')
        eta=(.04 if task==0 else .02)*(.1+.9*(1+math.cos(math.pi*(coord[1]-1)/50))/2)
        if not math.isclose(r['nominal_lr'],eta,rel_tol=1e-12,abs_tol=1e-12):raise AssertionError('Epoch cosine schedule changed')
        if r['response_probe_variant']!=job['response_probe_variant']:raise AssertionError('Wrong active probe variant')
        if job['radius_variant']!='original':
            if r['radius_mode']!='independent_inner_outer_v1' or r['radius_variant']!=job['radius_variant'] or r['h_ref']!=r['h']:
                raise AssertionError('Independent mode absent in active step')
            for prefix in ('inner','outer'):
                mode=job[prefix+'_probe_radius_mode'];theory=job[prefix+'_probe_radius'] if mode=='fixed_l2' else r['h']
                if r[prefix+'_probe_radius_mode']!=mode or r[prefix+'_probe_radius_theoretical']!=theory:
                    raise AssertionError('Wrong registered probe distance')
                actual=r[prefix+'_probe_radius_actual']
                if not r['fallback_reason'] and not math.isclose(actual,theory,rel_tol=1e-4,abs_tol=1e-8):
                    raise AssertionError('Measured FP32 probe distance disagrees with theory')
                if actual!=r['future_actual_parameter_displacement_norm' if prefix=='inner' else 'relocation_actual_parameter_displacement_norm']:
                    raise AssertionError('Measured probe distance aliases disagree')
            if r['lambda']!=r['outer_probe_radius_theoretical'] or r['ell_f']!=r['inner_probe_radius_theoretical']:
                raise AssertionError('Radius formula metadata wrong')
        if r['history_used'] or r['history_available'] or r['history_cache_valid_after_update'] or r['effective_history_weight']!=0:
            raise AssertionError('Pure Current-G2 unexpectedly used history')
        if r['g2_sha256'] is not None and r['relocation_origin']!='base':raise AssertionError('Relocation started from inner point')
        if method=='sgd' and (calls!=1 or r['g1_sha256'] is not None or r['g2_sha256'] is not None or r['U_sha256']!=r['g0_sha256']):raise AssertionError('SGD contains extra probes or modified update')
        if method in ('current_g2','history_g2') and calls==3 and r['final_gradient_sha256']!=r['g2_sha256']:raise AssertionError('Relocation method did not submit only g2')
        if previous is None or previous['task']!=task:
            if r['history_used'] or r['previous_batch_identity'] is not None:raise AssertionError('History crossed task/head reset')
        if r['history_used']:
            used+=1
            if source is None or previous is None or previous['task']!=task or r['history_incoming_vector_source']!=source:raise AssertionError('Wrong incoming history source')
            expected_hash=previous['g0_sha256'] if source=='base' else previous['final_gradient_sha256']
            if r['history_incoming_vector_sha256']!=expected_hash or r['history_cache_actual_displacement_sha256']!=previous['actual_step_displacement_sha256']:
                raise AssertionError('History is not previous raw/final vector and actual displacement')
            if not math.isclose(r['ell'],previous['actual_parameter_step_norm'],rel_tol=1e-12,abs_tol=1e-12) or r['effective_history_weight']!=.5 or not math.isclose(r['history_coefficient'],.5*r['h']/r['ell'],rel_tol=1e-12,abs_tol=1e-12):
                raise AssertionError('Incorrect history scale')
        if source is None and r['history_cache_valid_after_update']:raise AssertionError('No-history method retained cache')
        if r['history_cache_valid_after_update']:
            expected_hash=r['g0_sha256'] if source=='base' else r['final_gradient_sha256']
            if r['history_cached_vector_sha256']!=expected_hash or r['history_cached_vector_source']!=source or not r['history_cache_keeps_full_amplitude']:
                raise AssertionError('Wrong outgoing cache source/amplitude')
        if abs(r['post_update_delta_ce']-(r['post_update_ce']-r['loss0']))>1e-9:raise AssertionError('Wrong actual post-update loss delta')
        previous=r
    if source is not None and used==0:raise AssertionError('History method never exercised history')
    if a['counters']['training_ce_gradient_passes']!=sum(r['ordinary_gradient_calls'] for r in rows) or a['counters']['training_sgd_updates']!=len(rows):raise AssertionError('Total gradient/SGD count mismatch')
    if a['counters']['diagnostic_loss_forward_count']!=len(rows) or a['counters']['training_forward_count']!=sum(r['ordinary_gradient_calls'] for r in rows):
        raise AssertionError('Total diagnostic/training forward count mismatch')
    if any(a['counters'][k] for k in ('fisher_estimation_passes','weight_merge_updates','forbidden_work_attempts','training_hvp_passes','training_create_graph_passes','training_global_projection_calls')):
        raise AssertionError('Forbidden total work counter')
    for name,key in [('step_trace.jsonl','step_trace_sha256'),('task_matrix.json','task_matrix_sha256')]:
        if sha256(out/name)!=a[key]:raise AssertionError('Trace/matrix changed')
    m=json.loads((out/'task_matrix.json').read_text());matrix=m['matrix'];n=len(matrix)
    finite_tree(m)
    if n!=expected[0] or any(len(row)!=i+1 for i,row in enumerate(matrix)):raise AssertionError('Incomplete task matrix')
    if any(not isinstance(v,(float,int)) or not 0<=v<=100 for row in matrix for v in row):raise AssertionError('Invalid task accuracy')
    recomputed={'FAA_task_macro':sum(matrix[-1])/n,'AAA_task_macro':sum(sum(row)/len(row) for row in matrix)/n,
      'LA':sum(matrix[i][i] for i in range(n))/n,'BWT':sum(matrix[-1][i]-matrix[i][i] for i in range(n-1))/(n-1) if n>1 else None,
      'forgetting':sum(max(matrix[k][i] for k in range(i,n))-matrix[-1][i] for i in range(n-1))/(n-1) if n>1 else None}
    for k,v in recomputed.items():
        if v is None:
            if m[k] is not None or a['final_metrics'][k] is not None:raise AssertionError('Single-task retention metric must be null')
        elif abs(m[k]-v)>1e-9 or abs(a['final_metrics'][k]-v)>1e-9:raise AssertionError('Metric recomputation failed')
    if layout=='single200':
        curve=json.loads((out/'epoch_test_curve.json').read_text())['epochs']
        if len(curve)!=expected[1] or any(x['epoch']!=i+1 for i,x in enumerate(curve)):raise AssertionError('Missing single-task epoch curve')
        if m['final_test_accuracy']!=m['FAA_task_macro'] or m['FAA_task_macro']!=m['AAA_task_macro']:raise AssertionError('Single-task metrics improperly regrouped')
    if len(a['checkpoint_dirs'])!=expected[0]:raise AssertionError('Incomplete checkpoint count')
    import torch
    for task,checkpoint in enumerate(a['checkpoint_dirs']):
        d=Path(checkpoint);manifest=json.loads((d/'manifest.json').read_text())
        if manifest['task']!=task or manifest['history_source']!=source:raise AssertionError('Wrong checkpoint condition')
        required={'training_state.pt',f'lora_w_a_{task}.pt',f'lora_w_b_{task}.pt',f'lora_meta_{task}.json',f'fc_state_{task}.pt'}
        if manifest.get('algorithm_cache_in_training_state') is not True or not required<=set(manifest['files_sha256']):raise AssertionError('Incomplete checkpoint/cache artifacts')
        for name,digest in manifest['files_sha256'].items():
            if sha256(d/name)!=digest:raise AssertionError('Checkpoint file changed')
        state=torch.load(d/'training_state.pt',map_location='cpu',weights_only=False)
        task_rows=[r for r in rows if r['task']==task]
        validate_cached_state(state,task_rows[-1],method,len(task_rows),job)
    return dict(summary_sha256=sha256(p),trace_sha256=a['step_trace_sha256'],updates=len(rows),tasks=n,
                history_used_steps=used,cache_source=source,full_class_partition_verified=True,
                checkpoint_tensor_cache_verified=True,metric_recomputation=True,full_formal_training=not job['smoke'])
