"""Accept complete runs only after validating the actual cached-response trace."""
import json
import math
import hashlib
from pathlib import Path
from common import HERE,sha256

def validate_cached_state(state, last, job, task_updates):
    """Inspect serialized tensors against the independently recorded last update."""
    import torch
    cache=state.get('algorithm_state')
    if not isinstance(cache,dict) or cache.get('format')!='HistoryResponseStep_v1':
        raise AssertionError('Missing real algorithm state')
    if state.get('algorithm')!=job['direction'] or cache.get('direction')!=job['direction'] or cache.get('history_weight')!=job['history_weight']:
        raise AssertionError('Checkpoint algorithm condition differs')
    if cache.get('real_updates')!=task_updates or state.get('previous_batch_identity')!=last['identity']:
        raise AssertionError('Checkpoint is not the final committed step of its task')
    layout=cache.get('layout')
    if not isinstance(layout,list) or not layout or len({x['name'] for x in layout})!=len(layout):
        raise AssertionError('Invalid serialized parameter layout')
    h=cache.get('history')
    if not last['history_cache_valid_after_update']:
        if h is not None:raise AssertionError('Invalid or disabled history was saved')
        return
    if job['direction']!='history_fused' or not isinstance(h,dict) or h.get('valid') is not True or h.get('layout')!=layout:
        raise AssertionError('Missing valid serialized history tensors')
    digests={};norm=0.
    for key in ('raw_gradient','displacement'):
        values=h.get(key)
        if not isinstance(values,(list,tuple)) or len(values)!=len(layout):
            raise AssertionError('Incorrect serialized history tensor count')
        digest=hashlib.sha256()
        for v,entry in zip(values,layout):
            if not torch.is_tensor(v) or list(v.shape)!=entry['shape'] or str(v.dtype)!=entry['dtype'] or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()):
                raise AssertionError('Invalid serialized history tensor')
            v=v.detach().cpu().contiguous()
            digest.update(str((tuple(v.shape),str(v.dtype))).encode())
            digest.update(v.numpy().tobytes())
            if key=='displacement':norm+=float((v.double()*v.double()).sum())
        digests[key]=digest.hexdigest()
    if digests['raw_gradient']!=h.get('raw_gradient_sha256') or digests['raw_gradient']!=last['raw_gradient_sha256']:
        raise AssertionError('Serialized cache differs from actual raw gradient')
    if digests['displacement']!=h.get('displacement_sha256') or digests['displacement']!=last['actual_step_displacement_sha256']:
        raise AssertionError('Serialized cache differs from actual committed displacement')
    length=math.sqrt(norm)
    if not math.isclose(length,h.get('length',float('nan')),rel_tol=1e-12,abs_tol=1e-12) or not math.isclose(length,last['actual_parameter_step_norm'],rel_tol=1e-12,abs_tol=1e-12):
        raise AssertionError('Serialized actual displacement distance differs')

def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):raise AssertionError('Nonfinite evidence')
    if isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite_tree(v)

def metrics(matrix):
    n=len(matrix)
    return dict(AAA_task_macro=sum(sum(row)/len(row) for row in matrix)/n,
                FAA_task_macro=sum(matrix[-1])/n,LA=sum(matrix[i][i] for i in range(n))/n,
                BWT=sum(matrix[-1][i]-matrix[i][i] for i in range(n-1))/(n-1) if n>1 else None,
                forgetting=sum(max(matrix[k][i] for k in range(i,n))-matrix[-1][i] for i in range(n-1))/(n-1) if n>1 else None)

def validate_job(job):
    out=Path(job['output_dir'])
    if job['kind']=='pairing':
        p=out/job['report_file'];r=json.loads(p.read_text())
        if r['status']!='passed' or r['smoke_count']!=2 or not all(r['checks'].values()):
            raise AssertionError('Paired short runs did not pass')
        return dict(summary_sha256=sha256(p),gate='passed')
    p=out/'final_summary.json';a=json.loads(p.read_text());finite_tree(a)
    cfg=json.loads((out/'effective_config.json').read_text())
    expected=(job['expected_tasks'],job['expected_epochs_per_task'],job['expected_steps'])
    if a['status']!='completed' or (a['completed_tasks'],a['epochs_per_task'],a['completed_steps'])!=expected:
        raise AssertionError('Incomplete budget')
    if a['completed_epochs']!=expected[0]*expected[1]:raise AssertionError('Incomplete epoch count')
    if a['seed']!=2 or a['update_direction']!=job['direction'] or a['history_weight']!=job['history_weight']:
        raise AssertionError('Wrong experimental condition')
    if a['r'] is not None or a['rho'] is not None or a['gam_radii_active']:
        raise AssertionError('Dynamic predictor incorrectly labelled fixed GAM radius')
    if a['input_config_sha256']!=job['config_sha256'] or a['trainer_sha256']!=sha256(HERE/'trainer/history_train.py'):
        raise AssertionError('Config/trainer changed')
    if a['algorithm_sha256']!=sha256(HERE/'trainer/history_step.py'):raise AssertionError('Algorithm changed')
    if (cfg['gam_grad_rho'],cfg['gam_grad_norm_rho'],cfg['lora_rank'],cfg['seed'])!=(.05,1.,10,2):
        raise AssertionError('Reference geometry/model/seed changed')
    if cfg['gam_radii_active'] or cfg['response_lambda_mode']!='sgd_prediction_distance':
        raise AssertionError('Reference GAM settings leaked into actual algorithm')
    if cfg['ewc_lambda']!=0 or not cfg['ewc_skip_task_fisher'] or any(cfg[k] for k in ['approximate_fisher_in_task','in_task_weight_merge','task_endpoint_weight_merge']):
        raise AssertionError('Forbidden module enabled')
    if not all(a['checks'].values()):raise AssertionError('Run-level check failed')
    rows=[json.loads(line) for line in (out/'step_trace.jsonl').read_text().splitlines()]
    if len(rows)!=expected[2]:raise AssertionError('Incomplete trace')
    batches=[job['smoke_batches_per_epoch']]*expected[0] if job['smoke'] else [10,8,8,5,4,3,3,3,4,3]
    coords=[(t,e,b) for t in range(expected[0]) for e in range(1,expected[1]+1) for b in range(batches[t])]
    if len(coords)!=len(rows):raise AssertionError('Unexpected task/epoch/batch budget')
    used=0
    previous=None
    for i,(r,co) in enumerate(zip(rows,coords)):
        finite_tree(r)
        if r['step_index']!=i or (r['task'],r['epoch'],r['batch'])!=co or r['seed']!=2:
            raise AssertionError('Incorrect trace coordinates')
        if r['direction']!=job['direction'] or r['history_weight']!=job['history_weight']:
            raise AssertionError('Step condition changed')
        if r['ordinary_gradient_calls'] not in (1,2) or r['sgd_updates']!=1 or not r['natural_sgd_exact']:
            raise AssertionError('Incorrect gradient/update budget')
        if r['ordinary_gradient_calls']==1 and not r['fallback_reason']:
            raise AssertionError('Missing degenerate-probe fallback reason')
        if not all(r['state_restore_checks'].values()) or not all(r['diagnostic_state_restoration'].values()):
            raise AssertionError('State restoration failed')
        if not r['only_base_R1_rng_advance'] or r['diagnostic_loss_forward_count']!=1:
            raise AssertionError('Incorrect RNG/diagnostic accounting')
        if not r['strict_deterministic'] or r['gam_radii_active']:
            raise AssertionError('Incorrect execution/radius flags')
        if r['actual_model_forward_count']!=r['ordinary_gradient_calls']+1:
            raise AssertionError('Unexpected forward count')
        if any(r[k] for k in ['fisher_estimation_passes','weight_merge_updates','projection_calls','hvp_calls','qr_calls','create_graph_calls','gradient_clipping_calls','final_norm_rescale_calls']):
            raise AssertionError('Forbidden algorithm work')
        if not r['history_cache_uses_raw_gradient_not_U']:
            raise AssertionError('Wrong cached gradient identity')
        if not math.isclose(r['ell_f'],r['nominal_lr']*r['g_grad_norm'],abs_tol=1e-12):
            raise AssertionError('Predictor radius differs from eta*norm(g)')
        if r['history_used']:
            used+=1
            if job['direction']!='history_fused' or previous is None or previous['task']!=r['task']:
                raise AssertionError('History crossed task or used by no-history baseline')
            if r['history_cache_raw_gradient_sha256']!=previous['raw_gradient_sha256']:
                raise AssertionError('Cache is not previous raw gradient')
            if r['history_cache_actual_displacement_sha256']!=previous['actual_step_displacement_sha256']:
                raise AssertionError('Cache is not actual previous step')
            if not math.isclose(r['ell_b'],previous['actual_parameter_step_norm'],abs_tol=1e-12):
                raise AssertionError('Cached distance differs from actual previous displacement')
            if r['effective_history_weight']!=.5 or not math.isclose(r['history_coefficient'],.5*r['ell_f']/r['ell_b'],abs_tol=1e-12):
                raise AssertionError('Incorrect fusion coefficients')
        elif r['effective_history_weight']!=0.:
            raise AssertionError('Invalid history has nonzero weight')
        if previous is None or previous['task']!=r['task']:
            if r['history_used'] or r['previous_batch_identity'] is not None:
                raise AssertionError('Task-entry history not reset')
        if abs(r['post_update_delta_ce']-(r['post_update_ce']-r['loss0']))>1e-9:
            raise AssertionError('Actual loss-delta inconsistency')
        previous=r
    if job['direction']=='history_fused' and used==0:raise AssertionError('History version never exercised history')
    if job['direction']=='future_only' and used:raise AssertionError('No-history ablation used history')
    c=a['counters']
    if c['training_ce_gradient_passes']!=sum(r['ordinary_gradient_calls'] for r in rows):raise AssertionError('Total gradient count mismatch')
    if c['training_sgd_updates']!=expected[2] or c['diagnostic_loss_forward_count']!=expected[2]:raise AssertionError('SGD/diagnostic total differs')
    if any(c[k] for k in ['fisher_estimation_passes','weight_merge_updates','forbidden_work_attempts','training_hvp_passes','training_create_graph_passes','training_global_projection_calls']):raise AssertionError('Forbidden work counter')
    for name,key in [('step_trace.jsonl','step_trace_sha256'),('task_matrix.json','task_matrix_sha256')]:
        if sha256(out/name)!=a[key]:raise AssertionError('Result hashes changed')
    m=json.loads((out/'task_matrix.json').read_text());matrix=m['matrix']
    if len(matrix)!=expected[0] or any(len(r)!=i+1 for i,r in enumerate(matrix)):raise AssertionError('Incomplete task matrix')
    if any(not math.isfinite(v) or not 0<=v<=100 for r in matrix for v in r):raise AssertionError('Invalid matrix entry')
    for k,v in metrics(matrix).items():
        if v is None:
            if m[k] is not None:raise AssertionError('Structural undefined metric must be null')
        elif abs(m[k]-v)>1e-9 or abs(a['final_metrics'][k]-v)>1e-9:raise AssertionError('Metric recomputation mismatch')
    if len(a['checkpoint_dirs'])!=expected[0]:raise AssertionError('Incomplete checkpoint count')
    for task,checkpoint in enumerate(a['checkpoint_dirs']):
        d=Path(checkpoint);manifest=json.loads((d/'manifest.json').read_text())
        if not manifest['algorithm_cache_in_training_state'] or sha256(d/'training_state.pt')!=manifest['files_sha256']['training_state.pt']:
            raise AssertionError('Checkpoint lost or changed cache state')
        if manifest['task']!=task:raise AssertionError('Checkpoint task differs')
        for name,digest in manifest['files_sha256'].items():
            if sha256(d/name)!=digest:raise AssertionError('Checkpoint tensor file changed')
        import torch
        state=torch.load(d/'training_state.pt',map_location='cpu',weights_only=False)
        task_rows=[r for r in rows if r['task']==task]
        validate_cached_state(state,task_rows[-1],job,len(task_rows))
    return dict(summary_sha256=sha256(p),trace_sha256=a['step_trace_sha256'],matrix_sha256=a['task_matrix_sha256'],
                updates=len(rows),tasks=len(matrix),history_used_steps=used,metric_recomputation=True,checkpoint_cache_tensor_validation=True)
