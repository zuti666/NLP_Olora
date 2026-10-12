"""Validate observed training records and derivative budgets before accepting endpoints."""
import json
import math
from pathlib import Path
from common import sha256
H=Path(__file__).resolve().parent

def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):raise AssertionError('Nonfinite evidence')
    if isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite_tree(v)

def metrics(matrix):
    n=len(matrix)
    a=dict(AAA_task_macro=sum(sum(row)/len(row) for row in matrix)/n,
           FAA_task_macro=sum(matrix[-1])/n,LA=sum(matrix[i][i] for i in range(n))/n)
    a['BWT']=sum(matrix[-1][i]-matrix[i][i] for i in range(n-1))/(n-1) if n>1 else 0.
    a['forgetting']=sum(max(matrix[k][i] for k in range(i,n))-matrix[-1][i] for i in range(n-1))/(n-1) if n>1 else 0.
    return a

def validate_job(job):
    out=Path(job['output_dir'])
    if job['kind']=='pairing':
        path=out/job['report_file'];r=json.loads(path.read_text())
        if r['status']!='passed' or r['smoke_count']!=2 or r['formal_count']!=2:
            raise AssertionError('Smoke pairing did not pass')
        if not all(r['checks'].values()):raise AssertionError('Smoke pairing check failed')
        return dict(summary_sha256=sha256(path),gate='passed')
    p=out/'final_summary.json';a=json.loads(p.read_text());finite_tree(a)
    expected=(job['expected_tasks'],job['expected_epochs_per_task'],job['expected_steps'])
    if a['status']!='completed' or (a['completed_tasks'],a['epochs_per_task'],a['completed_steps'])!=expected:
        raise AssertionError('Incomplete training budget')
    if a['completed_epochs']!=expected[0]*expected[1]:raise AssertionError('Incomplete epochs')
    if (a['seed'],a['r'],a['rho'],a['probe_branch'],a['update_direction'],a['step_mode'],a['attention_backend'])!=(2,job['radius'],.05,job['branch'],job['direction'],'natural','native'):
        raise AssertionError('Wrong scientific condition')
    c=a['counters']
    if c['training_ce_gradient_passes']!=4*expected[2]:raise AssertionError('Not four probes per update')
    if any(c[k] for k in ['fisher_estimation_passes','weight_merge_updates','forbidden_work_attempts','training_hvp_passes','training_create_graph_passes','training_hvp_reference_gradient_passes']):
        raise AssertionError('Forbidden Fisher/merge/second derivative work')
    required={'finite','all_steps_self_spec','exact_snapshot_restore','four_probes','fisher_empty_all_tasks','actual_weight_merges_zero',
              'g3_outer_probe','opposite_probe_omitted','natural_sgd_exact'}
    if not required<=set(a['checks']) or not all(a['checks'].values()):raise AssertionError('Failed scientific checks')
    if c['training_forward_count']!=4*expected[2] or c['training_sgd_updates']!=expected[2]:raise AssertionError('Forward/update accounting differs')
    if c['training_g3_gradient_passes']!=expected[2] or c['training_opposite_g1_gradient_passes']:raise AssertionError('Unused probe ran')
    if c['training_global_projection_calls']!=0:raise AssertionError('Projection accounting differs')
    if a['input_config_sha256']!=job['config_sha256'] or a['trainer_sha256']!=sha256(H/'trainer/four_probe_train.py'):
        raise AssertionError('Config/trainer hashes changed')
    for name,key in [('step_trace.jsonl','step_trace_sha256'),('task_matrix.json','task_matrix_sha256')]:
        if sha256(out/name)!=a[key]:raise AssertionError('Result hash changed')
    cfg=json.loads((out/'effective_config.json').read_text())
    if (cfg['gam_grad_rho'],cfg['gam_grad_norm_rho'],cfg['lora_rank'],cfg['seed'])!=(.05,job['radius'],10,2):
        raise AssertionError('Effective radius/rank/seed differs')
    if cfg['ewc_lambda']!=0 or not cfg['ewc_skip_task_fisher'] or any(cfg[k] for k in ['approximate_fisher_in_task','in_task_weight_merge','task_endpoint_weight_merge']):
        raise AssertionError('Forbidden enabled module')
    rows=[json.loads(v) for v in (out/'step_trace.jsonl').read_text().splitlines()]
    if len(rows)!=expected[2]:raise AssertionError('Trace count mismatch')
    batches=[1]*expected[0] if job['smoke'] else [10,8,8,5,4,3,3,3,4,3]
    coords=[(t,e,b) for t in range(expected[0]) for e in range(1,expected[1]+1) for b in range(batches[t])]
    for i,(r,co) in enumerate(zip(rows,coords)):
        finite_tree(r)
        if r['step_index']!=i or (r['task'],r['epoch'],r['batch'])!=co or r['seed']!=2:
            raise AssertionError('Coverage/seed mismatch')
        if (r['probe_branch'],r['update_direction'],r['step_mode'],r['attention_backend'])!=(job['branch'],job['direction'],'natural','native'):
            raise AssertionError('Step direction mismatch')
        if r['ordinary_gradient_calls']!=4 or r['training_ce_gradient_passes']!=4 or r['sgd_updates']!=1:
            raise AssertionError('Step gradient/update budget mismatch')
        if r['radius']!=job['radius'] or r['rho']!=.05 or r['training_forward_count']!=4 or not r['natural_sgd_exact']:
            raise AssertionError('Actual radius/forward/update differed')
        if any(r[k] for k in ['opposite_g1_gradient_calls','hvp_calls','create_graph_calls']):
            raise AssertionError('Unused gradient/HVP probe ran')
        if r['g3_gradient_calls']!=1 or r['gradient_projection_calls'] or r['projection'] is not None:
            raise AssertionError('Missing g3 or an unwanted projection')
        if not r['base_parameters_exact'] or not r['base_buffers_exact'] or not all(r['state_restoration'].values()):
            raise AssertionError('Step restoration failed')
        if r['class_slice']!=[20*co[0],20*(co[0]+1)] or r['seen_classes']!=20*(co[0]+1):
            raise AssertionError('Wrong task slice')
        if r['scale']!=r['nominal_lr'] or r.get('matched',False):raise AssertionError('Natural step was scaled')
        if any(r[k] for k in ['training_hvp_passes','training_create_graph_passes','training_hvp_reference_gradient_passes']):
            raise AssertionError('Second derivatives used')
        if not math.isfinite(r['train_ce']):raise AssertionError('Nonfinite train CE')
    m=json.loads((out/'task_matrix.json').read_text())
    matrix=m['matrix']
    if len(matrix)!=expected[0] or any(len(row)!=i+1 for i,row in enumerate(matrix)):raise AssertionError('Incomplete matrix')
    if any(not math.isfinite(v) or not 0<=v<=100 for row in matrix for v in row):raise AssertionError('Invalid accuracy')
    for k,v in metrics(matrix).items():
        if abs(m[k]-v)>1e-9 or abs(a['final_metrics'][k]-v)>1e-9:raise AssertionError('Metric recomputation failed: '+k)
    return dict(summary_sha256=sha256(p),trace_sha256=a['step_trace_sha256'],matrix_sha256=a['task_matrix_sha256'],updates=len(rows),tasks=len(matrix),metric_recomputation=True)
