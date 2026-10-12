"""Fail-closed completion evidence for the three-probe odd/even experiment."""
import json
import math
import re
from pathlib import Path
from common import HERE, sha256, verify_config, load_suite, verify_source, FORMULAS


def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):
        raise AssertionError('Nonfinite scientific evidence')
    if isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite_tree(v)


def read(path):
    value=json.loads(Path(path).read_text());finite_tree(value);return value


def checked_file(out,summary,name,field):
    path=out/name
    if summary[field]!=sha256(path):raise AssertionError('Evidence hash differs: '+name)
    return read(path)


def validate_training(job):
    suite=load_suite()
    registered=next((j for j in suite['jobs'] if j['id']==job['id']),None)
    if registered!=job:raise AssertionError('Completion job differs from registered suite')
    verify_source(suite)
    _,cfg,_=verify_config(job)
    seed=job['seed']
    out=Path(job['output_dir']);summary=read(out/'final_summary.json')
    n,epochs,steps=job['expected_tasks'],job['expected_epochs_per_task'],job['expected_steps']
    if summary['status']!='completed' or (summary['completed_tasks'],summary['epochs_per_task'],summary['completed_steps'])!=(n,epochs,steps):
        raise AssertionError('Incomplete training or wrong budget')
    if summary['completed_epochs']!=n*epochs or summary['step_count']!=steps:
        raise AssertionError('Epoch/update accounting differs')
    if (summary['probe_branch'],summary['update_direction'],summary['step_mode'],summary['attention_backend'])!=(job['branch'],job['direction'],'natural','math'):
        raise AssertionError('Wrong ordinary scientific condition')
    if (summary['seed'],summary['outer_radius'],summary['rho'])!=(seed,None,.05):
        raise AssertionError('Seed/radii differ')
    if summary['input_config_sha256']!=job['config_sha256']:
        raise AssertionError('Executed input config hash differs')
    if summary['source']!=suite['source_root'] or summary['trainer_sha256']!=sha256(HERE/'trainer/odd_even_train.py'):
        raise AssertionError('Executed source/trainer identity differs')
    if summary['method_name']!='odd_even_local_v1' or summary['protocol']!='odd_even_local_exact_restore_v1':
        raise AssertionError('Method/protocol differs')
    if summary['smoke'] is not job['smoke'] or summary['requested_tasks']!=n or summary['task_count']!=n or summary['scheduler_T_max']!=50 or summary['matched_steps']!=0:
        raise AssertionError('Actual smoke/natural schedule differs')
    counters=summary['counters']
    if counters['training_ce_gradient_passes']!=3*steps or counters['sgd_updates']!=steps:
        raise AssertionError('Actual ordinary three-probe update count differs')
    for name in ('diagnostic_replay_ce_gradient_passes','isolated_diagnostic_ce_gradient_passes'):
        if counters[name]!=6*n:raise AssertionError('Isolated repeatability budget differs: '+name)
    if counters['training_forward_calls']!=3*steps or counters['diagnostic_replay_forward_calls']!=6*n:
        raise AssertionError('Actual model forward accounting differs')
    for name in ('training_hvp_passes','training_hvp_reference_gradient_passes','training_create_graph_passes',
                 'training_gradient_projection_passes','fisher_estimation_passes','weight_merge_updates','forbidden_work_attempts'):
        if counters[name]!=0:raise AssertionError('Prohibited work occurred: '+name)
    checks=summary['checks']
    required_checks={'finite','all_steps_self_spec','exact_snapshot_restore','three_base_probes',
        'fisher_empty_all_tasks','actual_weight_merges_zero','strict_bitwise_first_batch_replay',
        'all_task_invariants_passed','natural_odd_even_only','complete_task_lifecycle'}
    if set(checks)!=required_checks or not all(v is True for v in checks.values()):
        raise AssertionError('Scientific checks failed')
    trace=out/'step_trace.jsonl'
    if sha256(trace)!=summary['step_trace_sha256']:raise AssertionError('Step trace changed')
    rows=[json.loads(line) for line in trace.read_text().splitlines()]
    if len(rows)!=steps:raise AssertionError('Trace count differs')
    batches=[1]*n if job['smoke'] else [10,8,8,5,4,3,3,3,4,3]
    coordinates=[(t,e,b) for t in range(n) for e in range(1,epochs+1) for b in range(batches[t])]
    for i,(row,coord) in enumerate(zip(rows,coordinates)):
        finite_tree(row)
        if (row['step_index'],row['task'],row['epoch'],row['batch'])!=(i,*coord) or row['seed']!=seed:
            raise AssertionError('Exact training matrix coordinates differ')
        if (row['probe_branch'],row['update_direction'],row['step_mode'],row['attention_backend'])!=(job['branch'],job['direction'],'natural','math'):
            raise AssertionError('Actual update condition differs')
        if row['sgd_updates']!=1 or row['training_ce_gradient_passes']!=3:
            raise AssertionError('Actual update derivative budget differs')
        if row['ordinary_gradient_calls']!=3 or row['closure_calls']!=3:
            raise AssertionError('Actual three CE closure accounting differs')
        if row['training_forward_calls']!=3 or row['base_sgd_exact'] is not True or row['natural_odd_even_only'] is not True:
            raise AssertionError('Actual forward count or unique restored-base SGD differs')
        if row['rho']!=.05 or row['outer_radius'] is not None or row['method_name']!='odd_even_local_v1' or row['eps']!=1e-12:
            raise AssertionError('Actual odd/even method/radius differs')
        if row['update_formula']!=FORMULAS[job['direction']] or row['probe_formula']!='theta0 +/- rho*g0/(norm(g0)+eps)':
            raise AssertionError('Registered FP32 odd/even arithmetic differs')
        if set(row['norms'])!={'g0','g1_plus','g1_minus','s','e','correction','U'} or set(row['probe_displacements'])!={'phi1_plus','phi1_minus'} or set(row['parameter_probe_displacements'])!={'phi1_plus','phi1_minus'}:
            raise AssertionError('Unexpected forward/outer probe evidence')
        if i==0:
            hashes=row.get('probe_tensor_sha256',{})
            if set(hashes)!={'g0','g1_plus','g1_minus','plus_point','minus_point'}:
                raise AssertionError('Missing exact shared-initial probe identities')
            values=list(hashes.values())+[row.get('rng0_sha256'),row.get('rng1_sha256')]
            if any(not isinstance(v,str) or not re.fullmatch(r'[0-9a-f]{64}',v) for v in values):
                raise AssertionError('Invalid shared-initial tensor/RNG hash')
        if row['matched'] is not False or row['scale_over_lr']!=1. or row['nominal_lr']!=row['scale'] or row['step_scale']!=row['scale']:
            raise AssertionError('Final odd/even direction was rescaled')
        lr=(.04 if coord[0]==0 else .02)*(.1+.9*(1+math.cos(math.pi*(coord[1]-1)/50))/2)
        if not math.isclose(row['nominal_lr'],lr,rel_tol=1e-12,abs_tol=1e-14):
            raise AssertionError('Actual natural cosine LR differs')
        if any(row[k]!=0 for k in ('hvp_calls','create_graph_calls','gradient_projection_calls','fisher_calls','weight_merge_calls','training_gradient_projection_passes')):
            raise AssertionError('Prohibited ordinary update module called')
        if any(row[name]!=0 for name in ('training_hvp_passes','training_create_graph_passes','training_hvp_reference_gradient_passes')):
            raise AssertionError('Higher-order derivative appeared in ordinary training')
        if row['class_slice']!=[20*coord[0],20*(coord[0]+1)] or row['seen_classes']!=20*(coord[0]+1):
            raise AssertionError('Actual task class slice differs')
        if not row['base_parameters_exact'] or not row['base_buffers_exact'] or not row['state_restoration'] or not all(row['state_restoration'].values()):
            raise AssertionError('Three-probe exact restoration failed')
        if row['identity'].get('data_seed') != seed*100000+coord[0]*1000+coord[1]:
            raise AssertionError('Actual data/augmentation seed differs from seed*100000+task*1000+epoch')
        if not row['identity']['input_sha256'] or not row['identity']['target_sha256']:
            raise AssertionError('Missing actual input identity')
        if not isinstance(row['train_ce'],(int,float)) or not math.isfinite(row['train_ce']):
            raise AssertionError('Invalid actual CE')
    matrix_artifact=checked_file(out,summary,'task_matrix.json','task_matrix_sha256')
    if summary['final_metrics']!=matrix_artifact:raise AssertionError('Final metrics were replaced')
    matrix=matrix_artifact['matrix']
    if len(matrix)!=n or any(len(row)!=i+1 for i,row in enumerate(matrix)):
        raise AssertionError('Observed task accuracy matrix incomplete')
    if not all(math.isfinite(v) and 0<=v<=100 for row in matrix for v in row):
        raise AssertionError('Invalid observed accuracy')
    expected_metrics=dict(FAA_task_macro=sum(matrix[-1])/n,AAA_task_macro=sum(sum(row)/len(row) for row in matrix)/n,
        LA=sum(matrix[t][t] for t in range(n))/n,
        BWT=sum(matrix[-1][j]-matrix[j][j] for j in range(n-1))/(n-1),
        forgetting=sum(max(matrix[t][j] for t in range(j,n))-matrix[-1][j] for j in range(n-1))/(n-1))
    if any(not math.isclose(matrix_artifact[k],v,rel_tol=1e-12,abs_tol=1e-12) for k,v in expected_metrics.items()):
        raise AssertionError('Final metrics disagree with observed raw matrix')
    replay=checked_file(out,summary,'replay_receipts.json','replay_receipts_sha256')
    if replay!=summary['replay_receipts'] or len(replay)!=n:
        raise AssertionError('Missing/replaced task replay gates')
    for t,receipt in enumerate(replay):
        required={'g0','g1_plus','g1_minus','s','e','correction','U','plus_point','minus_point','losses','logits','R1'}
        if (receipt['task'],receipt['seed'],receipt['branch'])!=(t,seed,job['branch']):
            raise AssertionError('Task replay identity differs')
        if receipt['class_slice']!=[20*t,20*(t+1)] or receipt['ordinary_gradient_calls']!=6:
            raise AssertionError('Replay class slice/ordinary CE count differs')
        first=next(r for r in rows if r['task']==t)
        if receipt['batch_identity']!=first['identity'] or receipt['forward_calls']!=6 or receipt['closure_calls']!=6 or receipt['sgd_updates']!=0:
            raise AssertionError('Replay did not use the exact first task batch or budget')
        if not all(receipt[k] is True for k in ('zero_updates','strict_bitwise','entry_state_restored')):
            raise AssertionError('Task repeatability/restoration did not pass')
        if receipt['method_name']!='odd_even_local_v1' or receipt['passed'] is not True:
            raise AssertionError('Replay method/status differs')
        if set(receipt['checks'])!=required or not all(v is True for v in receipt['checks'].values()):
            raise AssertionError('Actual odd/even repeated tensors were not bitwise exact')
    identities=checked_file(out,summary,'task_identities.json','task_identities_sha256')
    effective=checked_file(out,summary,'effective_config.json','effective_config_sha256')
    expected_effective=dict(cfg,seed=seed,device=['cuda:0'],nb_tasks=n,max_train_tasks=n,
        component_rule='odd_even_local_v1',optimizer_tag_override='odd_even_local_v1',
        method_name='odd_even_local_v1',outer_radius=None,protocol='odd_even_local_exact_restore_v1',
        actual_optimizer='torch.optim.SGD',actual_optimizer_momentum=0.,actual_optimizer_weight_decay=0.,
        actual_amp_enabled=False,precision='float32',dtype='float32',
        normalize_final_gradient=False,gradient_projection=False,hvp=False,create_graph=False,
        amp=False,use_amp=False,tf32=False,allow_tf32=False,deterministic_algorithms=True)
    if any(effective.get(k)!=v for k,v in expected_effective.items()):
        differing=[k for k,v in expected_effective.items() if effective.get(k)!=v]
        raise AssertionError('Effective runtime configuration differs: '+repr(differing))
    sizes=[1259,997,993,596,426,260,357,317,398,378]
    details=read(out/'tasks.json')
    if details!=summary['task_details'] or len(details)!=n:raise AssertionError('Native task details differ')
    for t,detail in enumerate(details):
        if (detail['task'],detail['train_dataset_samples'],detail['batches_per_epoch'],detail['epochs'],detail['steps'])!=(t,sizes[t],batches[t],epochs,batches[t]*epochs):
            raise AssertionError('Actual task sample/epoch/batch matrix differs')
    if len(identities)!=n:raise AssertionError('Missing per-task native lifecycle identity')
    for t,identity in enumerate(identities):
        if (identity['task'],identity['seed'],identity['class_slice'])!=(t,seed,[20*t,20*(t+1)]) or not identity['old_fc_rows_inherited_exact']:
            raise AssertionError('Native classifier growth/class identity differs')
        if identity.get('data_seed_rule') != 'seed*100000+task*1000+epoch':
            raise AssertionError('Task metadata data seed rule differs')
        if not identity['task_initial_parameter_sha256'] or not identity['frozen_origin_sha256'] or not identity['new_fc_rows_sha256']:
            raise AssertionError('Native initialization/origin/head identity missing')
    invariants=checked_file(out,summary,'task_invariant_receipts.json','task_invariant_receipts_sha256')
    if invariants!=summary['task_invariant_receipts'] or len(invariants)!=n:
        raise AssertionError('Incomplete whole-task invariant evidence')
    for t,receipt in enumerate(invariants):
        if receipt['task']!=t or receipt['status']!='passed':
            raise AssertionError('Task lifecycle invariant failed')
        required={'three_ce_per_update','three_forward_per_update','one_sgd_per_update',
            'six_isolated_replay_ce','six_isolated_replay_forwards','forbidden_modules_zero',
            'strict_replay_passed','old_fc_rows_inherited_exact'}
        if set(receipt['checks'])!=required or not all(v is True for v in receipt['checks'].values()):
            raise AssertionError('Whole-task Fisher/origin/parameter invariant failed')
        if receipt['seed']!=seed or receipt['branch']!='local_symmetric' or receipt['passed'] is not True:
            raise AssertionError('Whole-task identity differs')
        task_steps=batches[t]*epochs;c=receipt['counters']
        if c['training_forward_calls']!=3*task_steps or c['diagnostic_replay_forward_calls']!=6 or c['isolated_diagnostic_ce_gradient_passes']!=6:
            raise AssertionError('Whole-task forward/replay accounting differs')
        if any(c[k]!=0 for k in ('training_hvp_passes','training_hvp_reference_gradient_passes','training_create_graph_passes','training_gradient_projection_passes','fisher_estimation_passes','weight_merge_updates','forbidden_work_attempts')):
            raise AssertionError('Whole-task prohibited work occurred')
        if receipt['steps']!=task_steps or c['training_ce_gradient_passes']!=3*task_steps or c['sgd_updates']!=task_steps or c['diagnostic_replay_ce_gradient_passes']!=6:
            raise AssertionError('Whole-task actual compute accounting differs')
    after=read(out/'after_task_checks.json')
    if after!=summary['after_task_checks'] or len(after)!=n:
        raise AssertionError('Missing actual after_task preservation evidence')
    for t,receipt in enumerate(after):
        if receipt['task']!=t or receipt['known_classes']!=20*(t+1) or not all(receipt[k] is True for k in ('parameters_exact','parameter_references_exact','fisher_empty')):
            raise AssertionError('Whole-task after_task parameter/Fisher identity differs')
    return dict(summary_sha256=sha256(out/'final_summary.json'),trace_sha256=sha256(trace),
        matrix_sha256=sha256(out/'task_matrix.json'),replay_sha256=sha256(out/'replay_receipts.json'),
        invariant_sha256=sha256(out/'task_invariant_receipts.json'),identities_sha256=sha256(out/'task_identities.json'),
        updates=steps,training_ce=3*steps,isolated_replay_ce=6*n,tasks=n)


def validate_pairing(job):
    suite=load_suite()
    if next((j for j in suite['jobs'] if j['id']==job['id']),None)!=job:
        raise AssertionError('Unregistered pairing job')
    out=Path(job['output_dir']);path=out/'smoke_pairing_report.json';report=read(path)
    from verify_pairing import compute_pairing
    expected=compute_pairing(suite)
    for k,v in expected.items():
        if report.get(k)!=v:raise AssertionError('Pairing evidence differs: '+k)
    return dict(summary_sha256=sha256(path),training_runs=8,seeds=[0,1],optimizer_updates=0)


def validate_job(job):
    if job['kind']=='training':return validate_training(job)
    if job['kind']=='pairing':return validate_pairing(job)
    raise AssertionError('Only registered training and CPU pairing jobs are authorized')
