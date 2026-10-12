"""Fail-closed completion evidence for the ordinary P/C/CD experiment."""
import json
import math
from pathlib import Path
from common import sha256, verify_config


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
    _,cfg,_=verify_config(job)
    out=Path(job['output_dir']);summary=read(out/'final_summary.json')
    n,epochs,steps=job['expected_tasks'],job['expected_epochs_per_task'],job['expected_steps']
    if summary['status']!='completed' or (summary['completed_tasks'],summary['epochs_per_task'],summary['completed_steps'])!=(n,epochs,steps):
        raise AssertionError('Incomplete training or wrong budget')
    if summary['completed_epochs']!=n*epochs or summary['step_count']!=steps:
        raise AssertionError('Epoch/update accounting differs')
    if (summary['probe_branch'],summary['update_direction'],summary['step_mode'],summary['attention_backend'])!=(job['branch'],'g2','natural','math'):
        raise AssertionError('Wrong ordinary scientific condition')
    if (summary['seed'],summary['r'],summary['rho'])!=(2,.25,.05):
        raise AssertionError('Seed/radii differ')
    if summary['input_config_sha256']!=job['config_sha256']:
        raise AssertionError('Executed input config hash differs')
    counters=summary['counters']
    if counters['training_ce_gradient_passes']!=4*steps or counters['sgd_updates']!=steps:
        raise AssertionError('Actual ordinary four-probe update count differs')
    for name in ('diagnostic_replay_ce_gradient_passes','isolated_diagnostic_ce_gradient_passes'):
        if counters[name]!=8*n:raise AssertionError('Isolated repeatability budget differs: '+name)
    if counters['training_forward_calls']!=4*steps or counters['diagnostic_replay_forward_calls']!=8*n:
        raise AssertionError('Actual model forward accounting differs')
    for name in ('training_hvp_passes','training_hvp_reference_gradient_passes','training_create_graph_passes',
                 'training_gradient_projection_passes','fisher_estimation_passes','weight_merge_updates','forbidden_work_attempts'):
        if counters[name]!=0:raise AssertionError('Prohibited work occurred: '+name)
    checks=summary['checks']
    required_checks={'finite','all_steps_self_spec','exact_snapshot_restore','four_base_probes',
        'fisher_empty_all_tasks','actual_weight_merges_zero','strict_bitwise_first_batch_replay',
        'all_task_invariants_passed','natural_g2_only','complete_task_lifecycle'}
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
        if (row['step_index'],row['task'],row['epoch'],row['batch'])!=(i,*coord) or row['seed']!=2:
            raise AssertionError('Exact training matrix coordinates differ')
        if (row['probe_branch'],row['update_direction'],row['step_mode'],row['attention_backend'])!=(job['branch'],'g2','natural','math'):
            raise AssertionError('Actual update condition differs')
        if row['sgd_updates']!=1 or row['training_ce_gradient_passes']!=4:
            raise AssertionError('Actual update derivative budget differs')
        if row['ordinary_gradient_calls']!=4 or row['closure_calls']!=4:
            raise AssertionError('Actual four CE closure accounting differs')
        if any(row[k]!=0 for k in ('hvp_calls','create_graph_calls','gradient_projection_calls','fisher_calls','weight_merge_calls','training_gradient_projection_passes')):
            raise AssertionError('Prohibited ordinary update module called')
        if any(row[name]!=0 for name in ('training_hvp_passes','training_create_graph_passes','training_hvp_reference_gradient_passes')):
            raise AssertionError('Higher-order derivative appeared in ordinary training')
        if row['class_slice']!=[20*coord[0],20*(coord[0]+1)] or row['seen_classes']!=20*(coord[0]+1):
            raise AssertionError('Actual task class slice differs')
        if not row['base_parameters_exact'] or not row['base_buffers_exact'] or not row['state_restoration'] or not all(row['state_restoration'].values()):
            raise AssertionError('Four-probe exact restoration failed')
        if not row['identity']['input_sha256'] or not row['identity']['target_sha256']:
            raise AssertionError('Missing actual input identity')
        if not isinstance(row['train_ce'],(int,float)) or not math.isfinite(row['train_ce']):
            raise AssertionError('Invalid actual CE')
    matrix=checked_file(out,summary,'task_matrix.json','task_matrix_sha256')['matrix']
    if len(matrix)!=n or any(len(row)!=i+1 for i,row in enumerate(matrix)):
        raise AssertionError('Observed task accuracy matrix incomplete')
    if not all(math.isfinite(v) and 0<=v<=100 for row in matrix for v in row):
        raise AssertionError('Invalid observed accuracy')
    replay=checked_file(out,summary,'replay_receipts.json','replay_receipts_sha256')
    if replay!=summary['replay_receipts'] or len(replay)!=n:
        raise AssertionError('Missing/replaced task replay gates')
    for t,receipt in enumerate(replay):
        required={'g0','g1_plus','g1_minus','q','outer','g2','U','losses','logits','R1'}
        if (receipt['task'],receipt['seed'],receipt['branch'])!=(t,2,job['branch']):
            raise AssertionError('Task replay identity differs')
        if receipt['class_slice']!=[20*t,20*(t+1)] or receipt['ordinary_gradient_calls']!=8:
            raise AssertionError('Replay class slice/ordinary CE count differs')
        first=next(r for r in rows if r['task']==t)
        if receipt['batch_identity']!=first['identity'] or receipt['forward_calls']!=8 or receipt['closure_calls']!=8 or receipt['sgd_updates']!=0:
            raise AssertionError('Replay did not use the exact first task batch or budget')
        if not all(receipt[k] is True for k in ('zero_updates','strict_bitwise','entry_state_restored')):
            raise AssertionError('Task repeatability/restoration did not pass')
        if set(receipt['checks'])!=required or not all(v is True for v in receipt['checks'].values()):
            raise AssertionError('Actual P/C/CD repeated tensors were not bitwise exact')
    identities=checked_file(out,summary,'task_identities.json','task_identities_sha256')
    checked_file(out,summary,'effective_config.json','effective_config_sha256')
    if len(identities)!=n:raise AssertionError('Missing per-task native lifecycle identity')
    for t,identity in enumerate(identities):
        if (identity['task'],identity['seed'],identity['class_slice'])!=(t,2,[20*t,20*(t+1)]) or not identity['old_fc_rows_inherited_exact']:
            raise AssertionError('Native classifier growth/class identity differs')
        if not identity['task_initial_parameter_sha256'] or not identity['frozen_origin_sha256'] or not identity['new_fc_rows_sha256']:
            raise AssertionError('Native initialization/origin/head identity missing')
    invariants=checked_file(out,summary,'task_invariant_receipts.json','task_invariant_receipts_sha256')
    if invariants!=summary['task_invariant_receipts'] or len(invariants)!=n:
        raise AssertionError('Incomplete whole-task invariant evidence')
    for t,receipt in enumerate(invariants):
        if receipt['task']!=t or receipt['status']!='passed':
            raise AssertionError('Task lifecycle invariant failed')
        required={'four_ce_per_update','four_forward_per_update','one_sgd_per_update',
            'eight_isolated_replay_ce','eight_isolated_replay_forwards','forbidden_modules_zero',
            'strict_replay_passed','old_fc_rows_inherited_exact'}
        if set(receipt['checks'])!=required or not all(v is True for v in receipt['checks'].values()):
            raise AssertionError('Whole-task Fisher/origin/parameter invariant failed')
        task_steps=batches[t]*epochs;c=receipt['counters']
        if receipt['steps']!=task_steps or c['training_ce_gradient_passes']!=4*task_steps or c['sgd_updates']!=task_steps or c['diagnostic_replay_ce_gradient_passes']!=8:
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
        updates=steps,training_ce=4*steps,isolated_replay_ce=8*n,tasks=n)


def validate_job(job):
    if job['kind']=='training':return validate_training(job)
    out=Path(job['output_dir']);path=out/job['report_file'];report=read(path)
    if job['kind']=='pairing':
        if report['status']!='passed' or report['training_runs']!=3 or report['optimizer_updates']!=0 or not all(report['checks'].values()):
            raise AssertionError('Smoke pairing failed')
        for path_str,digest in report['input_files_sha256'].items():
            if sha256(path_str)!=digest:raise AssertionError('Smoke pairing input changed')
        return dict(summary_sha256=sha256(path),gate='passed',paired_runs=3)
    if report['status']!='passed' or report['ordinary_gradient_calls']!=108 or report['state_batch_count']!=18:
        raise AssertionError('Offline g2 diagnostic status/budget differs')
    for name in ('optimizer_updates','sgd_updates','hvp_calls','create_graph_gradient_calls','fisher_calls','weight_merge_calls','projection_calls'):
        if report[name]!=0:raise AssertionError('Offline diagnostic performed prohibited work: '+name)
    for name in ('entry_state_restored','source_unchanged','reference_assets_unchanged','diagnostic_code_unchanged'):
        if report[name] is not True:raise AssertionError('Offline diagnostic integrity failed: '+name)
    if len(report['state_results'])!=9:raise AssertionError('Missing offline observation states')
    run_dir=Path(report['run_dir']);source=Path(report['source'])
    formal=[f'G2-CD0{i}_O_{b}_natural_s2' for i,b in enumerate(('P','C','CD'),1)]
    if report['runs']!=formal or report['tasks']!=[0,4,9] or report['seed']!=2 or report['formal_whole_task_matrix_identity'] is not True:
        raise AssertionError('Offline complete formal matrix pairing/coverage failed')
    if set(report['input_assets_sha256'])!=set(formal):raise AssertionError('Offline input runs differ')
    for run,files in report['input_assets_sha256'].items():
        for filename,digest in files.items():
            if sha256(run_dir/'outputs'/run/filename)!=digest:raise AssertionError('Offline pinned input changed: '+filename)
    for filename,digest in report['source_python_sha256'].items():
        if sha256(source/filename)!=digest:raise AssertionError('Offline source changed: '+filename)
    for filename,digest in report['diagnostic_code_sha256'].items():
        if sha256(run_dir/'trainer'/filename)!=digest:raise AssertionError('Offline diagnostic code changed: '+filename)
    registry=checked_file(out,report,'batch_registry.json','batch_registry_sha256')
    checked_file(out,report,'diagnostic_data_manifest.json','diagnostic_data_manifest_sha256')
    seen=set()
    for state in report['state_results']:
        key=(state['run_id'],state['task'])
        if key in seen or len(state['batches'])!=2:raise AssertionError('Duplicate/incomplete offline state')
        seen.add(key)
        if sha256(Path(state['checkpoint'])/'manifest.json')!=state['checkpoint_manifest_sha256']:
            raise AssertionError('Offline checkpoint manifest changed')
        for b,batch in enumerate(state['batches']):
            if batch['identity']!=registry['tasks'][str(state['task'])]['batches'][b] or batch['status']!='passed' or batch['ordinary_gradient_calls']!=6:
                raise AssertionError('Fixed offline augmented input/accounting differs')
            if batch['entry_state_restored'] is not True or not all(batch['entry_state_restore'].values()):
                raise AssertionError('Offline full state restoration failed')
            if any(batch[k]!=0 for k in ('optimizer_updates','sgd_updates','hvp_calls','create_graph_gradient_calls','fisher_calls','weight_merge_calls','projection_calls')):
                raise AssertionError('Prohibited work in actual offline state batch')
    if seen!={(run,t) for run in formal for t in (0,4,9)}:raise AssertionError('Offline state coordinates differ')
    return dict(summary_sha256=sha256(path),gate='passed',state_batches=18,ordinary_ce=108,optimizer_updates=0)
