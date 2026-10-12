"""Accept pinned minimal-read training and isolated finite-step diagnostics."""
import hashlib,json,math
from pathlib import Path
from common import HERE,PROTOCOL,bad_loss_lines,read_json,sha256,verify_config,expected_budget,snapshot_rng_fingerprint

def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):raise AssertionError('Nonfinite evidence')
    if isinstance(value,dict):
        for child in value.values():finite_tree(child)
    elif isinstance(value,(list,tuple)):
        for child in value:finite_tree(child)

def metrics(matrix):
    n=len(matrix)
    return dict(AAA_task_macro=sum(sum(row)/len(row) for row in matrix)/n,FAA_task_macro=sum(matrix[-1])/n,
                LA=sum(matrix[i][i] for i in range(n))/n,
                BWT=sum(matrix[-1][i]-matrix[i][i] for i in range(n-1))/(n-1) if n>1 else 0.,
                forgetting=sum(max(matrix[t][i] for t in range(i,n))-matrix[-1][i] for i in range(n-1))/(n-1) if n>1 else 0.)

def rows(path):
    with open(path) as f:return [json.loads(line) for line in f if line.strip()]

def snapshot_check(job,summary,trace):
    meta=summary['same_state_snapshot']
    if not job['smoke']:
        if meta is not None:raise AssertionError('Formal run contains smoke-only snapshot')
        return None
    if meta!=trace[0].get('same_state_snapshot') or sha256(meta['path'])!=meta['sha256']:raise AssertionError('Snapshot hash/trace mismatch')
    import torch
    payload=torch.load(meta['path'],map_location='cpu',weights_only=False)
    if payload['protocol']!=PROTOCOL or payload['factors']!={'direction':job['direction']}:raise AssertionError('Snapshot condition differs')
    required={'g0','g1','gm'} if job['direction']=='two_midpoint' else ({'g0','g1','g2','g3'} if job['direction']=='sum13' else {'g0','g1','g2'})
    if set(payload['grads'])!=required or payload['identity']!=trace[0]['identity']:raise AssertionError('Snapshot reads/input mismatch')
    names=payload['parameter_names']
    if not names or len(names)!=len(set(names)):raise AssertionError('Incomplete parameter layout')
    for group,key in [('grads','g012_fingerprints'),('points','point012_fingerprints')]:
        observed={}
        for label,tensors in payload[group].items():
            if len(tensors)!=len(names) or any(v.device.type!='cpu' or not torch.isfinite(v).all().item() for v in tensors):raise AssertionError('Invalid snapshot tensor')
            observed[label]=[hashlib.sha256(v.detach().contiguous().numpy().tobytes()).hexdigest() for v in tensors]
        if observed!=payload[key] or observed!=trace[0][key]:raise AssertionError('Snapshot tensor fingerprint mismatch')
    return dict(verified=True,sha256=meta['sha256'],rng0_sha256=snapshot_rng_fingerprint(payload['rng0']),rng1_sha256=snapshot_rng_fingerprint(payload['rng1']),parameter_layout_sha256=hashlib.sha256(json.dumps([(n,str(v.dtype),list(v.shape)) for n,v in zip(names,payload['points']['phi0'])]).encode()).hexdigest())

def validate_job(job):
    out=Path(job['output_dir']);path=out/job.get('report_file','final_summary.json');report=read_json(path);finite_tree(report)
    exit_record=read_json(HERE/'runtime/jobs'/job['id']/'exit.json')
    if exit_record['job_id']!=job['id'] or exit_record['returncode']!=0:raise AssertionError('Runner did not exit successfully')
    if job['kind']=='pairing':
        if report['status']!='passed' or report['protocol']!=PROTOCOL or report['smoke_count']!=4 or report['formal_count']!=4 or not all(report['checks'].values()):raise AssertionError('Pairing gate failed')
        return dict(summary_sha256=sha256(path),gate='passed')
    config=verify_config(job)
    if job['kind']=='model_gate':
        if report['status']!='passed' or not report['checks'] or not all(v is True for v in report['checks'].values()):raise AssertionError('Real model native/audit regression failed')
        if report['gradient_calls_total']!=33 or report['independent_sgd_updates']!=8 or set(report['directions'])!={'sum12','base02','two_midpoint','sum13'}:raise AssertionError('Real model gate grid/accounting mismatch')
        return dict(summary_sha256=sha256(path),gate='passed',real_model_native_audit=True)
    if job['kind']=='diagnostic':
        expected={'combination_count':1 if job['smoke'] else 24,'ordinary_gradient_calls':6 if job['smoke'] else 144,
                  'sgd_updates':0,'old_data_gradient_calls':0,'fp64_calibrated_count':1 if job['smoke'] else 6,
                  'precision_uncalibrated_count':0 if job['smoke'] else 18}
        if report['status']!='completed' or report['accepted'] is not True:raise AssertionError('Diagnostic did not complete')
        for key,value in expected.items():
            if report[key]!=value:raise AssertionError('Diagnostic accounting mismatch: '+key)
        for key in ['restoration_all_exact','saved_virtual_fp32_coordinates','source_hashes_unchanged']:
            if report[key] is not True:raise AssertionError('Diagnostic evidence check failed: '+key)
        if report['diagnostic_sha256']!=sha256(out/'diagnostics.jsonl'):raise AssertionError('Diagnostic trace hash differs')
        entries=rows(out/'diagnostics.jsonl')
        if len(entries)!=expected['combination_count']:raise AssertionError('Incomplete diagnostic entries')
        for entry in entries:
            finite_tree(entry)
            if not all(entry['state_restoration'].values()) or entry['virtual_update_coordinate_dtype']!='torch.float32':raise AssertionError('Diagnostic restoration/coordinate mismatch')
            if sha256(entry['coordinate_file'])!=entry['coordinate_file_sha256']:raise AssertionError('Virtual coordinate archive changed')
        if sum(bool(e['fp64_calibrated']) for e in entries)!=expected['fp64_calibrated_count']:raise AssertionError('FP64 calibration grid differs')
        if any(sha256(filename)!=digest for filename,digest in report['source_hashes'].items()):raise AssertionError('Reference artifact changed during diagnosis')
        return dict(summary_sha256=sha256(path),diagnostic=True,**expected)
    summary=report;budget=expected_budget(config);steps=job['expected_steps']
    if (summary['status'],summary['protocol'],summary['completed_tasks'],summary['epochs_per_task'],summary['completed_steps'],summary['completed_epochs'])!=('completed',PROTOCOL,job['expected_tasks'],job['expected_epochs_per_task'],steps,job['expected_tasks']*job['expected_epochs_per_task']):raise AssertionError('Incomplete training')
    if (summary['seed'],summary['r'],summary['rho'],summary['update_direction'],summary['step_mode'],summary['attention_backend'])!=(2,1.,.05,job['direction'],'natural','native'):raise AssertionError('Wrong scientific condition')
    if (summary['deterministic_enabled'],summary['deterministic_warn_only'],summary['cublas_workspace_config'])!=(True,False,':4096:8'):raise AssertionError('Actual deterministic flags differ')
    if summary['input_config_sha256']!=job['config_sha256'] or summary['trainer_sha256']!=sha256(HERE/'trainer/four_probe_train.py'):raise AssertionError('Input/code evidence hash differs')
    if summary['extra_diagnostic_gradient_calls'] or summary['gradients_per_step']!=budget['train_ce_per_update']:raise AssertionError('Wrong native gradient budget')
    if not summary['checks'] or not all(v is True for v in summary['checks'].values()):raise AssertionError('Training self-audit failed')
    c=summary['counters']
    expected_counts={'training_batches':steps,'training_sgd_updates':steps,'training_ce_gradient_passes':steps*budget['train_ce_per_update'],'training_forward_count':steps*budget['train_ce_per_update']}
    for key in ['g0','g1','g2','gm','g3']:expected_counts['training_'+key+'_gradient_passes']=steps*budget[key+'_ce_per_update']
    for key,value in expected_counts.items():
        if c[key]!=value:raise AssertionError('Observed derivative count differs: '+key)
    forbidden=['fisher_estimation_passes','weight_merge_updates','forbidden_work_attempts','training_global_projection_calls','training_hvp_passes','training_create_graph_passes','training_opposite_g1_gradient_passes','diagnostic_gradient_calls','diagnostic_forward_count','diagnostic_sgd_updates']
    if any(c[key] for key in forbidden):raise AssertionError('Unrequested derivative/Fisher/merge/update work')
    trace=rows(out/'step_trace.jsonl')
    if len(trace)!=steps:raise AssertionError('Incomplete update trace')
    for index,row in enumerate(trace):
        finite_tree(row)
        if row['step_index']!=index or row['update_direction']!=job['direction'] or row['ordinary_gradient_calls']!=budget['train_ce_per_update'] or row['sgd_updates']!=1 or row['training_forward_count']!=budget['train_ce_per_update']:raise AssertionError('Step budget/identity mismatch')
        if not row['natural_sgd_exact'] or not all(row['state_restoration'].values()) or row['audit_all'] or row['additional_diagnostic_gradient_calls']:raise AssertionError('Step restoration/SGD/native budget failed')
        if any(row[k] for k in ['fisher_calls','weight_merge_calls','gradient_projection_calls','hvp_calls','create_graph_calls']):raise AssertionError('Forbidden step work')
        for key in ['g0','g1','g2','gm','g3']:
            if row[key+'_gradient_calls']!=budget[key+'_ce_per_update']:raise AssertionError('Unused/omitted read count differs')
        for geom in ['first_probe_parameter_audit','outer_location_parameter_audit']:
            if row[geom]['passed'] is not True:raise AssertionError('Probe geometry failed')
        if job['direction']=='sum13' and row['second_probe_parameter_audit']['passed'] is not True:raise AssertionError('Terminal geometry failed')
    contexts=rows(out/'context_diagnostics.jsonl')
    if len(contexts)!=steps or any(r['diagnostic_gradient_calls'] or r['scope']!='training_reads_only_no_extra_gradient' for r in contexts):raise AssertionError('Hidden context readouts')
    epochs=read_json(out/'metrics.json')['epochs'];finite_tree(epochs)
    if len(epochs)!=job['expected_tasks']*job['expected_epochs_per_task']:raise AssertionError('Incomplete epoch metrics')
    identities=read_json(out/'task_identities.json')
    if len(identities)!=job['expected_tasks'] or any(not t['old_fc_rows_inherited_exact'] or t['class_slice']!=[20*i,20*(i+1)] for i,t in enumerate(identities)):raise AssertionError('FC/data lifecycle mismatch')
    after=read_json(out/'after_task_checks.json')
    if len(after)!=job['expected_tasks'] or any(not all(r[k] for k in ['parameters_exact','parameter_references_exact','fisher_empty']) for r in after):raise AssertionError('Task-end work changed model')
    matrix_record=read_json(out/'task_matrix.json');matrix=matrix_record['matrix'];finite_tree(matrix_record)
    if len(matrix)!=job['expected_tasks'] or any(len(row)!=i+1 or any(not 0<=v<=100 for v in row) for i,row in enumerate(matrix)):raise AssertionError('Incomplete task matrix')
    recomputed=metrics(matrix)
    for key,value in recomputed.items():
        if abs(matrix_record[key]-value)>1e-9 or abs(summary['final_metrics'][key]-value)>1e-9:raise AssertionError('Metric recomputation failed')
    for file,key in [('step_trace.jsonl','step_trace_sha256'),('task_matrix.json','task_matrix_sha256'),('context_diagnostics.jsonl','context_diagnostics_sha256'),('observations.json','observations_sha256')]:
        if sha256(out/file)!=summary[key]:raise AssertionError('Receipt output hash differs')
    observations=read_json(out/'observations.json');points=['entry']+['epoch'+str(e) for e in [1,5,25,50] if e<=job['expected_epochs_per_task']]
    if sorted((r['task'],r['point']) for r in observations)!=sorted((t,p) for t in range(job['expected_tasks']) for p in points):raise AssertionError('Missing planned snapshots')
    for obs in observations:
        if sha256(obs['path'])!=obs['sha256']:raise AssertionError('Observation snapshot changed')
    for directory in summary['checkpoint_dirs']:
        manifest=read_json(Path(directory)/'manifest.json')
        for name,digest in manifest['files_sha256'].items():
            if sha256(Path(directory)/name)!=digest:raise AssertionError('Checkpoint receipt mismatch')
        if not (Path(directory)/'training_state.pt').is_file():raise AssertionError('No checkpoint optimizer/RNG state')
    witness=read_json(HERE/'runtime/jobs'/job['id']/'cuda_witness.json')
    if witness['status']!='passed':raise AssertionError('Kernel witness failed')
    bad=bad_loss_lines(HERE/'runtime/jobs'/job['id']/'console.log')
    if bad:raise AssertionError('Nonfinite logged loss: '+repr(bad))
    snap=snapshot_check(job,summary,trace)
    return dict(summary_sha256=sha256(path),matrix_sha256=sha256(out/'task_matrix.json'),trace_sha256=sha256(out/'step_trace.jsonl'),tasks=len(matrix),updates=steps,metric_recomputation=True,same_state_snapshot=snap,**recomputed)
