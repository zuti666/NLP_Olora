"""CPU-only, receipt-verified aggregation of four directions over seeds 0/1/2.

Historical seed 2 is admitted only by PARENT_REUSE_AUDIT.json. A full matrix
without the completed receipt is progress, never a final comparative result.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from common import atomic_text, atomic_json, now, sha256

H = Path(__file__).resolve().parent
PARENT = H.parent / 'odd-even-rho005-gpu123-20261003T185629Z'
CONDITIONS = [('OE01_SAM', 'g1_plus'), ('OE02_Backward', 'backward_reflected'),
              ('OE03_Central', 'central'), ('OE04_CE', 'g0')]
METRICS = ('AAA', 'FAA', 'LA', 'BWT', 'forgetting')
METRIC_KEYS = dict(AAA='AAA_task_macro', FAA='FAA_task_macro', LA='LA',
                   BWT='BWT', forgetting='forgetting')
SEEDS = (0, 1, 2)
BOUND_OUTPUTS = dict(summary_sha256='final_summary.json', trace_sha256='step_trace.jsonl',
    matrix_sha256='task_matrix.json', replay_sha256='replay_receipts.json',
    invariant_sha256='task_invariant_receipts.json', identities_sha256='task_identities.json')


def read(path):
    return json.loads(Path(path).read_text())


def metric_values(matrix_doc):
    """Recompute task-macro metrics from a complete finite triangular matrix."""
    m = matrix_doc['matrix']
    if len(m) != 10 or any(len(row) != i + 1 for i, row in enumerate(m)):
        raise ValueError('Expected ten completed lower-triangular task rows')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or not 0 <= x <= 100 for row in m for x in row):
        raise ValueError('Invalid task accuracy')
    n = len(m)
    values = dict(AAA=sum(sum(row)/len(row) for row in m)/n,
        FAA=sum(m[-1])/n, LA=sum(m[i][i] for i in range(n))/n,
        BWT=sum(m[-1][j]-m[j][j] for j in range(n-1))/(n-1),
        forgetting=sum(max(m[i][j] for i in range(j,n))-m[-1][j]
                       for j in range(n-1))/(n-1))
    for key, value in values.items():
        stored = matrix_doc[METRIC_KEYS[key]]
        if not isinstance(stored,(int,float)) or not math.isfinite(stored) or abs(stored-value)>1e-9:
            raise ValueError('Reported metric differs from raw matrix: '+key)
    return values


def verify_receipt(run, job, expected_seed):
    """Verify the immutable artifacts bound by the queue's successful validator."""
    run = Path(run); out = Path(job['output_dir'])
    runtime = run/'runtime/jobs'/job['id']
    completed = read(runtime/'completed.json'); exited = read(runtime/'exit.json')
    if completed['job_id'] != job['id'] or exited['job_id'] != job['id']:
        raise ValueError('Completion identity differs')
    if completed['exit'] != exited or exited['returncode'] != 0:
        raise ValueError('Missing successful matching process exit')
    bindings = {str(runtime/name): sha256(runtime/name)
                for name in ('completed.json','exit.json','job.json')}
    registered = read(runtime/'job.json')
    expected_registration = dict(job, cwd=read(run/'suite.json')['source_root'],
                                 cuda_witness_output=str(runtime/'cuda_witness.json'))
    if registered != expected_registration:
        raise ValueError('Runtime registration differs from suite')
    for field, name in BOUND_OUTPUTS.items():
        actual = sha256(out/name)
        if completed[field] != actual:
            raise ValueError('Completed artifact changed: '+name)
        bindings[str(out/name)] = actual
    config = read(job['config']); config_sha = sha256(job['config'])
    if config_sha != job['config_sha256'] or config['seed'] != [expected_seed]:
        raise ValueError('Configuration hash or seed differs')
    bindings[job['config']] = config_sha
    summary = read(out/'final_summary.json')
    required = dict(status='completed', seed=expected_seed, smoke=False,
        update_direction=job['direction'], probe_branch='local_symmetric',
        attention_backend='math', step_mode='natural', method_name='odd_even_local_v1',
        protocol='odd_even_local_exact_restore_v1', outer_radius=None, rho=.05,
        completed_tasks=10, completed_epochs=500, completed_steps=2550,
        input_config_sha256=config_sha)
    if any(summary.get(k) != v for k,v in required.items()):
        raise ValueError('Completed scientific identity or budget differs')
    if summary['trainer_sha256'] != sha256(run/'trainer/odd_even_train.py'):
        raise ValueError('Completed trainer hash differs')
    expected_counts = dict(sgd_updates=2550, training_ce_gradient_passes=7650,
        training_forward_calls=7650, diagnostic_replay_ce_gradient_passes=60,
        diagnostic_replay_forward_calls=60, fisher_estimation_passes=0,
        weight_merge_updates=0, training_gradient_projection_passes=0,
        training_hvp_passes=0, training_create_graph_passes=0,
        training_hvp_reference_gradient_passes=0, forbidden_work_attempts=0)
    if any(summary['counters'].get(k) != v for k,v in expected_counts.items()):
        raise ValueError('Actual compute accounting differs')
    if not summary['checks'] or not all(v is True for v in summary['checks'].values()):
        raise ValueError('Scientific checks failed')
    if (completed['tasks'], completed['updates'], completed['training_ce'],
        completed['isolated_replay_ce']) != (10,2550,7650,60):
        raise ValueError('Completion budget differs')
    values = metric_values(read(out/'task_matrix.json'))
    return dict(job=job['id'], direction=job['direction'], seed=expected_seed,
                status='validated_complete', output_dir=str(out), **values), bindings


def audit_parent(parent=PARENT, target=H/'PARENT_REUSE_AUDIT.json'):
    """Independently re-run the original CPU validator; do not touch parent data."""
    parent = Path(parent).resolve(); suite = read(parent/'suite.json')
    if Path(sys.executable).resolve() != Path(suite['python']).resolve():
        raise RuntimeError('Historical audit must use the registered suite Python')
    pins = read(parent/'RUN_PINS.json'); bindings = {}
    for name, expected in pins['files'].items():
        path = parent/name
        if sha256(path) != expected: raise ValueError('Parent pin changed: '+name)
        bindings[str(path)] = expected
    for name, expected in pins['external_files'].items():
        if sha256(name) != expected: raise ValueError('Parent external pin changed: '+name)
        bindings[name] = expected
    bindings[str(parent/'RUN_PINS.json')] = sha256(parent/'RUN_PINS.json')
    source_manifest = read(suite['source_manifest'])
    if len(source_manifest['files']) != 190 or sha256(suite['source_manifest']) != suite['source_manifest_sha256']:
        raise ValueError('Historical source manifest differs')
    for relative, expected in source_manifest['files'].items():
        path = Path(suite['source_root'])/relative
        if sha256(path) != expected: raise ValueError('Historical source changed: '+relative)
        bindings[str(path)] = expected
    rows=[]; parent_jobs=[]; original_validation=[]
    for ident, direction in CONDITIONS:
        job = next(j for j in suite['jobs'] if j['id'] == ident+'_s2')
        if job['smoke'] or job['direction'] != direction:
            raise ValueError('Wrong historical condition')
        code = ('import json,sys;sys.path.insert(0,sys.argv[1]);'
                'from validate_job import validate_job;from common import load_suite;'
                'j=next(j for j in load_suite()["jobs"] if j["id"]==sys.argv[2]);'
                'print(json.dumps(validate_job(j),sort_keys=True))')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        result = subprocess.run([suite['python'],'-B','-c',code,str(parent),job['id']],
                cwd=str(parent),env=env,capture_output=True,text=True,timeout=180)
        if result.returncode != 0:
            raise RuntimeError('Original parent validator failed: '+job['id']+' '+result.stderr[-4000:])
        verdict=json.loads(result.stdout)
        completed=read(parent/'runtime/jobs'/job['id']/'completed.json')
        if any(completed.get(k) != v for k,v in verdict.items()):
            raise ValueError('Original validator differs from completion receipt')
        row,bound=verify_receipt(parent,job,2)
        rows.append(row); parent_jobs.append(job); bindings.update(bound)
        original_validation.append(dict(job=job['id'],returncode=0,receipt=verdict))
    assets=read(parent/'gates/INPUT_ASSETS.json'); spec=read(parent/'environment_spec.json')
    if assets['status']!='passed' or not assets['data']['all_expected_hashes_exact'] or not assets['pretrained']['verified']:
        raise ValueError('Historical asset acceptance is missing')
    if assets['data']['asset_hashes'] != spec['data']['asset_hashes'] or assets['pretrained']['sha256'] != spec['weight_dirs']['vit_b16']['sha256']:
        raise ValueError('Historical inputs differ from frozen specification')
    audit=dict(schema_version=1,status='passed',audited_at=now(),cpu_only=True,gpu_calls=0,
        python_executable=sys.executable,python_version=sys.version,
        parent_run=str(parent),allowed_seed=2,allowed_directions=[d for _,d in CONDITIONS],
        parent_suite_sha256=sha256(parent/'suite.json'),parent_pins_sha256=sha256(parent/'RUN_PINS.json'),
        parent_pins_all_exact=True,source_files_verified=190,
        historical_environment_spec_sha256=sha256(parent/'environment_spec.json'),
        historical_asset_receipt_sha256=sha256(parent/'gates/INPUT_ASSETS.json'),
        historical_data_asset_hashes=spec['data']['asset_hashes'],
        historical_pretrained_sha256=spec['weight_dirs']['vit_b16']['sha256'],
        historical_environment=spec['software'],
        historical_asset_scope='Frozen accepted historical receipt; current assets are checked by this campaign asset preflight.',
        accepted_config_differences=['seed','prefix','outputs_root'],
        jobs=parent_jobs,rows=rows,original_validator_results=original_validation,bound_files=bindings,
        excluded_cohorts='All old native/four-probe runs and all other directions or seeds')
    atomic_json(target,audit)
    return audit


def parent_rows(audit):
    if audit['status']!='passed' or audit['allowed_seed']!=2:
        raise ValueError('Parent reuse has not passed')
    for path,expected in audit['bound_files'].items():
        if sha256(path)!=expected: raise ValueError('Historical reuse binding changed: '+path)
    rows=[]
    for job in audit['jobs']:
        row,_=verify_receipt(audit['parent_run'],job,2)
        saved=next(r for r in audit['rows'] if r['job']==job['id'])
        if row!=saved: raise ValueError('Historical validated result differs')
        rows.append(row)
    if {(r['direction'],r['seed']) for r in rows}!={(d,2) for _,d in CONDITIONS}:
        raise ValueError('Unexpected parent cohort')
    return rows


def aggregate(rows):
    """Withhold estimates until each condition/pair has all three planned seeds."""
    lookup={}
    for row in rows:
        if row['status']!='validated_complete': continue
        key=(row['direction'],row['seed'])
        if key in lookup: raise ValueError('Duplicate condition/seed')
        if key[0] not in {d for _,d in CONDITIONS} or key[1] not in SEEDS:
            raise ValueError('Unregistered aggregate cohort')
        if any(not isinstance(row[m],(int,float)) or not math.isfinite(row[m]) for m in METRICS):
            raise ValueError('Invalid aggregate metric')
        lookup[key]=row
    groups=[]
    for ident,direction in CONDITIONS:
        seeds=[s for s in SEEDS if (direction,s) in lookup]
        row=dict(condition=ident,direction=direction,n_complete=len(seeds),expected_n=3,
                 complete_seeds=','.join(map(str,seeds)),status='complete' if len(seeds)==3 else 'incomplete',
                 eligible_for_three_seed_comparison=len(seeds)==3)
        for metric in METRICS:
            for seed in SEEDS: row[f'{metric}_seed{seed}']=lookup.get((direction,seed),{}).get(metric)
            values=[lookup[(direction,s)][metric] for s in seeds]
            row[metric+'_mean']=statistics.mean(values) if len(seeds)==3 else None
            row[metric+'_sample_sd']=statistics.stdev(values) if len(seeds)==3 else None
        groups.append(row)
    paired=[]
    for baseline in ('g0','g1_plus'):
        for _,direction in CONDITIONS:
            if direction==baseline: continue
            shared=[s for s in SEEDS if (direction,s) in lookup and (baseline,s) in lookup]
            for metric in METRICS:
                deltas={s:lookup[(direction,s)][metric]-lookup[(baseline,s)][metric] for s in shared}
                row=dict(direction=direction,baseline=baseline,metric=metric,
                    difference='candidate minus baseline',n_complete=len(shared),expected_n=3,
                    complete_seeds=','.join(map(str,shared)),status='complete' if len(shared)==3 else 'incomplete',
                    **{f'delta_seed{s}':deltas.get(s) for s in SEEDS},
                    paired_mean=statistics.mean(deltas.values()) if len(shared)==3 else None,
                    paired_sample_sd=statistics.stdev(deltas.values()) if len(shared)==3 else None,
                    improvement_direction='lower' if metric=='forgetting' else 'higher',
                    wins=sum(v<0 if metric=='forgetting' else v>0 for v in deltas.values()) if len(shared)==3 else None,
                    ties=sum(v==0 for v in deltas.values()) if len(shared)==3 else None)
                paired.append(row)
    return groups,paired


def write_csv(path,rows):
    stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);atomic_text(path,stream.getvalue())


def collect():
    suite=read(H/'suite.json'); audit=read(H/'PARENT_REUSE_AUDIT.json')
    if Path(sys.executable).resolve() != Path(suite['python']).resolve():
        raise RuntimeError('Result collection must use the registered suite Python')
    if suite['parent_run']!=audit['parent_run']:
        raise ValueError('Suite parent differs from reuse audit')
    old=parent_rows(audit)
    state=read(H/'runtime/queue_state.json') if (H/'runtime/queue_state.json').exists() else {'jobs':{}}
    rows=[]; validation_errors=[]
    formal=[j for j in suite['jobs'] if j['kind']=='training' and not j['smoke']]
    if len(formal)!=8: raise ValueError('Exactly eight new formal runs required')
    for job in formal:
        config=read(job['config']); seeds=config['seed']
        if len(seeds)!=1 or seeds[0] not in (0,1): raise ValueError('Only new seeds 0/1 are registered')
        seed=seeds[0]
        if job.get('seed',seed)!=seed: raise ValueError('Job and config seeds differ')
        parent_job=next(j for j in audit['jobs'] if j['direction']==job['direction'])
        parent_config=read(parent_job['config'])
        differing={k for k in set(config)|set(parent_config) if config.get(k)!=parent_config.get(k)}
        if differing != {'seed','prefix','outputs_root'}:
            raise ValueError('Unexpected config differences: '+job['id']+' '+repr(differing))
        record=state['jobs'].get(job['id'],{}); out=Path(job['output_dir'])
        matrix=read(out/'task_matrix.json') if (out/'task_matrix.json').exists() else {}
        row=dict(job=job['id'],phase=job['phase'],branch=job['branch'],direction=job['direction'],
            mode=job['mode'],backend=job['backend'],seed=seed,status=record.get('status','pending'),
            validation_status='not_completed',tasks=len(matrix.get('matrix',[])),
            **{m:None for m in METRICS},optimizer_updates=0,training_ce=0,isolated_replay_ce=0,
            output_dir=str(out),validation_error='')
        receipt=H/'runtime/jobs'/job['id']/'completed.json'
        if receipt.exists():
            try:
                # Replay the current CPU validator, not merely the queue label.
                # This also verifies per-step and effective-configuration files.
                from validate_job import validate_job
                verdict=validate_job(job)
                completed=read(receipt)
                if any(completed.get(k)!=v for k,v in verdict.items()):
                    raise ValueError('Current validator differs from completion receipt')
                verified,_=verify_receipt(H,job,seed)
                row.update({m:verified[m] for m in METRICS})
                row.update(status='validated_complete',validation_status='passed',tasks=10,
                    optimizer_updates=2550,training_ce=7650,isolated_replay_ce=60)
            except Exception as exc:
                row.update(status='validation_failed',validation_status='failed',validation_error=str(exc))
                validation_errors.append(dict(job=job['id'],error=str(exc)))
        elif record.get('status')=='completed':
            row.update(status='completed_unverified',validation_error='Missing completed.json')
        rows.append(row)
    if {(r['direction'],r['seed']) for r in rows}!={(d,s) for _,d in CONDITIONS for s in (0,1)}:
        raise ValueError('New cohort is not exactly the four directions crossed with seeds 0/1')
    groups,pairs=aggregate(rows+old)
    write_csv(H/'summary.csv',rows);write_csv(H/'THREE_SEED_SUMMARY.csv',groups)
    write_csv(H/'PAIRED_DIFFERENCES.csv',pairs)
    atomic_json(H/'results.json',dict(updated_at=now(),rows=rows,reused_seed2=old,
        three_seed=groups,paired_differences=pairs,validation_errors=validation_errors,
        metrics_scope='task-macro percentages; forgetting and paired differences in percentage points',
        sample_sd_ddof=1,ranking_performed=False))
    fmt=lambda value:'—' if value is None else f'{value:.3f}'
    lines=['# 正向、校正反向、中心与基点：补充 seed0/1','',
        '更新时间：'+now()+'。四方向均为 rho=.05、无外层位置、math FP32、三探针自然步长。',
        '只有完成凭据与原始矩阵核验通过的结果进入统计；每条件齐 seed0/1/2 后才计算均值和样本标准差，不对未完成组排名。','',
        '| 新实验 | seed | 状态 | 已完成任务 | AAA | FAA |',
        '|---|---:|---|---:|---:|---:|']
    for row in rows: lines.append(f"| {row['job']} | {row['seed']} | {row['status']} | {row['tasks']} | {fmt(row['AAA'])} | {fmt(row['FAA'])} |")
    lines+=['','| 条件 | 已完成/预期 | seed0 AAA/FAA | seed1 AAA/FAA | seed2 AAA/FAA | 三seed AAA均值±SD | 三seed FAA均值±SD |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in groups:
        vals=[fmt(row[f'AAA_seed{s}'])+' / '+fmt(row[f'FAA_seed{s}']) for s in SEEDS]
        stats=[fmt(row[m+'_mean'])+' ± '+fmt(row[m+'_sample_sd']) for m in ('AAA','FAA')]
        lines.append('| '+row['condition']+' | '+str(row['n_complete'])+'/3 | '+' | '.join(vals+stats)+' |')
    if validation_errors: lines+=['','存在验证错误，相关结果已排除：',json.dumps(validation_errors,ensure_ascii=False)]
    atomic_text(H/'RESULTS.md','\n'.join(lines)+'\n')
    return dict(new_validated=sum(r['status']=='validated_complete' for r in rows),
                expected_new=8,reused_validated=len(old),errors=validation_errors)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-parent',action='store_true')
    args=parser.parse_args()
    if args.audit_parent:
        audit=audit_parent();print(json.dumps(dict(status=audit['status'],parent_runs=len(audit['rows']),
                                                   bound_files=len(audit['bound_files']))))
    else: print(json.dumps(collect(),ensure_ascii=False))

if __name__=='__main__': main()
