#!/usr/bin/env python3
"""Native scientific completion and same-seed smoke pairing checks."""
import argparse,json,math
from pathlib import Path
from native_support import HERE,read_json,sha256,verify_config,inspect_metrics,now,atomic_json
from audit_validation import validate_job_audit

def validate(suite,job):
    _,cfg,output=verify_config(job)
    summary=read_json(output/'final_summary.json')
    if summary['status']!='completed' or summary['seed']!=job['seed'] or summary['id']!=job['id']:raise ValueError('Native summary identity differs')
    if summary['config_sha256']!=job['config_sha256'] or summary['scheduler_T_max']!=50:raise ValueError('Native summary config/scheduler differs')
    result=inspect_metrics(suite,job);proof=validate_job_audit(suite,job,require_receipt=False)
    if summary['result']!=result or summary['direction_proof']!=proof:raise ValueError('Native summary evidence changed')
    return {'passed':True,'job':job['id'],'seed':job['seed'],'summary_sha256':sha256(output/'final_summary.json'),
            'direction_proof':proof,'result':result,'config_sha256':job['config_sha256'],'completed_steps':job['expected_steps'],'completed_tasks':job['expected_tasks']}

def validate_smokes(suite):
    jobs=[j for j in suite['jobs'] if j['phase']=='smoke' and j['condition'].startswith('N')]
    if len(jobs)!=28:raise ValueError('Expected 28 native smokes')
    proofs={};traces={};seed_pairing={}
    for job in jobs:
        proofs[job['id']]=validate(suite,job)
        traces[(job['seed'],job['condition'])]=[json.loads(line) for line in (HERE/'runtime/jobs'/job['id']/'direction_audit.jsonl').read_text().splitlines()]
    for seed in (0,1):
        control=traces[(seed,'N03')]
        for (row_seed,condition),rows in traces.items():
            if row_seed!=seed:continue
            if len(rows)!=len(control):raise ValueError('Native paired count differs')
            for a,b in zip(rows,control):
                for key in ['input_sha256','target_sha256','rng_before_batch_sha256','rng_after_batch_sha256','task','task_step_index','class_offset','seen_classes','batch_samples','parameter_names','parameter_shapes']:
                    if a[key]!=b[key]:raise ValueError(f'Native same-seed pairing differs: {seed}/{condition}/{key}')
            if rows[0]['probe_parameter_sha256'][0]!=control[0]['probe_parameter_sha256'][0]:raise ValueError('Native initialization differs within seed')
        summed=traces[(seed,'N02')][0]['actual_effective_delta']['joint_norm']
        mean2=traces[(seed,'N13')][0]['actual_effective_delta']['joint_norm']
        difference=abs(summed-mean2)/summed
        if not math.isfinite(difference) or difference>1e-3:raise ValueError('Sum/Mean2 first effective displacement inconsistent')
        seed_pairing[str(seed)]={'conditions':14,'aligned_batches_per_condition':18,'initial_Sum_Mean2_norm_relative_difference':difference}
    return {'passed':True,'validated_at':now(),'jobs':proofs,'actual_verified_gpu_steps':504,'pairing_by_seed':seed_pairing,
            'scope':'exact own source decomposition/projector/SGD; within-seed data/RNG/init pairing; independent same-seed completed-S Gmatch'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--job');p.add_argument('--smokes',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();suite=read_json(a.suite)
    if a.smokes:result=validate_smokes(suite)
    else:result=validate(suite,next(j for j in suite['jobs'] if j['id']==a.job))
    atomic_json(a.output,result);print(json.dumps(result,allow_nan=False))
if __name__=='__main__':main()
