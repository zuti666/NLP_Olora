#!/usr/bin/env python3
"""Read-only pairing gate for 16 upward training smokes (no torch/CUDA calls)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

READS={'g0':1,'g1':2,'g2_minus_s':3,'g2':3,'g2_plus_s':3,
       'g0_plus_d':4,'g1_plus_d':4,'g3':4}
SINGLE={'g0','g1','g2_minus_s','g2','g1_plus_d','g3'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text())

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as handle:
            json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')
            handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)

def validate(suite_path):
    suite_path=Path(suite_path).resolve();suite=read_json(suite_path)
    jobs=[j for j in suite['jobs'] if j.get('smoke') and j.get('kind')=='training']
    parents=[j for j in jobs if j['phase']=='parent']
    branches=[j for j in jobs if j['phase']=='branch']
    singles=[j for j in jobs if j['phase']=='single200']
    checks={};evidence={};cells={}
    def check(name,condition):
        checks[name]=bool(condition)
        if not checks[name]:raise AssertionError('Smoke pairing check failed: '+name)
    def equal(name,values):
        check(name,bool(values) and all(v is not None for v in values) and all(v==values[0] for v in values))
    check('exact_16_training_smokes',len(jobs)==16)
    check('one_parent_nine_branches_six_single200',len(parents)==1 and len(branches)==9 and len(singles)==6)
    check('eight_unique_r1_rules_plus_r025_g2_bridge',
          {j['upward_direction'] for j in branches if j['r']==1.}==set(READS)
          and len([j for j in branches if j['r']==1.])==8
          and len([j for j in branches if j['r']==.25 and j['upward_direction']=='g2'])==1)
    check('six_registered_single200_rules',{j['upward_direction'] for j in singles}==SINGLE)
    for job in jobs:
        identifier=job['id'];out=Path(job['output_dir']).resolve()
        summary_path=out/'final_summary.json';trace_path=out/'step_trace.jsonl'
        summary=read_json(summary_path)
        trace=[json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        direction=job['upward_direction'];config_path=Path(job['config']).resolve()
        check(identifier+'_completed',summary['status']=='completed' and summary['smoke'] is True)
        check(identifier+'_seed2',summary['seed']==job['seed']==2)
        check(identifier+'_two_real_updates',summary['completed_steps']==len(trace)==2)
        check(identifier+'_config_pin',digest(config_path)==job['config_sha256']==summary['input_config_sha256'])
        check(identifier+'_trace_pin',digest(trace_path)==summary['step_trace_sha256'])
        check(identifier+'_registered_direction',summary['upward_direction']==direction)
        check(identifier+'_one_epoch',summary['completed_epochs']==1)
        for i,row in enumerate(trace):
            check(identifier+'_batch%d_native_probe_budget'%i,
                row['ordinary_gradient_calls']==row['gradient_reads']==READS[direction]
                and row['native_probe_skipping'] is True and row['direction']==direction)
            check(identifier+'_batch%d_single_natural_sgd'%i,
                row['sgd_updates']==1 and row['natural_sgd_exact'] is True)
            check(identifier+'_batch%d_no_auxiliary_work'%i,
                all(row[key]==0 for key in ('fisher_estimation_passes','weight_merge_updates',
                    'projection_calls','hvp_calls','create_graph_calls','final_norm_rescale_calls'))
                and row['history_used'] is False)
            check(identifier+'_batch%d_exact_rng_commit'%i,row['only_base_R1_rng_advance'] is True)
            check(identifier+'_batch%d_finite_loss_lr'%i,
                math.isfinite(row['train_ce']) and math.isfinite(row['nominal_lr']))
        identity=summary['initial_identity']
        check(identifier+'_old_fc_rows_exact',identity['old_fc_rows_inherited_exact'] is True)
        cells[identifier]=dict(job=job,summary=summary,trace=trace,identity=identity)
        evidence[identifier]=dict(output_dir=str(out),config_sha256=digest(config_path),
            summary_sha256=digest(summary_path),step_trace_sha256=digest(trace_path),
            first_g0_sha256=trace[0]['g0_sha256'],initial_identity=identity,
            first_batch_identity=trace[0]['identity'],second_batch_identity=trace[1]['identity'],
            first_nominal_lr=trace[0]['nominal_lr'],second_nominal_lr=trace[1]['nominal_lr'])
    parent=cells[parents[0]['id']]
    b=[cells[j['id']] for j in branches];s=[cells[j['id']] for j in singles]
    equal('nine_branches_complete_parent_provenance',[c['summary']['parent_restore_receipt'] for c in b])
    check('nine_branches_parent_model_is_accepted_parent',
        all(c['summary']['parent_restore_receipt']['parent_model_sha256']==parent['summary']['final_model_parameter_sha256'] for c in b))
    equal('nine_branches_expanded_initial_parameter_identity',[c['identity']['task_initial_parameter_sha256'] for c in b])
    equal('nine_branches_expanded_new_fc_rows',[c['identity']['new_fc_rows_sha256'] for c in b])
    equal('nine_branches_fc40_shape',[c['identity']['fc_shape'] for c in b])
    check('branches_use_full_fc40',all(c['identity']['fc_shape'][0]==40 for c in b))
    equal('nine_branches_frozen_origin',[c['identity']['frozen_origin_sha256'] for c in b])
    check('branch_frozen_origin_matches_parent',all(c['identity']['frozen_origin_sha256']==parent['identity']['frozen_origin_sha256'] for c in b))
    equal('nine_branches_pre_expand_baseline',[c['summary']['fork_baselines']['pre_expand'] for c in b])
    equal('nine_branches_post_expand_baseline',[c['summary']['fork_baselines']['post_expand'] for c in b])
    for i in (0,1):
        equal('nine_branches_batch%d_realized_inputs_targets_indices'%i,[c['trace'][i]['identity'] for c in b])
        equal('nine_branches_batch%d_nominal_lr'%i,[c['trace'][i]['nominal_lr'] for c in b])
        check('nine_branches_batch%d_task1_current20_ce'%i,all(c['trace'][i]['class_slice']==[20,40] for c in b))
    equal('nine_branches_first_g0_sha256',[c['trace'][0]['g0_sha256'] for c in b])
    equal('eight_g1_readers_first_g1_sha256_including_bridge',
        [c['trace'][0]['g1_sha256'] for c in b if READS[c['job']['upward_direction']]>=2])
    equal('six_r1_g2_readers_first_g2_sha256',
        [c['trace'][0]['g2_sha256'] for c in b if c['job']['r']==1. and READS[c['job']['upward_direction']]>=3])
    equal('three_r1_g3_readers_first_g3_sha256',
        [c['trace'][0]['g3_sha256'] for c in b if READS[c['job']['upward_direction']]>=4])
    equal('six_single200_initial_parameter_identity',[c['identity']['task_initial_parameter_sha256'] for c in s])
    equal('six_single200_initial_fc_rows',[c['identity']['new_fc_rows_sha256'] for c in s])
    equal('six_single200_frozen_origin',[c['identity']['frozen_origin_sha256'] for c in s])
    equal('six_single200_fc200_shape',[c['identity']['fc_shape'] for c in s])
    check('single200_full_classifier',all(c['identity']['fc_shape'][0]==200 for c in s))
    for i in (0,1):
        equal('six_single200_batch%d_realized_inputs_targets_indices'%i,[c['trace'][i]['identity'] for c in s])
        equal('six_single200_batch%d_nominal_lr'%i,[c['trace'][i]['nominal_lr'] for c in s])
        check('six_single200_batch%d_all200_ce'%i,all(c['trace'][i]['class_slice']==[0,200] for c in s))
    equal('six_single200_first_g0_sha256',[c['trace'][0]['g0_sha256'] for c in s])
    return dict(status='passed',schema='UpwardSixteenSmokePairing_v1',smoke_count=16,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),suite=str(suite_path),suite_sha256=digest(suite_path),
        checks=checks,evidence=evidence,
        scope='same expanded parent/model/head/first probe; same first and second realized batch/LR; native skipped-probe budgets',
        caveat='second gradient hashes are not compared after different real first updates; r=.25 bridge shares g1 but not g2',
        training_mutated=False,gpu_used=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True);parser.add_argument('--output-dir',required=True)
    cli=parser.parse_args();out=Path(cli.output_dir).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Pairing output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    try:report=validate(cli.suite)
    except BaseException as exc:
        report=dict(status='failed',schema='UpwardSixteenSmokePairing_v1',error=repr(exc),
            checked_at_utc=datetime.now(timezone.utc).isoformat(),training_mutated=False,gpu_used=False)
        atomic_json(out/'pairing_report.json',report)
        raise
    atomic_json(out/'pairing_report.json',report)
    print(json.dumps(dict(status=report['status'],smoke_count=report['smoke_count'],checks=len(report['checks']))))

if __name__=='__main__':main()
