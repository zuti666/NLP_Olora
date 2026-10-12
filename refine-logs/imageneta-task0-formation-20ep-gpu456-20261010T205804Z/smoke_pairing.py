#!/usr/bin/env python3
"""Passive pairing of six real 2x20-epoch smoke runs, including epoch11."""
import argparse,json
from pathlib import Path
from common import atomic_json,now,sha256
CONDITIONS=('sgd','sam','gam_residual','sf_g2','sf_g2_then_sgd','sgd_then_sf_g2')

def validate(suite_path):
    suite=json.loads(Path(suite_path).read_text());jobs=[j for j in suite['jobs'] if j['kind']=='training' and j['smoke']]
    checks={};evidence={};cells={}
    def check(name,value):
        checks[name]=bool(value)
        if not checks[name]:raise AssertionError('Formation pairing failed: '+name)
    def equal(name,values):check(name,bool(values) and all(v is not None and v==values[0] for v in values))
    check('six_unique_seed2_native_smokes',len(jobs)==6 and {j['formation_condition_id'] for j in jobs}==set(CONDITIONS) and all(j['seed']==2 for j in jobs))
    for j in jobs:
        out=Path(j['output_dir']);sp=out/'final_summary.json';tp=out/'step_trace.jsonl';summary=json.loads(sp.read_text());rows=[json.loads(l) for l in tp.read_text().splitlines() if l.strip()]
        c=j['formation_condition_id'];cells[c]=(j,summary,rows)
        check(c+'_full40_updates',summary['status']=='completed' and summary['smoke'] and summary['completed_tasks']==2 and summary['completed_epochs']==40 and summary['completed_steps']==len(rows)==40)
        check(c+'_config_hash',summary['input_config_sha256']==j['config_sha256']==sha256(j['config']))
        check(c+'_trace_hash',summary['step_trace_sha256']==sha256(tp))
        check(c+'_task1_always_native_sgd',all(r['task']==1 and r['epoch']==i+1 and r['active_method']=='sgd' and r['ordinary_gradient_calls']==1 and r['U_sha256']==r['g0_sha256'] for i,r in enumerate(rows[20:])))
        check(c+'_switch_without_reset',len(summary['switch_receipts'])==1 and all(summary['switch_receipts'][0][k] for k in ('optimizer_same','scheduler_same','model_parameters_unchanged_at_switch','parameter_references_unchanged_at_switch','rng_unchanged_at_switch')))
        for path in (sp,tp,Path(j['config'])):evidence[str(path)]=sha256(path)
    allcells=list(cells.values())
    equal('six_task0_initial_model', [a['initial_identity']['task_initial_parameter_sha256'] for _,a,_ in allcells])
    equal('six_frozen_origin', [a['initial_identity']['frozen_origin_sha256'] for _,a,_ in allcells])
    equal('six_task0_fc20_new_rows', [a['task_identities'][0]['new_fc_rows_sha256'] for _,a,_ in allcells])
    equal('six_task1_fc40_new_rows', [a['task_identities'][1]['new_fc_rows_sha256'] for _,a,_ in allcells])
    check('six_task1_old_fc_rows_inherited',all(a['task_identities'][1]['old_fc_rows_inherited_exact'] for _,a,_ in allcells))
    for index in range(40):
        equal('all40_realized_inputs_'+str(index),[rows[index]['identity'] for _,_,rows in allcells])
        equal('all40_common_lr_'+str(index),[rows[index]['nominal_lr'] for _,_,rows in allcells])
        equal('all40_common_coordinate_'+str(index),[(rows[index]['task'],rows[index]['epoch'],rows[index]['batch']) for _,_,rows in allcells])
    equal('six_first_g0_exact',[rows[0]['g0_sha256'] for _,_,rows in allcells])
    equal('sam_and_gam_first_ascent_g1_exact',[cells[c][2][0]['g1_sha256'] for c in ('sam','gam_residual')])
    equal('sf_and_early_sf_first_descent_g1_exact',[cells[c][2][0]['g1_sha256'] for c in ('sf_g2','sf_g2_then_sgd')])
    for a,b,name in (('sgd','sgd_then_sf_g2','sgd_vs_late'),('sf_g2','sf_g2_then_sgd','sf_vs_early')):
        for index in range(10):
            left,right=cells[a][2][index],cells[b][2][index]
            for key in ('g0_sha256','U_sha256','base_parameter_sha256','actual_post_parameter_sha256','actual_step_displacement_sha256','identity','nominal_lr','active_method'):
                equal(name+'_first10_'+str(index)+'_'+key,[left[key],right[key]])
    check('early_sf_switches_only_epoch11', [r['active_method'] for r in cells['sf_g2_then_sgd'][2][:20]]==['sf_g2']*10+['sgd']*10)
    check('late_sf_switches_only_epoch11', [r['active_method'] for r in cells['sgd_then_sf_g2'][2][:20]]==['sgd']*10+['sf_g2']*10)
    check('native_gradient_budget400',sum(sum(r['ordinary_gradient_calls'] for r in rows) for _,_,rows in allcells)==400)
    return dict(status='passed',schema='FormationSixSmokePairing_v1',smoke_count=6,checked_at_utc=now(),checks=checks,
        evidence_hashes=evidence,training_mutated=False,gpu_used=False,
        scope='Task0 common initializer; full40 realized inputs and cosine LR; first10 paired trajectory oracles; epoch11 switches; Task1 new FC initialization and native SGD',
        caveat='Task0 endpoints and Task1 full model/g0 intentionally differ between methods; those hashes are not forced equal')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite',required=True);parser.add_argument('--output-dir',required=True);a=parser.parse_args();out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Pairing output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    try:report=validate(a.suite)
    except BaseException as exc:
        atomic_json(out/'pairing_report.json',dict(status='failed',schema='FormationSixSmokePairing_v1',error=repr(exc),checked_at_utc=now(),training_mutated=False,gpu_used=False));raise
    atomic_json(out/'pairing_report.json',report);print(json.dumps(dict(status='passed',smoke_count=6,checks=len(report['checks']))),flush=True)
if __name__=='__main__':main()
