"""Four real radius smokes: initial pairing and frozen legacy trajectory checks."""
import json
from pathlib import Path
from common import HERE,read_json,atomic_json,sha256
from validate_job import validate_job

def main():
    suite=read_json(HERE/'suite.json');jobs=[j for j in suite['jobs'] if j['kind']=='training' and j['smoke']]
    if len(jobs)!=4 or {j['radius_variant'] for j in jobs}!={'original','fixed_inner','fixed_outer','fixed_both'}:
        raise AssertionError('Expected original plus three new bounded radius smokes')
    rows={};summaries={};evidence=[]
    for j in jobs:
        receipt=read_json(HERE/'runtime/jobs'/j['id']/'completed.json')
        checked=validate_job(j)
        if receipt['summary_sha256']!=checked['summary_sha256'] or receipt['exit']['returncode']!=0:raise AssertionError('Missing accepted smoke receipt')
        evidence.append(dict(job=j['id'],**checked))
        out=Path(j['output_dir']);summaries[j['radius_variant']]=read_json(out/'final_summary.json')
        rows[j['radius_variant']]=[json.loads(x) for x in (out/'step_trace.jsonl').read_text().splitlines()]
    ref=rows['original'];first=summaries['original']['initial_identity'];checks={}
    checks['same_initial_model_and_head']=all(all(s['initial_identity'][k]==first[k] for k in ('task_initial_parameter_sha256','frozen_origin_sha256','new_fc_rows_sha256')) for s in summaries.values())
    checks['same_realized_inputs_all_four_steps']=all([r['identity'] for r in rr]==[r['identity'] for r in ref] for rr in rows.values())
    checks['same_nominal_learning_rates']=all([r['nominal_lr'] for r in rr]==[r['nominal_lr'] for r in ref] for rr in rows.values())
    checks['same_first_base_gradient']=len({rr[0]['g0_sha256'] for rr in rows.values()})==1
    checks['same_inner_geometry_coupled_pair']=rows['original'][0]['g1_sha256']==rows['fixed_outer'][0]['g1_sha256']
    checks['same_inner_geometry_fixed_pair']=rows['fixed_inner'][0]['g1_sha256']==rows['fixed_both'][0]['g1_sha256']
    checks['no_history_or_forbidden_work']=all(not r['history_used'] and not r['history_cache_valid_after_update'] and not any(r[k] for k in ('fisher_estimation_passes','weight_merge_updates','projection_calls','hvp_calls','qr_calls','create_graph_calls','gradient_clipping_calls','final_norm_rescale_calls')) for rr in rows.values() for r in rr)
    checks['three_probes_only_g2_from_base']=all(r['ordinary_gradient_calls']==3 and not r['fallback_reason'] and r['relocation_origin']=='base' and r['U_sha256']==r['g2_sha256'] for rr in rows.values() for r in rr)
    checks['two_task_heads_20_then_40']=all([r['fc_shape'][0] for r in rr]==[20,20,40,40] for rr in rows.values())
    old=read_json(HERE/'gates/ORIGINAL_RESULT_REUSE.json');oldout=Path(old['output_dir'])
    oldrows=[]
    with (oldout/'step_trace.jsonl').open() as handle:
        for _,line in zip(range(2),handle):oldrows.append(json.loads(line))
    keys=('identity','base_parameter_sha256','future_parameter_sha256','relocation_parameter_sha256','g0_sha256','g1_sha256','g2_sha256','U_sha256','actual_step_displacement_sha256','nominal_lr','h','loss0','loss_future','loss_relocation','post_update_ce')
    checks['original_matches_frozen_formal_first_two_updates']=all(all(a[k]==b[k] for k in keys) for a,b in zip(ref[:2],oldrows))
    checks['original_matches_frozen_initial_state']=all(first[k]==read_json(oldout/'final_summary.json')['initial_identity'][k] for k in ('task_initial_parameter_sha256','frozen_origin_sha256','new_fc_rows_sha256'))
    report=dict(status='passed' if all(checks.values()) else 'failed',smoke_count=4,formal_count=0,checks=checks,evidence=evidence,limitation='Four updates each; smoke accuracy is not a formal result')
    atomic_json(HERE/'gates/smoke_pairing/radius_smoke_pairing.json',report)
    if not all(checks.values()):raise AssertionError('Radius smoke pairing failed: '+repr(checks))
    print(json.dumps(report))

if __name__=='__main__':main()
