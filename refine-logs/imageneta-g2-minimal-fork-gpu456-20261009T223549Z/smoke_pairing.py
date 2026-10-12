"""Verify one actual SGD parent and six paired native Task1 GPU branches."""
import json
from pathlib import Path
from common import HERE as H,read_json,atomic_json,sha256
from validate_job import validate_job

def main():
    suite=read_json(H/'suite.json');jobs=[j for j in suite['jobs'] if j['kind']=='training' and j['smoke']]
    if len(jobs)!=7:raise AssertionError('Expected 1 parent+6 native fork smokes')
    rows={};summaries={};evidence=[]
    for j in jobs:
        checked=validate_job(j);receipt=read_json(H/'runtime/jobs'/j['id']/'completed.json')
        if receipt['summary_sha256']!=checked['summary_sha256'] or receipt['exit']['returncode']!=0:raise AssertionError('Unaccepted smoke')
        evidence.append(dict(job=j['id'],**checked));out=Path(j['output_dir']);key=j['study_condition_id']
        summaries[key]=read_json(out/'final_summary.json');rows[key]=[json.loads(x) for x in (out/'step_trace.jsonl').read_text().splitlines()]
    branches={k:v for k,v in summaries.items() if k!='parent_sgd'};ref=rows['B0'];first=summaries['B0']['initial_identity']
    checks={}
    checks['shared_accepted_SGD_parent']=len({s['parent_provenance']['shared_parent_sha256'] for s in branches.values()})==1 and len({s['parent_provenance']['parent_completion_receipt_sha256'] for s in branches.values()})==1
    checks['same_expanded_complete_initial_model']=all(all(s['initial_identity'][k]==first[k] for k in ('task_initial_parameter_sha256','frozen_origin_sha256','new_fc_rows_sha256')) for s in branches.values())
    checks['same_augmentation_data_order_and_LR']=all([r['identity'] for r in rows[k]]==[r['identity'] for r in ref] and [r['nominal_lr'] for r in rows[k]]==[r['nominal_lr'] for r in ref] for k in branches)
    checks['same_first_base_gradient']=len({rows[k][0]['g0_sha256'] for k in branches})==1
    checks['fixed_inner_pair_g1_identity']=len({rows[k][0]['g1_sha256'] for k in ('R25','F_S','R100')})==1
    checks['natural_inner_pair_g1_identity']=rows['S_F'][0]['g1_sha256']==rows['S_S'][0]['g1_sha256']
    checks['SGD_one_gradient_no_extra_probe']=all(r['ordinary_gradient_calls']==1 and r['U_sha256']==r['g0_sha256'] for k in ('parent_sgd','B0') for r in rows[k])
    checks['G2_three_gradients_unscaled_from_base']=all(r['ordinary_gradient_calls']==3 and not r['fallback_reason'] and r['relocation_origin']=='base' and r['U_sha256']==r['g2_sha256'] for k in branches if k!='B0' for r in rows[k])
    checks['no_task0_retraining_in_branches']=all(s['trained_task_ids']==[1] and s['completed_steps']==2 and len(s['checkpoint_dirs'])==1 for s in branches.values())
    checks['strict_zero_history_Fisher_fusion_projection']=all(not r['history_used'] and not r['history_cache_valid_after_update'] and not any(r[k] for k in ('fisher_estimation_passes','weight_merge_updates','projection_calls','hvp_calls','qr_calls','create_graph_calls','gradient_clipping_calls','final_norm_rescale_calls')) for rr in rows.values() for r in rr)
    audit=next(j for j in suite['jobs'] if j['id']=='smoke_local_probe_audit_s2');accepted=validate_job(audit);receipt=read_json(H/'runtime/jobs'/audit['id']/'completed.json')
    checks['real_native_offline_zero_update_smoke']=receipt['summary_sha256']==accepted['summary_sha256'] and receipt['exit']['returncode']==0
    report=dict(status='passed' if all(checks.values()) else 'failed',smoke_count=7,formal_count=0,checks=checks,evidence=evidence,offline_evidence=accepted,limitation='One epoch and two minibatches each; no accuracy ranking')
    atomic_json(H/'gates/smoke_pairing/fork_smoke_pairing.json',report)
    if not all(checks.values()):raise AssertionError('Native fork pairing failed: '+repr(checks))
    print(json.dumps(report))
if __name__=='__main__':main()
