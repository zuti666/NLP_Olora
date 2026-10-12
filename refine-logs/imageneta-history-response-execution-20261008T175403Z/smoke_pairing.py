"""Check paired short runs and exact equality before any history exists."""
import json
from pathlib import Path
from common import HERE,atomic_json,read_json,sha256
from validate_job import validate_job

def main():
    suite=read_json(HERE/'suite.json')
    jobs=[j for j in suite['jobs'] if j['kind']=='training' and j['smoke']]
    if len(jobs)!=2:raise AssertionError('Expected two smoke variants')
    by_direction={j['direction']:j for j in jobs}
    if set(by_direction)!={'history_fused','future_only'}:raise AssertionError('Expected the two distinct algorithm arms')
    jobs=[by_direction['history_fused'],by_direction['future_only']]
    evidence=[];rows=[];summaries=[]
    for j in jobs:
        checked=validate_job(j);p=Path(j['output_dir'])
        summaries.append(read_json(p/'final_summary.json'))
        rows.append([json.loads(line) for line in (p/'step_trace.jsonl').read_text().splitlines()])
        evidence.append(dict(job=j['id'],output=str(p),**checked))
    initial=[s['initial_identity'] for s in summaries]
    hashes=('task_initial_parameter_sha256','frozen_origin_sha256','new_fc_rows_sha256')
    checks=dict(initial_parameters_identical=all(initial[0][k]==initial[1][k] for k in hashes),
                realized_inputs_identical=[r['identity'] for r in rows[0]]==[r['identity'] for r in rows[1]],
                learning_rates_identical=[r['nominal_lr'] for r in rows[0]]==[r['nominal_lr'] for r in rows[1]],
                first_base_gradient_identical=rows[0][0]['raw_gradient_sha256']==rows[1][0]['raw_gradient_sha256'],
                first_future_gradient_identical=rows[0][0]['future_gradient_sha256']==rows[1][0]['future_gradient_sha256'],
                first_final_direction_identical=rows[0][0]['U_sha256']==rows[1][0]['U_sha256'],
                first_actual_step_identical=rows[0][0]['actual_step_displacement_sha256']==rows[1][0]['actual_step_displacement_sha256'],
                history_exercised=any(r['history_used'] for r in rows[0]),
                no_history_never_used=not any(r['history_used'] for r in rows[1]),
                task_reset_passed=all(not r['history_used'] for rows_ in rows for r in rows_ if r['task_step_index']==0))
    if not all(checks.values()):raise AssertionError('Pairing failed: '+repr(checks))
    out=HERE/'gates/smoke_pairing';out.mkdir(exist_ok=True)
    report=dict(status='passed',smoke_count=2,formal_count=2,checks=checks,evidence=evidence)
    atomic_json(out/'smoke_pairing.json',report);print(json.dumps(report))

if __name__=='__main__':main()
