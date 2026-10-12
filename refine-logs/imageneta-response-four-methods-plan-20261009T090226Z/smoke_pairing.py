"""Same-layout realized inputs and first probes, without cross-layout comparisons."""
import json
from pathlib import Path
from common import HERE, read_json, atomic_json
from validate_job import validate_job

def main():
    suite=read_json(HERE/'suite.json');jobs=[j for j in suite['jobs'] if j['kind']=='training' and j['smoke']]
    if len(jobs)!=8:raise AssertionError('Expected four methods under two layouts')
    checks={};evidence=[]
    for layout in ('single200','cl10'):
        selected={j['method']:j for j in jobs if j['layout']==layout}
        if set(selected)!={'sgd','current_g2','history_fusion','history_g2'}:raise AssertionError('Incomplete method group')
        rows={};summaries={}
        for method,j in selected.items():
            evidence.append(dict(job=j['id'],**validate_job(j)))
            out=Path(j['output_dir']);summaries[method]=read_json(out/'final_summary.json')
            rows[method]=[json.loads(x) for x in (out/'step_trace.jsonl').read_text().splitlines()]
        first=summaries['sgd']['initial_identity'];ref=rows['sgd']
        checks[layout+'_initial_state']=all(all(s['initial_identity'][k]==first[k] for k in ('task_initial_parameter_sha256','frozen_origin_sha256','new_fc_rows_sha256')) for s in summaries.values())
        checks[layout+'_realized_inputs']=all([x['identity'] for x in r]==[x['identity'] for x in ref] for r in rows.values())
        checks[layout+'_learning_rates']=all([x['nominal_lr'] for x in r]==[x['nominal_lr'] for x in ref] for r in rows.values())
        checks[layout+'_first_base_gradient']=all(r[0]['g0_sha256']==ref[0]['g0_sha256'] for r in rows.values())
        checks[layout+'_first_future_gradient']=len({rows[m][0]['g1_sha256'] for m in ('current_g2','history_fusion','history_g2')})==1
        checks[layout+'_first_relocation_and_update']=all(rows['current_g2'][0][k]==rows['history_g2'][0][k] for k in ('g2_sha256','U_sha256','actual_step_displacement_sha256'))
        checks[layout+'_actual_history']=all(any(x['history_used'] for x in rows[m]) for m in ('history_fusion','history_g2'))
        checks[layout+'_no_history']=not any(x['history_used'] or x['history_cache_valid_after_update'] for m in ('sgd','current_g2') for x in rows[m])
        checks[layout+'_task_reset']=all(not x['history_used'] for r in rows.values() for x in r if x['task_step_index']==0)
    if not all(checks.values()):raise AssertionError('Pairing checks failed: '+repr(checks))
    report=dict(status='passed',smoke_count=8,formal_count=0,checks=checks,evidence=evidence,
                limitation='Bounded implementation validation, no complete 50-epoch result')
    atomic_json(HERE/'gates/smoke_pairing/smoke_pairing.json',report);print(json.dumps(report))

if __name__=='__main__':main()
