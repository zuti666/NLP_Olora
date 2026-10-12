"""Compare smoke data identities and Task0 entry parameters, not evolving trajectories."""
import json
from pathlib import Path
from common import atomic_json,sha256
H=Path(__file__).resolve().parent

def main():
    s=json.loads((H/'suite.json').read_text());items=[]
    smoke=[j for j in s['jobs'] if j['kind']=='training' and j['smoke']]
    formal=[j for j in s['jobs'] if j['kind']=='training' and not j['smoke']]
    for j in smoke:
        out=Path(j['output_dir']);a=json.loads((out/'final_summary.json').read_text())
        rows=[json.loads(v) for v in (out/'step_trace.jsonl').read_text().splitlines()]
        ids=json.loads((out/'task_identities.json').read_text())
        items.append(dict(job=j['id'],summary_sha256=sha256(out/'final_summary.json'),r=a['r'],direction=a['update_direction'],
                          batch_identities=[r['identity'] for r in rows],
                          initial_parameter_hash=ids[0]['task_initial_parameter_sha256'],
                          task_class_slices=[r['class_slice'] for r in rows],steps=len(rows),
                          third_probe_count=a['counters']['training_ce_gradient_passes']))
    first=items[0]
    checks=dict(two_smokes=len(items)==2,two_requested_formal=len(formal)==2,
                initial_parameters_equal=all(r['initial_parameter_hash']==first['initial_parameter_hash'] for r in items),
                paired_realized_batches=all(r['batch_identities']==first['batch_identities'] for r in items),
                class_slices_equal=all(r['task_class_slices']==[[0,20],[20,40]] for r in items),
                exactly_four_probes=all(r['steps']==2 and r['third_probe_count']==8 for r in items),
                radius_direction_grid={(r['r'],r['direction']) for r in items}=={(.25,'g0_plus_g1_plus_g2'),(1.,'g0_plus_g1_plus_g2')})
    r=dict(status='passed' if all(checks.values()) else 'failed',smoke_count=len(smoke),formal_count=len(formal),checks=checks,smokes=items,
           scope='Inputs/labels/sampler and initial state pairing. Not a GPU bitwise-gradient or evolving-state identity claim.')
    atomic_json(H/'gates/smoke_pairing/smoke_pairing.json',r)
    if r['status']!='passed':raise AssertionError('Smoke data/state pairing failed')
    print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
