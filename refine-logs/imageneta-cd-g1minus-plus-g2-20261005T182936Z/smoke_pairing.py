"""Validate equal input/initial state pairing without requiring equal trained trajectories."""
import json
from pathlib import Path
from common import atomic_json,sha256
R=Path(__file__).resolve().parent

def main():
    suite=json.loads((R/'suite.json').read_text())
    smokes=[j for j in suite['jobs'] if j['kind']=='training' and j['smoke']]
    formals=[j for j in suite['jobs'] if j['kind']=='training' and not j['smoke']]
    items=[]
    for job in smokes:
        out=Path(job['output_dir']);a=json.loads((out/'final_summary.json').read_text())
        trace=[json.loads(x) for x in (out/'step_trace.jsonl').read_text().splitlines()]
        identities=json.loads((out/'task_identities.json').read_text())
        items.append(dict(job=job['id'],branch=a['probe_branch'],r=a['r'],
            direction=a['update_direction'],summary_sha256=sha256(out/'final_summary.json'),
            initial_parameter_hash=identities[0]['task_initial_parameter_sha256'],
            batch_identities=[x['identity'] for x in trace],class_slices=[x['class_slice'] for x in trace],
            steps=len(trace),ce_gradients=a['counters']['training_ce_gradient_passes'],
            first_plus=a['counters']['training_first_plus_gradient_passes'],
            first_minus=a['counters']['training_first_minus_gradient_passes'],
            g3=a['counters']['training_g3_gradient_passes']))
    first=items[0]
    checks=dict(four_smokes=len(items)==4,four_formal=len(formals)==4,
        initial_parameters_equal=all(x['initial_parameter_hash']==first['initial_parameter_hash'] for x in items),
        realized_batches_equal=all(x['batch_identities']==first['batch_identities'] for x in items),
        class_slices_equal=all(x['class_slices']==[[0,20],[20,40]] for x in items),
        actual_probe_budgets=all(x['steps']==2 and x['ce_gradients']==8 and x['first_plus']==2 and x['first_minus']==2 and x['g3']==0 for x in items),
        radius_branch_grid={(x['r'],x['branch'],x['direction']) for x in items}==
            {(r,b,'g1_minus_plus_g2') for r in [.25,1.] for b in ['C','D']})
    report=dict(status='passed' if all(checks.values()) else 'failed',smoke_count=4,formal_count=4,
        checks=checks,smokes=items,
        scope='Realized inputs and Task0 entry parameters only; evolving trajectories and cross-GPU bitwise gradients need not be equal.')
    atomic_json(R/'gates/smoke_pairing/smoke_pairing.json',report)
    if report['status']!='passed':raise AssertionError('Smoke input/initial-state pairing failed')
    print(json.dumps(report))

if __name__=='__main__':main()
