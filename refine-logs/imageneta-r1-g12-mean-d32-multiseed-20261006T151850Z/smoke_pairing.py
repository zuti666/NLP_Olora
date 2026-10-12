"""Pair realized inputs and initial state within each seed, across all three directions."""
import json
from pathlib import Path
from common import atomic_json, sha256

H = Path(__file__).resolve().parent
DIRECTIONS = {'g1_plus_g2', 'half_g1_plus_g2', 'g1_plus_d32'}


def main():
    suite = json.loads((H / 'suite.json').read_text())
    smokes = [j for j in suite['jobs'] if j['kind'] == 'training' and j['smoke']]
    formal = [j for j in suite['jobs'] if j['kind'] == 'training' and not j['smoke']]
    items = []
    for j in smokes:
        out = Path(j['output_dir'])
        a = json.loads((out / 'final_summary.json').read_text())
        rows = [json.loads(v) for v in (out / 'step_trace.jsonl').read_text().splitlines()]
        identities = json.loads((out / 'task_identities.json').read_text())
        items.append(dict(job=j['id'], seed=j['seed'], direction=j['direction'],
                          summary_sha256=sha256(out / 'final_summary.json'),
                          identities=[r['identity'] for r in rows],
                          initial_parameters=identities[0]['task_initial_parameter_sha256'],
                          class_slices=[r['class_slice'] for r in rows],
                          steps=len(rows), gradient_calls=a['counters']['training_ce_gradient_passes']))
    checks = dict(six_smokes=len(items) == 6, six_formal=len(formal) == 6,
                  complete_grid={(i['seed'], i['direction']) for i in items} ==
                  {(s, d) for s in (0, 1) for d in DIRECTIONS},
                  budget=all(i['steps'] == 2 and i['gradient_calls'] == 8 for i in items),
                  class_slices=all(i['class_slices'] == [[0, 20], [20, 40]] for i in items))
    for seed in (0, 1):
        group = [i for i in items if i['seed'] == seed]
        checks[f'seed{seed}_same_initial_model'] = len(group) == 3 and all(
            i['initial_parameters'] == group[0]['initial_parameters'] for i in group)
        checks[f'seed{seed}_same_realized_inputs'] = len(group) == 3 and all(
            i['identities'] == group[0]['identities'] for i in group)
    result = dict(status='passed' if all(checks.values()) else 'failed',
                  smoke_count=len(smokes), formal_count=len(formal), checks=checks, smokes=items,
                  scope='Within-seed realized data and Task0 initialization pairing; evolving parameters differ.')
    atomic_json(H / 'gates/smoke_pairing/smoke_pairing.json', result)
    if result['status'] != 'passed':
        raise AssertionError('Within-seed pairing failed')
    print(json.dumps(dict(status=result['status'], checks=checks)))


if __name__ == '__main__':
    main()
