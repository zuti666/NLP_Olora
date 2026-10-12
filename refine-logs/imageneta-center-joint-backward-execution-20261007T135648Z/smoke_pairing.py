"""Strict family-aware comparison of accepted real-GPU smoke common snapshots."""
import json
from collections import defaultdict
from pathlib import Path
from common import HERE,PROTOCOL,VARIANTS,atomic_json,expected_budget,read_json,sha256
from validate_job import validate_job

def main():
    suite=read_json(HERE/'suite.json');state=read_json(HERE/'runtime/queue_state.json')
    smokes=[j for j in suite['jobs'] if j['kind']=='training' and j['smoke']]
    formals=[j for j in suite['jobs'] if j['kind']=='training' and not j['smoke']]
    items=[];accepted=True
    for job in smokes:
        checked=validate_job(job)
        receipt=read_json(HERE/'runtime/jobs'/job['id']/'completed.json')
        accepted &= state['jobs'][job['id']]['status']=='completed'
        accepted &= all(receipt[k]==checked[k] for k in ('summary_sha256','trace_sha256','matrix_sha256','diagnostics_sha256','updates','tasks'))
        out=Path(job['output_dir']);trace=[json.loads(l) for l in (out/'step_trace.jsonl').read_text().splitlines()]
        identities=read_json(out/'task_identities.json');summary=read_json(out/'final_summary.json')
        counters=summary['counters']
        items.append(dict(job=job['id'],variant=job['variant'],family=job['first_location_estimator'],
                          summary_sha256=checked['summary_sha256'],trace_sha256=checked['trace_sha256'],
                          initial_parameter_hash=identities[0]['task_initial_parameter_sha256'],
                          initial_head_hash=identities[0]['new_fc_rows_sha256'],
                          frozen_parameter_hash=identities[0]['frozen_origin_sha256'],
                          batch_identities=[r['identity'] for r in trace],
                          task_class_slices=[r['class_slice'] for r in trace],
                          g012_fingerprints=trace[0]['g012_fingerprints'],
                          point012_fingerprints=trace[0]['point012_fingerprints'],
                          same_state_snapshot=checked['same_state_snapshot'],
                          diagnostic_gradient_calls=counters['diagnostic_gradient_calls'],
                          steps=len(trace),training_ce_passes=counters['training_ce_gradient_passes'],
                          expected_ce=job['expected_training_ce'],
                          first_plus_count=counters['training_first_positive_gradient_passes'],
                          third_plus_count=counters['training_g3_positive_gradient_passes']))
    first=items[0];groups=defaultdict(list)
    for item in items:groups[item['family']].append(item)
    checks=dict(all_smokes_accepted=bool(accepted),four_smokes=len(items)==4,four_formals=len(formals)==4,
                initial_parameters_equal=all(i['initial_parameter_hash']==first['initial_parameter_hash'] for i in items),
                initial_head_equal=all(i['initial_head_hash']==first['initial_head_hash'] for i in items),
                initial_frozen_parameters_equal=all(i['frozen_parameter_hash']==first['frozen_parameter_hash'] for i in items),
                initial_rng_all_streams_equal=all(all(i['same_state_snapshot'][k]==first['same_state_snapshot'][k] for k in ('rng0_sha256','rng1_sha256')) for i in items),
                initial_parameter_layout_equal=all(i['same_state_snapshot']['parameter_layout_sha256']==first['same_state_snapshot']['parameter_layout_sha256'] for i in items),
                paired_realized_batches=all(i['batch_identities']==first['batch_identities'] for i in items),
                class_slices_equal=all(i['task_class_slices']==[[0,20],[20,40]] for i in items),
                explicit_training_ce_budget=all(i['steps']==2 and i['training_ce_passes']==i['expected_ce'] for i in items),
                same_state_tensor_snapshots_verified=all(i['same_state_snapshot']['verified'] is True for i in items),
                no_extra_diagnostic_gradients=all(i['diagnostic_gradient_calls']==0 for i in items),
                family_initial_g012=all(all(i['g012_fingerprints']==g[0]['g012_fingerprints'] for i in g) for g in groups.values()),
                family_initial_points=all(all(i['point012_fingerprints']==g[0]['point012_fingerprints'] for i in g) for g in groups.values()),
                cross_family_initial_g0=all(i['g012_fingerprints']['g0']==first['g012_fingerprints']['g0'] for i in items),
                no_positive_BB=all(i['first_plus_count']==0 and i['third_plus_count']==0 for i in items if i['variant']=='BB'),
                family_grid=set(groups)=={'forward','backward_corrected'} and {i['variant'] for i in items}==VARIANTS)
    # FF/FB initial shared geometry is unchanged. Compare actual first-step
    # fingerprints to both original strict baseline traces, never old final scores.
    refs=[]
    for reference in suite['reused_results']:
        tracefile=reference['evidence_paths']['trace']
        with open(tracefile) as f:oldrow=json.loads(next(f))
        refs.append(dict(variant=reference['variant'],g012=oldrow['g012_fingerprints'],points=oldrow['point012_fingerprints']))
    fgroup=groups['forward']
    checks['old_F_reference_initial_g012']=all(i['g012_fingerprints']==r['g012'] for i in fgroup for r in refs)
    checks['old_F_reference_initial_points']=all(i['point012_fingerprints']==r['points'] for i in fgroup for r in refs)
    report=dict(status='passed' if all(checks.values()) else 'failed',protocol=PROTOCOL,
                checks=checks,smoke_count=4,formal_count=4,smokes=items,reference_initial_snapshots=refs,
                scope='Compare realized data/initial base globally; g012/points only within first-location family. No evolving trajectory equality.')
    atomic_json(HERE/'gates/smoke_pairing/smoke_pairing.json',report)
    if report['status']!='passed':raise AssertionError('Real-model family/data/reference pairing failed')
    print(json.dumps(dict(status=report['status'],checks=checks)))

if __name__=='__main__':main()
