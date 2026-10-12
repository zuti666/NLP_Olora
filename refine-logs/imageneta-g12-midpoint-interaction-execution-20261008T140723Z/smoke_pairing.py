"""Real-model initial-state and realized-data pairing across four minimal-read updates."""
import json
from common import HERE,PROTOCOL,atomic_json,load_suite,read_json,sha256
from validate_job import validate_job,rows

def main():
    suite=load_suite();jobs=[j for j in suite['jobs'] if j['kind']=='training' and j['smoke']]
    items=[]
    for job in jobs:
        checked=validate_job(job);out=job['output_dir'];summary=read_json(out+'/final_summary.json');trace=rows(out+'/step_trace.jsonl');first=trace[0]
        items.append(dict(direction=job['direction'],initial=summary['initial_identity'],snapshot=checked['same_state_snapshot'],
                          batch_identities=[r['identity'] for r in trace],class_slices=[r['class_slice'] for r in trace],
                          points=first['point012_fingerprints'],grads=first['g012_fingerprints'],steps=len(trace),calls=summary['counters']['training_ce_gradient_passes']))
    first=items[0]
    checks=dict(all_smokes_accepted=len(items)==4,initial_parameters_equal=all(i['initial']['task_initial_parameter_sha256']==first['initial']['task_initial_parameter_sha256'] for i in items),
        initial_frozen_parameters_equal=all(i['initial']['frozen_origin_sha256']==first['initial']['frozen_origin_sha256'] for i in items),
        initial_head_equal=all(i['initial']['new_fc_rows_sha256']==first['initial']['new_fc_rows_sha256'] for i in items),
        paired_realized_batches=all(i['batch_identities']==first['batch_identities'] for i in items),
        class_slices_equal=all(i['class_slices']==first['class_slices'] for i in items),
        initial_rng_all_streams_equal=all(all(i['snapshot'][k]==first['snapshot'][k] for k in ['rng0_sha256','rng1_sha256']) for i in items),
        initial_parameter_layout_equal=all(i['snapshot']['parameter_layout_sha256']==first['snapshot']['parameter_layout_sha256'] for i in items),
        common_initial_g01=all(all(i['grads'][g]==first['grads'][g] for g in ['g0','g1']) for i in items),
        common_initial_points=all(all(i['points'][g]==first['points'][g] for g in ['phi0','phi1','phi2','phim']) for i in items),
        common_g2_where_read=all(i['grads']['g2']==next(x['grads']['g2'] for x in items if x['direction']=='sum12') for i in items if 'g2' in i['grads']),
        same_state_tensor_snapshots_verified=all(i['snapshot']['verified'] for i in items),
        explicit_training_ce_budget=all(i['steps']==4 and i['calls']==4*(4 if i['direction']=='sum13' else 3) for i in items))
    report=dict(status='passed' if all(checks.values()) else 'failed',protocol=PROTOCOL,smoke_count=4,formal_count=4,checks=checks,smokes=items,
                scope='Same initial phi/g01/outer/midpoint and realized data; subsequent trained parameters intentionally differ')
    atomic_json(HERE/'gates/smoke_pairing/smoke_pairing.json',report)
    if report['status']!='passed':raise AssertionError('Real-model pairing failed: '+repr(checks))
    print(json.dumps({'status':report['status'],'checks':checks}))
if __name__=='__main__':main()
