"""Read existing endpoint receipts only; no model loading, GPU use or experiment changes."""
import csv
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
EX = OUT.parent


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


rows, extra, historical, sources = [], [], [], []
hist = read(EX / 'analysis_setting_20260920/HISTORICAL_COMPARISON.json')['method_comparison']
aud = read(EX / 'analysis_setting_20260920/AUDIT_EXECUTION.json')['runs']
for method in ['sgd', 'gam']:
    path = EX / f'results_seed0_recovery1/{method}/table_a/summary.json'
    a = read(path)
    assert a['status'] == 'complete'
    oldpath = Path(a['current_anchor_training_metrics']['path'])
    old = {r['task']: r for r in map(json.loads, oldpath.read_text().splitlines())}
    metrics = read(a['metrics_json'])
    mech = metrics['mechanism']['steps']['1']
    row = dict(dataset='ImageNet-A', seed=0, task1_update=method.upper(),
               task2_update='SGD+EWC', a11=old[0]['test']['seen_accuracy'],
               a21=old[1]['test']['seen_accuracy'],
               a22_recorded_rounded=a['accuracy_CIL']['a22'],
               BWT_pp=old[1]['test']['seen_accuracy']-old[0]['test']['seen_accuracy'],
               average_from_recorded_accuracies=(a['accuracy_CIL']['a21']+a['accuracy_CIL']['a22'])/2,
               F_old_raw_at_task1=a['anchor']['fisher']['raw_trace'],
               F_old_raw_at_task2=a['endpoint']['fisher']['raw_trace'],
               effective_QV_displacement=a['effective_weight_drift'],
               old_test_local_CE_increment=a['D0']['test'])
    assert all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
    rows.append(row)
    extra.append(dict(task1_update=method.upper(), task2_update='SGD+EWC',
                      task2_batch_proxy_raw_trace=mech['in_task_fisher_final_raw_trace'],
                      inherited_normalized_protection_trace=aud[method]['fisher_trace_float64'],
                      delta_norm_squared=mech['delta_norm'],
                      sqrt_logged_delta_norm=math.sqrt(mech['delta_norm']),
                      fisher_weighted_drift=mech['fisher_weighted_drift'],
                      normalized_fisher_drift=mech['normalized_fisher_drift'],
                      endpoint_EWC_penalty=mech['ewc_penalty_value']))
    tr_path = EX / f'results_seed0_recovery1/{method}/trajectory/summary.json'
    tr = read(tr_path)['tasks'][1]
    h = hist[method]['historical_task1']
    hm = read(hist[method]['historical_source'])['mechanism']['steps']['1']
    historical.append(dict(dataset='ImageNet-A', seed=0, task1_update=method.upper(),
                           task2_update=method.upper()+'+EWC',
                           a11_historical=h['a11'], a21_historical=h['a21'],
                           a22_historical=h['a22'], BWT_historical_pp=h['a21']-h['a11'],
                           effective_QV_displacement=tr['effective_weight_drift'],
                           F_old_raw_at_task2=None,
                           F_old_raw_at_task2_status='not measured for this historical task1 endpoint',
                           task2_batch_proxy_raw_trace=hm.get('in_task_fisher_final_raw_trace')))
    for q in [path, oldpath, Path(a['metrics_json']), tr_path, Path(hist[method]['historical_source'])]:
        sources.append({'path':str(q.resolve()), 'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})

write_csv('current_imageneta_fisher_drift.csv', rows)
write_csv('current_imageneta_training_diagnostics.csv', extra)
write_csv('historical_imageneta_reference.csv', historical)
result = dict(current=rows, training_diagnostics=extra, historical_reference=historical,
              definitions={
                  'task_numbering':'paper Task 1/2 corresponds to code Task0/Task1',
                  'current_raw_trace':'mean_i ||grad_W CE_i||^2 on the same stratified 256 old-task train images with fixed old 20-class head, full FP32, 24 effective Q/V BA matrices; no normalization, floor, regularizer',
                  'current_displacement':'sqrt(sum_lc ||B2A2-B1A1||_F^2), 24 Q/V blocks; excludes classifier and frozen base',
                  'current_batch_proxy':'new-task training batch-mean CE gradient squared, before trace normalization, 997 samples; not old-task per-example Fisher',
                  'legacy_table_trace':'endpoint_fisher_total_mass; current-task batch-gradient proxy trace, effective K/V; Task2 value uses new-task data/head, not old-task sensitivity',
                  'legacy_table_displacement':'mean_seed sqrt(delta_norm), delta_norm=sum_lc ||B_new A_new||_F^2; fresh K/V adapter, pre-merge endpoint; excludes classifier',
                  'legacy_scope':'ImageNet-R, seeds0/1/2, low-rank-cl-new, Adam-based update, lambda1e7; no direct pooling with current ImageNet-A SGD lambda2000 seed0'},
              validated_current_rows=2, expected_current_rows=2,
              no_gpu_used=True, source_receipts=sources)
(OUT / 'FISHER_DRIFT_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
