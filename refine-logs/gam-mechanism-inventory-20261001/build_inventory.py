"""Index existing GAM evidence; write only beside this script; launch no jobs."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs')
TASK0 = BASE / 's3-stage1-task0-run-20260930T213643Z'
TEN = BASE / 's3-g1g3-mechanism-run-20260930T210711Z'
FORMATION = BASE / 's3-formation-run-20261001T180712Z'
SOURCE = BASE / 's3-hparam-search-20260921/source'
FILES = {}


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    FILES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def hp(config):
    keys = ['dataset', 'class_shuffle', 'lora_rank', 'batch_size', 'init_epoch',
            'init_lr', 'lrate', 'init_momentum', 'init_weight_decay', 'lambda_flat',
            'gam_grad_beta_1', 'gam_grad_beta_2', 'gam_grad_beta_3', 'gam_grad_gamma',
            'gam_grad_rho', 'gam_grad_norm_rho', 'ewc_lambda', 'ewc_skip_task_fisher',
            'in_task_weight_merge', 'task_endpoint_weight_merge', 'stage1_direction']
    return {k: config[k] for k in keys if k in config}


records = []
suite = read(TASK0 / 'suite.json')
sam_suite = read(TASK0 / 'recovery_sam_v2_20261001/formal50_suite.json')
for job in suite['jobs']:
    if job['direction'] == 'true_sam':
        job = sam_suite['jobs'][0]
        entry = sam_suite['script']
        runtime = Path(sam_suite['runtime'])
    else:
        entry = suite['script']
        runtime = TASK0 / 'runtime/formal'
        if '_g_r' in job['id']:
            runtime = TASK0 / 'runtime/recovery_radii_20261001T132842Z'
    output = Path(job['output'])
    summary_path = output / 'final_summary.json'
    summary = read(summary_path)
    effective = output / 'effective_config.json'
    config = read(effective)
    receipt = runtime / 'jobs' / job['id'] / 'completed.json'
    read(receipt)
    final = summary['final']
    assert summary['status'] == 'completed' and summary['completed_epochs'] == 50
    assert summary['finite_check'] and final['eval_samples'] == 307
    records.append(dict(
        campaign='independent_task0_seed2', id=job['id'], seed=summary['seed'],
        phase='task0', arm=summary['direction'], status=summary['status'],
        completed_epochs=50, expected_epochs=50, config_path=job['config'],
        effective_config_path=str(effective), entrypoint=entry,
        optimizer_path=str(SOURCE / 'optimer_PerturabtionType/gam.py'),
        result_path=str(summary_path), completed_receipt=str(receipt),
        hyperparameters=hp(config), accuracy=final['eval_accuracy'],
        macro_accuracy=final['eval_macro_accuracy'], eval_ce=final['eval_ce'],
        train_ce=final['train_ce'], correct=sum(final['eval_per_class_correct']),
        eval_samples=final['eval_samples'], initial_identity=summary['initial_identity']))

ten_results = read(TASK0 / 'results_audit_20261001_final/ten_task/training_results.json')
for row in ten_results:
    config = read(row['config_path'])
    metrics = read(row['metrics_path'])
    receipt = read(row['completed_receipt'])
    matrix = metrics['cnn']['matrices']['final']
    t = len(matrix)
    aaa = sum(sum(matrix[i][:i+1])/(i+1) for i in range(t))/t
    faa = sum(matrix[-1])/t
    assert t == 10 and abs(aaa-row['aaa_task_macro']) < 1e-8
    assert abs(faa-row['faa_task_macro']) < 1e-8
    previous = row['origin'] != 'new_training'
    entry = (BASE / 's3-puregam-direction-run-20260930T180800Z/run_job.py'
             if previous else TEN / 'training/run_job.py')
    assert entry.is_file()
    records.append(dict(
        campaign='ten_task_seed2', id=row['id'], seed=row['seed'], phase='ten_task',
        arm=row['arm'], status=row['status'], origin=row['origin'],
        completed_epochs=500, expected_epochs=500, config_path=row['config_path'],
        entrypoint=str(entry), optimizer_path=str(SOURCE / 'optimer_PerturabtionType/gam.py'),
        result_path=row['metrics_path'], completed_receipt=row['completed_receipt'],
        hyperparameters=hp(config), aaa=aaa, faa=faa,
        acquisition=row['acquisition_task_macro'], bwt=row['bwt'],
        forgetting=row['forgetting'], tasks_completed=t,
        stratum='gamma0_controlled' if row['gamma'] == 0 else 'gamma0.1_reference'))

formation_suite = read(FORMATION / 'suite.json')
state = read(FORMATION / 'runtime/queue_state.json')
for job in formation_suite['jobs']:
    if job['smoke']:
        continue
    config = read(job['config'])
    live = state['jobs'][job['id']]
    output = Path(job['output_dir'])
    metrics_path = output / 'metrics.json'
    epochs = None
    if metrics_path.is_file():
        metrics = read(metrics_path)
        epochs = max((e['epoch'] for e in metrics['epochs']), default=0)
    summary_path = output / 'final_summary.json'
    row = dict(campaign='formation_seeds012', id=job['id'], seed=job['seed'],
               phase=job['phase'], arm=job['arm'], status=live['status'],
               completed_epochs=epochs, expected_epochs=job.get('expected_epochs'),
               expected_steps=job['expected_steps'], config_path=job['config'],
               entrypoint=job['command'][1], result_path=str(summary_path),
               output_dir=str(output), hyperparameters=hp(config),
               depends_on=job['depends_on'], formation_epoch=job.get('formation_epoch'),
               future_rule=job.get('future_rule'), queue_updated_at=state['updated_at'])
    if live['status'] == 'completed':
        receipt = FORMATION / 'runtime/jobs' / job['id'] / 'completed.json'
        read(receipt)
        row['completed_receipt'] = str(receipt)
    records.append(row)

ledger = {}
for campaign in sorted({r['campaign'] for r in records}):
    subset = [r for r in records if r['campaign'] == campaign]
    ledger[campaign] = dict(expected=len(subset), statuses=dict(Counter(r['status'] for r in subset)))
ledger['formation_by_phase'] = {
    phase: dict(Counter(r['status'] for r in records if r['campaign'] == 'formation_seeds012' and r['phase'] == phase))
    for phase in ['formation', 'future_s', 'switch']}

evidence = dict(
    snapshot_utc=datetime.now(timezone.utc).isoformat(), ledger=ledger,
    interpretation_limits=[
        'Previous task0/ten-task results are seed2 exploratory; batches and anchors are not seeds.',
        'Independent task0 results cannot be inserted into the ten-task trajectory.',
        'Ten-task AAA is recomputed as mean of stage task means; native final.AAA differs.',
        'Residual/interpolation scans are restored virtual one-step interventions, not training runs.',
        'Formation and future/switch scientific results require completed receipt, not a final_summary filename.',
        'Current formation rollout is 100 future steps, not full two-task or three-task retraining.'],
    diagnostic_ledger=dict(
        ten_task=dict(anchors=9, batches=33, all_candidates_matched=825, all_candidates_expected=825,
                      scanned_family_candidates=429),
        independent_task0=dict(anchors=5, batches=20, all_candidates_matched=500,
                               all_candidates_expected=500, scanned_family_candidates=260),
        coefficient_family_total=689,
        effective_step_target='same-state eta=.02 g1; concat(LoRA delta BA, FC delta W, FC delta b)',
        sam_probe_matching='A/B+FC parameter L2 distance OR all-seen output KL; not BA probe distance'),
    diagnostic_evidence=dict(
        protocol=str(TEN / 'EXPERIMENT_PLAN.md'),
        code=str(TEN / 'diagnostics/paired_gam_diagnostic.py'),
        ten_task_outputs=str(TEN / 'diagnostics/results'),
        ten_task_suite=str(TEN / 'diagnostics/suite.json'),
        ten_task_recovery_suite=str(TEN / 'diagnostics/suite_recovery_20260930T214003Z.json'),
        task0_suite=str(TASK0 / 'diagnostics/suite_completion_recovery_20261001.json'),
        task0_outputs=str(TASK0 / 'diagnostics'),
        recovered_mean_suite=str(TASK0 / 'diagnostics/recovery_mean_selfspec_v2_20261001T142100Z/recovery_suite.json'),
        recovered_mean_result=str(TASK0 / 'diagnostics/recovery_mean_selfspec_v2_20261001T142100Z/results/mean_epoch50_v2.json'),
        recovered_mean_code=str(TASK0 / 'diagnostics/recovery_mean_selfspec_v2_20261001T142100Z/paired_gam_diagnostic_v2.py'),
        residual_report=str(TASK0 / 'residual_direction_analysis_20261001/REPORT.md'),
        residual_coefficients=str(TASK0 / 'residual_direction_analysis_20261001/coefficient_curves.csv'),
        pair_report=str(TASK0 / 'g1_vs_g3_analysis_20261001/ANALYSIS.md'),
        future_plan=str(FORMATION / 'PLAN.md'),
        future_gates=str(FORMATION / 'FROZEN_GATES.json'),
        earlier_component_report=str(BASE / 's3-mechanisms-20260920/MECHANISM_RESULTS.md'),
        earlier_component_results=str(BASE / 's3-mechanisms-20260920/results.json'),
        earlier_component_configs=str(BASE / 's3-mechanisms-20260920/configs'),
        earlier_component_suite=str(BASE / 's3-mechanisms-20260920/suite.json')),
    records=records, input_sha256=FILES)
for path in evidence['diagnostic_evidence'].values():
    assert Path(path).exists(), path
(OUT / 'evidence_index.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
fields = ['campaign', 'id', 'seed', 'phase', 'arm', 'status', 'stratum', 'origin',
          'completed_epochs', 'expected_epochs', 'expected_steps', 'accuracy', 'macro_accuracy',
          'eval_ce', 'train_ce', 'correct', 'eval_samples', 'aaa', 'faa', 'acquisition', 'bwt',
          'forgetting', 'gam_grad_beta_1', 'gam_grad_beta_2', 'gam_grad_beta_3', 'gam_grad_gamma',
          'gam_grad_rho', 'gam_grad_norm_rho', 'init_lr', 'config_path', 'effective_config_path',
          'entrypoint', 'optimizer_path', 'result_path', 'completed_receipt', 'queue_updated_at']
with (OUT / 'experiments.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in records:
        writer.writerow({**row, **row['hyperparameters']})
print(json.dumps(dict(records=len(records), ledger=ledger, queue_updated_at=state['updated_at']), ensure_ascii=False))
