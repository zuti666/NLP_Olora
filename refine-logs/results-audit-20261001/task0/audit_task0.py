"""Read-only audit of source artifacts; writes only this audit directory. CPU only."""
import csv
import hashlib
import itertools
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-stage1-task0-run-20260930T213643Z')
OUT = Path(__file__).resolve().parent
DIRECTIONS = ['s', 'g', 'mean', 'sum', 'ce']

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def nonfinite(value, path=''):
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in nonfinite(v, f'{path}.{k}')]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in nonfinite(v, f'{path}[{i}]')]
    return []

def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, torch.nn.Module):
        yield from value.state_dict().values()
    elif isinstance(value, dict):
        for v in value.values():
            yield from tensors(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from tensors(v)

def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

torch.serialization.add_safe_globals([torch.nn.Linear, set])
suite = read(ROOT / 'suite.json')
state = read(ROOT / 'runtime/formal/state.json')
source = Path(suite['source_root'])
manifest = read(ROOT / 'source_manifest.json')
source_mismatches = [k for k, h in manifest['files'].items() if sha(source / k) != h]
pin_mismatches = [p for p, h in suite['file_pins'].items() if sha(p) != h]
jobs, curves, checkpoint_rows = [], [], []
all_epochs, identities, epoch0s = {}, {}, {}

for direction in DIRECTIONS:
    jid = f'task0_{direction}_k10_s2'
    output = ROOT / 'outputs' / jid
    runtime = ROOT / 'runtime/formal/jobs' / jid
    cfg_path = ROOT / 'configs' / f'{jid}.json'
    config = read(cfg_path)
    effective = read(output / 'effective_config.json')
    started = read(runtime / 'started.json')
    receipt = read(runtime / 'completed.json')
    summary = read(output / 'final_summary.json')
    metric_doc = read(output / 'metrics.json')
    epochs = metric_doc['epochs']
    geometry = read(output / 'geometry.json')['records']
    identity = read(output / 'identity.json')
    epoch0 = read(output / 'epoch0_eval.json')
    assert receipt['returncode'] == 0 and not receipt['errors']
    assert receipt['summary'] == summary
    assert summary['status'] == 'completed' and summary['completed_epochs'] == 50
    assert summary['trained_tasks'] == [0] and not summary['smoke']
    assert summary['final'] == epochs[-1]
    assert [x['epoch'] for x in epochs] == list(range(1, 51))
    assert len(geometry) == 50 and [x['epoch'] for x in geometry] == list(range(1, 51))
    assert not nonfinite(epochs + geometry)
    assert started['config_sha256'] == sha(cfg_path)
    assert started['script_sha256'] == summary['trainer_sha256'] == sha(suite['script'])
    assert summary['source_gam_sha256'] == sha(source / 'optimer_PerturabtionType/gam.py')
    fixed = {'seed': 2, 'lora_rank': 10, 'init_epoch': 50, 'epochs': 50,
             'max_train_tasks': 1, 'gam_grad_rho': .05, 'gam_grad_norm_rho': 1.,
             'gam_grad_gamma': 0., 'init_lr': .04, 'batch_size': 128,
             'ewc_lambda': 0., 'ewc_skip_task_fisher': True,
             'in_task_weight_merge': False, 'task_endpoint_weight_merge': False,
             'approximate_fisher_in_task': False, 'in_task_fisher_omega': 0.,
             'in_task_weight_merge_omega': 0., 'ewc_fisher_damping_alpha': 0.,
             'ewc_normalize_fisher': False, 'ewc_normalize_accumulated_fisher': False,
             'compute_full_fisher': False, 'posterior_state_save': False}
    assert all(effective.get(k) == v for k, v in fixed.items())
    for row in epochs:
        counts, correct = row['eval_per_class_counts'], row['eval_per_class_correct']
        assert row['task'] == 0 and row['train_samples'] == 1259 and row['eval_samples'] == 307
        assert row['batches'] == 10 and sum(counts) == 307 and len(counts) == 20
        assert all(0 <= a <= b for a, b in zip(correct, counts))
        assert abs(sum(correct) / 307 * 100 - row['eval_accuracy']) < 1e-10
        assert abs(sum(a / b * 100 for a, b in zip(correct, counts)) / 20 - row['eval_macro_accuracy']) < 1e-10
        curves.append({'job': jid, 'direction': direction, **{k: row[k] for k in
            ['epoch', 'train_ce', 'train_accuracy', 'eval_ce', 'eval_accuracy', 'eval_macro_accuracy']},
            'eval_correct': sum(correct), 'eval_samples': 307,
            'lr_used': row['lr_used'][0], 'sampler_order_sha256': row['sampler_order_sha256'],
            'first_input_sha256': row['first_batch']['input_sha256'],
            'first_target_sha256': row['first_batch']['target_sha256']})
    assert metric_doc['counters']['training_batches'] == 500
    assert metric_doc['counters']['training_samples'] == 62950
    assert metric_doc['counters']['training_ce_gradient_passes'] == 2500
    assert metric_doc['counters']['isolated_diagnostic_ce_gradient_passes'] == 0
    log_lines = (runtime / 'console.log').read_bytes().decode(errors='replace').split('\n')
    marker_patterns = {'fisher_update': r'Updated past Fisher for',
        'fisher_snapshot': r'Computing raw Fisher snapshot|Saved raw Fisher snapshot',
        'weight_capture': r'\[AS1-NormFisher-Damp-WMergeThenFisher\]\[Checkpoint\]',
        'weight_merge_applied': r'\[WeightMerge\].*applied=True',
        'has_fisher_true': r'\[TaskStart\].*has_fisher=True',
        'traceback': r'Traceback \(most recent call last\)'}
    markers = {k: [i for i, line in enumerate(log_lines, 1) if re.search(p, line)]
               for k, p in marker_patterns.items()}
    assert all(not hits for hits in markers.values())
    ewc = [float(m.group(1)) for line in log_lines for m in re.finditer(r'EWC\(λ·F·Δθ²\)\s+([+\-0-9.eE]+)', line)]
    assert ewc and all(x == 0 for x in ewc)
    for ep in [0, 10, 20, 30, 40, 50]:
        checkpoint = output / 'checkpoints' / f'epoch_{ep:03d}'
        for name in ['lora_w_a_0.pt', 'lora_w_b_0.pt', 'fc_state_0.pt']:
            path = checkpoint / name
            values = list(tensors(torch.load(path, map_location='cpu', weights_only=True)))
            assert values and all(bool(torch.isfinite(t).all()) for t in values)
            checkpoint_rows.append({'job': jid, 'epoch': ep, 'file': name,
                'sha256': sha(path), 'tensor_count': len(values), 'finite': True})
            if ep == 50:
                final_values = list(tensors(torch.load(output / 'final_native_checkpoint' / name,
                                                      map_location='cpu', weights_only=True)))
                assert len(values) == len(final_values) and all(torch.equal(a, b) for a, b in zip(values, final_values))
    final = epochs[-1]
    best_value = max(x['eval_accuracy'] for x in epochs)
    best_epochs = [x['epoch'] for x in epochs if x['eval_accuracy'] == best_value]
    best_ce = min(epochs, key=lambda x: x['eval_ce'])
    jobs.append({'job': jid, 'direction': direction, 'completed_at_utc': receipt['time'],
        'epochs': 50, 'eval_samples': 307, 'correct': sum(final['eval_per_class_correct']),
        'a11_sample': final['eval_accuracy'], 'a11_class_macro': final['eval_macro_accuracy'],
        'eval_ce': final['eval_ce'], 'train_ce': final['train_ce'], 'train_accuracy': final['train_accuracy'],
        'best_epoch_accuracy_exploratory': best_value, 'best_epochs_exploratory': best_epochs,
        'best_minus_final_correct_exploratory': round((best_value-final['eval_accuracy']) * 307 / 100),
        'lowest_eval_ce_epoch_exploratory': best_ce['epoch'], 'lowest_eval_ce_exploratory': best_ce['eval_ce'],
        'geometry_records': len(geometry), 'geometry_max_selected_gradient_relative_error': max(x['selected_gradient_relative_error'] for x in geometry),
        'geometry_max_actual_step_relative_error': max(x['actual_step_vs_selected_relative_error'] for x in geometry),
        'zero_ewc_log_occurrences': len(ewc), 'execution_marker_lines': markers,
        'initial_model_sha256': identity['initial_model_sha256'],
        'frozen_origin_sha256': identity['frozen_origin_sha256'],
        'source_paths': {'output': str(output), 'completed_receipt': str(runtime/'completed.json')},
        'artifact_sha256': {n: sha(output/n) for n in ['metrics.json', 'geometry.json', 'final_summary.json', 'identity.json', 'epoch0_eval.json', 'effective_config.json']},
        'config_sha256_verified': sha(cfg_path), 'receipt_sha256': sha(runtime/'completed.json')})
    all_epochs[direction], identities[direction], epoch0s[direction] = epochs, identity, epoch0

assert all(x == identities['s'] for x in identities.values())
assert all(x == epoch0s['s'] for x in epoch0s.values())
pairing = []
for ep in range(50):
    ref = all_epochs['s'][ep]
    for direction in DIRECTIONS[1:]:
        other = all_epochs[direction][ep]
        assert ref['sampler_order_sha256'] == other['sampler_order_sha256']
        assert ref['first_batch'] == other['first_batch']
        assert ref['eval_per_class_counts'] == other['eval_per_class_counts']
        assert ref['lr_used'] == other['lr_used'] and ref['lr_after_scheduler'] == other['lr_after_scheduler']
    pairing.append({'epoch': ep+1, 'all_five_same_order': True, 'all_five_same_first_augmented_batch': True,
                    'sampler_order_sha256': ref['sampler_order_sha256'],
                    'first_input_sha256': ref['first_batch']['input_sha256']})
initial_checkpoint_hashes = {}
for row in checkpoint_rows:
    if row['epoch'] == 0:
        initial_checkpoint_hashes.setdefault(row['file'], set()).add(row['sha256'])
assert all(len(v) == 1 for v in initial_checkpoint_hashes.values())
pairs = [{'left': a['direction'], 'right': b['direction'],
          'delta_sample_pp_left_minus_right': a['a11_sample']-b['a11_sample'],
          'delta_correct_left_minus_right': a['correct']-b['correct'],
          'delta_class_macro_pp': a['a11_class_macro']-b['a11_class_macro'],
          'delta_eval_ce': a['eval_ce']-b['eval_ce']}
         for a, b in itertools.combinations(jobs, 2)]
failed = read(ROOT/'runtime/formal/jobs/task0_true_sam_k10_s2/failed.json')
error = failed['summary']['error']
failed_numeric = {k: float(re.search(r"'"+k+r"': ([0-9.eE+-]+)", error).group(1))
                  for k in ['selected_gradient_relative_error', 'actual_step_vs_selected_relative_error']}
true_epochs = read(ROOT/'outputs/task0_true_sam_k10_s2/metrics.json')['epochs']
report = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'campaign': str(ROOT),
    'scope': 'Task 0, 20 classes, seed 2, rank 10; partial campaign, no formal ranking or G* selection.',
    'campaign_state': {'status': state['status'], 'planned': 10, 'completed': len(state['completed']),
                       'failed': len(state['failed']), 'pending': state['pending'], 'running': state['running']},
    'source_files_checked': len(manifest['files']), 'source_mismatches': source_mismatches,
    'runtime_pins_checked': len(suite['file_pins']), 'runtime_pin_mismatches': pin_mismatches,
    'completed_jobs': jobs, 'pairwise_final_differences': pairs,
    'pairing': {'initial_identity_all_equal': True, 'epoch0_evaluation_all_equal': True,
               'epoch0_adapter_fc_file_hashes_all_equal': True, 'paired_epochs': pairing,
               'limit': 'All sampler indices and first augmented batch per epoch are recorded; later augmented tensors are not individually hashed. Loader code resets epoch data RNG, and rejects stochastic model layers.'},
    'true_sam_failure': {'completed_epochs': len(true_epochs), 'failure_epoch': 13, 'failure_batch': 0,
        'failed_at_utc': failed['time'], **failed_numeric, 'threshold': 1e-4,
        'completed_epoch_metrics_finite': not nonfinite(true_epochs),
        'description': 'Selected-gradient/update diagnostic assertion; missing final metrics/checkpoints are consequences of early termination, not evidence of NaN divergence.',
        'source': str(ROOT/'runtime/formal/jobs/task0_true_sam_k10_s2/failed.json')},
    'metric_definitions': {'a11_sample': '100*total correct/307, final epoch 50',
        'a11_class_macro': 'Mean of 20 per-class accuracies; not task-macro AAA from the ten-task campaign.',
        'train_ce': 'Epoch mean clean training CE over 1259 augmented training samples, at evolving parameters.',
        'eval_ce': 'Fixed test-set CE at epoch endpoint; distinct data/augmentation/parameter timing from train_ce.'},
    'limitations': ['Single Task 0 and one seed; not a continual-learning AAA/FAA/forgetting result.',
        'Peak epoch and peak accuracy are exploratory test-trajectory descriptions, not selected checkpoints.',
        'Five completed of ten planned; do not declare a final winner or select G*.',
        'Per-class counts do not identify which individual images differ; pairwise correct-count deltas are net counts, not McNemar discordance.',
        'Trainer does not invoke after_task; absence of skip log is expected. Effective flags plus no Fisher/fusion execution markers and zero EWC establish the Task 0 scope only.',
        'Metrics files have no pre-existing recorded digest in the completion receipt; receipt embeds identical final summary, launch pins bind config/trainer, and this audit records current artifact hashes.',
        'Checkpoint finite audit loads adapter and classifier weights at epochs 0/10/20/30/40/50 with CPU weights_only and explicit Linear/set allowlist. Full optimizer/RNG pickles are not deserialized.']}
assert not source_mismatches and not pin_mismatches
(OUT/'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
write_csv('epoch_curves.csv', curves)
write_csv('checkpoint_tensor_audit.csv', checkpoint_rows)
write_csv('pairwise_final_differences.csv', pairs)
summary_rows = [{k:v for k,v in row.items() if k in ['job','direction','epochs','correct','eval_samples','a11_sample','a11_class_macro','eval_ce','train_ce','train_accuracy','best_epoch_accuracy_exploratory','best_minus_final_correct_exploratory']} for row in jobs]
write_csv('final_results.csv', summary_rows)
print(json.dumps({'status':'passed','completed_jobs':5,'finite_checkpoint_files':len(checkpoint_rows),
                  'source_files_checked':len(manifest['files']),'runtime_pins_checked':len(suite['file_pins']),
                  'audit':str(OUT/'audit.json')},ensure_ascii=False))
