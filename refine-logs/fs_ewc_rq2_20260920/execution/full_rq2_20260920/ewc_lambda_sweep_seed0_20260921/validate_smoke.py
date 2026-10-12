"""Validate actual raw-Fisher EWC training and evaluation before the sweep."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    module_spec = importlib.util.spec_from_file_location('raw_sweep_train', HERE / 'train.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = json.loads((HERE / 'sweep_spec.json').read_text())
    lam = spec['smoke_lambda']
    folder = HERE / 'smoke/training' / f'smoke_lambda{lam}_seed0'
    receipt_path = folder / 'SMOKE_COMPLETE.json'
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads((folder / 'run_manifest.json').read_text())
    assert not (folder / 'COMPLETE.json').exists()
    assert receipt['status'] == 'smoke_complete' and receipt['completed_tasks'] == 2
    assert manifest['seed'] == 0 and manifest['smoke_only'] is True
    assert manifest['anchor_method'] == 'sgd' and manifest['future_optimizer'] == 'sgd'
    assert receipt['identity_sha256'] == manifest['identity_sha256']
    assert manifest['runner_sha256'] == sha(HERE / 'train.py')
    cfg = manifest['requested_config']
    assert cfg['rq2_smoke_only'] is True and cfg['epochs'] == 1
    assert cfg['optimizer_type'] == 'sgd' and cfg['lambda_flat'] == 0
    assert cfg['ewc_lambda'] == lam and cfg['resume_from_task'] == 0 and cfg['max_train_tasks'] == 2
    assert cfg['ewc_normalize_fisher'] is False and cfg['ewc_normalize_accumulated_fisher'] is False
    assert cfg['mechanism_eval'] and cfg['mechanism_grad_stats']
    assert not manifest['matmul_allow_tf32'] and not manifest['cudnn_allow_tf32']
    validation = module.validate_completion(folder, cfg, manifest['mode'])
    assert validation == receipt['validation'], 'Smoke outputs changed after completion'
    content = json.loads(Path(validation['metrics']).read_text())
    step = content['mechanism']['steps']['1']
    for key in ('train_clean_loss_mean', 'train_fisher_loss_mean', 'in_task_fisher_final_raw_trace'):
        assert math.isfinite(float(step[key])) and step[key] > 0, key
    assert step['fisher_num_samples'] == 997
    assert step['in_task_fisher_used_path'] is False and step['in_task_weight_merge_applied'] == 0
    # Step0 is at the reference, where P and its gradient are exactly zero.
    # A one-epoch smoke has eight batches; inspect the positive loss after step0.
    batches = [json.loads(line) for line in (folder / 'training_batch_metrics.jsonl').read_text().splitlines()]
    assert len(batches) == 8 and any(row['ewc'] > 0 for row in batches[1:])
    assert any(row.get('grad_fisher_norm', 0) > 0 for row in batches[1:])
    assert all(row['parameters_finite_after_step'] for row in batches)
    local = [json.loads(line) for line in (folder / 'task0_local_metrics.jsonl').read_text().splitlines()]
    old = next(row for row in local if row['task'] == 1)
    assert old['test']['n'] == 307 and old['train']['n'] == 1259
    assert abs(old['test']['seen_accuracy'] - content['cnn']['matrices']['t01'][1][0]) < .0051
    witness_path = HERE / 'witness/smoke_sgd_lambda60.json'
    witness = json.loads(witness_path.read_text())
    assert witness['status'] == 'passed' and witness['finite_forward_backward']
    prepared_path = HERE / 'prepared_fisher/PREPARED.json'
    prepared = json.loads(prepared_path.read_text())
    assert prepared['status'] == 'complete' and prepared['weights_unchanged'] is True
    result = {'status': 'passed', 'validated_smokes': 1, 'formal_results': False,
              'source_train_sha256': sha(HERE / 'train.py'), 'prepared_receipt_sha256': sha(prepared_path),
              'smoke_receipt_sha256': sha(receipt_path), 'witness_sha256': sha(witness_path),
              'validation': validation, 'actual_CIL_test_evaluated': True,
              'actual_old_task_loss_evaluated': True, 'actual_positive_protection_loss': True}
    target = HERE / 'smoke/VALIDATED.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'formal_training_released': True}))


if __name__ == '__main__':
    main()
