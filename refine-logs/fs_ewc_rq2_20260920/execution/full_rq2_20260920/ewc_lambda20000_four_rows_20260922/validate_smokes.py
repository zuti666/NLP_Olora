#!/usr/bin/env python3
"""CPU-only acceptance gate for all four real seed0 one-epoch continuations."""
import importlib.util
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_train():
    spec = importlib.util.spec_from_file_location('lambda20000_smoke_train', HERE / 'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_smoke_contract(module, arm, manifest, receipt):
    anchor, future = module.ARMS[arm]
    cfg = manifest['requested_config']
    expected = {'seed': [0], 'epochs': 1, 'init_epoch': 50, 'batch_size': 128,
                'lrate': .02, 'train_num_workers': 4, 'eval_num_workers': 4,
                'resume_from_task': 0, 'max_train_tasks': 2, 'ewc_lambda': 20000.,
                'optimizer_type': future, 'lambda_flat': float(future == 'gam'),
                'ewc_normalize_fisher': False, 'ewc_normalize_accumulated_fisher': False,
                'rq2_smoke_only': True, 'rq2_extension_arm': arm, 'rq2_anchor_method': anchor,
                'mechanism_eval': True, 'mechanism_grad_stats': True,
                'approximate_fisher_in_task': False, 'weight_merge': False,
                'in_task_weight_merge': False, 'endpoint_weight_merge': False,
                'delta_w_penalty': False, 'ewc_project_grad': False}
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{arm}: smoke scientific configuration does not match the fixed contract.')
    if (receipt.get('status') != 'smoke_complete' or receipt.get('completed_tasks') != 2
            or manifest.get('arm') != arm or manifest.get('seed') != 0
            or manifest.get('anchor_method') != anchor or manifest.get('future_optimizer') != future
            or manifest.get('mode') != 'continuation' or manifest.get('smoke_only') is not True
            or receipt.get('identity_sha256') != manifest.get('identity_sha256')
            or manifest.get('matmul_allow_tf32') is not False or manifest.get('cudnn_allow_tf32') is not False):
        raise ValueError(f'{arm}: smoke identity/status/precision mismatch.')
    if manifest.get('runner_sha256') != module.sha256(HERE / 'train.py'):
        raise ValueError(f'{arm}: smoke was produced by a different training driver.')
    return cfg


def validate_all(module=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    module = module or load_train()
    checked = []
    for arm, (anchor, future) in module.ARMS.items():
        folder = HERE / 'smoke/training' / f'smoke_{arm}_seed0'
        receipt_path = folder / 'SMOKE_COMPLETE.json'
        receipt = json.loads(receipt_path.read_text())
        manifest = json.loads((folder / 'run_manifest.json').read_text())
        if (folder / 'COMPLETE.json').exists() or (folder / 'DIVERGED.json').exists():
            raise ValueError('A smoke cannot be a formal result or a diverged run.')
        cfg = validate_smoke_contract(module, arm, manifest, receipt)
        ns = module.parser().parse_args([
            '--arm', arm, '--anchor-method', anchor, '--future-optimizer', future,
            '--seed', '0', '--ewc-lambda', '20000', '--run-name', f'smoke_{arm}_seed0',
            '--anchor-dir', cfg['resume_from_checkpoint_dir'],
            '--anchor-metrics', cfg['resume_from_metrics_json'],
            '--prepared-fisher', cfg['rq2_prepared_fisher'], '--smoke'])
        rebuilt, rebuilt_dir = module.build_config(ns)
        identity = module.experiment_identity(ns, rebuilt)
        if rebuilt_dir != folder or rebuilt != cfg or identity['identity_sha256'] != manifest['identity_sha256']:
            raise ValueError('Smoke identity cannot be reproduced from the current configuration and actual parent.')
        validation = module.validate_completion(folder, cfg, 'continuation')
        if validation != receipt['validation']:
            raise ValueError('Smoke outputs changed after completion.')
        content = json.loads(Path(validation['metrics']).read_text())
        mechanism = content['mechanism']['steps']['1']
        finite_fields = ['train_clean_loss_mean', 'train_fisher_loss_mean', 'in_task_fisher_final_raw_trace']
        if future == 'gam':
            finite_fields += ['grad_clean_norm_mean', 'grad_gam_norm_mean', 'grad_fisher_norm_mean', 'grad_combined_norm_mean']
        if any(not math.isfinite(float(mechanism[key])) for key in finite_fields):
            raise ValueError('Smoke native mechanism contains a nonfinite value.')
        if (mechanism['train_fisher_loss_mean'] <= 0 or mechanism['in_task_fisher_final_raw_trace'] <= 0
                or mechanism['fisher_num_samples'] != 997
                or mechanism['in_task_fisher_used_path'] is not False
                or mechanism['in_task_weight_merge_applied'] != 0):
            raise ValueError('Smoke did not exercise the required native raw-Fisher path.')
        old = next(row for row in module.read_jsonl(folder / 'task0_local_metrics.jsonl') if row['task'] == 1)
        if (old['test']['n'] != 307 or old['train']['n'] != 1259
                or abs(old['test']['seen_accuracy'] - content['cnn']['matrices']['t01'][1][0]) >= .0051):
            raise ValueError('Smoke endpoint old-task local/CIL evaluation mismatch.')
        checked.append({'arm': arm, 'seed': 0, 'receipt': str(receipt_path),
                        'receipt_sha256': module.sha256(receipt_path),
                        'manifest': str(folder / 'run_manifest.json'),
                        'manifest_sha256': module.sha256(folder / 'run_manifest.json'),
                        'validation': validation,
                        'mechanism_fields': {key: mechanism[key] for key in finite_fields},
                        'actual_CIL_test_evaluated': True, 'actual_old_task_loss_evaluated': True,
                        'positive_step1_ewc_gradient': True})
    return {'status': 'passed', 'validated_smokes': 4, 'formal_results': False,
            'formal_training_released': True, 'validation_device': 'cpu',
            'source_train_sha256': module.sha256(HERE / 'train.py'),
            'validator_sha256': module.sha256(__file__), 'checked': checked}


def main():
    module = load_train()
    result = validate_all(module)
    module.write_json(HERE / 'smoke/VALIDATED.json', result)
    print(json.dumps({'status': 'passed', 'validated_smokes': 4, 'formal_training_released': True}))


if __name__ == '__main__':
    main()
