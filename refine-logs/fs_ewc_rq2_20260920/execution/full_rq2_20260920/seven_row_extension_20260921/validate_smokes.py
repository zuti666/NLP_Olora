"""Require both real one-epoch integration tests before releasing formal training."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('extension_train', HERE / 'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked = []
    for arm, anchor, lam in [('shaping_only', 'gam', 0.), ('task2_shaping', 'sgd', 2000.)]:
        folder = HERE / 'smoke' / 'training' / f'{arm}_seed0'
        receipt_path = folder / 'SMOKE_COMPLETE.json'
        receipt = json.loads(receipt_path.read_text())
        manifest = json.loads((folder / 'run_manifest.json').read_text())
        assert not (folder / 'COMPLETE.json').exists(), 'Smoke must not create formal COMPLETE'
        assert receipt['status'] == 'smoke_complete' and receipt['completed_tasks'] == 2
        assert manifest['arm'] == arm and manifest['seed'] == 0
        assert manifest['anchor_method'] == anchor and manifest['future_optimizer'] == 'gam'
        assert manifest['smoke_only'] is True
        assert receipt['identity_sha256'] == manifest['identity_sha256']
        cfg = manifest['requested_config']
        assert cfg['rq2_smoke_only'] is True and cfg['epochs'] == 1
        assert cfg['optimizer_type'] == 'gam' and cfg['lambda_flat'] == 1
        assert cfg['ewc_lambda'] == lam and cfg['resume_from_task'] == 0 and cfg['max_train_tasks'] == 2
        assert cfg['mechanism_eval'] and cfg['mechanism_grad_stats']
        assert not manifest['matmul_allow_tf32'] and not manifest['cudnn_allow_tf32']
        validation = module.validate_completion(folder, cfg, manifest['mode'])
        assert validation == receipt['validation'], 'Smoke outputs changed after completion'
        content = json.loads(Path(validation['metrics']).read_text())
        m = content['mechanism']['steps']['1']
        finite_fields = ['train_clean_loss_mean', 'train_fisher_loss_mean', 'grad_gam_norm_mean',
                         'grad_fisher_norm_mean', 'in_task_fisher_final_raw_trace']
        assert all(math.isfinite(float(m[k])) for k in finite_fields)
        assert m['grad_gam_norm_mean'] > 0 and m['in_task_fisher_final_raw_trace'] > 0
        if lam:
            assert m['grad_fisher_norm_mean'] > 0, 'Protected smoke did not exercise EWC gradient'
        else:
            assert m['grad_fisher_norm_mean'] == 0 and m['train_fisher_loss_mean'] == 0
        assert m['fisher_num_samples'] == 997
        assert m['in_task_fisher_used_path'] is False and m['in_task_weight_merge_applied'] == 0
        local = [json.loads(x) for x in (folder / 'task0_local_metrics.jsonl').read_text().splitlines()]
        old = next(x for x in local if x['task'] == 1)
        assert old['test']['n'] == 307 and old['train']['n'] == 1259
        assert abs(old['test']['seen_accuracy'] - content['cnn']['matrices']['t01'][1][0]) < .0051
        witness_path = HERE / 'witness' / f'smoke_{arm}_seed0.json'
        witness = json.loads(witness_path.read_text())
        assert witness['status'] == 'passed' and witness['finite_forward_backward']
        checked.append({'arm': arm, 'receipt': str(receipt_path), 'receipt_sha256': sha(receipt_path),
                        'witness': str(witness_path), 'witness_sha256': sha(witness_path),
                        'validation': validation, 'mechanism_fields': {k:m[k] for k in finite_fields},
                        'actual_CIL_test_evaluated': True, 'actual_old_task_loss_evaluated': True})
    result = {'status': 'passed', 'validated_smokes': 2, 'formal_results': False,
              'source_train_sha256': sha(HERE / 'train.py'), 'checked': checked}
    target = HERE / 'smoke' / 'VALIDATED.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status':'passed','validated_smokes':2,'formal_training_released':True}))


if __name__ == '__main__':
    main()
