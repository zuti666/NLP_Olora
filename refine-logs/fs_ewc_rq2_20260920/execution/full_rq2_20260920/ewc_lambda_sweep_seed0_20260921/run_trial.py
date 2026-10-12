"""Observe a formal lambda point without mistaking divergence for completed training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def name_for(lam):
    assert math.isfinite(lam) and lam >= 0 and lam == int(lam)
    return f'lambda_{int(lam)}_seed0'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ewc-lambda', type=float, required=True)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    spec = json.loads((HERE / 'sweep_spec.json').read_text())
    assert args.ewc_lambda in spec['new_lambdas']
    run_name = name_for(args.ewc_lambda)
    folder = HERE / 'training' / run_name
    sys.argv = [str(HERE / 'train.py'), '--seed', '0', '--ewc-lambda', str(args.ewc_lambda),
                '--run-name', run_name, '--gpu', args.gpu, '--resume-last',
                '--prepared-fisher', spec['prepared_fisher']]
    code = 0
    try:
        # Run in this process so the scheduler's inherited GPU/pipeline locks
        # remain held throughout training and receipt creation.
        runpy.run_path(str(HERE / 'train.py'), run_name='__main__')
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    manifest_path = folder / 'run_manifest.json'
    if not manifest_path.exists():
        raise RuntimeError(f'Training did not create a manifest; exit={code}')
    manifest = json.loads(manifest_path.read_text())
    cfg = manifest['requested_config']
    assert manifest['seed'] == 0 and manifest['smoke_only'] is False
    assert cfg['optimizer_type'] == 'sgd' and cfg['lambda_flat'] == 0
    assert cfg['ewc_lambda'] == args.ewc_lambda and cfg['epochs'] == 50
    assert cfg['max_train_tasks'] == 2 and cfg['resume_from_task'] == 0
    assert cfg['ewc_normalize_fisher'] is False and cfg['ewc_normalize_accumulated_fisher'] is False
    assert manifest['runner_sha256'] == sha(HERE / 'train.py')
    complete, divergence = folder / 'COMPLETE.json', folder / 'DIVERGED.json'
    if code == 0 and complete.exists() and not divergence.exists():
        receipt_path, outcome = complete, 'complete'
    elif code == 86 and divergence.exists() and not complete.exists():
        receipt_path, outcome = divergence, 'diverged'
    else:
        raise RuntimeError(f'Unclassified trial failure: exit={code}; COMPLETE={complete.exists()}; DIVERGED={divergence.exists()}')
    native = json.loads(receipt_path.read_text())
    assert native['status'] == outcome
    assert native['identity_sha256'] == manifest['identity_sha256']
    if outcome == 'complete':
        assert native['completed_tasks'] == 2
        assert native['validation']['epochs_per_new_task'] == 50
    result = {'status': 'observed', 'outcome': outcome, 'seed': 0, 'ewc_lambda': args.ewc_lambda,
              'time_unix': time.time(), 'native_exit_code': code,
              'native_receipt': str(receipt_path), 'native_receipt_sha256': sha(receipt_path),
              'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
              'identity_sha256': manifest['identity_sha256'], 'formal_accuracy_available': outcome == 'complete',
              'note': 'observed includes explicitly detected numerical divergence; it does not mean every training run succeeded'}
    target = folder / 'TRIAL_RESULT.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
