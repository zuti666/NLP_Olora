"""Observe one spec-identified float-lambda refinement; never mutate the initial sweep."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import runpy
import sys
import time

HERE = Path(__file__).resolve().parent
INITIAL_SWEEP = HERE.parent / 'ewc_lambda_sweep_seed0_20260921'
PREPARED_FISHER = INITIAL_SWEEP / 'prepared_fisher/fisher_state_raw.pt'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite_lambda(value):
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1e9:
        raise argparse.ArgumentTypeError('EWC lambda must be finite and in [0, 1e9].')
    return result


def load_trial(spec_path, trial_id, ewc_lambda):
    specification = json.loads(Path(spec_path).read_text())
    if specification.get('round') != 2 or specification.get('seed') != 0:
        raise ValueError('Only refinement round2, seed0 is authorized.')
    if ('initial_sweep' in specification
            and Path(specification['initial_sweep']).resolve() != INITIAL_SWEEP.resolve()):
        raise ValueError('The refinement must refer to the completed initial sweep.')
    if ('prepared_fisher' in specification
            and Path(specification['prepared_fisher']).resolve() != PREPARED_FISHER.resolve()):
        raise ValueError('The refinement must reuse the exact initial prepared raw Fisher path.')
    for path, digest in specification.get('input_fingerprints', {}).items():
        if sha(path) != digest:
            raise ValueError(f'Refinement input fingerprint changed: {path}')
    rows = specification.get('trials')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Refinement specification requires a nonempty trials list.')
    identities, lambdas = set(), set()
    for row in rows:
        label = row.get('trial_id')
        if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', label):
            raise ValueError('Each trial_id must be one safe path component.')
        value = row.get('ewc_lambda')
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError('Each ewc_lambda must be a JSON number.')
        value = finite_lambda(value)
        if label in identities or value in lambdas:
            raise ValueError('Refinement trial IDs and numeric lambda values must be unique.')
        identities.add(label)
        lambdas.add(value)
    selected = [row for row in rows if row['trial_id'] == trial_id]
    if len(selected) != 1 or float(selected[0]['ewc_lambda']) != finite_lambda(ewc_lambda):
        raise ValueError('--trial-id and --ewc-lambda must exactly match one specification row.')
    return dict(selected[0])


def native_argv(trial, gpu, resume_last=False):
    argv = [str(HERE / 'train.py'), '--seed', '0', '--ewc-lambda', repr(float(trial['ewc_lambda'])),
            '--run-name', trial['trial_id'], '--gpu', str(gpu),
            '--prepared-fisher', str(PREPARED_FISHER)]
    if resume_last:
        argv.append('--resume-last')
    return argv


def expected_identity(argv):
    specification = importlib.util.spec_from_file_location('refinement_native_train', HERE / 'train.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    ns = module.parser().parse_args(argv[1:])
    cfg, folder = module.build_config(ns)
    return module.experiment_identity(ns, cfg), folder


def observe_native(folder, trial, code, expected):
    manifest_path = folder / 'run_manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError(f'Training did not create a manifest; exit={code}')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('identity_sha256') != expected['identity_sha256']
            or manifest.get('runner_sha256') != sha(HERE / 'train.py')
            or manifest.get('requested_config') != expected['requested_config']
            or manifest.get('prepared_fisher') != expected['prepared_fisher']
            or manifest.get('seed') != 0 or manifest.get('smoke_only') is not False
            or manifest.get('run_name') != trial['trial_id']):
        raise ValueError('Native manifest differs from the exact current refinement identity.')
    complete, divergence = folder / 'COMPLETE.json', folder / 'DIVERGED.json'
    if code == 0 and complete.is_file() and not divergence.exists():
        receipt_path, outcome = complete, 'complete'
    elif code == 86 and divergence.is_file() and not complete.exists():
        receipt_path, outcome = divergence, 'diverged'
    else:
        raise RuntimeError(f'Unclassified trial failure: exit={code}; COMPLETE={complete.exists()}; DIVERGED={divergence.exists()}')
    native = json.loads(receipt_path.read_text())
    if native.get('status') != outcome or native.get('identity_sha256') != manifest['identity_sha256']:
        raise ValueError('Native terminal receipt identity/status does not match its manifest.')
    if outcome == 'complete':
        if native.get('completed_tasks') != 2 or native.get('validation', {}).get('epochs_per_new_task') != 50:
            raise ValueError('Only a fully completed 50-epoch two-task run is a finite result.')
    elif (native.get('seed') != 0 or native.get('ewc_lambda') != float(trial['ewc_lambda'])
          or native.get('smoke_only') is not False):
        raise ValueError('Divergence receipt belongs to a different lambda/seed/scope.')
    return {'status': 'observed', 'outcome': outcome, 'seed': 0, 'round': 2,
            'trial_id': trial['trial_id'], 'ewc_lambda': float(trial['ewc_lambda']),
            'time_unix': time.time(), 'native_exit_code': code,
            'native_receipt': str(receipt_path), 'native_receipt_sha256': sha(receipt_path),
            'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
            'identity_sha256': manifest['identity_sha256'],
            'prepared_fisher': expected['prepared_fisher'],
            'formal_accuracy_available': outcome == 'complete',
            'note': 'observed includes explicitly detected numerical divergence; it does not mean training succeeded'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=HERE / 'refinement_spec.json')
    parser.add_argument('--trial-id', required=True)
    parser.add_argument('--ewc-lambda', type=finite_lambda, required=True)
    parser.add_argument('--gpu', choices=list(map(str, range(7))), required=True)
    parser.add_argument('--resume-last', action='store_true')
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    trial = load_trial(spec_path, args.trial_id, args.ewc_lambda)
    spec_sha = sha(spec_path)
    argv = native_argv(trial, args.gpu, args.resume_last)
    expected, folder = expected_identity(argv)
    existing_result = folder / 'TRIAL_RESULT.json'
    if existing_result.exists():
        previous = json.loads(existing_result.read_text())
        if (previous.get('spec_sha256') != spec_sha or previous.get('trial_id') != args.trial_id
                or previous.get('identity_sha256') != expected['identity_sha256']):
            raise ValueError('Existing trial result has a different specification or native identity.')
        if not args.resume_last:
            raise FileExistsError('Trial already observed; --resume-last is required for receipt verification.')
    recovered_divergence = args.resume_last and (folder / 'DIVERGED.json').is_file()
    code = 86 if recovered_divergence else 0
    if not recovered_divergence:
        previous_argv = sys.argv
        sys.argv = argv
        try:
            # Same process retains scheduler GPU/pipeline locks throughout training.
            runpy.run_path(str(HERE / 'train.py'), run_name='__main__')
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        finally:
            sys.argv = previous_argv
    result = observe_native(folder, trial, code, expected)
    if sha(spec_path) != spec_sha:
        raise ValueError('Refinement specification changed during the trial.')
    result.update(spec=str(spec_path), spec_sha256=spec_sha, wrapper_sha256=sha(__file__),
                  recovered_terminal_divergence=bool(recovered_divergence))
    target = folder / 'TRIAL_RESULT.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
