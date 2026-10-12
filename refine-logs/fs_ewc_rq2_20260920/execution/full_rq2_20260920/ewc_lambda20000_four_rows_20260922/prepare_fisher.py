#!/usr/bin/env python3
"""Recompute native raw fit-Fisher from any of six unchanged formal Task0 parents.

GPU execution requires only its selected GPU UUID and inherited exclusive lock.
--validate-only checks provenance and exact input replay on CPU, without a model
forward/backward, CUDA context, output tensor, or completed preparation receipt.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent
EXECUTION = FORMAL.parent
SOURCE = EXECUTION / 'source'
PARENT = None
OUTPUT = None
ANCHOR_METHOD = None
SEED = None
REPO = EXECUTION.parents[2]
PROTOCOL = 'rq2_raw_fit_fisher_replay_multiseed_v1'
GPU_UUIDS = {'0': 'GPU-fd7e2955-3733-e30a-dff1-5da992423839', '1': 'GPU-d59d54d6-e68a-ffb7-4b2b-5e8d69d4b78a', '2': 'GPU-fe4ca05a-b7fd-efa7-8192-201e24573825', '3': 'GPU-9aa16f75-eb1c-c1a4-e900-f7b5afcddf26', '4': 'GPU-135554e2-aa66-e230-7d2f-bb14d2efddc5', '5': 'GPU-6eac7f06-8173-e1a3-c938-239f6f6eb19e', '6': 'GPU-b74c0891-02cf-152f-704a-bd7edbd0b50a'}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = Path(path).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def tensor_hash(items):
    result = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(str(value.dtype).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def per_tensor_statistics(raw):
    return {name: {'shape': list(value.shape), 'numel': value.numel(),
                   'trace': value.double().sum().item(),
                   'mean': value.double().mean().item(), 'max': value.max().item(),
                   'min': value.min().item(), 'dtype': str(value.dtype)}
            for name, value in sorted(raw.items())}


def verify_provenance():
    manifest = read_json(PARENT / 'run_manifest.json')
    complete = read_json(PARENT / 'COMPLETE.json')
    if (complete.get('status') != 'complete' or manifest.get('seed') != SEED
            or manifest.get('anchor_method') != ANCHOR_METHOD
            or complete.get('identity_sha256') != manifest.get('identity_sha256')):
        raise ValueError('The selected formal Task0 identity is not complete.')
    if sha256(FORMAL / 'train.py') != manifest['runner_sha256']:
        raise ValueError('Original training runner has changed.')
    source_hash = sha256(EXECUTION / 'source_frozen.json')
    if source_hash != manifest['source_manifest_sha256']:
        raise ValueError('Frozen-source manifest differs from the parent.')
    for entry in read_json(EXECUTION / 'source_frozen.json')['files']:
        if sha256(SOURCE / entry['path']) != entry['sha256']:
            raise ValueError(f"Frozen source has changed: {entry['path']}")
    assets = {}
    for relative, expected in complete['validation']['checkpoint_assets'].items():
        actual = sha256(PARENT / relative)
        if actual != expected['sha256']:
            raise ValueError(f'Parent checkpoint asset changed: {relative}')
        assets[relative] = actual
    receipt_path = PARENT / 'rng_receipts.jsonl'
    receipt_hash = sha256(receipt_path)
    if receipt_hash != complete['validation']['rng_receipts_sha256']:
        raise ValueError('Parent RNG receipts changed.')
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    iterations = [r for r in receipts if r.get('event') == 'loader_iteration'
                  and r.get('task') == 0 and r.get('loader_serial') == 0]
    if [r['iteration'] for r in iterations] != list(range(51)):
        raise ValueError('Exactly 50 fit epochs and one Fisher iteration are required.')
    target = iterations[-1]
    if any(target[k] != v for k, v in {
        'sample_count': 1259, 'samples': 1259, 'batches': 10,
        'num_workers': 4, 'shuffle': True, 'generator_seed': 30260939 + 100003 * SEED,
    }.items()):
        raise ValueError('Unexpected original fit-loader protocol.')
    if any(r['generator_seed'] != target['generator_seed'] for r in iterations):
        raise ValueError('The dedicated loader seed changed during the parent run.')
    if manifest['config'].get('fisher_mode') != 'batch_empirical' or manifest['config'].get('ewc_fisher_source') != 'fit':
        raise ValueError('The parent did not use the native batch_empirical fit estimator.')
    return manifest, iterations, {
        'anchor_method': ANCHOR_METHOD, 'seed': SEED,
        'parent_checkpoint_path': str(PARENT / 'checkpoints/task_00.pt'),
        'parent_checkpoint_sha256': assets['checkpoints/task_00.pt'],
        'parent_fisher_sha256': assets['checkpoints/fisher_state_0.pt'],
        'parent_asset_sha256': assets,
        'parent_manifest_sha256': sha256(PARENT / 'run_manifest.json'),
        'parent_rng_receipts_sha256': receipt_hash,
        'source_manifest_sha256': source_hash,
        'formal_runner_sha256': manifest['runner_sha256'],
    }


def replay_generator(torch, iterations):
    """Use the installed DataLoader implementation, not an assumed RNG stride."""
    generator = torch.Generator().manual_seed(iterations[0]['generator_seed'])
    loader = torch.utils.data.DataLoader(torch.arange(1259), batch_size=128,
                                        shuffle=True, num_workers=0, generator=generator)
    before_fisher = None
    for iteration, expected in enumerate(iterations):
        if iteration == 50:
            before_fisher = generator.get_state().clone()
        digest = hashlib.sha256()
        for indices in loader:
            digest.update(indices.numpy().tobytes())
        if digest.hexdigest() != expected['indices_sha256']:
            raise ValueError(f'Dummy sampler replay mismatch at iteration {iteration}.')
    return before_fisher, {
        'iterations_verified': 51,
        'generator_state_before_fisher_sha256': hashlib.sha256(before_fisher.numpy().tobytes()).hexdigest(),
        'sampler_indices_matched': True,
    }


def build_fit_loader(torch, manifest, generator_state, target):
    import numpy as np
    from utils.data_manager import DataManager
    cfg = copy.deepcopy(manifest['config'])
    cfg.update(seed=SEED, ewc_normalize_fisher=False,
               ewc_normalize_accumulated_fisher=False, persistent_workers=False)
    manager = DataManager('imageneta', False, SEED, 20, 20, cfg)
    dataset = manager.get_dataset(np.arange(20), source='train', mode='train')
    ids = list(zip(map(str, dataset.images), map(int, dataset.labels)))
    ids_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    if ids_hash != target['dataset_ids_sha256'] or len(dataset) != 1259:
        raise ValueError('Current Task0 dataset/order differs from the formal fit loader.')
    generator = torch.Generator()
    generator.set_state(generator_state)
    loader = manager.build_dataloader(dataset, batch_size=128, shuffle=True,
                                      num_workers=4, generator=generator,
                                      persistent_workers=False)
    if loader.generator is not loader.sampler.generator:
        raise ValueError('Sampler and loader must share their dedicated generator.')
    return cfg, manager, loader


class VerifiedInputPass:
    """Fail before the first model forward if original augmentations do not match."""
    def __init__(self, loader, target):
        self.loader, self.target = loader, target
        self.result = None

    def __iter__(self):
        digest = hashlib.sha256()
        all_pixels = hashlib.sha256()
        first = None
        samples = batches = 0
        for indices, inputs, targets in self.loader:
            digest.update(indices.detach().cpu().numpy().tobytes())
            batch_hash = tensor_hash([('input', inputs), ('target', targets)])
            all_pixels.update(batch_hash.encode())
            if first is None:
                first = batch_hash
                if first != self.target['first_batch_pixels_targets_sha256']:
                    raise ValueError('Task0 Fisher first-batch pixels/targets do not match the original.')
            samples += len(targets)
            batches += 1
            yield indices, inputs, targets
        result = {k: self.target[k] for k in (
            'dataset_ids_sha256', 'generator_seed', 'num_workers', 'sample_count', 'iteration')}
        result.update(indices_sha256=digest.hexdigest(),
                      first_batch_pixels_targets_sha256=first,
                      all_batch_pixels_targets_sha256=all_pixels.hexdigest(),
                      samples=samples, batches=batches)
        for key in ('indices_sha256', 'first_batch_pixels_targets_sha256', 'samples', 'batches'):
            if result[key] != self.target[key]:
                raise ValueError(f'Native fit-Fisher input receipt mismatch: {key}.')
        result.update(matched=True, original_full_pixel_digest_available=False)
        self.result = result


def verify_inherited_gpu_lock(lock_path):
    """Check a live inherited open description owns this GPU's exclusive lock."""
    lock_path = Path(lock_path).resolve(strict=True)
    inherited = []
    for entry in Path('/proc/self/fd').iterdir():
        try:
            if entry.resolve(strict=True) == lock_path:
                inherited.append(int(entry.name))
        except FileNotFoundError:
            continue
    if not inherited:
        raise ValueError('GPU preparation requires the queue-inherited UUID lock fd.')
    with lock_path.open('a') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise ValueError('The inherited UUID lock is not held exclusively.')
    for fd in inherited:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return {'path': str(lock_path), 'inherited_fd': inherited[0],
            'exclusive_lock_verified': True}


def validate_gpu_admission(gpu):
    if str(gpu) not in GPU_UUIDS:
        raise ValueError('Only physical GPUs 0 through 6 are admitted.')
    gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if gpu_uuid != GPU_UUIDS[str(gpu)]:
        raise ValueError('Requested physical GPU differs from its frozen UUID.')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
        raise ValueError('GPU preparation requires the queue-selected CUDA UUID.')
    lock = verify_inherited_gpu_lock(REPO / '.cache/gpu_locks' / f'{gpu_uuid}.lock')
    # Check this card only; other GPUs may already run independent queued jobs.
    status = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True).strip()
    memory, utilization = [int(value.strip()) for value in status.split(',')]
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'], text=True).splitlines()
    if memory >= 500 or utilization >= 5 or gpu_uuid in [v.strip() for v in occupied]:
        raise ValueError('Selected GPU is no longer idle before CUDA initialization.')
    return {'status': 'admitted', 'scope': 'selected_gpu_only', 'physical_gpu': str(gpu),
            'expected_gpu_uuid': gpu_uuid, 'shared_lock': lock,
            'memory_mib': memory, 'utilization': utilization,
            'poll_seconds': 900, 'all_gpu_gate_required': False}, gpu_uuid


def load_formal_runner():
    spec = importlib.util.spec_from_file_location('rq2_original_fisher_runtime', FORMAL / 'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(torch, cfg, manager, loader, target, provenance, gate, gpu_uuid):
    import argparse
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runner = load_formal_runner()
    # validate_cpu here suppresses an unrelated old-task evaluation in patched_load;
    # the actual model device below is CUDA and this is a real Fisher recomputation.
    ns = argparse.Namespace(seed=SEED, workers=4, anchor_dir=PARENT / 'checkpoints', validate_cpu=True)
    cfg.update(device=[torch.device('cuda:0')], filepath=str(OUTPUT),
               save_task_checkpoint_dir=str(OUTPUT), seed=SEED,
               ewc_normalize_fisher=False, ewc_normalize_accumulated_fisher=False,
               approximate_fisher_in_task=False, ewc_skip_task_fisher=False)
    cfg['task_increments'], cfg['nb_classes'] = list(manager._increments), manager.nb_classes
    runtime = runner.install_runtime(ns, cfg, OUTPUT)
    runtime['reset_rng'](SEED)
    model = runtime['factory'].get_model(cfg['model_name'], cfg)
    model.load_task_checkpoint(str(PARENT / 'checkpoints'), 0, manager)
    net = model._unwrap_network()
    if model._normalize_fisher or model._normalize_accumulated_fisher:
        raise ValueError('Both Fisher normalization flags must be disabled at runtime.')
    if (model._cur_task, model._total_classes) != (0, 20):
        raise ValueError('Expected unchanged Task0 with its original 20-class head.')
    before = tensor_hash(net.state_dict().items())
    model._fisher_past_delta = {}
    model._delta_reference = {}
    model._fisher_past_trace_mass = 0.0
    model._current_fit_loader = model.train_loader = loader
    verified = VerifiedInputPass(loader, target)
    # Never call after_task() on a resumed model: known_classes is already 20.
    raw = model._compute_delta_fisher(loader=verified, model=net, class_offset=0)
    if verified.result is None or not verified.result['matched']:
        raise ValueError('Fisher computation did not finish the exact input pass.')
    after = tensor_hash(net.state_dict().items())
    if before != after:
        raise ValueError('Fisher recomputation changed network weights or buffers.')
    expected_keys = {f'backbone.lora_vit.blocks.{block}.attn.qkv.{role}'
                     for block in range(12) for role in ('q', 'v')}
    if set(raw) != expected_keys:
        raise ValueError('Raw Fisher does not contain the expected 24 Q/V matrices.')
    for name, value in raw.items():
        if (tuple(value.shape) != (768, 768) or not torch.isfinite(value).all().item()
                or not (value >= 0).all().item()):
            raise ValueError(f'Invalid raw Fisher tensor: {name}')
    trace = sum(v.double().sum().item() for v in raw.values())
    old = torch.load(PARENT / 'checkpoints/fisher_state_0.pt', map_location='cpu',
                     weights_only=False, mmap=True)
    expected_trace = float(old['fisher_new_task_raw_trace'])
    if not (math.isfinite(trace) and trace > 0 and
            math.isclose(trace, expected_trace, rel_tol=1e-4, abs_tol=1e-5)):
        raise ValueError(f'Recomputed raw trace {trace} differs from original {expected_trace}.')
    del old
    numel = sum(v.numel() for v in raw.values())
    stats = {'tensors': len(raw), 'numel': numel, 'sum': trace,
             'mean': trace / numel, 'max': max(v.max().item() for v in raw.values())}
    model._fisher_past_delta = model._accumulate_fisher({}, raw)
    if any(not torch.equal(raw[k], model._fisher_past_delta[k]) for k in raw):
        raise ValueError('Raw accumulation unexpectedly modified Fisher values.')
    for relative, expected in provenance['parent_asset_sha256'].items():
        if sha256(PARENT / relative) != expected:
            raise ValueError('An original parent checkpoint asset changed.')
    payload = {'fisher_past_delta': {k: v.detach().cpu() for k, v in raw.items()},
               'raw_trace': trace, 'stats': stats,
               'fisher_new_task_raw_trace': trace, 'last_task_new_fisher_stats': stats,
               'fisher_past_trace_mass': 0.0}
    output_path = OUTPUT / 'fisher_state_raw.pt'
    temporary = output_path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(output_path)
    receipt = dict(provenance, status='complete', protocol=PROTOCOL,
                   preparation_kind='native_fit_fisher_recomputation',
                   payload_path=str(output_path), payload_sha256=sha256(output_path),
                   ewc_normalize_fisher=False, ewc_normalize_accumulated_fisher=False,
                   fisher_mode='batch_empirical', fisher_source='fit', task=0,
                   raw_trace=trace, tensor_count=len(raw), numel=numel, mean=trace / numel,
                   tensor_shapes={key: list(value.shape) for key, value in raw.items()},
                   stats=stats, tensor_stats=per_tensor_statistics(raw), weights_unchanged=True,
                   full_checkpoint_restoration_verified=True,
                   network_state_before_sha256=before, network_state_after_sha256=after,
                   expected_parent_raw_trace=expected_trace,
                   raw_trace_tolerance={'rtol': 1e-4, 'atol': 1e-5},
                   exact_inputs_receipt=verified.result, admission=gate, gpu_uuid=gpu_uuid,
                   torch_version=torch.__version__, source=str(SOURCE),
                   producer_runner_path=str(Path(__file__).resolve()),
                   producer_runner_sha256=sha256(__file__),
                   prepare_script_sha256=sha256(__file__), time_unix=time.time(),
                   fisher_estimation_metrics=dict(model._last_fisher_estimation_metrics),
                   model_training_steps=0, native_fisher_passes=1,
                   original_normalized_fisher_used_for_computation=False)
    write_json(OUTPUT / 'PREPARED.json', receipt)
    return receipt


def verify_prepared(torch, directory, provenance, *, legacy=False):
    """Validate the real raw tensor payload and its completed provenance."""
    directory = Path(directory)
    receipt = read_json(directory / 'PREPARED.json')
    required = {'status': 'complete', 'seed': SEED, 'task': 0,
                'ewc_normalize_fisher': False, 'ewc_normalize_accumulated_fisher': False,
                'fisher_mode': 'batch_empirical', 'fisher_source': 'fit',
                'tensor_count': 24, 'weights_unchanged': True,
                'full_checkpoint_restoration_verified': True,
                'original_normalized_fisher_used_for_computation': False}
    if legacy:
        if (ANCHOR_METHOD, SEED) != ('sgd', 0):
            raise ValueError('The audited existing raw Fisher belongs only to SGD seed0.')
        required['protocol'] = 'rq2_raw_fit_fisher_replay_v1'
    else:
        required.update(anchor_method=ANCHOR_METHOD, protocol=PROTOCOL)
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f'Prepared receipt mismatch: {key}.')
    for key in ('parent_checkpoint_sha256', 'parent_fisher_sha256',
                'parent_asset_sha256', 'parent_manifest_sha256',
                'parent_rng_receipts_sha256', 'source_manifest_sha256',
                'formal_runner_sha256'):
        if receipt.get(key) != provenance[key]:
            raise ValueError(f'Prepared provenance mismatch: {key}.')
    if (receipt.get('exact_inputs_receipt', {}).get('matched') is not True or
            receipt.get('network_state_before_sha256') != receipt.get('network_state_after_sha256') or
            not receipt.get('network_state_before_sha256')):
        raise ValueError('Prepared state/input verification is absent or failed.')
    producer = Path(receipt['producer_runner_path'])
    if sha256(producer) != receipt['producer_runner_sha256']:
        raise ValueError('The original raw-Fisher producer has changed.')
    payload_path = directory / 'fisher_state_raw.pt'
    if sha256(payload_path) != receipt['payload_sha256']:
        raise ValueError('Prepared raw-Fisher payload SHA differs.')
    payload = torch.load(payload_path, map_location='cpu', weights_only=True, mmap=True)
    raw = payload['fisher_past_delta']
    keys = {f'backbone.lora_vit.blocks.{block}.attn.qkv.{role}'
            for block in range(12) for role in ('q', 'v')}
    if set(raw) != keys:
        raise ValueError('Expected exactly 24 raw Q/V matrices.')
    for name, value in raw.items():
        if (tuple(value.shape) != (768, 768) or value.dtype != torch.float32 or
                not torch.isfinite(value).all().item() or not (value >= 0).all().item()):
            raise ValueError(f'Invalid raw Fisher matrix: {name}.')
    trace = sum(value.double().sum().item() for value in raw.values())
    if not math.isfinite(trace) or trace <= 0:
        raise ValueError('Raw Fisher trace must be finite and positive.')
    for recorded in (receipt['raw_trace'], payload['raw_trace'], payload['stats']['sum']):
        if not math.isclose(trace, recorded, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Raw Fisher payload/receipt trace mismatch.')
    old = torch.load(PARENT / 'checkpoints/fisher_state_0.pt', map_location='cpu',
                     weights_only=True, mmap=True)
    if not math.isclose(trace, float(old['fisher_new_task_raw_trace']), rel_tol=1e-4, abs_tol=1e-5):
        raise ValueError('Prepared trace differs from the original native raw trace.')
    if receipt['numel'] != sum(value.numel() for value in raw.values()):
        raise ValueError('Prepared element count mismatch.')
    return receipt, per_tensor_statistics(raw)


def reuse_existing(torch, provenance, exact_inputs):
    original_dir = FORMAL / 'ewc_lambda_sweep_seed0_20260921/prepared_fisher'
    original, tensor_stats = verify_prepared(torch, original_dir, provenance, legacy=True)
    # Unlike the six historical training logs, the raw preparation receipt also
    # retained every augmented batch digest. Reuse must agree with that too.
    for key in ('dataset_ids_sha256', 'indices_sha256', 'first_batch_pixels_targets_sha256',
                'all_batch_pixels_targets_sha256', 'generator_seed', 'samples', 'batches'):
        if original['exact_inputs_receipt'][key] != exact_inputs[key]:
            raise ValueError(f'Existing raw Fisher CPU replay differs: {key}.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / 'fisher_state_raw.pt'
    temporary = output_path.with_suffix('.pt.tmp')
    shutil.copyfile(original_dir / 'fisher_state_raw.pt', temporary)
    if sha256(temporary) != original['payload_sha256']:
        raise ValueError('Existing raw Fisher copy changed bytes.')
    temporary.replace(output_path)
    result = dict(original)
    result.update(provenance)
    result.update(protocol=PROTOCOL, payload_path=str(output_path),
                  tensor_stats=tensor_stats, exact_inputs_receipt=exact_inputs,
                  preparation_kind='verified_existing_raw_payload_reuse',
                  reuse_provenance={'receipt_path': str(original_dir / 'PREPARED.json'),
                                    'receipt_sha256': sha256(original_dir / 'PREPARED.json'),
                                    'payload_path': str(original_dir / 'fisher_state_raw.pt'),
                                    'payload_sha256': original['payload_sha256'],
                                    'producer_runner_path': original['producer_runner_path'],
                                    'producer_runner_sha256': original['producer_runner_sha256'],
                                    'cpu_replay_all_pixels_matched': True},
                  producer_runner_path=str(Path(__file__).resolve()),
                  producer_runner_sha256=sha256(__file__),
                  prepare_script_sha256=sha256(__file__),
                  reuse_runner_path=str(Path(__file__).resolve()),
                  reuse_runner_sha256=sha256(__file__), reuse_time_unix=time.time(),
                  native_fisher_passes_this_invocation=0, cuda_initialized=False)
    write_json(OUTPUT / 'PREPARED.json', result)
    return result


def main():
    global PARENT, OUTPUT, ANCHOR_METHOD, SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor-method', required=True, choices=['sgd', 'gam'])
    parser.add_argument('--seed', required=True, type=int, choices=range(3))
    parser.add_argument('--gpu', choices=list(map(str, range(7))))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--reuse-existing', action='store_true')
    parser.add_argument('--result-json', type=Path)
    ns = parser.parse_args()
    ANCHOR_METHOD, SEED = ns.anchor_method, ns.seed
    PARENT = FORMAL / f'formal_training/anchor_{ANCHOR_METHOD}_seed{SEED}'
    OUTPUT = HERE / f'prepared_fisher/{ANCHOR_METHOD}_seed{SEED}'
    cpu_only = ns.validate_only or ns.reuse_existing
    if not cpu_only and ns.gpu is None:
        parser.error('GPU preparation requires --gpu from a queued job with its UUID lock.')
    if ns.reuse_existing and (ANCHOR_METHOD, SEED) != ('sgd', 0):
        parser.error('Only the audited SGD seed0 payload may use --reuse-existing.')
    if ns.result_json and ns.result_json.name == 'PREPARED.json':
        parser.error('--result-json must not replace the completed PREPARED.json receipt.')
    gate = gpu_uuid = None
    if cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        gate, gpu_uuid = validate_gpu_admission(ns.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ['TMPDIR'] = '/tmp'
    os.environ['FLATNESS_CV_DATA_ROOT'] = '/data/115-2/users/liying/datasets'
    sys.path.insert(0, str(SOURCE))
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manifest, iterations, provenance = verify_provenance()
    generator_state, replay = replay_generator(torch, iterations)
    provenance['sampler_replay'] = replay
    if not cpu_only and (OUTPUT / 'PREPARED.json').exists():
        verify_prepared(torch, OUTPUT, provenance)
        result = {'status': 'already_complete', 'anchor_method': ANCHOR_METHOD,
                  'seed': SEED, 'prepared': str(OUTPUT / 'PREPARED.json')}
    else:
        cfg, manager, loader = build_fit_loader(torch, manifest, generator_state, iterations[-1])
        if cpu_only:
            verified = VerifiedInputPass(loader, iterations[-1])
            for _ in verified:
                pass
            if torch.cuda.is_initialized():
                raise RuntimeError('CPU validation/reuse must not initialize CUDA.')
            if ns.reuse_existing:
                result = reuse_existing(torch, provenance, verified.result)
            else:
                result = dict(provenance, status='cpu_validated', protocol=PROTOCOL,
                              exact_inputs_receipt=verified.result,
                              cuda_initialized=False, model_forward_backward_executed=False,
                              native_fisher_passes=0, prepared_payload_written=False,
                              prepare_script_sha256=sha256(__file__),
                              torch_version=torch.__version__, time_unix=time.time())
        else:
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('Preparation requires exactly the admitted visible GPU.')
            result = prepare(torch, cfg, manager, loader, iterations[-1], provenance, gate, gpu_uuid)
    if ns.result_json:
        ns.result_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(ns.result_json, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
