#!/usr/bin/env python3
"""Recompute native raw fit-Fisher once from the unchanged formal SGD Task0.

GPU execution is a queued preparation job after the campaign admission gate.
--validate-only checks provenance and exact input replay on CPU, without a model
forward/backward, CUDA context, output tensor, or completed preparation receipt.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent
EXECUTION = FORMAL.parent
SOURCE = EXECUTION / 'source'
PARENT = FORMAL / 'formal_training/anchor_sgd_seed0'
OUTPUT = HERE / 'prepared_fisher'
PROTOCOL = 'rq2_raw_fit_fisher_replay_v1'


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


def verify_provenance():
    manifest = read_json(PARENT / 'run_manifest.json')
    complete = read_json(PARENT / 'COMPLETE.json')
    if (complete.get('status') != 'complete' or manifest.get('seed') != 0
            or manifest.get('anchor_method') != 'sgd'
            or complete.get('identity_sha256') != manifest.get('identity_sha256')):
        raise ValueError('The formal SGD seed0 Task0 identity is not complete.')
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
        'num_workers': 4, 'shuffle': True, 'generator_seed': 30260939,
    }.items()):
        raise ValueError('Unexpected original fit-loader protocol.')
    return manifest, iterations, {
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
    cfg.update(seed=0, ewc_normalize_fisher=False,
               ewc_normalize_accumulated_fisher=False, persistent_workers=False)
    manager = DataManager('imageneta', False, 0, 20, 20, cfg)
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


def validate_gate(path, gpu):
    receipt = read_json(path)
    statuses = receipt.get('gpu_statuses', {})
    policy = receipt.get('policy', {})
    if (receipt.get('status') != 'admitted' or not isinstance(policy, dict)
            or policy.get('required') is not True or policy.get('poll_seconds') != 900
            or set(map(str, policy.get('gpus', []))) != set(map(str, range(7)))
            or set(map(str, receipt.get('gpus', []))) != set(map(str, range(7)))
            or set(map(str, receipt.get('shared_locks_acquired', []))) != set(map(str, range(7)))
            or receipt.get('blocked_gpus') or not receipt.get('all_locks_acquired_utc')
            or not receipt.get('admitted_utc')):
        raise ValueError('A successful all-GPU admission receipt is required.')
    for index in map(str, range(7)):
        status = statuses.get(index, {})
        if status.get('blocked_reasons') or status.get('shared_lock') != 'acquired':
            raise ValueError('All seven GPU shared locks must have been acquired at admission.')
    return {'path': str(path.resolve()), 'sha256': sha256(path),
            'admitted_utc': receipt.get('admitted_utc'),
            'expected_gpu_uuid': statuses[str(gpu)]['uuid']}


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
    ns = argparse.Namespace(seed=0, workers=4, anchor_dir=PARENT / 'checkpoints', validate_cpu=True)
    cfg.update(device=[torch.device('cuda:0')], filepath=str(OUTPUT),
               save_task_checkpoint_dir=str(OUTPUT), seed=0,
               ewc_normalize_fisher=False, ewc_normalize_accumulated_fisher=False,
               approximate_fisher_in_task=False, ewc_skip_task_fisher=False)
    cfg['task_increments'], cfg['nb_classes'] = list(manager._increments), manager.nb_classes
    runtime = runner.install_runtime(ns, cfg, OUTPUT)
    runtime['reset_rng'](0)
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
                   payload_path=str(output_path), payload_sha256=sha256(output_path),
                   ewc_normalize_fisher=False, ewc_normalize_accumulated_fisher=False,
                   fisher_mode='batch_empirical', fisher_source='fit', task=0, seed=0,
                   raw_trace=trace, tensor_count=len(raw), numel=numel, mean=trace / numel,
                   tensor_shapes={key: list(value.shape) for key, value in raw.items()},
                   stats=stats, weights_unchanged=True,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=list(map(str, range(7))))
    parser.add_argument('--gate-receipt', type=Path, default=HERE / 'campaign/ADMISSION.json')
    parser.add_argument('--validate-only', action='store_true')
    ns = parser.parse_args()
    if not ns.validate_only and ns.gpu is None:
        parser.error('GPU preparation requires --gpu and an admitted campaign gate receipt.')
    gate = gpu_uuid = None
    if ns.validate_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        gate = validate_gate(ns.gate_receipt, ns.gpu)
        gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', ns.gpu,
            '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        if gpu_uuid != gate['expected_gpu_uuid']:
            raise ValueError('Requested physical GPU differs from admission UUID.')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible.startswith('GPU-') and visible != gpu_uuid:
            raise ValueError('Queue CUDA visibility differs from the requested GPU.')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    # Local IPC sockets must not live on NFS: workers otherwise leave .nfs files.
    os.environ['TMPDIR'] = '/tmp'
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
    if not ns.validate_only and (OUTPUT / 'PREPARED.json').exists():
        previous = read_json(OUTPUT / 'PREPARED.json')
        for key in ('parent_checkpoint_sha256', 'source_manifest_sha256'):
            if previous.get(key) != provenance[key]:
                raise ValueError('Existing preparation has incompatible provenance.')
        if (previous.get('status') != 'complete' or
                previous.get('weights_unchanged') is not True or
                previous.get('ewc_normalize_fisher') is not False or
                previous.get('ewc_normalize_accumulated_fisher') is not False or
                previous.get('exact_inputs_receipt', {}).get('matched') is not True or
                previous.get('payload_sha256') != sha256(OUTPUT / 'fisher_state_raw.pt')):
            raise ValueError('Existing completed preparation failed verification.')
        print(json.dumps({'status': 'already_complete', 'prepared': str(OUTPUT / 'PREPARED.json')}))
        return
    cfg, manager, loader = build_fit_loader(torch, manifest, generator_state, iterations[-1])
    if ns.validate_only:
        verified = VerifiedInputPass(loader, iterations[-1])
        for _ in verified:
            pass
        result = dict(provenance, status='cpu_validated', exact_inputs_receipt=verified.result,
                      cuda_initialized=torch.cuda.is_initialized(),
                      model_forward_backward_executed=False, native_fisher_passes=0,
                      prepared_payload_written=False)
        if result['cuda_initialized']:
            raise RuntimeError('CPU validation must not initialize CUDA.')
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Preparation requires exactly the admitted visible GPU.')
        result = prepare(torch, cfg, manager, loader, iterations[-1], provenance, gate, gpu_uuid)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
