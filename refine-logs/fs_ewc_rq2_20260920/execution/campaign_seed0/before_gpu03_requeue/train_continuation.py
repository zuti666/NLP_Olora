#!/usr/bin/env python3
"""RQ2 paired continuation / additional-seed training using frozen CopyFlatCV.

The historical learner implements every optimizer update. Runtime adapters only
make pretrained loading, complete resume, RNG pairing, and receipts explicit.
Nothing is written to the original CopyFlatCV tree or frozen source snapshot.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ORIGINAL = Path('/data/140-1/users/liying/CopyFlatCV')
ANCHORS = ORIGINAL / 'outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50'
PROTOCOL = 'rq2_shared_task_rng_v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + '\n')
    temporary.replace(path)


def append_json(path, payload):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=str) + '\n')


def task_seed(seed, task):
    return 20260920 + 100003 * int(seed) + 1009 * int(task)


def anchor_assets(method):
    manifest = ANCHORS / f'manifests/{method}_seed0.env'
    values = dict(line.split('=', 1) for line in manifest.read_text().splitlines() if '=' in line)
    return Path(values['S1_CKPT_DIR']), Path(values['S1_METRICS_JSON'])


def is_frozen_backbone(key):
    return key.startswith('backbone.') and not any(
        part in key for part in ('linear_a_', 'linear_b_', 'w_As.', 'w_Bs.')
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['continuation', 'sequence'], required=True)
    p.add_argument('--method', choices=['sgd', 'gam'], required=True,
                   help='Anchor method for continuation; training method for sequence.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', choices=['4', '6'])
    p.add_argument('--source', type=Path, default=HERE / 'source')
    p.add_argument('--output-root', type=Path, default=HERE / 'training')
    p.add_argument('--anchor-dir', type=Path,
                   help='Optional Task0 directory, for later-seed A branches.')
    p.add_argument('--anchor-metrics', type=Path)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--resume-last', action='store_true')
    p.add_argument('--print-config', action='store_true')
    p.add_argument('--validate-cpu', action='store_true',
                   help='Build and restore model on CPU; no forward, gradients, or training.')
    return p


def build_config(ns):
    import yaml
    base = ns.source / 'config_Expand/sgd_gam_s1c/base_s1c_normaccum_e50_initlr004_lr002.yaml'
    cfg = yaml.safe_load(base.read_text())
    label = (f'A_{ns.method}_to_sgd_seed{ns.seed}' if ns.mode == 'continuation'
             else f'sequence_{ns.method}_seed{ns.seed}')
    run_dir = ns.output_root.resolve() / label
    method = 'sgd' if ns.mode == 'continuation' else ns.method
    cfg.update({
        'seed': [ns.seed], 'device': ['0'], 'prefix': label,
        'outputs_root': str(run_dir / 'trainer_outputs'),
        'save_task_checkpoint_dir': str(run_dir / 'checkpoints'),
        'save_task_checkpoint_required': True, 'save_final_model': True,
        'checkpoint_require_nonempty_fisher': True, 'resume_require_fisher_state': True,
        'resume_require_experiment_identity': False, 'resume_require_model_sidecars': False,
        'optimizer_type': method, 'optimizer_type_by_task': None,
        'optimizer_tag_override': label, 'lambda_flat': 1.0 if method == 'gam' else 0.0,
        'batch_size': 128, 'init_epoch': 50, 'epochs': 50,
        'init_lr': .04, 'lrate': .02, 'momentum': 0, 'init_momentum': 0,
        'weight_decay': 0., 'init_weight_decay': 0., 'scheduler': 'cosine',
        'ewc_lambda': 2000., 'ewc_gamma': .9, 'ewc_eta': 0.,
        'ewc_max_batches': 100, 'fisher_mode': 'batch_empirical',
        'ewc_fisher_source': 'fit', 'fisher_split': False,
        'ewc_normalize_fisher': True, 'ewc_normalize_accumulated_fisher': True,
        'ewc_dual_ascent': False, 'ewc_project_grad': False,
        'approximate_fisher_in_task': False, 'weight_merge': False,
        'in_task_weight_merge': False, 'endpoint_weight_merge': False,
        'delta_w_penalty': False, 'gam_base_optimizer': 'sgd',
        'gam_adaptive': False, 'gam_grad_rho': .05, 'gam_grad_norm_rho': .2,
        'as1_rho': .05, 'as1_norm_rho': .2,
        'gam_grad_gamma': .1, 'gam_grad_beta_1': 1., 'gam_grad_beta_2': 1.,
        'gam_grad_beta_3': 1., 'data_root': '/data/115-2/users/liying/datasets',
        'train_num_workers': ns.workers, 'eval_num_workers': ns.workers,
        'persistent_workers': False, 'compute_nme_class_means': False,
        'mechanism_eval': True, 'mechanism_grad_stats': True,
        'compute_full_fisher': False, 'endpoint_eval': False,
        'task0_trajectory_eval': False, 'task0_fisher_recompute_eval': False,
        'flat_eval': False, 'probe_flatness_eval': False, 'multi_fisher_eval': False,
        'feature_drift_mode': 'none', 'feature_drift_eval': False,
        'max_train_tasks': 2 if ns.mode == 'continuation' else 10,
        'rq2_rng_protocol': PROTOCOL,
    })
    if ns.mode == 'continuation':
        if ns.anchor_dir is None:
            if ns.seed != 0:
                raise ValueError('Nonzero-seed continuation requires --anchor-dir and --anchor-metrics.')
            anchor_dir, anchor_metrics = anchor_assets(ns.method)
        else:
            if ns.anchor_metrics is None:
                raise ValueError('--anchor-dir requires --anchor-metrics.')
            anchor_dir, anchor_metrics = ns.anchor_dir.resolve(), ns.anchor_metrics.resolve()
        cfg.update(resume_from_task=0, resume_from_checkpoint_dir=str(anchor_dir),
                   resume_from_metrics_json=str(anchor_metrics))
    return cfg, run_dir


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def install_runtime(ns, cfg, run_dir):
    import numpy as np
    import torch
    import timm

    # Timm's mutable default alias must never select/download another pretrained
    # state. Reuse the audited frozen base in the complete historical manifest.
    reference_dir, _ = anchor_assets('sgd')
    reference_path = reference_dir / 'task_00.pt'
    reference = torch.load(reference_path, map_location='cpu', weights_only=True,
                           mmap=True)['network_state']
    frozen = {key: value for key, value in reference.items() if is_frozen_backbone(key)}
    if len(frozen) != 300:
        raise ValueError(f'Unexpected frozen-backbone state count: {len(frozen)}')
    original_create = timm.create_model

    def local_create_model(model_name, *a, **kw):
        if model_name != 'vit_base_patch16_224':
            raise ValueError(f'Unexpected model constructor {model_name!r}')
        kw['pretrained'] = False
        return original_create(model_name, *a, **kw)

    timm.create_model = local_create_model
    from utils import factory
    from utils import data_manager as dm_module
    from utils.inc_net import IncrementalNet
    from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
    from models_LoRAbasedCL.seqlora import Learner as SeqLearner
    original_factory = factory.get_model
    original_load = LoraBaseLearner.load_task_checkpoint
    original_save = LoraBaseLearner.save_task_checkpoint
    original_incremental = SeqLearner.incremental_train
    original_update_fc = IncrementalNet.update_fc
    original_build_loader = dm_module.DataManager.build_dataloader
    original_loader_class = dm_module.DataLoader
    context = {'task': -1, 'loader_serial': 0, 'learner': None, 'anchor_delta': None,
               'anchor_head': None}

    def reset_rng(seed):
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def capture_rng():
        return {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    @contextlib.contextmanager
    def preserve_rng():
        state = capture_rng()
        try:
            yield
        finally:
            random.setstate(state['python'])
            np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch_cpu'])
            if state['torch_cuda'] is not None:
                torch.cuda.set_rng_state_all(state['torch_cuda'])

    def patched_factory(name, args):
        model = original_factory(name, args)
        net = model._unwrap_network()
        current = net.state_dict()
        expected = {key for key in current if is_frozen_backbone(key)}
        if expected != set(frozen):
            raise ValueError('Frozen-backbone keys differ from audited historical architecture.')
        current.update(frozen)
        net.load_state_dict(current, strict=True)
        if not all(torch.equal(net.state_dict()[key].cpu(), val) for key, val in frozen.items()):
            raise ValueError('Frozen backbone restoration failed.')
        context['learner'] = model
        append_json(run_dir / 'runtime_receipts.jsonl', {
            'event': 'frozen_backbone_loaded', 'reference': str(reference_path),
            'frozen_state_sha256': tensor_hash(frozen.items()), 'frozen_keys': len(frozen),
            'new_adapter_state_sha256': tensor_hash((k, v) for k, v in current.items() if not is_frozen_backbone(k)),
        })
        return model

    def remember_anchor(model):
        net = model._unwrap_network()
        context['anchor_delta'] = {k: value.detach().cpu().clone()
                                   for k, value in model._iter_delta_terms(net)}
        context['anchor_head'] = {k: value[:20].detach().cpu().clone()
                                  for k, value in net.fc.state_dict().items()}

    def patched_load(model, checkpoint_dir, task_idx, data_manager):
        # Preserve lifecycle/Fisher sidecar logic, then override *every* weight.
        original_load(model, checkpoint_dir, task_idx, data_manager)
        checkpoint = Path(checkpoint_dir) / f'task_{task_idx:02d}.pt'
        full = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        net = model._unwrap_network()
        net.load_state_dict(full['network_state'], strict=True)
        if not all(torch.equal(net.state_dict()[key].cpu(), value)
                   for key, value in full['network_state'].items()):
            raise ValueError('Complete manifest resume did not restore every parameter.')
        model._num_finished_tasks = int(task_idx) + 1
        model._data_manager = model.data_manager = data_manager
        append_json(run_dir / 'runtime_receipts.jsonl', {
            'event': 'complete_resume', 'task': task_idx, 'checkpoint': str(checkpoint),
            'checkpoint_sha256': sha256(checkpoint),
            'fisher_sha256': sha256(Path(checkpoint_dir) / f'fisher_state_{task_idx}.pt'),
            'state_keys': len(full['network_state']), 'verified_exact': True,
        })
        if task_idx == 0:
            remember_anchor(model)
            if not ns.validate_cpu:
                evaluate_old_task(model, task_idx)
        else:
            anchor_path = run_dir / 'checkpoints/task_00.pt'
            anchor_state = torch.load(anchor_path, map_location='cpu', weights_only=True,
                                     mmap=True)['network_state']
            # Shapes beyond the old head differ; only old-head prefix is needed.
            delta = {}
            actual_keys = [key for key, _ in model._iter_delta_terms(net)]
            for block in range(12):
                for role in ('q', 'v'):
                    prefix = f'backbone.lora_vit.blocks.{block}.attn.qkv.'
                    # Use actual learner keys instead of assuming their spelling.
                    actual_key = actual_keys[2 * block + (role == 'v')]
                    delta[actual_key] = (anchor_state[prefix + f'linear_b_{role}.weight'] @
                                         anchor_state[prefix + f'linear_a_{role}.weight'])
            context['anchor_delta'] = delta
            context['anchor_head'] = {k: anchor_state['fc.' + k][:20].clone() for k in net.fc.state_dict()}

    def patched_update_fc(network, nb_classes):
        old_classes = 0 if network.fc is None else network.fc.out_features
        result = original_update_fc(network, nb_classes)
        if context['task'] >= 0:
            append_json(run_dir / 'rng_receipts.jsonl', {
                'event': 'head_initialized', 'task': context['task'],
                'old_classes': old_classes, 'total_classes': nb_classes,
                'task_seed': task_seed(ns.seed, context['task']),
                'new_head_sha256': tensor_hash((k, v[old_classes:]) for k, v in network.fc.state_dict().items()),
            })
        return result

    class AuditedLoader(original_loader_class):
        def __iter__(self):
            receipt = getattr(self, '_rq2_receipt', None)
            if receipt is None or not receipt['shuffle']:
                yield from super().__iter__()
                return
            iteration = getattr(self, '_rq2_iterations', 0)
            self._rq2_iterations = iteration + 1
            sample_digest = hashlib.sha256()
            first_pixels = None
            batches = samples = 0
            try:
                for batch in super().__iter__():
                    idx, inputs, targets = batch
                    sample_digest.update(idx.detach().cpu().numpy().tobytes())
                    if first_pixels is None:
                        first_pixels = tensor_hash([('input', inputs), ('target', targets)])
                    batches += 1
                    samples += len(targets)
                    yield batch
            finally:
                append_json(run_dir / 'rng_receipts.jsonl', {
                    **receipt, 'event': 'loader_iteration', 'iteration': iteration,
                    'batches': batches, 'samples': samples,
                    'indices_sha256': sample_digest.hexdigest(),
                    'first_batch_pixels_targets_sha256': first_pixels,
                })

    def patched_build_loader(manager, dataset, *, batch_size, shuffle, num_workers, **kwargs):
        serial = context['loader_serial']
        context['loader_serial'] += 1
        seed = task_seed(ns.seed, max(0, context['task'])) + 10000019 + 7919 * serial
        kwargs['generator'] = torch.Generator().manual_seed(seed)
        kwargs['persistent_workers'] = False
        loader = original_build_loader(manager, dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, **kwargs)
        ids = list(zip(map(str, getattr(dataset, 'images', [])),
                       map(int, getattr(dataset, 'labels', []))))
        loader._rq2_receipt = {'task': context['task'], 'loader_serial': serial,
                               'shuffle': bool(shuffle), 'generator_seed': seed,
                               'num_workers': num_workers, 'sample_count': len(dataset),
                               'dataset_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        append_json(run_dir / 'rng_receipts.jsonl', {**loader._rq2_receipt, 'event': 'loader_created'})
        return loader

    def patched_incremental(model, data_manager):
        task = model._cur_task + 1
        context.update(task=task, loader_serial=0)
        reset_rng(task_seed(ns.seed, task))
        append_json(run_dir / 'rng_receipts.jsonl', {
            'event': 'task_rng_reset', 'task': task, 'task_seed': task_seed(ns.seed, task),
            'rng_protocol': PROTOCOL, 'optimizer_type': cfg['optimizer_type'],
        })
        return original_incremental(model, data_manager)

    def evaluate_old_task(model, task):
        if context['anchor_delta'] is None:
            return
        net = model._unwrap_network()
        for name, value in context['anchor_head'].items():
            if not torch.equal(net.fc.state_dict()[name][:20].cpu(), value):
                raise ValueError(f'Old Task0 head changed at task {task}.')
        device = model._device
        record = {'task': task, 'coordinate': 'effective_QV_BA', 'old_local_head': [0, 20],
                  'train_transform': 'deterministic_test', 'trace_measured': False}
        with torch.no_grad():
            record['effective_weight_displacement'] = float(sum(
                (value.double() - context['anchor_delta'][key].to(device).double()).square().sum()
                for key, value in model._iter_delta_terms(net)).sqrt().item())
        was_training = net.training
        with preserve_rng(), torch.no_grad():
            net.eval()
            for split in ('train', 'test'):
                dataset = model._data_manager.get_dataset(np.arange(20), source=split, mode='test')
                loader = original_loader_class(dataset, batch_size=64, shuffle=False,
                                               num_workers=ns.workers)
                total = local_correct = seen_correct = 0
                local_loss = seen_loss = 0.
                for _, images, targets in loader:
                    targets = targets.to(device)
                    logits = net(images.to(device))['logits']
                    local_logits = logits[:, :20]
                    local_loss += torch.nn.functional.cross_entropy(local_logits, targets, reduction='none').double().sum().item()
                    seen_loss += torch.nn.functional.cross_entropy(logits, targets, reduction='none').double().sum().item()
                    local_correct += int((local_logits.argmax(1) == targets).sum().item())
                    seen_correct += int((logits.argmax(1) == targets).sum().item())
                    total += len(targets)
                record[split] = {'n': total, 'local_ce': local_loss / total,
                                 'seen_ce': seen_loss / total, 'local_accuracy': 100 * local_correct / total,
                                 'seen_accuracy': 100 * seen_correct / total}
            net.train(was_training)
        append_json(run_dir / 'task0_local_metrics.jsonl', record)

    def patched_save(model, ckpt_dir, task_idx):
        for name, parameter in model._unwrap_network().named_parameters():
            if parameter.requires_grad and not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f'Non-finite trained parameter at task {task_idx}: {name}')
        original_save(model, ckpt_dir, task_idx)
        torch.save({'rng_protocol': PROTOCOL, 'seed': ns.seed, 'task': task_idx,
                    'state': capture_rng()}, Path(ckpt_dir) / f'rng_state_{task_idx:02d}.pt')
        if task_idx == 0:
            remember_anchor(model)
        evaluate_old_task(model, task_idx)
        append_json(run_dir / 'runtime_receipts.jsonl', {
            'event': 'task_complete', 'task': task_idx, 'time_unix': time.time(),
            'checkpoint': str(Path(ckpt_dir) / f'task_{task_idx:02d}.pt'),
        })

    factory.get_model = patched_factory
    LoraBaseLearner.load_task_checkpoint = patched_load
    LoraBaseLearner.save_task_checkpoint = patched_save
    SeqLearner.incremental_train = patched_incremental
    IncrementalNet.update_fc = patched_update_fc
    dm_module.DataLoader = AuditedLoader
    dm_module.DataManager.build_dataloader = patched_build_loader
    return {'factory': factory, 'DataManager': dm_module.DataManager,
            'reset_rng': reset_rng, 'context': context, 'reference_path': reference_path}


def main():
    ns = parser().parse_args()
    if ns.seed not in (0, 1, 2):
        raise ValueError('This protocol only defines seeds 0, 1, and 2.')
    if ns.workers < 1:
        raise ValueError('At least one worker is required to isolate augmentation RNG from model RNG.')
    ns.source = ns.source.resolve()
    sys.path.insert(0, str(ns.source))
    cfg, run_dir = build_config(ns)
    if ns.print_config:
        print(json.dumps({'run_dir': str(run_dir), 'config': cfg}, indent=2))
        return
    if ns.validate_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        gpu_uuid = None
        run_dir = ns.output_root.resolve() / 'cpu_validation' / run_dir.name
    else:
        if ns.gpu is None:
            raise ValueError('Training requires --gpu 4 or --gpu 6.')
        gpu_uuid = subprocess.check_output([
            'nvidia-smi', '-i', ns.gpu, '--query-gpu=uuid', '--format=csv,noheader'
        ], text=True).strip()
        if not gpu_uuid.startswith('GPU-') or '\n' in gpu_uuid:
            raise RuntimeError(f'Cannot resolve requested physical GPU {ns.gpu}: {gpu_uuid!r}')
        current_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if current_visible.startswith('GPU-') and current_visible != gpu_uuid:
            raise RuntimeError(f'Queue GPU UUID {current_visible} differs from physical GPU {ns.gpu}: {gpu_uuid}')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    os.environ.setdefault('TMPDIR', '/tmp')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ['PYTHONHASHSEED'] = str(ns.seed)
    os.environ['FLATNESS_CV_DATA_ROOT'] = cfg['data_root']
    if not ns.validate_cpu and (run_dir / 'COMPLETE.json').exists():
        print(f'Already complete: {run_dir}', flush=True)
        return
    if not ns.validate_cpu and (run_dir / 'run_manifest.json').exists() and not ns.resume_last:
        raise FileExistsError(f'Run already exists: {run_dir}; use --resume-last for task-boundary resume.')
    run_dir.mkdir(parents=True, exist_ok=True)
    if ns.resume_last and not ns.validate_cpu:
        saved = sorted((run_dir / 'checkpoints').glob('task_??.pt'))
        if saved:
            latest = int(saved[-1].stem.split('_')[1])
            metrics = sorted((run_dir / 'trainer_outputs').rglob('*_cl_metrics.json'))
            if len(metrics) != 1:
                raise ValueError(f'Expected one metrics JSON for resume, found {metrics}')
            if latest + 1 >= cfg['max_train_tasks']:
                # A task checkpoint alone does not establish all metrics are saved.
                events = [json.loads(line) for line in (run_dir / 'runtime_receipts.jsonl').read_text().splitlines()]
                if any(e.get('event') == 'task_complete' and e.get('task') == latest for e in events):
                    write_json(run_dir / 'COMPLETE.json', {'status': 'complete', 'completed_tasks': latest + 1,
                                                          'recovered_after_final_task': True})
                    return
                raise RuntimeError('Final checkpoint exists but final receipt is absent; inspect partial run.')
            cfg.update(resume_from_task=latest, resume_from_checkpoint_dir=str(run_dir / 'checkpoints'),
                       resume_from_metrics_json=str(metrics[0]))
    import torch
    import timm
    import torchvision
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = {'protocol': PROTOCOL, 'mode': ns.mode, 'anchor_method': ns.method,
                'physical_gpu': ns.gpu if not ns.validate_cpu else None, 'seed': ns.seed,
                'gpu_uuid': gpu_uuid,
                'source': str(ns.source), 'source_manifest_sha256': sha256(HERE / 'source_frozen.json'),
                'runner_sha256': sha256(__file__), 'torch': torch.__version__,
                'torchvision': torchvision.__version__, 'timm': timm.__version__,
                'production_precision': 'FP32; CUDA matmul and cuDNN TF32 explicitly disabled',
                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                'config': cfg, 'start_unix': time.time(),
                'scope': 'CPU restoration only' if ns.validate_cpu else 'training',
                'history_compatibility': 'Same training hyperparameters; new explicit paired RNG protocol.'}
    write_json(run_dir / ('cpu_manifest.json' if ns.validate_cpu else 'run_manifest.json'), manifest)
    runtime = install_runtime(ns, cfg, run_dir)
    if ns.validate_cpu:
        cpu_cfg = copy.deepcopy(cfg)
        cpu_cfg.update(seed=ns.seed, device=[torch.device('cpu')], filepath=str(run_dir / 'checkpoints'))
        Path(cpu_cfg['filepath']).mkdir(parents=True, exist_ok=True)
        runtime['reset_rng'](ns.seed)
        manager = runtime['DataManager']('imageneta', False, ns.seed, 20, 20, cpu_cfg)
        cpu_cfg['task_increments'] = list(manager._increments)
        cpu_cfg['nb_classes'] = manager.nb_classes
        model = runtime['factory'].get_model(cpu_cfg['model_name'], cpu_cfg)
        if ns.mode == 'continuation':
            model.load_task_checkpoint(cpu_cfg['resume_from_checkpoint_dir'], 0, manager)
        next_task = 1 if ns.mode == 'continuation' else 0
        runtime['context']['task'] = next_task
        runtime['reset_rng'](task_seed(ns.seed, next_task))
        model._unwrap_network().update_fc(20 * (next_task + 1))
        result = {'status': 'passed', 'scope': 'CPU model construction/full-state/Fisher restoration/new-head initialization',
                  'forward_tested': False, 'training_started': False,
                  'new_head_sha256': tensor_hash((k, v[next_task * 20:])
                                                for k, v in model._unwrap_network().fc.state_dict().items()),
                  'fisher_tensors': len(model._fisher_past_delta), 'run_dir': str(run_dir)}
        write_json(run_dir / 'CPU_VALIDATED.json', result)
        print(json.dumps(result), flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one authorized GPU must be visible.')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from src.trainer import train
    try:
        train(cfg)
    except BaseException as exc:
        write_json(run_dir / 'FAILED.json', {'error': repr(exc), 'time_unix': time.time()})
        raise
    write_json(run_dir / 'COMPLETE.json', {'status': 'complete', 'completed_tasks': cfg['max_train_tasks'],
                                          'time_unix': time.time(), 'manifest': str(run_dir / 'run_manifest.json')})


if __name__ == '__main__':
    main()
