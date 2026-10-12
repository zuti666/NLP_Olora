#!/usr/bin/env python3
"""ImageNet-R 20+20 controlled native GAM-EWC versus EWC experiment.

Task0 and Task1 can run together or as two explicit stages. A common frozen
ViT base is restored from a local R checkpoint, while A/B and heads are freshly
initialized. Both task endpoints save complete network/Fisher state.
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
PROTOCOL = 'imagenetr_gam_ewc_1x1_v1'
SOURCE_FILES = (
    'optimer_PerturabtionType/gam.py', 'optimer_PerturabtionType/util.py',
    'models_LoRAbasedCL/as1_normfisher_gam.py',
    'models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py',
    'models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py',
    'models_LoRAbasedCL/seqlora.py', 'models_LoRAbasedCL/baseLoRA.py',
    'utils/inc_net.py', 'utils/data.py', 'utils/data_manager.py',
    'backbone/lora.py', 'backbone/linears.py', 'src/trainer.py',
)


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


def is_frozen_backbone(key):
    return key.startswith('backbone.') and not any(
        part in key for part in ('linear_a_', 'linear_b_', 'w_As.', 'w_Bs.')
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=HERE / 'config.json')
    p.add_argument('--method', choices=['ewc', 'gam_ewc'], required=True)
    p.add_argument('--stage', choices=['task0', 'task1', 'both'], default='both')
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--gpu', help='Explicit physical GPU index; no implicit queue.')
    p.add_argument('--source', type=Path, default=ORIGINAL)
    p.add_argument('--output-root', type=Path,
                   default=HERE.parents[1] / 'refine-logs/imagenetr_gam_ewc_1x1/runs')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs', type=int, help='Override BOTH task budgets, e.g. 1 for smoke.')
    p.add_argument('--print-config', action='store_true')
    p.add_argument('--validate-cpu', action='store_true',
                   help='Real model/data construction and complete initialization only; no updates.')
    return p


def build_config(ns):
    cfg = json.loads(ns.config.read_text())
    if cfg.get('dataset') != 'imagenetr' or cfg.get('class_shuffle') is not False:
        raise ValueError('This experiment requires fixed-order ImageNet-R.')
    if cfg.get('init_cls') != 20 or cfg.get('increment') != 20 or cfg.get('lora_rank') != 10:
        raise ValueError('This experiment is rank10 LoRA with 20+20 classes.')
    if cfg['momentum'] != 0 or cfg['init_momentum'] != 0 or cfg['weight_decay'] != 0 or cfg['init_weight_decay'] != 0:
        raise ValueError('Zero momentum/weight decay is required for frozen old head rows.')
    if cfg['comparison_base_optimizer'] not in ('sgd', 'adam'):
        raise ValueError('Supported base optimizers: sgd, adam.')
    if not 0 <= cfg['ewc_fisher_damping_beta'] <= 1:
        raise ValueError('Invalid damping.')
    if not cfg['ewc_normalize_fisher'] or cfg['ewc_normalize_accumulated_fisher']:
        raise ValueError('Protection protocol: per-task trace normalization, no accumulated renormalization.')
    if ns.epochs is not None:
        if ns.epochs < 1:
            raise ValueError('--epochs must be positive.')
        cfg.update(init_epoch=ns.epochs, epochs=ns.epochs)
    run_dir = ns.output_root.resolve() / ns.method / f'seed{ns.seed}'
    optimizer_type = 'gam' if ns.method == 'gam_ewc' else cfg['comparison_base_optimizer']
    cfg.update({
        'seed': [ns.seed], 'device': ['0'], 'prefix': f'{ns.method}_seed{ns.seed}_{ns.stage}',
        'source': str(ns.source.resolve()), 'comparison_method': ns.method,
        'outputs_root': str(run_dir / 'trainer_outputs' / ns.stage),
        'save_task_checkpoint_dir': str(run_dir / 'checkpoints'),
        'save_task_checkpoint_required': True, 'save_final_model': False,
        'checkpoint_require_nonempty_fisher': True, 'resume_require_fisher_state': True,
        'resume_require_experiment_identity': False, 'resume_require_model_sidecars': False,
        'optimizer_type': optimizer_type, 'optimizer_type_by_task': None,
        'optimizer': cfg['comparison_base_optimizer'], 'gam_base_optimizer': cfg['comparison_base_optimizer'],
        'optimizer_tag_override': f'{ns.method}_1x1',
        'lambda_flat': 1.0 if ns.method == 'gam_ewc' else 0.0,
        'train_num_workers': ns.workers, 'eval_num_workers': ns.workers,
        'persistent_workers': False, 'max_train_tasks': 1 if ns.stage == 'task0' else 2,
        'rq2_rng_protocol': PROTOCOL,
    })
    if ns.stage == 'task1':
        marker = run_dir / 'TASK0_COMPLETE.json'
        if not marker.is_file():
            raise FileNotFoundError(f'Task1 requires completed task0: {marker}')
        previous = json.loads(marker.read_text())
        if sha256(run_dir / 'checkpoints/task_00.pt') != previous['checkpoint_sha256']:
            raise ValueError('Task0 checkpoint changed after its completion record.')
        saved_cfg = json.loads((run_dir / 'resolved_task0.json').read_text())
        ignored = {'prefix', 'outputs_root', 'max_train_tasks'}
        differences = [key for key in cfg if key not in ignored and saved_cfg.get(key) != cfg[key]]
        if differences:
            raise ValueError(f'Task1 config differs from its own Task0: {differences}')
        cfg.update(resume_from_task=0,
                   resume_from_checkpoint_dir=str(run_dir / 'checkpoints'),
                   resume_from_metrics_json=previous['metrics'])
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
    reference_path = Path(cfg['frozen_backbone_checkpoint'])
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
        from learner import Learner
        model = Learner(args)
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
            torch.save({k: v.detach().cpu().clone() for k, v in network.fc.state_dict().items()},
                       run_dir / f'head_initial_task_{context["task"]:02d}.pt')
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
        state_now = model._unwrap_network().state_dict()
        if any(not torch.equal(state_now[key].cpu(), value) for key, value in frozen.items()):
            raise RuntimeError('Frozen backbone changed during training.')
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
    if ns.seed < 0 or ns.workers < 1:
        raise ValueError('Nonnegative seed and >=1 worker required for separate augmentation RNG.')
    ns.source = ns.source.resolve()
    cfg, run_dir = build_config(ns)
    training_run_dir = run_dir
    if ns.print_config:
        print(json.dumps({'run_dir': str(run_dir), 'stage': ns.stage, 'config': cfg}, indent=2))
        return
    if not Path(cfg['frozen_backbone_checkpoint']).is_file():
        raise FileNotFoundError(cfg['frozen_backbone_checkpoint'])
    if ns.validate_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        gpu_uuid = None
        run_dir = ns.output_root.resolve() / 'cpu_validation' / ns.method / f'seed{ns.seed}'
    else:
        if ns.gpu is None or not ns.gpu.isdigit():
            raise ValueError('Training requires an explicit physical --gpu index.')
        gpu_uuid = subprocess.check_output([
            'nvidia-smi', '-i', ns.gpu, '--query-gpu=uuid', '--format=csv,noheader'
        ], text=True).strip()
        if not gpu_uuid.startswith('GPU-') or '\n' in gpu_uuid:
            raise RuntimeError(f'Invalid GPU UUID: {gpu_uuid!r}')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['FLATNESS_CV_DATA_ROOT'] = cfg['data_root']
    sys.path.insert(0, str(ns.source))
    # Keep this experiment's learner.py ahead of upstream packages.
    sys.path.insert(0, str(HERE))
    if not ns.validate_cpu:
        stage_manifest = run_dir / f'run_manifest_{ns.stage}.json'
        if stage_manifest.exists() or (run_dir / 'COMPLETE.json').exists():
            raise FileExistsError(f'Existing run/stage: {run_dir}; choose a new output root for a rerun.')
        if ns.stage != 'task1' and (run_dir / 'checkpoints').exists():
            raise FileExistsError(f'Checkpoints already exist: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    import torch
    import timm
    import torchvision
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manifest = {'protocol': PROTOCOL, 'method': ns.method, 'stage': ns.stage,
                'seed': ns.seed, 'source': str(ns.source),
                'physical_gpu': ns.gpu if not ns.validate_cpu else None, 'gpu_uuid': gpu_uuid,
                'torch': torch.__version__, 'torchvision': torchvision.__version__, 'timm': timm.__version__,
                'precision': 'FP32; TF32 disabled', 'config': cfg,
                'class_order': list(range(200)), 'start_unix': time.time(),
                'scope': 'CPU construction only' if ns.validate_cpu else 'training',
                'local_sources': {f.name: sha256(f) for f in HERE.glob('*.py')},
                'training_source_sha256': {name: sha256(ns.source / name) for name in SOURCE_FILES},
                'reference_checkpoint': cfg['frozen_backbone_checkpoint']}
    if ns.stage == 'task1':
        original_manifest = json.loads((training_run_dir / 'run_manifest_task0.json').read_text())
        if original_manifest['training_source_sha256'] != manifest['training_source_sha256']:
            raise ValueError('Upstream training source changed since Task0.')
        for name in ('train.py', 'learner.py'):
            if original_manifest['local_sources'][name] != manifest['local_sources'][name]:
                raise ValueError(f'{name} changed since Task0.')
    if not ns.validate_cpu:
        write_json(run_dir / 'run_manifest.json', manifest)
        write_json(run_dir / f'run_manifest_{ns.stage}.json', manifest)
        write_json(run_dir / f'resolved_{ns.stage}.json', cfg)
    runtime = install_runtime(ns, cfg, run_dir)
    if ns.validate_cpu:
        cpu_cfg = copy.deepcopy(cfg)
        cpu_cfg.update(seed=ns.seed, device=[torch.device('cpu')], filepath=str(run_dir / 'checkpoints') + '/')
        Path(cpu_cfg['filepath']).mkdir(parents=True, exist_ok=True)
        runtime['reset_rng'](ns.seed)
        manager = runtime['DataManager']('imagenetr', False, ns.seed, 20, 20, cpu_cfg)
        cpu_cfg['task_increments'] = list(manager._increments)
        cpu_cfg['nb_classes'] = manager.nb_classes
        model = runtime['factory'].get_model(cpu_cfg['model_name'], cpu_cfg)
        if ns.stage == 'task1':
            model.load_task_checkpoint(cpu_cfg['resume_from_checkpoint_dir'], 0, manager)
        next_task = 1 if ns.stage == 'task1' else 0
        runtime['context']['task'] = next_task
        runtime['reset_rng'](task_seed(ns.seed, next_task))
        model._unwrap_network().update_fc(20 * (next_task + 1))
        result = {'status': 'passed', 'scope': 'CPU model/data construction and initialization',
                  'forward_tested': False, 'training_started': False,
                  'initial_network_sha256': tensor_hash(model._unwrap_network().state_dict().items()),
                  'new_head_sha256': tensor_hash((k, v[next_task * 20:])
                                                for k, v in model._unwrap_network().fc.state_dict().items()),
                  'fisher_tensors': len(model._fisher_past_delta), 'config': cfg}
        write_json(run_dir / 'CPU_VALIDATED.json', result)
        print(json.dumps(result), flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one GPU must be visible.')
    # All incidental outputs stay in this run, never in the reference repository.
    os.chdir(run_dir)
    from src.trainer import train
    try:
        train(cfg)
        metrics = sorted((run_dir / 'trainer_outputs' / ns.stage).rglob('*_cl_metrics.json'))
        if len(metrics) != 1:
            raise RuntimeError(f'Expected one stage metrics JSON, found {metrics}')
        expected = 0 if ns.stage == 'task0' else 1
        ckpt = run_dir / 'checkpoints' / f'task_{expected:02d}.pt'
        if not ckpt.is_file():
            raise RuntimeError(f'Missing final full checkpoint: {ckpt}')
        marker = 'TASK0_COMPLETE.json' if ns.stage == 'task0' else 'COMPLETE.json'
        write_json(run_dir / marker, {'status': 'complete', 'completed_tasks': expected + 1,
                                     'time_unix': time.time(), 'metrics': str(metrics[0]),
                                     'checkpoint': str(ckpt), 'checkpoint_sha256': sha256(ckpt),
                                     'stage': ns.stage})
    except BaseException as exc:
        write_json(run_dir / f'FAILED_{ns.stage}.json', {'error': repr(exc), 'time_unix': time.time()})
        raise


if __name__ == '__main__':
    main()
