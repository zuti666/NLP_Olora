#!/usr/bin/env python3
"""ImageNet-A first two tasks: two shared anchors and six continuation rules."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = Path('/data/140-1/users/liying/CopyFlatCV')
DEFAULT_RUN = ROOT / 'refine-logs/imageneta_two_task_table_20260920/runs'
RULES = {
    'vanilla': ('sgd', 'sgd', False),
    'shaping_only': ('gam', 'gam', False),
    'ewc': ('sgd', 'sgd', True),
    'gam_ewc': ('gam', 'gam', True),
    'task0_shaping': ('gam', 'sgd', True),
    'task1_shaping': ('sgd', 'gam', True),
}
spec = importlib.util.spec_from_file_location('shared_training_runtime', HERE.parent / 'imagenetr_gam_ewc_1x1/train.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
runtime.HERE = HERE
runtime.PROTOCOL = 'imageneta_two_task_table_weighted_merge_v1'
runtime.SOURCE_FILES += (
    'models_LoRAbasedCL/as1_normfisher_gam_intask.py',
    'models_LoRAbasedCL/as1_normfisher_gam_intask_damp.py',
    'models_LoRAbasedCL/as1_normfisher_gam_intask_damp_weightmergethenfisher.py',
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['task0', 'task1'], required=True)
    p.add_argument('--method', choices=['sgd', 'gam', *RULES], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', help='Physical GPU index, scheduler checks idleness.')
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--config', type=Path, default=HERE / 'config.json')
    p.add_argument('--output-root', type=Path, default=DEFAULT_RUN)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--smoke', action='store_true', help='3 epochs, snapshots 1/2; separate root required.')
    p.add_argument('--validate-cpu', action='store_true')
    p.add_argument('--print-config', action='store_true')
    return p


def resolve(ns):
    cfg = json.loads(ns.config.read_text())
    if cfg['dataset'] != 'imageneta' or cfg['class_shuffle'] or cfg['init_cls'] != 20 or cfg['increment'] != 20:
        raise ValueError('Fixed ImageNet-A 10x20 protocol required; execute only its first two tasks.')
    if ns.stage == 'task0':
        if ns.method not in ('sgd', 'gam'):
            raise ValueError('Task0 method must be sgd or gam.')
        first = current = ns.method
        protected = False
        run_dir = ns.output_root.resolve() / 'anchors' / first / f'seed{ns.seed}'
        anchor = None
    else:
        if ns.method not in RULES:
            raise ValueError('Task1 method must name one of the six table variants.')
        first, current, protected = RULES[ns.method]
        run_dir = ns.output_root.resolve() / ns.method / f'seed{ns.seed}'
        anchor = ns.output_root.resolve() / 'anchors' / first / f'seed{ns.seed}'
    if ns.smoke:
        cfg.update(init_epoch=3, epochs=3, snapshot_epochs=[1, 2],
                   in_task_weight_checkpoint_interval=1, in_task_fisher_interval=1)
    if cfg.get('trajectory_merge_policy') not in ('all', 'gam_only'):
        raise ValueError('Explicit trajectory merge policy required.')
    cfg.update({
        'seed': [ns.seed], 'device': ['0'], 'prefix': f'{ns.method}_s{ns.seed}_{ns.stage}',
        'source': str(ns.source.resolve()), 'comparison_method': ns.method,
        'table_run_dir': str(run_dir), 'stage0_optimizer': first,
        'stage1_optimizer': current if ns.stage == 'task1' else None,
        'optimizer_type': current, 'optimizer_type_by_task': None,
        'optimizer': 'sgd', 'gam_base_optimizer': 'sgd', 'comparison_base_optimizer': 'sgd',
        'optimizer_tag_override': f'table_{ns.method}', 'lambda_flat': 1. if current == 'gam' else 0.,
        'ewc_lambda': 2000. if protected else 0.,
        'outputs_root': str(run_dir / 'trainer_outputs'),
        'save_task_checkpoint_dir': str(run_dir / 'checkpoints'),
        'save_task_checkpoint_required': True, 'save_final_model': False,
        'checkpoint_require_nonempty_fisher': True, 'resume_require_fisher_state': True,
        'resume_require_experiment_identity': False, 'resume_require_model_sidecars': False,
        'train_num_workers': ns.workers, 'eval_num_workers': ns.workers,
        'persistent_workers': False, 'max_train_tasks': 1 if ns.stage == 'task0' else 2,
        'rq2_rng_protocol': runtime.PROTOCOL,
        'in_task_weight_merge': cfg['trajectory_merge_policy'] == 'all' or current == 'gam',
    })
    if ns.stage == 'task1':
        marker_path = anchor / 'TASK0_COMPLETE.json'
        marker = json.loads(marker_path.read_text())
        if marker['status'] != 'complete' or runtime.sha256(anchor / 'checkpoints/task_00.pt') != marker['checkpoint_sha256']:
            raise ValueError('Anchor is incomplete or changed.')
        anchor_cfg = json.loads((anchor / 'run_manifest.json').read_text())['config']
        shared = ['dataset', 'class_shuffle', 'lora_rank', 'init_lr', 'lrate', 'init_epoch', 'epochs',
                  'batch_size', 'ewc_gamma', 'ewc_normalize_fisher', 'ewc_normalize_accumulated_fisher',
                  'ewc_fisher_damping_alpha', 'snapshot_epochs', 'trajectory_merge_policy',
                  'in_task_path_beta', 'in_task_weight_merge_omega', 'frozen_backbone_checkpoint']
        if any(cfg[k] != anchor_cfg[k] for k in shared):
            raise ValueError('Branch and anchor have inconsistent training protocols.')
        cfg.update(resume_from_task=0, resume_from_checkpoint_dir=str(run_dir / 'checkpoints'),
                   resume_from_metrics_json=marker['metrics'])
    return cfg, run_dir, anchor


def copy_anchor(anchor, run_dir):
    # Hard links save space; completed Task0 files are read-only to continuation.
    # A branch writes different task_01 files and its own manifests.
    for name in ('checkpoints', 'trajectory'):
        shutil.copytree(anchor / name, run_dir / name, copy_function=os.link)
    for name in ('rng_receipts.jsonl', 'head_initial_task_00.pt', 'runtime_receipts.jsonl'):
        if (anchor / name).is_file():
            shutil.copyfile(anchor / name, run_dir / name)
    runtime.write_json(run_dir / 'ANCHOR_SOURCE.json', {'anchor_dir': str(anchor),
                       'checkpoint_sha256': runtime.sha256(anchor / 'checkpoints/task_00.pt')})


def main():
    ns = parser().parse_args()
    if ns.seed < 0 or ns.workers < 1:
        raise ValueError('Nonnegative seed and >=1 DataLoader worker required.')
    ns.source = ns.source.resolve()
    cfg, run_dir, anchor = resolve(ns)
    if ns.print_config:
        print(json.dumps({'run_dir': str(run_dir), 'anchor': str(anchor), 'config': cfg}, indent=2))
        return
    if ns.validate_cpu:
        run_dir = ns.output_root.resolve() / 'cpu_validation' / ns.method / f'seed{ns.seed}'
        cfg['table_run_dir'] = str(run_dir)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        gpu_uuid = None
    else:
        if not ns.gpu or not ns.gpu.isdigit():
            raise ValueError('An explicit physical --gpu index is required.')
        if run_dir.exists():
            raise FileExistsError(f'Run already exists: {run_dir}; preserve it and choose a new root.')
        gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', ns.gpu, '--query-gpu=uuid',
                                            '--format=csv,noheader'], text=True).strip()
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      FLATNESS_CV_DATA_ROOT=cfg['data_root'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    sys.path[:0] = [str(HERE), str(ns.source), str(ROOT)]
    run_dir.mkdir(parents=True, exist_ok=True)
    if anchor is not None and not ns.validate_cpu:
        copy_anchor(anchor, run_dir)
    import torch
    import timm
    import torchvision
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest = {'protocol': runtime.PROTOCOL, 'method': ns.method, 'variant': ns.method,
                'stage': ns.stage, 'seed': ns.seed, 'config': cfg, 'source': str(ns.source),
                'anchor_dir': str(anchor) if anchor else None, 'gpu_uuid': gpu_uuid,
                'smoke': ns.smoke, 'start_unix': time.time(), 'precision': 'FP32; TF32 disabled',
                'torch': torch.__version__, 'timm': timm.__version__, 'torchvision': torchvision.__version__,
                'training_source_sha256': {f: runtime.sha256(ns.source / f) for f in runtime.SOURCE_FILES},
                'local_sources': {f.name: runtime.sha256(f) for f in HERE.glob('*.py')},
                'shared_runtime_sha256': runtime.sha256(HERE.parent / 'imagenetr_gam_ewc_1x1/train.py'),
                'shared_step_sha256': runtime.sha256(HERE.parent / 'imagenetr_gam_ewc_1x1/learner.py')}
    if anchor is not None:
        previous = json.loads((anchor / 'run_manifest.json').read_text())
        if previous['training_source_sha256'] != manifest['training_source_sha256']:
            raise ValueError('Training source changed after anchor creation.')
        if any(previous[k] != manifest[k] for k in ('shared_runtime_sha256', 'shared_step_sha256')):
            raise ValueError('Shared runtime or GAM step changed after anchor creation.')
        for name in ('train.py', 'learner.py'):
            if previous['local_sources'][name] != manifest['local_sources'][name]:
                raise ValueError(f'{name} changed after anchor creation.')
    runtime.write_json(run_dir / 'run_manifest.json', manifest)
    runtime.write_json(run_dir / 'config.json', cfg)
    installed = runtime.install_runtime(ns, cfg, run_dir)
    if ns.validate_cpu:
        local_cfg = copy.deepcopy(cfg)
        local_cfg.update(seed=ns.seed, device=[torch.device('cpu')], filepath=str(run_dir / 'checkpoints') + '/')
        Path(local_cfg['filepath']).mkdir(parents=True, exist_ok=True)
        installed['reset_rng'](ns.seed)
        manager = installed['DataManager']('imageneta', False, ns.seed, 20, 20, local_cfg)
        local_cfg.update(task_increments=list(manager._increments), nb_classes=manager.nb_classes)
        model = installed['factory'].get_model(local_cfg['model_name'], local_cfg)
        task = 0 if ns.stage == 'task0' else 1
        if task:
            model.load_task_checkpoint(str(anchor / 'checkpoints'), 0, manager)
        installed['context']['task'] = task
        installed['reset_rng'](runtime.task_seed(ns.seed, task))
        model._unwrap_network().update_fc(20 * (task + 1))
        runtime.write_json(run_dir / 'CPU_VALIDATED.json', {
            'status': 'passed', 'training_started': False, 'forward_tested': False,
            'initial_network_sha256': runtime.tensor_hash(model._unwrap_network().state_dict().items()),
            'task_increments': list(manager._increments), 'config': cfg})
        print(f'CPU model/data construction passed: {run_dir}', flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA device must be visible.')
    os.chdir(run_dir)
    from src.trainer import train
    try:
        train(cfg)
        task = 0 if ns.stage == 'task0' else 1
        ckpt = run_dir / 'checkpoints' / f'task_{task:02d}.pt'
        metrics = list((run_dir / 'trainer_outputs').rglob('*_cl_metrics.json'))
        if len(metrics) != 1 or not ckpt.is_file():
            raise RuntimeError('Missing complete endpoint or unique CL metrics.')
        if anchor is not None and runtime.sha256(run_dir / 'checkpoints/task_00.pt') != json.loads((anchor / 'TASK0_COMPLETE.json').read_text())['checkpoint_sha256']:
            raise RuntimeError('Continuation changed its Task0 checkpoint.')
        receipt = {'status': 'complete', 'completed_tasks': task + 1, 'method': ns.method,
                   'seed': ns.seed, 'checkpoint_sha256': runtime.sha256(ckpt), 'checkpoint': str(ckpt),
                   'metrics': str(metrics[0]), 'smoke': ns.smoke, 'end_unix': time.time()}
        runtime.write_json(run_dir / ('TASK0_COMPLETE.json' if task == 0 else 'COMPLETE.json'), receipt)
    except BaseException as exc:
        runtime.write_json(run_dir / 'FAILED.json', {'error': repr(exc), 'time_unix': time.time()})
        raise


if __name__ == '__main__':
    main()
