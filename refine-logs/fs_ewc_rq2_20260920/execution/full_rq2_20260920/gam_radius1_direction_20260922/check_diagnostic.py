#!/usr/bin/env python3
"""CPU checks for new-anchor binding and both fixed gradient directions.

Writes no files and starts no GPU work. Redirect stdout to retain this receipt.
Pending training is explicitly reported; synthetic checks are not measurements.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_gam_radius_intervention_worker', HERE / 'diagnostic.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def must_reject(name, action, faults):
    try:
        action()
    except (ValueError, KeyError, IndexError, RuntimeError) as error:
        faults.append(dict(fault=name, rejected=True, reason=str(error)))
    else:
        raise AssertionError(f'Fault accepted: {name}')


def main():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision import datasets
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-training', action='store_true')
    args = parser.parse_args()
    preflights, faults, head_replays = [], [], []
    evidence_by_seed = {}
    for arm in worker.ARMS:
        for seed in range(3):
            config_path = HERE / f'configs/{arm}_seed{seed}.json'
            config = worker.read(config_path)
            for task in (0, 1):
                command = [sys.executable, str(HERE / 'diagnostic.py'), '--config', str(config_path),
                           '--gradient-task', str(task), '--validate-only']
                if not args.require_training:
                    command.append('--allow-pending-training')
                process = subprocess.run(command, check=True, capture_output=True, text=True, cwd='/tmp')
                evidence = json.loads(process.stdout)
                worker.require(evidence['status'] in ('passed', 'pending_training'), 'Unexpected preflight status')
                worker.require(evidence['trained_checkpoint_weight_reads'] == [config['checkpoints']['gam']['0']], 'Future or historical anchor weights requested')
                worker.require(evidence['radii'] == worker.RADII and evidence['expected_states'] == 11, 'Unexpected grid')
                worker.require(evidence['gradient_task'] == task and evidence['direction_sign'] == (1 if task == 0 else -1), 'Gradient sign changed')
                preflights.append({k: evidence[k] for k in ('status', 'intervention_arm', 'seed', 'job', 'gradient_task',
                    'direction_sign', 'evaluation_task', 'gradient_samples', 'fisher_samples', 'expected_states', 'config', 'trained_checkpoint_weight_reads')})
                if task == 1:
                    evidence_by_seed[seed] = evidence
                    head = worker.negative.reconstruct_task1_head(Path(config['source_path']), seed)
                    head_hash = worker.negative.named_tensor_hash(head.items())
                    worker.require(head_hash == evidence['task1_formal']['head_initialization']['new_head_sha256'], 'Untrained Task1 head replay mismatch')
                    head_replays.append(dict(arm=arm, seed=seed, head_sha256=head_hash, paired_formal_match=True))
    old_checkpoint = copy.deepcopy(config)
    old_checkpoint['checkpoints']['gam']['0'] = str(worker.BASE / f'formal_training/anchor_gam_seed{config["training_seed"]}/checkpoints/task_00.pt')
    must_reject('historical_anchor_substitution', lambda: worker.validate_configuration(old_checkpoint), faults)
    bad_radius = copy.deepcopy(config)
    bad_radius['training_norm_radius'] = .2
    must_reject('cancelled_norm_radius_point2', lambda: worker.validate_configuration(bad_radius), faults)
    missing_origin = copy.deepcopy(config)
    missing_origin['radii'] = worker.RADII[1:]
    must_reject('missing_radius_zero', lambda: worker.validate_configuration(missing_origin), faults)
    # Real running manifests exercise the strict controlled-intervention check.
    checked_running = []
    for arm in worker.ARMS:
        for seed in range(3):
            cfg = worker.read(HERE / f'configs/{arm}_seed{seed}.json')
            path = Path(cfg['training_receipt']).parent / 'run_manifest.json'
            if path.exists():
                manifest = worker.read(path)
                worker.validate_training_manifest(cfg, manifest)
                checked_running.append(worker.file_record(path))
                if len(checked_running) == 1:
                    wrong_alias = copy.deepcopy(manifest)
                    wrong_alias['config']['as1_norm_rho'] = .2
                    must_reject('active_optimizer_alias_mismatch', lambda: worker.validate_training_manifest(cfg, wrong_alias), faults)
                    wrong_learner = copy.deepcopy(manifest)
                    wrong_learner['config']['gam_grad_gamma'] = .9
                    must_reject('unapproved_nonradius_training_change', lambda: worker.validate_training_manifest(cfg, wrong_learner), faults)
    dataset = datasets.ImageFolder(Path(config['data_root']) / 'train')
    members = [(str(path), int(label)) for path, label in dataset.samples if 20 <= label < 40]
    member_hash = hashlib.sha256(json.dumps(members).encode()).hexdigest()
    worker.require(len(members) == 997 and member_hash == worker.negative.TASK1_DATASET_IDS_SHA256, 'Task1 training sample membership changed')
    torch.manual_seed(314159)
    backbone = nn.Linear(7, 768, bias=False)
    old_head = nn.Linear(768, 20).requires_grad_(False)
    fresh = worker.negative.reconstruct_task1_head(Path(config['source_path']), 0)

    class Objective(nn.Module):
        def __init__(self, weight, bias):
            super().__init__()
            self.backbone = backbone
            self.register_buffer('head_weight', weight)
            self.register_buffer('head_bias', bias)

        def forward(self, x):
            return F.linear(self.backbone(x), self.head_weight, self.head_bias)

    old_model = Objective(old_head.weight.detach(), old_head.bias.detach()).eval()
    task1_model = Objective(fresh['weight'], fresh['bias']).eval()
    old_before = worker.negative.named_tensor_hash(old_head.state_dict().items())
    new_before = worker.negative.named_tensor_hash(fresh.items())
    params = list(backbone.parameters())
    anchor = [value.detach().clone() for value in params]
    numeric_checks = []
    for task, model, count in ((0, old_model, 19), (1, task1_model, 21)):
        # 19 -> 8+8+3 and 21 -> 8+8+5 mirror real 1259/997 remainder handling.
        x, labels = torch.randn(count, 7), torch.randint(0, 20, (count,))
        full_loss = F.cross_entropy(model(x), labels)
        full_loss.backward()
        direct = [value.grad.detach().double().clone() for value in params]
        objective = float(F.cross_entropy(model(x), labels, reduction='none').detach().double().mean())
        model.zero_grad(set_to_none=True)
        helper = worker.helper_for(task)
        gradient, reduced_loss = helper.accumulate_full_gradient(model, params, x, labels, 'cpu', 8)
        worker.require(all(torch.allclose(a, b, atol=1e-7, rtol=1e-5) for a, b in zip(gradient, direct)), 'Batch means were weighted incorrectly')
        worker.close(reduced_loss, objective, 'CPU sample-mean objective', atol=3e-7, rtol=1e-7)
        normalize = helper.normalize_positive_gradient if task == 0 else helper.normalize_negative_gradient
        direction, norm, normu, dot = normalize(gradient)
        helper.validate_gradient_tensors(gradient, direction)
        worker.close(normu, 1., 'CPU unit direction', atol=5e-7, rtol=0.)
        worker.close(dot, norm if task == 0 else -norm, 'CPU signed derivative', atol=1e-9, rtol=2e-6)
        must_reject(f'task{task}_actual_direction_sign_flip', lambda: helper.validate_gradient_tensors(gradient, [-value for value in direction]), faults)
        must_reject(f'task{task}_nonfinite_gradient', lambda: normalize([torch.full_like(gradient[0], float('nan'))]), faults)
        probes = []
        for h in worker.WITNESS_STEPS:
            losses = {}
            for sign in (-1, 1):
                with torch.no_grad():
                    for value, start, u in zip(params, anchor, direction):
                        value.copy_(start).add_(u, alpha=sign * h)
                    losses[sign] = float(F.cross_entropy(model(x), labels))
                    for value, start in zip(params, anchor):
                        value.copy_(start)
            derivative = (losses[1] - losses[-1]) / (2 * h)
            probes.append(dict(h=h, loss_plus=losses[1], loss_minus=losses[-1],
                               central_derivative=derivative, relative_error=abs(derivative - dot) / norm))
        worker.require((probes[0]['loss_plus'] - objective) * (1 if task == 0 else -1) > 0, 'CPU witness direction sign wrong')
        worker.require(min(row['relative_error'] for row in probes) <= .05, 'CPU finite difference disagrees')
        worker.require(all(torch.equal(value, initial) for value, initial in zip(params, anchor)), 'CPU exact anchor restoration failed')
        numeric_checks.append(dict(gradient_task=task, fixture_samples=count, batch_sizes=[8, 8, count - 16],
            direct_autograd_max_error=max(float((a - b).abs().max()) for a, b in zip(gradient, direct)),
            direction_norm=normu, gradient_dot_direction=dot, raw_gradient_norm=norm, finite_difference=probes))
    must_reject('global_Task1_labels_without_local_mapping', lambda: F.cross_entropy(task1_model(x), labels + 20), faults)
    worker.require(worker.negative.named_tensor_hash(old_head.state_dict().items()) == old_before, 'Old head mutated')
    worker.require(worker.negative.named_tensor_hash(fresh.items()) == new_before, 'Untrained Task1 head mutated')
    # Exercise the real loader import without constructing a model or device.
    sys.path.insert(0, str(worker.BASE))
    import diagnostic_full
    import diagnostic as canonical_loader
    worker.require(Path(canonical_loader.__file__).resolve() == worker.EXECUTION / 'diagnostic.py', 'New filename shadows canonical loader')
    worker.require(diagnostic_full.required_checkpoints('gam', 'sparse_train', 0) == {'gam': (0,)}, 'Loader would read additional trained checkpoints')
    print(json.dumps(dict(status='passed', gpu_used=False, production_models_constructed=0,
        wrapper_sha256=worker.sha(HERE / 'diagnostic.py'), checks_sha256=worker.sha(__file__),
        preflight_jobs=len(preflights), complete_training_preflights=sum(item['status'] == 'passed' for item in preflights),
        pending_training_preflights=sum(item['status'] == 'pending_training' for item in preflights),
        expected_production_states=132, cli_preflights=preflights, checked_running_training_manifests=checked_running,
        task1_head_replays=head_replays, task1_membership=dict(samples=len(members), sha256=member_hash),
        numeric_fixture='CPU synthetic two-head model; arithmetic checks only, not production measurements',
        numeric_checks=numeric_checks, fixed_heads_unchanged=True, canonical_loader_import=True,
        fault_rejections=faults), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
