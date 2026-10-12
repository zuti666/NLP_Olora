#!/usr/bin/env python3
"""CPU-only provenance and two-head Task1-gradient tests; no production model."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import diagnostic_task1_gradient as worker


def main():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision import datasets
    torch.set_num_threads(2)
    preflights, heads = [], []
    for seed in range(3):
        config = worker.read(worker.HERE / f'configs/seed{seed}.json')
        head = worker.reconstruct_task1_head(Path(config['source_path']), seed)
        head_sha = worker.named_tensor_hash(head.items())
        for method in ('sgd', 'gam'):
            command = [sys.executable, str(worker.HERE / 'diagnostic_task1_gradient.py'),
                       '--config', str(worker.HERE / f'configs/seed{seed}.json'),
                       '--method', method, '--job', 'task1_gradient', '--gradient-task', '1', '--validate-only']
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            evidence = json.loads(result.stdout)
            worker.require(evidence['status'] == 'passed' and evidence['direction_sign'] == -1 and evidence['evaluation_task'] == 0, 'CPU preflight failed')
            worker.require(head_sha == evidence['task1_formal']['head_initialization']['new_head_sha256'], 'Fresh head does not match formal initialization')
            worker.require(evidence['trained_checkpoint_weight_reads'] == [config['checkpoints'][method]['0']], 'Future-trained checkpoint requested')
            preflights.append(evidence)
        heads.append(dict(seed=seed, reconstructed_new20_sha256=head_sha, both_methods_match=True))
    dataset = datasets.ImageFolder(Path(config['data_root']) / 'train')
    members = [(str(path), int(label)) for path, label in dataset.samples if 20 <= label < 40]
    membership_sha = hashlib.sha256(json.dumps(members).encode()).hexdigest()
    worker.require(len(members) == 997 and membership_sha == worker.TASK1_DATASET_IDS_SHA256, 'Task1 membership differs from formal loader')
    # A small independent two-head classifier checks gradient arithmetic and
    # head separation. 21 examples yield 8+8+5, matching Task1's last-batch size.
    torch.manual_seed(314159)
    backbone = nn.Linear(7, 768, bias=False)
    old_head = nn.Linear(768, 20).requires_grad_(False)
    fresh_head = worker.reconstruct_task1_head(Path(config['source_path']), 0)
    fresh_before = worker.named_tensor_hash(fresh_head.items())
    old_before = worker.named_tensor_hash(old_head.state_dict().items())
    class Objective(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.register_buffer('new_weight', fresh_head['weight'])
            self.register_buffer('new_bias', fresh_head['bias'])

        def forward(self, x):
            return F.linear(self.backbone(x), self.new_weight, self.new_bias)
    model = Objective().eval()
    params = list(backbone.parameters())
    x, global_labels = torch.randn(21, 7), torch.randint(20, 40, (21,))
    local_labels = global_labels - 20
    full_loss = F.cross_entropy(model(x), local_labels)
    full_loss.backward()
    direct = [param.grad.detach().double().clone() for param in params]
    expected_loss = float(full_loss.detach())
    model.zero_grad(set_to_none=True)
    gradient, loss = worker.accumulate_full_gradient(model, params, x, local_labels, 'cpu', 8)
    errors = [float((a - b).abs().max()) for a, b in zip(gradient, direct)]
    worker.require(all(torch.allclose(a, b, atol=1e-7, rtol=1e-5) for a, b in zip(gradient, direct)), 'Batched gradient is not the sample mean')
    worker.close(loss, expected_loss, 'Toy Task1 CE', atol=3e-7, rtol=1e-7)
    direction, norm, normu, dot = worker.normalize_negative_gradient(gradient)
    worker.validate_gradient_tensors(gradient, direction)
    worker.close(normu, 1., 'Toy direction norm', atol=5e-7, rtol=0.)
    worker.close(dot, -norm, 'Toy negative Task1 derivative', atol=1e-9, rtol=2e-6)
    faults = []
    for name, action in [
        ('actual_direction_sign_flip', lambda: worker.validate_gradient_tensors(gradient, [-value for value in direction])),
        ('nonfinite_gradient', lambda: worker.normalize_negative_gradient([torch.full_like(gradient[0], float('nan'))])),
    ]:
        try:
            action()
        except ValueError as error:
            faults.append(dict(fault=name, rejected=True, reason=str(error)))
        else:
            raise AssertionError(f'Fault accepted: {name}')
    try:
        F.cross_entropy(model(x), global_labels)
    except (IndexError, RuntimeError) as error:
        faults.append(dict(fault='global_labels_passed_to_local20', rejected=True, reason=str(error)))
    else:
        raise AssertionError('Unmapped global Task1 labels accepted')
    from backbone.linears import SimpleLinear
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260920 + 1009)
        incorrect_twenty = SimpleLinear(768, 20)
    worker.require(worker.named_tensor_hash(incorrect_twenty.state_dict().items()) != heads[0]['reconstructed_new20_sha256'], 'Incorrect 20-row constructor unexpectedly matches')
    faults.append(dict(fault='construct20_instead_of40_then_slice', rejected=True, reason='New-row hash differs from formal RNG receipt'))
    anchor = [param.detach().clone() for param in params]
    probes = []
    for h in worker.WITNESS_STEPS:
        values, old_values = {}, {}
        for sign in (-1, 1):
            with torch.no_grad():
                for param, initial, u in zip(params, anchor, direction):
                    param.copy_(initial).add_(u, alpha=sign * h)
                values[sign] = float(F.cross_entropy(model(x), local_labels))
                old_values[sign] = float(F.cross_entropy(old_head(backbone(x)), local_labels))
                for param, initial in zip(params, anchor):
                    param.copy_(initial)
        derivative = (values[1] - values[-1]) / (2 * h)
        probes.append(dict(h=h, loss_plus=values[1], loss_minus=values[-1],
                           central_derivative=derivative, relative_error=abs(derivative + norm) / norm,
                           old_loss_plus=old_values[1], old_loss_minus=old_values[-1]))
    worker.require(probes[0]['loss_plus'] < expected_loss, 'Task1 negative gradient did not decrease Task1 CE')
    worker.require(min(row['relative_error'] for row in probes) < .05, 'Task1 finite difference disagrees')
    worker.require(all(torch.equal(param, initial) for param, initial in zip(params, anchor)), 'Anchor restoration failed')
    worker.require(worker.named_tensor_hash(old_head.state_dict().items()) == old_before, 'Old20 classifier changed')
    worker.require(worker.named_tensor_hash(fresh_head.items()) == fresh_before, 'New20 classifier changed')
    worker.require(all(param.grad is None for param in old_head.parameters()), 'Old classifier unexpectedly received gradients')
    receipt = dict(status='passed', checked_utc=datetime.now(timezone.utc).isoformat(), gpu_used=False,
        wrapper_sha256=worker.sha(worker.HERE / 'diagnostic_task1_gradient.py'), checks_sha256=worker.sha(__file__),
        cli_preflights=preflights, exact_formal_head_replays=heads,
        task1_membership=dict(samples=len(members), dataset_ids_sha256=membership_sha, formal_match=True),
        tensor_validation=dict(status='passed', fixture='CPU synthetic two-head classifier with 21 examples; not experiment measurements',
            batch_sizes=[8, 8, 5], direct_autograd_max_errors=errors, objective_error=abs(loss - expected_loss),
            direction_norm=normu, raw_grad_norm=norm, gradient_dot_direction=dot,
            finite_difference=probes, exact_anchor_restoration=True, old_head_unchanged=True, new_head_unchanged=True),
        fault_rejections=faults)
    destination = worker.HERE / 'worker_cpu_check.json'
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='passed', preflight_jobs=6, head_replays=6, membership_samples=997,
                         fault_rejections=len(faults), receipt=str(destination))))


if __name__ == '__main__':
    main()
