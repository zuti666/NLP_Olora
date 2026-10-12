#!/usr/bin/env python3
"""CPU-only fixed-gradient implementation checks; never produces experiment data."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import subprocess
import sys

import diagnostic_fixed_gradient as worker


def main():
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(2)
    preflights = []
    for seed in range(3):
        for method in ('sgd', 'gam'):
            command = [sys.executable, str(worker.HERE / 'diagnostic_fixed_gradient.py'),
                       '--config', str(worker.HERE / f'configs/seed{seed}.json'),
                       '--method', method, '--job', 'fixed_gradient', '--validate-only']
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            evidence = json.loads(result.stdout)
            worker.require(evidence['status'] == 'passed' and evidence['direction_sign'] == 1, 'CPU preflight failed')
            preflights.append(evidence)
    # Independent small CPU linear classifier: 19 examples give 8+8+3, so a
    # mistaken mean of batch means differs from the actual sample mean.
    torch.manual_seed(271828)
    model = nn.Linear(7, 3).eval()
    x, y = torch.randn(19, 7), torch.randint(0, 3, (19,))
    params = list(model.parameters())
    full_loss = F.cross_entropy(model(x), y, reduction='mean')
    full_loss.backward()
    direct = [p.grad.detach().double().clone() for p in params]
    expected_loss = float(full_loss.detach())
    model.zero_grad(set_to_none=True)
    accumulated, loss = worker.accumulate_full_gradient(model, params, x, y, 'cpu', batch_size=8)
    errors = [float((a - b).abs().max()) for a, b in zip(accumulated, direct)]
    worker.require(all(torch.allclose(a, b, atol=1e-7, rtol=1e-5) for a, b in zip(accumulated, direct)), 'Batched gradient is not the full sample mean')
    worker.close(loss, expected_loss, 'Toy full objective', atol=2e-7, rtol=1e-7)
    u, norm, direction_norm, dot = worker.normalize_positive_gradient(accumulated)
    worker.validate_gradient_tensors(accumulated, u)
    worker.close(direction_norm, 1., 'Toy direction norm', atol=5e-7, rtol=0.)
    worker.close(dot, norm, 'Toy positive directional derivative', atol=1e-9, rtol=2e-6)
    faults = []
    # This flips the actual vector while its surrounding metadata would still
    # claim positive ascent. The tensor-based validator must reject it.
    try:
        worker.validate_gradient_tensors(accumulated, [-value for value in u])
    except ValueError as error:
        faults.append(dict(fault='actual_vector_wrong_sign', rejected=True, reason=str(error)))
    else:
        raise AssertionError('Wrong-sign actual direction was accepted')
    try:
        worker.normalize_positive_gradient([torch.full_like(accumulated[0], float('nan'))])
    except ValueError as error:
        faults.append(dict(fault='nonfinite_actual_gradient', rejected=True, reason=str(error)))
    else:
        raise AssertionError('Nonfinite gradient was accepted')
    anchor = [p.detach().clone() for p in params]
    probe_rows = []
    for h in worker.WITNESS_STEPS:
        losses = {}
        for sign in (-1, 1):
            with torch.no_grad():
                for p, a, direction in zip(params, anchor, u):
                    p.copy_(a).add_(direction, alpha=sign * h)
                losses[sign] = float(F.cross_entropy(model(x), y))
                for p, a in zip(params, anchor):
                    p.copy_(a)
        derivative = (losses[1] - losses[-1]) / (2 * h)
        probe_rows.append(dict(h=h, loss_plus=losses[1], loss_minus=losses[-1],
                               central_derivative=derivative,
                               relative_error=abs(derivative - norm) / norm))
    worker.require(probe_rows[0]['loss_plus'] > expected_loss, 'Positive gradient did not increase toy CE')
    worker.require(min(row['relative_error'] for row in probe_rows) < .05, 'Toy finite difference failed')
    worker.require(all(torch.equal(p, a) for p, a in zip(params, anchor)), 'Toy anchor not restored')
    config = worker.read(worker.HERE / 'configs/seed0.json')
    config['fisher_samples'] = 128
    try:
        worker.validate_configuration(config, worker.read(worker.BASE / 'configs/seed0.json'))
    except ValueError as error:
        faults.append(dict(fault='changed_sample_scope', rejected=True, reason=str(error)))
    else:
        raise AssertionError('Changed formal sample scope accepted')
    receipt = dict(status='passed', checked_utc=datetime.now(timezone.utc).isoformat(), gpu_used=False,
        wrapper_sha256=worker.sha(worker.HERE / 'diagnostic_fixed_gradient.py'),
        checks_sha256=worker.sha(__file__), cli_preflights=preflights,
        tensor_validation=dict(status='passed', fixture='CPU synthetic 19-example linear classifier; not experiment measurements',
            batch_sizes=[8, 8, 3], direct_autograd_max_errors=errors,
            objective_error=abs(loss - expected_loss), direction_norm=direction_norm,
            gradient_dot_direction=dot, raw_grad_norm=norm, finite_difference=probe_rows,
            exact_anchor_restoration=True), fault_rejections=faults)
    destination = worker.HERE / 'worker_cpu_check.json'
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='passed', preflight_jobs=len(preflights), faults_rejected=len(faults),
                         gradient_batching_and_sign='passed', receipt=str(destination))))


if __name__ == '__main__':
    main()
