#!/usr/bin/env python3
"""Complete ImageNet-A RQ2 diagnostics; isolated, resumable measurement jobs.

Dense evaluation is forward-only.  All Fisher measurements use the unchanged,
validated per-example effective-Q/V collector in ../rq2_common.py.  A new job
protocol is deliberately distinct from the historical pilot protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXECUTION = HERE.parent
sys.path.insert(0, str(EXECUTION))

import torch
from diagnostic import Journal, Runner
from rq2_common import (
    atomic_json, canonical_hash, convert_effective, coordinates,
    effective_weights, evaluate, evaluate_local_and_seen, fisher, load_network,
    norm_sq, prepare_data, random_direction, read_state, set_point, sha_file,
    tensor_hash, unit_direction,
)

JOBS = ('dense', 'sparse_train', 'sparse_test', 'sparse_train_full',
        'trajectory_fisher', 'path', 'bridge_full')


def required_checkpoints(method, job, path_task):
    if job == 'bridge_full':
        return {m: (0, 9) for m in ('sgd', 'gam')}
    tasks = tuple(range(10)) if job == 'trajectory_fisher' else (
        (0, path_task) if job == 'path' else (0,))
    return {method: tasks}


def grid(job, config):
    """Protocol defaults implement the attachment, not the pilot config keys."""
    if job == 'dense':
        return list(config.get('dense_radii', [i / 4 for i in range(81)])), list(range(int(config.get('dense_directions', 100))))
    if job == 'sparse_train':
        return list(config.get('sparse_train_radii', [0, .5, 1, 2, 5, 10, 20])), list(range(int(config.get('sparse_directions', 16))))
    if job == 'sparse_test':
        return list(config.get('sparse_test_radii', [0, 1, 10, 20])), list(range(int(config.get('sparse_directions', 16))))
    if job == 'sparse_train_full':
        # Chosen before observing results; these are indices in the same library.
        return list(config.get('sparse_train_full_radii', [0, 1, 10, 20])), list(config.get('full_train_direction_indices', [0, 1, 2, 3]))
    raise ValueError(job)


def integrate_simpson(values):
    n = len(values)
    assert n >= 3 and n % 2 == 1
    return (values[0] + values[-1] + 4 * sum(values[1:-1:2]) +
            2 * sum(values[2:-1:2])) / (3 * (n - 1))


def quadrature_gate(coarse, fine, damage, values, config):
    absolute = abs(fine - coarse)
    tolerance = float(config.get('quadrature_atol', 1e-8)) + float(config.get('quadrature_rtol', .05)) * max(abs(fine), abs(coarse))
    bound_tolerance = max(1e-8, float(config.get('bound_rtol', .02)) * abs(fine))
    # A narrow sampled spike calls for the next nested grid even if two Simpson
    # estimates accidentally cancel.  The factor is fixed in the protocol.
    peak_factor = float(config.get('path_peak_neighbor_factor', 4.0))
    unresolved_peak = any(values[i] > peak_factor * max(values[i-1], values[i+1], 1e-12)
                          for i in range(1, len(values)-1))
    return {'absolute_difference': absolute,
            'relative_difference': absolute / max(abs(fine), abs(coarse), 1e-30),
            'stability_tolerance': tolerance, 'stable': absolute <= tolerance,
            'bound_tolerance': bound_tolerance,
            'bound_passed': damage * damage <= fine + bound_tolerance,
            'bound_passed_without_tolerance': damage * damage <= fine,
            'sampled_peak_needs_resolution_check': unresolved_peak,
            'request_refinement': absolute > tolerance or damage * damage > fine + bound_tolerance or unresolved_peak,
            # A physical peak is not a theoretical/numerical failure.  It
            # requests the next nested grid, while the actual acceptance test
            # remains the pre-registered quadrature stability and bound check.
            'passed': absolute <= tolerance and damage * damage <= fine + bound_tolerance}


def summarize_damage(rows, origin, split):
    signed = torch.tensor([r['metrics'][split]['loss'] - origin['metrics'][split]['loss'] for r in rows], dtype=torch.float64)
    positive = signed.clamp_min(0)
    return {'D_mean': float(signed.mean()), 'G': float(positive.mean()),
            'Q95': float(torch.quantile(positive, .95)), 'sampled_max': float(positive.max()),
            'D_each': signed.tolist(),
            'absolute_loss_mean': sum(r['metrics'][split]['loss'] for r in rows) / len(rows),
            'accuracy_mean': sum(r['metrics'][split]['accuracy'] for r in rows) / len(rows),
            'accuracy_drop_mean_pp': origin['metrics'][split]['accuracy'] - sum(r['metrics'][split]['accuracy'] for r in rows) / len(rows)}


class FullRunner(Runner):
    def __init__(self, args):
        # Do not call pilot Runner.__init__: it requires both t9 files even for
        # B1 and defines a single 256-sample origin shared between scopes.
        self.args = args
        self.config = json.loads(Path(args.config).read_text())
        self.seed = int(self.config.get('training_seed', 0))
        self.device = torch.device(args.device)
        torch.set_num_threads(int(self.config.get('cpu_threads', 4)))
        torch.manual_seed(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.source = Path(self.config.get('source_path', EXECUTION / 'source'))
        self.output = Path(self.config['output_dir']) / args.method
        self.output.mkdir(parents=True, exist_ok=True)
        self.eval_batch = int(self.config.get('eval_batch_size', 64))
        self.fisher_batch = int(self.config.get('fisher_batch_size', 8))
        print(json.dumps({'event': 'loading_data', 'seed': self.seed, 'method': args.method, 'job': args.job}), flush=True)
        self.x, self.y, self.data = prepare_data(self.config, self.output)
        assert len(self.y['train']) == 1259 and len(self.y['test']) == 307 and len(self.y['subset']) == 256
        required = required_checkpoints(args.method, args.job, args.path_task)
        self.paths = {}
        for method, tasks in required.items():
            configured = self.config.get('checkpoints', {}).get(method, {})
            self.paths[method] = {}
            for task in tasks:
                assert str(task) in configured, f'Explicit seed{self.seed} checkpoint required: {method}/t{task}'
                path = Path(configured[str(task)])
                assert path.is_file(), path
                self.paths[method][task] = path
        self.checksums = {m: {str(t): sha_file(p) for t, p in paths.items()} for m, paths in self.paths.items()}
        self.anchor_state = read_state(self.paths[args.method][0])
        self.anchor_cpu = effective_weights(self.anchor_state)
        self.delta_cpu, self.drift = {}, {}
        if args.job in ('path', 'bridge_full'):
            for method, paths in self.paths.items():
                target = args.path_task if args.job == 'path' else 9
                state0, state1 = read_state(paths[0]), read_state(paths[target])
                self.validate_fixed_state(state0, state1, f'{method}/t{target}')
                name = f'{method}_t{target:02d}'
                delta = [b - a for a, b in zip(effective_weights(state0), effective_weights(state1))]
                self.delta_cpu[name] = delta
                self.drift[name] = norm_sq(delta) ** .5
        self.deltas = {name: [v.to(self.device) for v in values] for name, values in self.delta_cpu.items()}
        self.model = load_network(self.paths[args.method][0], self.source, self.device)
        with torch.no_grad():
            self.original_logits = self.model(self.x['test'][:8].to(self.device)).cpu()
        self.collector, self.params = convert_effective(self.model)
        self.anchor = [p.detach().clone() for p in self.params]
        job_directory = f'path_t{args.path_task:02d}' if args.job == 'path' else args.job
        self.directory = self.output / job_directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scopes = {'dense': [], 'sparse_train': ['subset'], 'sparse_test': ['test'],
                       'sparse_train_full': ['train'], 'trajectory_fisher': ['train', 'test@t0,1,5,9'],
                       'path': ['train'], 'bridge_full': []}[args.job]
        self.meta = {
            'schema': 'rq2_full_v1', 'seed': self.seed, 'method': args.method, 'job': args.job,
            'path_task': args.path_task if args.job == 'path' else None,
            'diagnostic_code_sha256': sha_file(Path(__file__)),
            'common_code_sha256': sha_file(EXECUTION / 'rq2_common.py'),
            'pilot_runner_code_sha256': sha_file(EXECUTION / 'diagnostic.py'),
            'source_path': str(self.source),
            'source_files': {str(p.relative_to(self.source)): sha_file(p) for p in
                             (self.source / 'backbone/lora.py', self.source / 'utils/data.py')},
            'config': self.config, 'config_sha256': sha_file(Path(args.config)),
            'checkpoints': {m: {str(t): str(p) for t, p in paths.items()} for m, paths in self.paths.items()},
            'checkpoint_sha256': self.checksums,
            'data_manifest_sha256': self.data['manifest_sha256'],
            'fisher_scopes': self.scopes, 'counts': {s: len(self.y[s]) for s in ('train', 'test', 'subset')},
            'coordinate_order': coordinates(), 'coordinate_numel': 14155776,
            'coordinate_sha256': canonical_hash(coordinates()),
            'head_sha256': tensor_hash([self.model.fc.weight[:20], self.model.fc.bias[:20]]),
            'anchor_effective_sha256': tensor_hash(self.anchor),
            'direction_seed': int(self.config.get('direction_seed', 20260920)),
            'actual_direction_sha256': {m: tensor_hash(d) for m, d in self.delta_cpu.items()},
            'actual_drift': self.drift,
            'production_precision': 'FP32 model/gradients, FP64 scalar reductions; TF32 off',
            'fisher_definition': 'mean per-example true-label CE gradient outer product in effective Q/V; fixed old20 head; no normalization/floor/regularizer',
            'torch_version': torch.__version__,
            'point_restoration': 'exact tensor equality after every point; anchor logits/loss/predictions checked before and after job',
        }
        self.protocol = canonical_hash(self.meta)
        self.meta['protocol_sha256'] = self.protocol
        manifest_path = self.directory / 'manifest.json'
        if manifest_path.exists():
            assert json.loads(manifest_path.read_text())['protocol_sha256'] == self.protocol, 'Protocol changed; choose a new output directory'
        atomic_json(manifest_path, self.meta)
        self.journal = Journal(self.directory, self.protocol)
        self.start_checks = self.check_anchor()
        atomic_json(self.directory / 'sanity.json', {'passed': True, 'protocol_sha256': self.protocol, **self.start_checks})

    @staticmethod
    def validate_fixed_state(anchor, state, label):
        assert torch.equal(anchor['fc.weight'][:20], state['fc.weight'][:20]), f'Old head changed: {label}'
        assert torch.equal(anchor['fc.bias'][:20], state['fc.bias'][:20]), f'Old bias changed: {label}'
        for key, value in anchor.items():
            if not key.startswith('fc.') and 'linear_a_' not in key and 'linear_b_' not in key:
                assert key in state and torch.equal(value, state[key]), f'Fixed state changed: {label}/{key}'

    def restore(self):
        set_point(self.params, self.anchor)
        assert all(torch.equal(p, a) for p, a in zip(self.params, self.anchor)), 'Anchor restoration failed'

    def check_anchor(self):
        self.restore()
        with torch.no_grad():
            logits = self.model(self.x['test'][:8].to(self.device)).cpu()
        error = float((logits - self.original_logits).abs().max())
        assert error < 3e-5, f'Effective conversion/restoration error: {error}'
        assert torch.equal(logits.argmax(1), self.original_logits.argmax(1))
        loss = evaluate(self.model, self.x['test'], self.y['test'], self.device, self.eval_batch)
        assert math.isfinite(loss['loss'])
        return {'max_logit_error': error, 'predictions_match': True, 'test': loss,
                'effective_sha256': tensor_hash(self.params)}

    def measure_full(self, state_id, perturbation=None, amount=0.0, fisher_scopes=(), directions=None, **point):
        if state_id in self.journal.rows:
            return self.journal.rows[state_id]
        started = time.monotonic()
        set_point(self.params, self.anchor, perturbation, amount)
        try:
            metrics = {s: self.evaluate(s) for s in ('train', 'test')}
            gradients = {s: fisher(self.model, self.collector, self.params, self.x[s], self.y[s],
                                  directions or {}, self.device, self.fisher_batch) for s in fisher_scopes}
            realized = norm_sq([p.detach() - a for p, a in zip(self.params, self.anchor)]) ** .5
            expected = 0.0 if perturbation is None else abs(float(amount)) * norm_sq(perturbation) ** .5
            assert abs(realized - expected) <= max(1e-5, expected * 2e-6), (realized, expected)
            row = {'state_id': state_id, 'method': self.args.method, 'seed': self.seed,
                   'point': point, 'amount': amount, 'metrics': metrics, 'fisher_by_scope': gradients,
                   'effective_weight_drift': realized, 'effective_weight_drift_squared': realized ** 2,
                   'coordinate_sha256': self.meta['coordinate_sha256'],
                   'data_manifest_sha256': self.data['manifest_sha256'],
                   'checkpoint_sha256': self.checksums[self.args.method],
                   'seconds': time.monotonic() - started}
            # Validation precedes a durable journal commit.
            json.dumps(row, allow_nan=False)
        finally:
            self.restore()
        row['anchor_restored_exactly'] = True
        return self.journal.add(row)

    def origin_full(self, scopes=(), directions=None):
        return self.measure_full('r0', fisher_scopes=scopes, directions=directions, kind='anchor', radius=0)

    def finish(self, payload):
        ending = self.check_anchor()
        assert ending['effective_sha256'] == self.start_checks['effective_sha256']
        assert abs(ending['test']['loss'] - self.start_checks['test']['loss']) < 1e-10
        assert ending['test']['accuracy'] == self.start_checks['test']['accuracy']
        super().finish({'seed': self.seed, 'end_restoration': ending, **payload})

    def neighborhood_full(self):
        radii, indices = grid(self.args.job, self.config)
        assert radii == sorted(set(radii)) and radii[0] == 0 and all(r >= 0 for r in radii)
        assert indices and len(indices) == len(set(indices)) and all(isinstance(i, int) and i >= 0 for i in indices)
        scopes = self.scopes
        origin = self.origin_full(scopes)
        for index in indices:
            needed = [r for r in radii if r and f'r{r:.12g}_u{index:03d}' not in self.journal.rows]
            if not needed:
                continue
            direction, info = random_direction(index, self.meta['direction_seed'], self.device)
            for radius in needed:
                self.measure_full(f'r{radius:.12g}_u{index:03d}', direction, float(radius),
                                  fisher_scopes=scopes, kind=self.args.job, radius=radius, direction=info)
        summaries = []
        for radius in radii:
            rows = [origin] if radius == 0 else [self.journal.rows[f'r{radius:.12g}_u{i:03d}'] for i in indices]
            item = {'radius': radius, 'directions': len(rows),
                    **{s: summarize_damage(rows, origin, s) for s in ('train', 'test')}}
            for scope in scopes:
                traces = torch.tensor([r['fisher_by_scope'][scope]['raw_trace'] for r in rows], dtype=torch.float64)
                item.setdefault('raw_trace_by_scope', {})[scope] = {
                    'mean': float(traces.mean()), 'Q95': float(torch.quantile(traces, .95)),
                    'sampled_max': float(traces.max()), 'each': traces.tolist(), 'samples': len(self.y[scope])}
            summaries.append(item)
        tolerance = []
        for split in ('train', 'test'):
            for metric in ('G', 'Q95'):
                for epsilon in self.config.get('damage_thresholds', [.001, .005, .01, .02, .05]):
                    radius = 0
                    for item in summaries:
                        if item[split][metric] > epsilon:
                            break
                        radius = item['radius']
                    tolerance.append({'split': split, 'metric': metric, 'epsilon': epsilon,
                                      'prefix_radius': radius, 'right_censored': radius == max(radii)})
        expected = 1 + (len(radii) - 1) * len(indices)
        assert len(self.journal.rows) == expected
        self.finish({'radius_summary': summaries, 'tolerance': tolerance, 'expected_states': expected,
                     'direction_indices': indices, 'paired_public_direction_library': True,
                     'gradient_scope': scopes, 'forward_only': not scopes,
                     'full_train_direction_subset_fixed_before_results': self.args.job == 'sparse_train_full'})

    def endpoint_reference(self, task):
        path = self.paths[self.args.method][task]
        state = read_state(path)
        self.validate_fixed_state(self.anchor_state, state, f'{self.args.method}/t{task}')
        assert state['fc.weight'].shape[0] == 20 * (task + 1)
        delta_cpu = [b - a for a, b in zip(self.anchor_cpu, effective_weights(state))]
        delta = [v.to(self.device) for v in delta_cpu]
        direct = load_network(path, self.source, self.device)
        metrics = {s: evaluate_local_and_seen(direct, self.x[s], self.y[s], self.device, self.eval_batch)
                   for s in ('train', 'test')}
        with torch.no_grad():
            logits = direct(self.x['test'][:8].to(self.device)).cpu()
        del direct
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        set_point(self.params, self.anchor, delta, 1.0)
        with torch.no_grad():
            interpolated = self.model(self.x['test'][:8].to(self.device)).cpu()
        error = float((interpolated - logits).abs().max())
        assert error < 5e-5, f'Endpoint mismatch at t{task}: {error}'
        reconstructed = {s: self.evaluate(s) for s in ('train', 'test')}
        loss_errors = {s: reconstructed[s]['loss'] - metrics[s]['local']['loss'] for s in ('train', 'test')}
        assert all(abs(v) < 3e-6 for v in loss_errors.values()), loss_errors
        self.restore()
        return delta, metrics, {'task': task, 'checkpoint': str(path),
                               'checkpoint_sha256': self.checksums[self.args.method][str(task)],
                               'max_logit_error': error, 'local_loss_errors': loss_errors,
                               'passed': True, 'effective_delta_sha256': tensor_hash(delta_cpu)}

    def trajectory_fisher(self):
        origin = self.origin_full(['train', 'test'])
        endpoints = []
        for task in range(10):
            state_id = f'task{task:02d}'
            if state_id in self.journal.rows:
                endpoints.append(self.journal.rows[state_id])
                continue
            delta, direct_metrics, recovery = self.endpoint_reference(task)
            scopes = ['train', 'test'] if task in (0, 1, 5, 9) else ['train']
            if task == 0:
                row = {k: v for k, v in origin.items() if k != 'protocol_sha256'}
                row.update(state_id=state_id, point={'kind': 'trajectory', 'task': task})
                row['direct_local_and_seen'] = direct_metrics
                row['endpoint_recovery'] = recovery
                row['D0'] = {'train': 0.0, 'test': 0.0}
                row = self.journal.add(row)
            else:
                # All direct metrics and recovery checks are persisted with the
                # same atomic state row; no result can be committed halfway.
                row = self.measure_full(state_id, delta, 1.0, fisher_scopes=scopes,
                                        kind='trajectory', task=task, endpoint_recovery=recovery,
                                        direct_local_and_seen=direct_metrics)
            endpoints.append(row)
        summaries = []
        for task, row in enumerate(endpoints):
            summaries.append({'task': task, 'state_id': row['state_id'],
                              'effective_weight_drift': row['effective_weight_drift'],
                              'effective_weight_drift_squared': row['effective_weight_drift_squared'],
                              'D0': {s: row['metrics'][s]['loss'] - origin['metrics'][s]['loss'] for s in ('train', 'test')},
                              'metrics': row['metrics'], 'fisher_by_scope': row['fisher_by_scope'],
                              'direct_local_and_seen': row.get('direct_local_and_seen', row['point'].get('direct_local_and_seen')),
                              'endpoint_recovery': row.get('endpoint_recovery', row['point'].get('endpoint_recovery'))})
        assert len(endpoints) == 10
        self.finish({'tasks': summaries, 'task_count': 10, 'train_fisher_samples': 1259,
                     'test_fisher_tasks': [0, 1, 5, 9], 'test_fisher_samples': 307})

    def path(self):
        task = self.args.path_task
        name = f'{self.args.method}_t{task:02d}'
        delta = self.deltas[name]
        _, direct_metrics, recovery = self.endpoint_reference(task)
        origin = self.origin_full(['train'], {name: delta})
        def get(fraction):
            if fraction == 0:
                return origin
            return self.measure_full(f's{fraction:.8f}', delta, fraction,
                                     fisher_scopes=['train'], directions={name: delta},
                                     kind='path', target_task=task, path_fraction=fraction,
                                     direction_sha256=self.meta['actual_direction_sha256'][name])
        integrals, gates = {}, []
        previous = None
        selected_rows, gate, selected_nodes = None, None, None
        for nodes in (5, 9, 17, 33):
            rows = [get(i / (nodes - 1)) for i in range(nodes)]
            q = [r['fisher_by_scope']['train']['q'][name] for r in rows]
            traces = [r['fisher_by_scope']['train']['raw_trace'] for r in rows]
            value = integrate_simpson(q)
            trace_integral = integrate_simpson(traces)
            damage = rows[-1]['fisher_by_scope']['train']['subset_loss'] - origin['fisher_by_scope']['train']['subset_loss']
            integrals[str(nodes)] = {'q_integral': value, 'trace_integral': trace_integral}
            if previous is not None:
                gate = quadrature_gate(previous, value, damage, q, self.config)
                gates.append({'nodes': nodes, **gate})
                selected_rows, selected_nodes = rows, nodes
                if gate['passed'] and (not gate['sampled_peak_needs_resolution_check'] or nodes == 33):
                    break
            previous = value
        assert selected_rows is not None and gate is not None
        end_errors = {s: selected_rows[-1]['metrics'][s]['loss'] - direct_metrics[s]['local']['loss'] for s in ('train', 'test')}
        assert all(abs(v) < 3e-6 for v in end_errors.values()), end_errors
        trace_bound = self.drift[name] ** 2 * trace_integral
        assert value <= trace_bound + max(1e-8, 1e-5 * abs(trace_bound)), 'Directional integral exceeds trace bound'
        result = {'target_task': task, 'fisher_scope': 'full old-task train1259', 'samples': 1259,
                  'integrals_simpson': integrals, 'quadrature_gates': gates,
                  'selected_nodes': selected_nodes, 'quadrature_converged': gate['stable'],
                  'numerical_gate_passed': gate['passed'],
                  'integral': value, 'trace_integral': trace_integral,
                  'trace_upper_bound': trace_bound,
                  'directional_to_trace_bound_ratio': value / trace_bound if trace_bound else None,
                  'S_path': value / self.drift[name] ** 2 if self.drift[name] else None,
                  'drift': self.drift[name], 'drift_squared': self.drift[name] ** 2,
                  'D_train': damage, 'D_train_squared': damage ** 2,
                  'D_test': selected_rows[-1]['metrics']['test']['loss'] - origin['metrics']['test']['loss'],
                  'bound_passed_with_numeric_tolerance': gate['bound_passed'],
                  'bound_gap_I_minus_D2': value - damage ** 2,
                  'endpoint_recovery': dict(recovery, path_endpoint_loss_errors=end_errors),
                  'path': [{'s': i / (selected_nodes - 1), 'q': r['fisher_by_scope']['train']['q'][name],
                            'raw_trace': r['fisher_by_scope']['train']['raw_trace'],
                            'train_loss_for_bound': r['fisher_by_scope']['train']['subset_loss'],
                            'metrics': r['metrics']} for i, r in enumerate(selected_rows)]}
        if not gate['passed']:
            atomic_json(self.directory / 'numerical_gate_failure.json', {'status': 'needs_review', 'protocol_sha256': self.protocol, **result})
            raise RuntimeError('Path failed fixed numerical gate after 33 nodes; all measurements preserved, no completed claim')
        self.finish(result)

    def bridge_full(self):
        origin = self.origin_full()
        radii = sorted(set(float(r) for r in self.config.get('bridge_base_radii', [0, .25, .5, 1, 1.5, 2])) |
                       set(self.drift.values()))
        k = int(self.config.get('bridge_random_directions', 100))
        assert k > 0 and radii[0] == 0
        actual = []
        for name, delta in self.deltas.items():
            unit, drift = unit_direction(delta)
            for radius in radii:
                row = origin if radius == 0 else self.measure_full(
                    f'actual_{name}_r{radius:.12g}', unit, radius, kind='bridge_actual',
                    radius=radius, direction_name=name, actual_drift=drift,
                    direction_sha256=self.meta['actual_direction_sha256'][name])
                actual.append({'direction_name': name, 'radius': radius, 'actual_drift': drift,
                               'state_id': row['state_id'], 'metrics': row['metrics'],
                               'D0': {s: row['metrics'][s]['loss'] - origin['metrics'][s]['loss'] for s in ('train', 'test')}})
        for index in range(k):
            missing = [r for r in radii if r and f'random_r{r:.12g}_u{index:03d}' not in self.journal.rows]
            if not missing:
                continue
            direction, info = random_direction(index, self.meta['direction_seed'], self.device)
            for radius in missing:
                self.measure_full(f'random_r{radius:.12g}_u{index:03d}', direction, radius,
                                  kind='bridge_random', radius=radius, direction=info)
        random_summaries = []
        for radius in radii:
            rows = [origin] if radius == 0 else [self.journal.rows[f'random_r{radius:.12g}_u{i:03d}'] for i in range(k)]
            random_summaries.append({'radius': radius, 'directions': len(rows),
                                     **{s: summarize_damage(rows, origin, s) for s in ('train', 'test')}})
        self.finish({'radii': radii, 'actual_drift_radii_included': self.drift,
                     'bridge_actual': actual, 'random_summary': random_summaries,
                     'random_directions': k, 'forward_only': True,
                     'interpretation': 'cross-anchor responses to actual t0-to-t9 directions plus the identical public random library at matched radii'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', required=True, choices=('sgd', 'gam'))
    parser.add_argument('--job', required=True, choices=JOBS)
    parser.add_argument('--path-task', type=int, default=9, choices=(1, 5, 9))
    parser.add_argument('--gpu', type=int, choices=range(7))
    parser.add_argument('--device', default='cuda:0', help='Use cpu only for explicit local implementation checks')
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        args.device = 'cuda:0'
    runner = FullRunner(args)
    if args.job in ('dense', 'sparse_train', 'sparse_test', 'sparse_train_full'):
        runner.neighborhood_full()
    else:
        getattr(runner, args.job)()


if __name__ == '__main__':
    main()
