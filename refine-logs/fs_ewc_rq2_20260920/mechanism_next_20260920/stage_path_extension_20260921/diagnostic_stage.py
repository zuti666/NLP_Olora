#!/usr/bin/env python3
"""Historical stage paths on an explicitly selected, fixed Fisher sample scope.

Only initialization and scope-dependent path aggregation extend FullRunner.
Checkpoint recovery, exact restoration, Fisher collection, and numerical gates
reuse its validated implementations. No optimizer or training is invoked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXECUTION = HERE.parents[1] / 'execution'
sys.path.insert(0, str(EXECUTION))
sys.path.insert(0, str(EXECUTION / 'full_rq2_20260920'))

import torch
from diagnostic import Journal
from diagnostic_full import FullRunner, integrate_simpson, quadrature_gate, required_checkpoints
from rq2_common import (atomic_json, canonical_hash, convert_effective, coordinates,
                        effective_weights, load_network, norm_sq, prepare_data,
                        read_state, sha_file, tensor_hash)


def path_scope(config):
    label = config.get('path_fisher_scope', 'subset256')
    if label not in ('subset256', 'full_train'):
        raise ValueError(f'Unknown path_fisher_scope: {label}')
    if int(config.get('fisher_samples', 256)) != 256:
        raise ValueError('The preserved data protocol requires fisher_samples=256')
    return label, ('subset' if label == 'subset256' else 'train')


def legacy_compatibility(config, method, task, data, meta):
    """Fail before measurements if the historical cohort or sample set changed."""
    root = config.get('legacy_results_dir', str(EXECUTION / 'results_seed0_recovery1'))
    if root is None:
        assert meta['path_fisher_scope'] == 'full_train', 'Historical subset256 requires legacy receipt comparison'
        return {'checked': False, 'reason': 'explicit full_train run without legacy reuse'}
    root = Path(root) / method
    data_path, manifest_path = root / 'data_manifest.json', root / 'path/manifest.json'
    legacy_data = json.loads(data_path.read_text())
    legacy = json.loads(manifest_path.read_text())
    # Full equality also checks subset IDs/order, decoded tensors, transforms,
    # class mapping, decoder version and the original data-manifest hash.
    assert data == legacy_data, 'Historical data/subset manifest changed'
    keys = ('seed', 'method', 'coordinate_order', 'coordinate_numel',
            'coordinate_sha256', 'data_manifest_sha256', 'head_sha256',
            'anchor_effective_sha256', 'source_files')
    for key in keys:
        assert meta[key] == legacy[key], f'Historical identity changed: {key}'
    assert meta['checkpoint_sha256'][method]['0'] == legacy['checkpoint_sha256'][method]['0'], 'Historical task0 checkpoint changed'
    if task == 9:
        assert meta['checkpoint_sha256'][method]['9'] == legacy['checkpoint_sha256'][method]['9']
        assert meta['actual_direction_sha256'][f'{method}_t09'] == legacy['actual_direction_sha256'][method]
    return {'checked': True, 'passed': True, 'data_manifest': str(data_path),
            'data_manifest_file_sha256': sha_file(data_path),
            'path_manifest': str(manifest_path), 'path_manifest_file_sha256': sha_file(manifest_path),
            'legacy_protocol_sha256': legacy['protocol_sha256'],
            'matched_identity_fields': list(keys), 'subset_ids_and_tensors_match': True}


class StageRunner(FullRunner):
    def __init__(self, args):
        # Construct the complete extension protocol before creating a journal.
        assert args.job == 'path' and 1 <= args.path_task <= 9
        self.args = args
        self.config = json.loads(Path(args.config).read_text())
        self.seed = int(self.config.get('training_seed', 0))
        self.scope_label, self.scope = path_scope(self.config)
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
        self.scopes = [self.scope]
        sample_ids = (self.data['subset_ids'] if self.scope == 'subset' else
                      [r['id'] for r in self.data['records']['train']])
        self.meta = {
            'schema': 'rq2_stage_path_extension_v1', 'seed': self.seed, 'method': args.method, 'job': args.job,
            'path_task': args.path_task if args.job == 'path' else None,
            'diagnostic_code_sha256': sha_file(Path(__file__)),
            'full_runner_code_sha256': sha_file(EXECUTION / 'full_rq2_20260920/diagnostic_full.py'),
            'common_code_sha256': sha_file(EXECUTION / 'rq2_common.py'),
            'pilot_runner_code_sha256': sha_file(EXECUTION / 'diagnostic.py'),
            'source_path': str(self.source),
            'source_files': {str(p.relative_to(self.source)): sha_file(p) for p in
                             (self.source / 'backbone/lora.py', self.source / 'utils/data.py')},
            'config': self.config, 'config_sha256': sha_file(Path(args.config)),
            'checkpoints': {m: {str(t): str(p) for t, p in paths.items()} for m, paths in self.paths.items()},
            'checkpoint_sha256': self.checksums,
            'data_manifest_sha256': self.data['manifest_sha256'],
            'path_fisher_scope': self.scope_label, 'fisher_samples': len(self.y[self.scope]),
            'fisher_sample_ids': sample_ids, 'fisher_sample_ids_sha256': canonical_hash(sample_ids),
            'fisher_scope_tensor_sha256': (self.data['subset_image_tensor_sha256'] if self.scope == 'subset'
                                           else self.data['decoded_tensor_sha256']['train']),
            'path_kind': 'separate straight chord from fixed task0 to target task',
            'quadrature_nodes': [5, 9, 17, 33],
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
        self.meta['legacy_compatibility'] = legacy_compatibility(
            self.config, args.method, args.path_task, self.data, self.meta)
        self.protocol = canonical_hash(self.meta)
        self.meta['protocol_sha256'] = self.protocol
        manifest_path = self.directory / 'manifest.json'
        if manifest_path.exists():
            assert json.loads(manifest_path.read_text())['protocol_sha256'] == self.protocol, 'Protocol changed; choose a new output directory'
        atomic_json(manifest_path, self.meta)
        self.journal = Journal(self.directory, self.protocol)
        self.start_checks = self.check_anchor()
        atomic_json(self.directory / 'sanity.json', {'passed': True, 'protocol_sha256': self.protocol, **self.start_checks})

    def endpoint_reference(self, task):
        try:
            return super().endpoint_reference(task)
        finally:
            self.restore()

    def path(self):
        task = self.args.path_task
        name = f'{self.args.method}_t{task:02d}'
        delta = self.deltas[name]
        _, direct_metrics, recovery = self.endpoint_reference(task)
        origin = self.origin_full([self.scope], {name: delta})
        def get(fraction):
            if fraction == 0:
                return origin
            return self.measure_full(f's{fraction:.8f}', delta, fraction,
                                     fisher_scopes=[self.scope], directions={name: delta},
                                     kind='path', target_task=task, path_fraction=fraction,
                                     direction_sha256=self.meta['actual_direction_sha256'][name])
        integrals, gates = {}, []
        previous = None
        selected_rows, gate, selected_nodes = None, None, None
        for nodes in (5, 9, 17, 33):
            rows = [get(i / (nodes - 1)) for i in range(nodes)]
            q = [r['fisher_by_scope'][self.scope]['q'][name] for r in rows]
            traces = [r['fisher_by_scope'][self.scope]['raw_trace'] for r in rows]
            value = integrate_simpson(q)
            trace_integral = integrate_simpson(traces)
            damage = rows[-1]['fisher_by_scope'][self.scope]['subset_loss'] - origin['fisher_by_scope'][self.scope]['subset_loss']
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
        result = {'target_task': task, 'fisher_scope': self.scope_label, 'samples': len(self.y[self.scope]),
                  'display_T': task + 1,
                  'integrals_simpson': integrals, 'quadrature_gates': gates,
                  'selected_nodes': selected_nodes, 'quadrature_converged': gate['stable'],
                  'numerical_gate_passed': gate['passed'],
                  'integral': value, 'trace_integral': trace_integral,
                  'trace_upper_bound': trace_bound,
                  'directional_to_trace_bound_ratio': value / trace_bound if trace_bound else None,
                  'S_path': value / self.drift[name] ** 2 if self.drift[name] else None,
                  'drift': self.drift[name], 'drift_squared': self.drift[name] ** 2,
                  'D_fisher_scope': damage, 'D_fisher_scope_squared': damage ** 2,
                  'D_train': selected_rows[-1]['metrics']['train']['loss'] - origin['metrics']['train']['loss'],
                  'D_train_squared': (selected_rows[-1]['metrics']['train']['loss'] - origin['metrics']['train']['loss']) ** 2,
                  'endpoint_metrics': selected_rows[-1]['metrics'],
                  'origin_metrics': origin['metrics'],
                  'D_test': selected_rows[-1]['metrics']['test']['loss'] - origin['metrics']['test']['loss'],
                  'bound_passed_with_numeric_tolerance': gate['bound_passed'],
                  'bound_gap_I_minus_D2': value - damage ** 2,
                  'endpoint_recovery': dict(recovery, path_endpoint_loss_errors=end_errors),
                  'path': [{'s': i / (selected_nodes - 1), 'q': r['fisher_by_scope'][self.scope]['q'][name],
                            'raw_trace': r['fisher_by_scope'][self.scope]['raw_trace'],
                            'fisher_scope_loss_for_bound': r['fisher_by_scope'][self.scope]['subset_loss'],
                            'metrics': r['metrics']} for i, r in enumerate(selected_rows)]}
        if not gate['passed']:
            atomic_json(self.directory / 'numerical_gate_failure.json', {'status': 'needs_review', 'protocol_sha256': self.protocol, **result})
            raise RuntimeError('Path failed fixed numerical gate after 33 nodes; all measurements preserved, no completed claim')
        self.finish(result)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', required=True, choices=('sgd', 'gam'))
    parser.add_argument('--path-task', required=True, type=int, choices=range(1, 10))
    parser.add_argument('--job', choices=('path',), default='path')
    parser.add_argument('--device', default='cuda:0')
    return parser


def main():
    runner = StageRunner(argument_parser().parse_args())
    runner.path()


if __name__ == '__main__':
    main()
