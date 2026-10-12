#!/usr/bin/env python3
"""Zero-update numerical revalidation of the registered Stage2 FP32 HVP.

The registered FP32 point and physical displacement are reconstructed once.
FP64 is an independent numerical reference at exact promotions of those saved
coordinates, never a replacement training protocol. A shard cannot authorize
Stage2. A failed cross-precision production HVP cannot be rescued by a passing
FP64 centered difference. Original run, source and checkpoints are read-only.
"""
import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

FINE_H = (.001, .0003, .0001, .00003, .00001, .000003)
REL_MAX = .005
COS_MIN = .999
SAME_GRAD_MAX = .00002


def accepted(stats):
    return (stats['relative_l2'] is not None and stats['relative_l2'] <= REL_MAX
            and stats['cosine'] is not None and stats['cosine'] >= COS_MIN
            and not stats['degenerate'])


def adjacent(scan, key):
    return any(a[key] and b[key] for a, b in zip(scan, scan[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--task', type=int, choices=(0, 4, 9), required=True,
                        help='One registered task per shard; both batches and both responses are mandatory.')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run, out = Path(args.run).resolve(), Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    target = out / 'hvp_revalidation_report.json'
    if target.exists():
        raise FileExistsError('Refusing to overwrite prior revalidation evidence: ' + str(target))
    original_path = run / 'outputs/HVP_full_gate/hvp_gate_report.json'
    original = json.loads(original_path.read_text())
    source = Path(original['source']).resolve()
    reference = Path(original['reference_output']).resolve()
    checkpoint = reference / 'checkpoints' / ('task_%03d' % args.task)
    oldtask = original['tasks'][str(args.task)]
    assert [b['batch'] for b in oldtask['batches']] == [0, 1]

    sys.path.insert(0, str(run / 'trainer'))
    from hvp_gate import (sha, tensor_sha, model_sha, pure_config,
                          seed_all, normalized, write_json)
    from response_hvp import State, rng_get, gradient_at, compare, vector_norm

    def source_pins():
        return {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*.py'))}

    def checkpoint_pins():
        return {p.name: sha(p) for p in checkpoint.iterdir() if p.is_file()}

    protected = [original_path, run / 'runtime/queue_state.json',
                 run / 'trainer/hvp_gate.py', run / 'trainer/response_hvp.py',
                 run / 'trainer/direction_step.py', run / 'trainer/four_probe_train.py',
                 reference / 'effective_config.json']
    protected_hashes = {str(p): sha(p) for p in protected}
    assert source_pins() == original['source_python_sha256'], 'Frozen source mismatch'
    assert checkpoint_pins() == oldtask['checkpoint_sha256'], 'Checkpoint mismatch'
    assert sha(run / 'trainer/response_hvp.py') == original['hvp_code_sha256']
    assert sha(run / 'trainer/hvp_gate.py') == original['gate_code_sha256']
    assert sha(reference / 'effective_config.json') == original['config_sha256']

    sys.path.insert(0, str(source))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.chdir(source)
    from utils.data_manager import DataManager
    from utils import factory
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    raw = json.loads((reference / 'effective_config.json').read_text())
    cfg = pure_config(raw, device, out)
    seed_all(cfg['seed'])
    dm = DataManager(cfg['dataset'], cfg.get('class_shuffle', False), cfg['seed'],
                     cfg['init_cls'], cfg['increment'], cfg)
    cfg.update(nb_classes=dm.nb_classes, task_increments=list(dm._increments), nb_tasks=dm.nb_tasks)
    learner = factory.get_model(cfg['model_name'], cfg)
    learner._unwrap_network().backbone.to(device)
    learner.load_task_checkpoint(str(checkpoint), args.task, dm)
    model = learner._unwrap_network().to(device)
    model.train()
    initial_hash = model_sha(model)
    assert initial_hash == oldtask['model_sha256'], 'Original model reconstruction mismatch'
    nonzero_dropout = [name for name, m in model.named_modules()
                       if (isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p > 0)
                       or float(getattr(m, 'drop_prob', 0.) or 0.) > 0]
    batchnorm = [name for name, m in model.named_modules()
                 if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training]
    assert not nonzero_dropout, 'Nonzero stochastic dropout prevents fixed chunked objective'
    assert not batchnorm, 'Training BatchNorm prevents sample-separable chunk equivalence'
    start, end = dm.get_task_class_range(args.task)
    assert [start, end] == oldtask['class_range']
    dataset = dm.get_dataset(np.arange(start, end), source='train', mode='train')
    data_seed = cfg['seed'] * 100000 + 70000 + args.task
    assert data_seed == oldtask['data_seed']
    seed_all(data_seed)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(data_seed), drop_last=False)
    names, params = zip(*[(name, p) for name, p in model.named_parameters() if p.requires_grad])
    assert all(p.dtype == torch.float32 for p in params)
    groups = {}
    for i, name in enumerate(names):
        match = re.search(r'\.blocks\.(\d+)\.', name)
        group = 'block_%02d' % int(match.group(1)) if match else 'fc' if name.startswith('fc.') else 'other'
        groups.setdefault(group, []).append(i)

    def block_errors(a, b):
        return {key: compare(tuple(a[i] for i in ix), tuple(b[i] for i in ix))
                for key, ix in groups.items()}

    def save_tensors(filename, items):
        torch.save({key: ({name: v.detach().cpu() for name, v in zip(names, values)}
                           if isinstance(values, (list, tuple)) else values)
                    for key, values in items.items()}, out / filename)
        return {'file': filename, 'sha256': sha(out / filename)}

    report = {
        'schema_version': 1, 'status': 'running', 'task': args.task, 'batches': [],
        'full_stage2_allowed': False, 'shard_only': True, 'optimizer_updates': 0,
        'fisher_calls': 0, 'weight_merge_calls': 0,
        'criteria': {'relative_l2_max': REL_MAX, 'cosine_min': COS_MIN,
                     'same_point_gradient_relative_max': SAME_GRAD_MAX,
                     'two_adjacent_h_required': True, 'fine_h_grid': list(FINE_H),
                     'production_fp32_vs_fp64_required': True,
                     'production_fp32_vs_fp64_center_fd_two_adjacent_required': True},
        'source': str(source), 'source_python_sha256': original['source_python_sha256'],
        'reference_output': str(reference), 'checkpoint': str(checkpoint),
        'checkpoint_sha256': oldtask['checkpoint_sha256'],
        'protected_files_before_sha256': protected_hashes,
        'script_sha256': sha(Path(__file__)), 'original_gate_sha256': sha(original_path),
        'model_sha256': initial_hash, 'class_range': [start, end], 'data_seed': data_seed,
        'coordinate_names': list(names), 'coordinate_shapes': [list(p.shape) for p in params],
        'precision': {'production': 'FP32 full batch128 math attention',
                      'reference': 'FP64 sample-weighted two64 chunks math attention',
                      'tf32': False, 'autocast': False},
        'chunk_objective': {'chunk_size': 64, 'same_fixed_samples': 128,
                            'same_scalar_mean_ce': True, 'nonzero_dropout': nonzero_dropout,
                            'training_batchnorm': batchnorm,
                            'numerical_path_changed': True,
                            'limit': 'Chunk shapes and reduction order differ; cross-precision discrepancy cannot be attributed to dtype alone.'},
        'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'},
        'acceptance_scope': 'One task shard only. Numerical pass never bypasses other Stage2 gates.',
    }
    write_json(target, report)
    beginning = time.perf_counter()
    initial_state = State(model)
    try:
        iterator = iter(loader)
        for batch_index, oldbatch in enumerate(oldtask['batches']):
            ids, x_cpu, y_cpu = next(iterator)
            assert len(y_cpu) == oldbatch['samples'] == 128
            assert ids.tolist() == oldbatch['indices'], 'Original batch indices mismatch'
            assert tensor_sha(x_cpu) == oldbatch['input_sha256'], 'Original input mismatch'
            assert tensor_sha(y_cpu) == oldbatch['target_sha256'], 'Original targets mismatch'
            x, y = x_cpu.to(device), y_cpu.to(device)
            entry = State(model)
            rng = rng_get()
            point0 = tuple(p.detach().clone() for p in params)

            def closure_for(m, inputs, targets):
                def closure():
                    logits = m(inputs)
                    logits = logits['logits'] if isinstance(logits, dict) else logits
                    if logits.ndim != 2 or logits.shape[1] != end:
                        raise ValueError('Current-task CE class slice mismatch')
                    return F.cross_entropy(logits[:, start:end], targets - start)
                return closure

            closure = closure_for(model, x, y)
            _, g0, _, _ = gradient_at(model, params, point0, closure, rng, 'math')
            phi1 = tuple((p + .05 * u).detach() for p, u in zip(point0, normalized(g0)))
            _, g1, _, _ = gradient_at(model, params, phi1, closure, rng, 'math')
            c10 = tuple(a - b for a, b in zip(g1, g0))
            phi2 = tuple((p + .25 * u).detach() for p, u in zip(point0, normalized(c10)))
            direction = tuple((b - a).detach() for a, b in zip(phi1, phi2))
            middle = tuple(((a + b) * .5).detach() for a, b in zip(phi1, phi2))
            assert abs(vector_norm(direction) - oldbatch['d12_norm']) <= 1e-12
            batch = {'batch': batch_index, 'samples': len(y), 'indices': ids.tolist(),
                     'input_sha256': tensor_sha(x_cpu), 'target_sha256': tensor_sha(y_cpu),
                     'd12_norm': vector_norm(direction), 'responses': {},
                     'coordinate_file': save_tensors('batch%d_coordinates.pt' % batch_index,
                        {'point0': point0, 'phi1': phi1, 'phi2': phi2, 'middle': middle,
                         'direction': direction, 'g0': g0, 'g1': g1,
                         'inputs': x_cpu, 'targets': y_cpu, 'indices': ids})}
            report['batches'].append(batch)
            write_json(target, report)
            double_model = None
            try:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                double_model = copy.deepcopy(model).double()
                double_names, double_params = zip(*[(n, p) for n, p in double_model.named_parameters() if p.requires_grad])
                assert tuple(double_names) == tuple(names)
                # Casting must preserve every registered weight coordinate exactly.
                for (n32, p32), (n64, p64) in zip(model.named_parameters(), double_model.named_parameters()):
                    assert n32 == n64 and torch.equal(p32.detach().double(), p64.detach())
                for (n32, b32), (n64, b64) in zip(model.named_buffers(), double_model.named_buffers()):
                    assert n32 == n64
                    expected = b32.double() if b32.is_floating_point() else b32
                    assert torch.equal(expected, b64)
                # Native loader clears historical eval artifacts. Assert no active
                # unregistered saved factors escaped Module.double().
                assert all(not getattr(m, key, {}) for m in double_model.modules()
                           for key in ('saved_A', 'saved_B') if hasattr(m, key))
                x64 = x.double()

                def evaluate64(point, need_hvp=False):
                    gs = [torch.zeros_like(p) for p in double_params]
                    hs = [torch.zeros_like(p) for p in double_params] if need_hvp else None
                    detail, loss_total = [], 0.
                    for lo in (0, 64):
                        hi = lo + 64
                        fun = closure_for(double_model, x64[lo:hi], y[lo:hi])
                        loss, gradient, hvp, info = gradient_at(double_model, double_params, point,
                            fun, rng, 'math', need_hvp, d64 if need_hvp else None)
                        for result, value in zip(gs, gradient):
                            result.add_(value, alpha=.5)
                        if need_hvp:
                            for result, value in zip(hs, hvp):
                                result.add_(value, alpha=.5)
                        loss_total += .5 * loss
                        detail.append({'slice': [lo, hi], 'weight': .5, 'evaluation': info})
                    return loss_total, tuple(gs), tuple(hs) if need_hvp else None, detail

                for label, point in [('hstart', phi1), ('hmid', middle)]:
                    prefix = 'batch%d_%s' % (batch_index, label)
                    response = {'status': 'running', 'float32_scan': [], 'float64_scan': []}
                    batch['responses'][label] = response
                    write_json(target, report)
                    _, ordinary32, _, ordinary32_meta = gradient_at(model, params, point, closure, rng, 'math')
                    _, graph32, hv32, hv32_meta = gradient_at(model, params, point, closure, rng, 'math', True, direction)
                    same32 = compare(graph32, ordinary32)
                    response['production_fp32'] = {'same_point_gradient': same32,
                        'ordinary_gradient': ordinary32_meta, 'hvp': hv32_meta,
                        'base_file': save_tensors(prefix + '_fp32_base.pt',
                            {'point': point, 'direction': direction, 'ordinary_gradient': ordinary32,
                             'graph_gradient': graph32, 'hvp': hv32})}
                    old = oldbatch['responses'][label]
                    norm_delta = abs(vector_norm(hv32) - old['hvp']['hvp_norm'])
                    response['original_fp32_hvp_norm_absolute_delta'] = norm_delta
                    if norm_delta >= 1e-8:
                        raise AssertionError('Original production HVP norm reproduction mismatch')

                    # FP32 is preserved in its exact production arithmetic. The
                    # fine grid diagnoses its saturation but cannot rescue a
                    # cross-precision failure.
                    for h in FINE_H:
                        plus = tuple((p + h * d).detach() for p, d in zip(point, direction))
                        minus = tuple((p - h * d).detach() for p, d in zip(point, direction))
                        _, gp, _, gp_info = gradient_at(model, params, plus, closure, rng, 'math')
                        _, gm, _, gm_info = gradient_at(model, params, minus, closure, rng, 'math')
                        fd = tuple((a - b) / (2 * h) for a, b in zip(gp, gm))
                        stats = compare(fd, hv32)
                        desired = tuple(h * d.double() for d in direction)
                        actual_plus = tuple(p.double() - v.double() for p, v in zip(plus, point))
                        actual_minus = tuple(v.double() - p.double() for p, v in zip(minus, point))
                        active_count = sum(int((d != 0).sum()) for d in direction)
                        active_lost = sum(int(((d != 0) & (p == v)).sum()) for d, p, v in zip(direction, plus, point))
                        row = {'h': h, 'central_vs_hvp': stats, 'accepted': accepted(stats),
                            'per_block': block_errors(fd, hv32),
                            'actual_plus_delta_vs_h_d': compare(actual_plus, desired),
                            'actual_minus_delta_vs_h_d': compare(actual_minus, desired),
                            'effective_direction_vs_d': compare(tuple((p.double() - m.double()) / (2*h)
                                for p, m in zip(plus, minus)), tuple(d.double() for d in direction)),
                            'midpoint_drift_norm': vector_norm(tuple((p.double() + m.double())*.5 - v.double()
                                for p, m, v in zip(plus, minus, point))),
                            'active_plus_coordinates_lost_fraction': active_lost / active_count if active_count else None,
                            'nominal_offset_norm': vector_norm(desired),
                            'gp_report': gp_info, 'gm_report': gm_info,
                            'tensor_file': save_tensors(prefix + '_fp32_h' + format(h, '.0e') + '.pt',
                                {'plus': plus, 'minus': minus, 'gp': gp, 'gm': gm, 'fd': fd, 'hvp': hv32})}
                        response['float32_scan'].append(row)
                        if h == .001:
                            old_row = next(v for v in old['scan'] if v['h'] == .001)
                            delta = abs(stats['relative_l2'] - old_row['central_vs_hvp']['relative_l2'])
                            response['original_fp32_fd_relative_error_absolute_delta'] = delta
                            if delta >= 1e-8:
                                raise AssertionError('Original production centered FD reproduction mismatch')
                        write_json(target, report)

                    point64 = tuple(p.double() for p in point)
                    d64 = tuple(d.double() for d in direction)
                    _, ordinary64, _, ordinary64_meta = evaluate64(point64)
                    _, graph64, hv64, hv64_meta = evaluate64(point64, True)
                    same64 = compare(graph64, ordinary64)
                    cross = compare(tuple(h.double() for h in hv32), hv64)
                    response['fp64_reference'] = {'same_point_gradient': same64,
                        'ordinary_gradient': ordinary64_meta, 'hvp': hv64_meta,
                        'production_fp32_vs_fp64_hvp': cross,
                        'production_fp32_vs_fp64_hvp_blocks': block_errors(tuple(h.double() for h in hv32), hv64),
                        'base_file': save_tensors(prefix + '_fp64_base.pt',
                            {'point': point64, 'direction': d64, 'ordinary_gradient': ordinary64,
                             'graph_gradient': graph64, 'hvp': hv64})}
                    write_json(target, report)
                    for h in FINE_H:
                        plus = tuple((p + h*d).detach() for p, d in zip(point64, d64))
                        minus = tuple((p - h*d).detach() for p, d in zip(point64, d64))
                        _, gp, _, gp_meta = evaluate64(plus)
                        _, gm, _, gm_meta = evaluate64(minus)
                        fd = tuple((a-b)/(2*h) for a, b in zip(gp, gm))
                        stats = compare(fd, hv64)
                        production_vs_fd = compare(tuple(v.double() for v in hv32), fd)
                        desired = tuple(h*d for d in d64)
                        row = {'h': h, 'central_vs_hvp': stats, 'accepted': accepted(stats),
                            'production_fp32_vs_fp64_center_fd': production_vs_fd,
                            'production_accepted': accepted(production_vs_fd),
                            'joint_accepted': accepted(stats) and accepted(production_vs_fd),
                            'per_block': block_errors(fd, hv64),
                            'production_fp32_vs_fp64_center_fd_blocks': block_errors(tuple(v.double() for v in hv32), fd),
                            'actual_plus_delta_vs_h_d': compare(tuple(p-v for p, v in zip(plus, point64)), desired),
                            'actual_minus_delta_vs_h_d': compare(tuple(v-m for v, m in zip(point64, minus)), desired),
                            'effective_direction_vs_d': compare(tuple((p-m)/(2*h) for p, m in zip(plus, minus)), d64),
                            'nominal_offset_norm': vector_norm(desired),
                            'gp_report': gp_meta, 'gm_report': gm_meta,
                            'tensor_file': save_tensors(prefix + '_fp64_h' + format(h, '.0e') + '.pt',
                                {'plus': plus, 'minus': minus, 'gp': gp, 'gm': gm, 'fd': fd, 'hvp': hv64})}
                        response['float64_scan'].append(row)
                        write_json(target, report)
                        print(json.dumps({'task': args.task, 'batch': batch_index, 'response': label,
                            'h': h, 'fp64_fd_error': stats['relative_l2'],
                            'production_fp32_vs_fp64_fd_error': production_vs_fd['relative_l2'],
                            'joint_accepted': row['joint_accepted']}), flush=True)
                    flags = {
                        'production_same_point_gradient': same32['relative_l2'] is not None and same32['relative_l2'] <= SAME_GRAD_MAX,
                        'reference_same_point_gradient': same64['relative_l2'] is not None and same64['relative_l2'] <= SAME_GRAD_MAX,
                        'fp64_center_fd_two_adjacent': adjacent(response['float64_scan'], 'accepted'),
                        'production_fp32_vs_fp64_hvp': accepted(cross),
                        'production_and_reference_joint_fd_two_adjacent': adjacent(response['float64_scan'], 'joint_accepted'),
                    }
                    response['acceptance_checks'] = flags
                    response['fp32_center_fd_two_adjacent'] = adjacent(response['float32_scan'], 'accepted')
                    response['status'] = 'passed' if all(flags.values()) else 'failed'
                    response['failed_checks'] = [key for key, value in flags.items() if not value]
                    write_json(target, report)
            finally:
                if double_model is not None:
                    del double_model, double_params
                entry.restore()
                batch['entry_state_restore'] = entry.check()
                assert model_sha(model) == initial_hash
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                write_json(target, report)
            batch['status'] = 'passed' if all(r['status'] == 'passed' for r in batch['responses'].values()) else 'failed'
            write_json(target, report)
        report['status'] = 'passed' if all(b['status'] == 'passed' for b in report['batches']) else 'failed'
        report['shard_numerical_pass'] = report['status'] == 'passed'
    except Exception as exc:
        report.update(status='error', error=repr(exc), shard_numerical_pass=False)
        raise
    finally:
        initial_state.restore()
        report['initial_state_restore'] = initial_state.check()
        report['model_unchanged'] = model_sha(model) == initial_hash
        report['source_unchanged'] = source_pins() == original['source_python_sha256']
        report['checkpoint_unchanged'] = checkpoint_pins() == oldtask['checkpoint_sha256']
        report['protected_files_after_sha256'] = {str(p): sha(p) for p in protected}
        report['original_protected_files_unchanged'] = report['protected_files_after_sha256'] == protected_hashes
        report['seconds'] = time.perf_counter() - beginning
        report['full_stage2_allowed'] = False
        integrity = (report['model_unchanged'] and report['source_unchanged'] and report['checkpoint_unchanged']
                     and report['original_protected_files_unchanged'] and all(report['initial_state_restore'].values()))
        if not integrity:
            report.update(status='error',shard_numerical_pass=False,error='Final integrity/restoration check failed')
        write_json(target, report)
        assert integrity
    print(json.dumps({'status': report['status'], 'shard_numerical_pass': report['shard_numerical_pass'],
                      'full_stage2_allowed': False, 'report': str(target)}), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
