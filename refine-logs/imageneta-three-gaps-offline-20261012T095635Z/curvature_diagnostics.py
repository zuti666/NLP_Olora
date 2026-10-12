#!/usr/bin/env python3
"""Separately versioned exploratory Hessian convergence study; no training.

The original (.001,.003,.01) gate is retained verbatim. The extended grid was
declared before any candidate-correction comparison and is labelled exploratory.
Passing first-order backend agreement never suffices to accept an HVP. Scientific
gate failures are saved and the next fixed cell proceeds; runtime faults raise.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
EXTENDED_DISTANCES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
PROTOCOL = 'response_chain_exploratory_hessian_convergence_v2_20261012'


def imports(campaign):
    """Resolve existing accepted kernels; never copy/edit campaign source."""
    sys.path.insert(0, str(Path(campaign) / 'trainer'))
    sys.path.insert(0, str(Path(campaign) / 'diagnostics'))
    global torch, DiagnosticEngine, comparison, norm, scaled, difference, dot
    global FD_DISTANCES, HVP_GATE_REL, GRADIENT_GATE_REL, GRADIENT_GATE_COS
    import torch
    from offline_core import (DiagnosticEngine, comparison, norm, scaled,
                              difference, dot, FD_DISTANCES, HVP_GATE_REL,
                              GRADIENT_GATE_REL, GRADIENT_GATE_COS)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def two_adjacent(records, threshold):
    """All grid points remain in output, including nonfinite/degenerate ones."""
    valid = [not r.get('degenerate', True) and
             r.get('relative_error') is not None and
             math.isfinite(r['relative_error']) and r['relative_error'] <= threshold
             for r in records]
    return any(left and right for left, right in zip(valid, valid[1:]))


def first_order_gate(engine, inputs, targets, reference=None):
    native = engine.gradient(inputs, targets) if reference is None else reference
    # For FP64 the immutable FP32 native gradient remains the reference. Repeat
    # means repeat the explicit FP64 read, not silently replace the reference.
    if reference is None:
        repeat = engine.gradient(inputs, targets)
        repeat_exact = all(torch.equal(a, b) for a, b in zip(native, repeat))
        explicit = engine.gradient(inputs, targets, 'math', chunked=True)
    else:
        explicit = engine.gradient(inputs, targets, 'math', chunked=True)
        repeat = engine.gradient(inputs, targets, 'math', chunked=True)
        repeat_exact = all(torch.equal(a, b) for a, b in zip(explicit, repeat))
    agree = comparison(explicit, native, norm(native))
    degenerate = norm(native) <= 1e-12 and norm(explicit) <= 1e-12 and repeat_exact
    accepted = bool(repeat_exact and not agree['degenerate'] and
                    agree['relative_error'] <= GRADIENT_GATE_REL and
                    agree['cosine'] >= GRADIENT_GATE_COS)
    return native, dict(repeated_gradient_exact=repeat_exact,
                       reference='native FP32 full128' if reference is not None
                       else 'native model full128', comparison=agree,
                       relative_error_max=GRADIENT_GATE_REL,
                       cosine_min=GRADIENT_GATE_COS, accepted=accepted,
                       degenerate=degenerate)


def curvature_gate(engine, inputs, targets, native_reference=None, progress=None):
    """Return detached g0,q,p plus all original/extended finite differences."""
    native, first = first_order_gate(engine, inputs, targets, native_reference)
    record = dict(first_order=first, accepted=False,
                  original_gate_accepted=False, extended_gate_accepted=False,
                  original_grid=list(FD_DISTANCES),
                  exploratory_grid=list(EXTENDED_DISTANCES),
                  hvp_relative_error_max=HVP_GATE_REL,
                  required_adjacent_points=2, microbatch=engine.microbatch,
                  precision=str(engine.params[0].dtype),
                  fixed_base=True, second_hvp_vector_detached=True,
                  first_order_agreement_is_not_higher_derivative_proof=True,
                  g0_norm=norm(native), q_norm=None, p_norm=None,
                  finite_difference={})
    if first['degenerate']:
        record['status'] = 'degenerate_gradient'
        return native, None, None, record
    if not first['accepted']:
        record['status'] = 'first_order_reference_disagreement'
        return native, None, None, record
    # In FP64 the HVP's input is the original native FP32 g0 converted exactly,
    # rather than an implicitly different first-order reference.
    vector = tuple(g.detach().to(p) for g, p in zip(native, engine.params))
    q = engine.hvp(inputs, targets, vector)
    record['q_norm'] = norm(q)
    if norm(q) <= 1e-12:
        record['status'] = 'degenerate_curvature'
        return native, q, None, record
    p = engine.hvp(inputs, targets, tuple(v.detach() for v in q))
    record['p_norm'] = norm(p)
    record['frozen_H_quadratic_form'] = .5 * dot(vector, q)
    record['quadratic_form_can_be_negative'] = True
    original_pass, extended_pass = [], []
    for name, direction, exact in (('H_g0', vector, q), ('H_detached_H_g0', q, p)):
        rows = []
        for distance in EXTENDED_DISTANCES:
            try:
                fd = engine.finite_difference_hvp(inputs, targets, direction, distance)
                row = dict(distance=distance, **comparison(fd, exact, norm(exact)))
                row['status'] = 'degenerate' if row['degenerate'] else 'finite'
            except FloatingPointError as error:
                row = dict(distance=distance, status='nonfinite',
                           degenerate=True, relative_error=None, cosine=None,
                           error=str(error))
            rows.append(row)
            if progress:
                progress(dict(event='finite_difference_point', product=name,
                              precision=record['precision'], **row))
        original = [dict(row) for row in rows if row['distance'] in FD_DISTANCES]
        op = two_adjacent(original, HVP_GATE_REL)
        ep = two_adjacent(rows, HVP_GATE_REL)
        original_pass.append(op)
        extended_pass.append(ep)
        record['finite_difference'][name] = dict(original=original,
                                                  original_passed=op,
                                                  exploratory=rows,
                                                  exploratory_passed=ep)
    record.update(original_gate_accepted=all(original_pass),
                  extended_gate_accepted=all(extended_pass),
                  accepted=all(extended_pass),
                  status='exploratory_converged' if all(extended_pass)
                  else 'hvp_convergence_not_verified')
    return native, q, p, record


def cast_cache(value, dtype):
    """Convert copied runtime floating tensors omitted by Module.to()."""
    if isinstance(value, torch.nn.Parameter):
        return value
    if torch.is_tensor(value):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {k: cast_cache(v, dtype) for k, v in value.items()}
    if isinstance(value, list):
        return [cast_cache(v, dtype) for v in value]
    if isinstance(value, tuple):
        return tuple(cast_cache(v, dtype) for v in value)
    return value


def float64_context(context, effective_step):
    """Independent model clone at exactly the represented FP32 BASE values."""
    from state_transaction import _cache_name
    model = copy.deepcopy(context.model).double()
    for module in model.modules():
        for name in list(vars(module)):
            if _cache_name(name) or name in ('momentum', 'fused_attn'):
                setattr(module, name, cast_cache(getattr(module, name), torch.float64))
    original = dict(context.model.named_parameters())
    for name, parameter in model.named_parameters():
        if not torch.equal(parameter.detach(), original[name].detach().double()):
            raise AssertionError('FP64 clone changed BASE parameter: ' + name)
    geometry = effective_step.from_model(model)
    if len(geometry.params) != len(context.geometry.params):
        raise AssertionError('FP64 coordinate count mismatch')
    for original_p, cloned_p in zip(context.geometry.params, geometry.params):
        if not torch.equal(cloned_p.detach(), original_p.detach().double()):
            raise AssertionError('FP64 trainable coordinate ordering mismatch')
    return SimpleNamespace(model=model, geometry=geometry,
                           loss_fn=lambda x, y: context.unbound_loss(model, x, y),
                           unbound_loss=context.unbound_loss,
                           State=context.State,
                           backend_context=context.backend_context)


def native_correction_rows(engine, inputs, targets, lr, g0, q, p, precision):
    """Native FP32 candidates, compared with accepted fixed-base curvature."""
    rows = []
    for scale in (1., .5):
        cand = engine.native_candidates(inputs, targets, lr, scale=scale, g0=g0)
        row = dict(scale=scale, status=cand['status'], degenerate=True,
                   cosine=None, relative_error=None,
                   delta_native_norm=None, delta_quadratic_norm=None,
                   curvature_precision=precision,
                   candidate_precision='torch.float32',
                   expression='Delta_native=g2-g0; Delta_quad=r*H.detach(H.g0)/||H.g0||',
                   native_candidate_kernel='immutable offline_core.DiagnosticEngine.native_candidates')
        for key in ('inner_radius', 'outer_radius', 'response_norm',
                    'response_degeneracy_threshold'):
            if key in cand:
                row[key] = cand[key]
        if cand['status'] == 'accepted':
            dn = difference(cand['g2'], g0)
            dq = scaled(p, .25 * scale / norm(q))
            row.update(comparison(dn, dq, norm(g0)),
                       delta_native_norm=norm(dn), delta_quadratic_norm=norm(dq),
                       g0_norm=norm(g0), g2_norm=norm(cand['g2']))
            if row['degenerate']:
                row['status'] = 'degenerate_correction'
        rows.append(row)
    return rows


def diagnostic_cell(context, inputs, targets, lr, allow_fp64=False,
                    effective_step=None, progress=None):
    started = time.perf_counter()
    engine = DiagnosticEngine(context, 16)
    record = dict(submitted_training_updates=0, scales=[],
                  fp64_fallback_declared=allow_fp64,
                  candidate_comparisons_permitted=False)
    try:
        g0, q, p, gate = curvature_gate(engine, inputs, targets, progress=progress)
        record['fp32'] = gate
        accepted, precision = gate['accepted'], 'torch.float32'
        if not accepted and allow_fp64 and gate['first_order']['accepted'] and q is not None:
            clone_context = float64_context(context, effective_step)
            engine64 = DiagnosticEngine(clone_context, 4)
            try:
                _, q64, p64, gate64 = curvature_gate(
                    engine64, inputs.double(), targets,
                    native_reference=g0, progress=progress)
                gate64['same_represented_BASE_parameters_verified'] = True
                gate64['original_FP32_g0_detached_and_converted'] = True
                if q64 is not None and q is not None:
                    gate64['FP32_vs_FP64_H_g0'] = comparison(q, q64, norm(q64))
                if p64 is not None and p is not None:
                    gate64['FP32_vs_FP64_H_detached_H_g0'] = comparison(p, p64, norm(p64))
                accepted = gate64['accepted']
                if accepted:
                    q, p, precision = q64, p64, 'torch.float64_exploratory_fallback'
                record['fp64'] = gate64
            finally:
                record['fp64_restoration'] = engine64.close()
                record['fp64_gradient_forward_calls'] = engine64.reads
                record['fp64_hvp_microbatch_calls'] = engine64.hvp_calls
            del clone_context, engine64
        record['candidate_comparisons_permitted'] = bool(accepted)
        if accepted:
            record['scales'] = native_correction_rows(engine, inputs, targets, lr,
                                                       g0, q, p, precision)
            record['status'] = 'exploratory_converged_corrections_measured'
            record['finite_displacement_analysis'] = dict(
                predeclared_joint_scales=[1., .5],
                both_inner_and_outer_distances_halved=True,
                candidate_agreement_is_separate_from_HVP_convergence=True,
                hypothesis='If correction error shrinks at half scale after HVP convergence, finite displacement contributes to approximation bias',
                no_posthoc_correction_agreement_threshold=True)
        else:
            record['status'] = 'inconclusive_no_candidate_comparison'
            record['candidate_comparisons_skip_reason'] = 'No independently accepted Hessian convergence gate'
        if not gate['first_order']['accepted']:
            record['failure_layer'] = 'first_order_reference'
        elif not accepted:
            record['failure_layer'] = 'higher_derivative_numerical_convergence'
        elif gate['original_gate_accepted']:
            record['failure_layer'] = None
        else:
            record['failure_layer'] = 'original_distance_grid_does_not_verify_HVP; exploratory_grid_or_precision_does'
    except FloatingPointError as error:
        record.update(status='inconclusive_nonfinite_numerics',
                      numerical_exception=str(error),
                      candidate_comparisons_permitted=False)
    finally:
        record.update(restoration=engine.close(),
                      elapsed_seconds=time.perf_counter() - started,
                      native_engine_gradient_forward_calls=engine.reads,
                      native_engine_hvp_microbatch_calls=engine.hvp_calls)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--fp64-pilot-fallback', action='store_true')
    cli = parser.parse_args()
    campaign = Path(cli.campaign).resolve()
    output = Path(cli.output_dir).resolve()
    checkpoints = [Path(p).resolve() for p in cli.checkpoint]
    if cli.fp64_pilot_fallback and not cli.pilot:
        raise ValueError('FP64 fallback is limited to declared pilot failure point')
    if cli.pilot and (cli.seed != 2 or len(checkpoints) != 1 or
                      checkpoints[0].parent.parent.name != 'formal_S_s2' or
                      checkpoints[0].name != 'epoch_001.pt'):
        raise ValueError('Pilot fixed to formal S seed2 epoch1 batch0')
    if not cli.pilot:
        required = {('formal_' + b + '_s' + str(cli.seed), 'epoch_%03d.pt' % e)
                    for b in ('S', 'R') for e in (1, 10, 25)}
        observed = {(p.parent.parent.name, p.name) for p in checkpoints}
        if len(checkpoints) != 6 or observed != required:
            raise ValueError('Full scope must contain S/R x epochs1/10/25 exactly once')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Exploratory output must be fresh')
    output.mkdir(parents=True, exist_ok=True)
    imports(campaign)
    from response_chain_train import load_diagnostic_context
    from state_transaction import State, backend_context
    from effective_step import EffectiveStep
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Inherited deterministic CUBLAS workspace required')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '2'))))
    aggregate = dict(protocol=PROTOCOL, campaign=str(campaign), seed=cli.seed,
                     pilot=cli.pilot, status='running', results=[],
                     submitted_training_updates=0,
                     original_preregistered_gate_unchanged=True,
                     original_failure_not_overwritten=True,
                     exploratory_grid_declared=list(EXTENDED_DISTANCES),
                     independent_repeat_unit='training seed; checkpoints/batches are not independent replicates',
                     code_sha256=digest(Path(__file__)),
                     immutable_core_sha256=digest(campaign / 'diagnostics/offline_core.py'),
                     time_start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    atomic_json(output / 'diagnostic_results.json', aggregate)
    try:
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            data = load_diagnostic_context(str(campaign), str(checkpoint), cli.seed)
            model, payload = data['model'], data['payload']
            geometry = EffectiveStep.from_model(model)
            context = SimpleNamespace(model=model, geometry=geometry,
                                      unbound_loss=data['loss_fn'],
                                      loss_fn=lambda x, y: data['loss_fn'](model, x, y),
                                      State=State, backend_context=backend_context)
            lr = float(payload['next_lr'])
            if any(float(group['lr']) != lr for group in payload['optimizer_state']['param_groups']):
                raise AssertionError('Saved next-update LR reference mismatch')
            epoch = int(payload['epoch'])
            batches = data['batches'][:1] if cli.pilot else data['batches']
            if len(batches) != (1 if cli.pilot else 2):
                raise AssertionError('Fixed diagnostic batch count mismatch')
            for batch_index, batch in enumerate(batches):
                inputs, targets = batch['inputs'], batch['targets']
                if targets.numel() != 128 or not bool(((targets >= 20) & (targets < 40)).all()):
                    raise AssertionError('Fixed Task1 training minibatch128 required')
                if next(model.parameters()).device.type != 'cuda':
                    raise RuntimeError('Full ViT diagnostics require scheduled CUDA')
                identity = dict(seed=cli.seed, branch=payload['branch'],
                                epoch=epoch, checkpoint=str(checkpoint),
                                checkpoint_sha256=digest(checkpoint),
                                checkpoint_index=checkpoint_index,
                                batch_index=batch_index, batch_identity=batch['identity'],
                                probe_lr=lr, epoch_lr_used=float(payload['epoch_lr_used']),
                                probe_lr_rule='saved next-update optimizer LR',
                                class_slice=[20, 40], loss='native mean current20 training CE')
                print(json.dumps(dict(event='cell_start', **identity)), flush=True)
                result = diagnostic_cell(context, inputs, targets, lr,
                                         cli.fp64_pilot_fallback, EffectiveStep,
                                         progress=lambda r: print(json.dumps(r), flush=True))
                row = dict(**identity, diagnostic=result)
                aggregate['results'].append(row)
                atomic_json(output / ('checkpoint%02d_batch%02d.json' % (checkpoint_index, batch_index)), row)
                atomic_json(output / 'diagnostic_results.json', aggregate)
                print(json.dumps(dict(event='cell_complete', status=result['status'],
                                      epoch=epoch, branch=payload['branch'],
                                      batch_index=batch_index)), flush=True)
            del context, geometry, model, data
            torch.cuda.empty_cache()
        valid = sum(r['diagnostic']['candidate_comparisons_permitted'] for r in aggregate['results'])
        original_valid = sum(r['diagnostic'].get('fp32', {}).get('original_gate_accepted', False)
                             for r in aggregate['results'])
        aggregate.update(status='completed_exploratory',
                         completed_cells=len(aggregate['results']),
                         candidate_comparisons_valid_cells=valid,
                         original_FP32_gate_valid_cells=original_valid,
                         scientific_outcome='some_converged' if valid else 'inconclusive_no_converged_cells',
                         time_end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        atomic_json(output / 'diagnostic_results.json', aggregate)
        atomic_json(output / 'completion_receipt.json', dict(
            status='accepted_execution', scientific_outcome=aggregate['scientific_outcome'],
            completed_cells=len(aggregate['results']), seed=cli.seed,
            submitted_training_updates=0,
            state_restoration_passed=all(all(r['diagnostic']['restoration'].values()) for r in aggregate['results']),
            results_sha256=digest(output / 'diagnostic_results.json')))
        # Gate failure is a scientific finding, not an execution failure.
        return 0
    except Exception as error:
        aggregate.update(status='runtime_failed', exception_type=type(error).__name__,
                         exception_message=str(error))
        atomic_json(output / 'diagnostic_results.json', aggregate)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
