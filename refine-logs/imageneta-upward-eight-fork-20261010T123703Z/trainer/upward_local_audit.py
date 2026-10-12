#!/usr/bin/env python3
"""Eight upward-probe updates at shared states; TRAIN-only and never committed.

This diagnostic is separate from formal training. All eight candidates are
constructed from ONE set of four gradients at each state/minibatch. The native
cost microbenchmark really skips unused probes and restores every trial. It is
not included in training purity counters or formal training wall time.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
import torch
import torch.nn.functional as F

from effective_step import EffectiveStep
from local_probe_audit import (LocalProbeEngine, atomic_json, cosine,
                               deterministic_batches, file_sha, norm,
                               normalized, tensor_sha)
from state_transaction import State, rng_equal, rng_get, rng_set

EPS = 1e-12
DIRECTIONS = ('g0', 'g1', 'g2_minus_s', 'g2', 'g2_plus_s',
              'g0_plus_d', 'g1_plus_d', 'g3')
READS = dict(zip(DIRECTIONS, (1, 2, 3, 3, 3, 4, 4, 4)))
FORMULAS = dict(zip(DIRECTIONS, ('g0', 'g1', 'g2-(g1-g0)', 'g2',
                     'g2+(g1-g0)', 'g0+(g3-g2)', 'g1+(g3-g2)', 'g3')))


def construct_candidates(g0, g1, g2, g3):
    """Prespecified coefficient-one residual matrix; no projection/scaling."""
    s, d = g1 - g0, g3 - g2
    # Match the registered native FP32 arithmetic, not just symbolic identities.
    values = (g0, g1, g2 - s, g2, g2 + s,
              g0 + d, g1 + d, g3)
    return {key: value.clone() for key, value in zip(DIRECTIONS, values)}


class UpwardLocalEngine(LocalProbeEngine):
    def native_norm(self, vector):
        # Native implementation sums FP64 tensor-local norms in parameter order.
        return math.sqrt(max(0., float(torch.stack([
            vector[a:b].double().square().sum() for a, b in self.slices]).sum())))

    @torch.no_grad()
    def _place(self, direction=None, scale=0.):
        """Allow sequential probe add_ operations, matching native FP32 order."""
        if not isinstance(direction, (tuple, list)):
            return super()._place(direction, scale)
        self.entry.restore()
        for vector, coefficient in direction:
            if vector.numel() != self.dimension or not bool(torch.isfinite(vector).all()):
                raise ValueError('Invalid sequential displacement')
            for p, (a, b) in zip(self.params, self.slices):
                p.add_(vector[a:b].reshape_as(p), alpha=float(coefficient))
        self.virtual_parameter_writes += 1

    @torch.no_grad()
    def loss_for_classes(self, batch, class_slice, direction=None, scale=0.):
        """CE on fixed TRAIN holdouts; old labels never enter gradients."""
        self._place(direction, scale)
        try:
            distance = self._distance()
            inputs, targets = batch
            output = self.model(inputs)
            logits = output['logits'] if isinstance(output, dict) else output
            lo, hi = class_slice
            if logits.shape[1] < hi or not bool(((targets >= lo) & (targets < hi)).all()):
                raise AssertionError('Holdout label/logit scope mismatch')
            ce = F.cross_entropy(logits[:, lo:hi], targets - lo)
            if not bool(torch.isfinite(ce)):
                raise FloatingPointError('Nonfinite heldout loss')
            self.loss_reads += 1
            return float(ce), distance
        finally:
            self._restore()


def audit_eight_batch(engine, probe_batch, current_holdout, old_holdout,
                      eta, rho=.05, radius=1.):
    """One numerical g0..g3 set, eight updates, native/matched step controls."""
    if eta <= 0 or rho <= 0 or radius <= 0:
        raise ValueError('Positive diagnostic radii and LR required')
    base = engine.gradient(probe_batch)
    g0 = base['gradient']
    first = ((g0, float(rho)/(engine.native_norm(g0)+EPS)),)
    read1 = engine.gradient(probe_batch, first)
    g1 = read1['gradient']
    response = g1 - g0
    outer = ((response, float(radius)/(engine.native_norm(response)+EPS)),)
    read2 = engine.gradient(probe_batch, outer)
    g2 = read2['gradient']
    read3 = engine.gradient(probe_batch, outer + ((g2, float(rho)/(engine.native_norm(g2)+EPS)),))
    g3 = read3['gradient']
    candidates = construct_candidates(g0, g1, g2, g3)
    current_before, _ = engine.loss_for_classes(current_holdout, (20, 40))
    current_seen_before, _ = engine.loss_for_classes(current_holdout, (0, 40))
    old_before, _ = engine.loss_for_classes(old_holdout, (0, 40))
    old20_before, _ = engine.loss_for_classes(old_holdout, (0, 20))
    reference_nominal = eta * norm(g0)
    reference_actual = engine.virtual_step_distance(g0, eta)['parameter_norm']
    rows = []
    for method in DIRECTIONS:
        u = candidates[method]
        magnitude = norm(u)
        coefficient = norm(g0) / magnitude if magnitude > EPS else 0.
        for mode, alpha in (('native', 1.), ('nominal_parameter_length_match', coefficient)):
            step_scale = -float(eta) * alpha
            # This is g(theta + Delta_U), not the curvature probe g2.
            after = engine.gradient(probe_batch, u, step_scale)
            actual = after['distance']['parameter_norm']
            current_after, _ = engine.loss_for_classes(current_holdout, (20, 40), u, step_scale)
            current_seen_after, _ = engine.loss_for_classes(current_holdout, (0, 40), u, step_scale)
            old_after, _ = engine.loss_for_classes(old_holdout, (0, 40), u, step_scale)
            old20_after, _ = engine.loss_for_classes(old_holdout, (0, 20), u, step_scale)
            nominal = eta * alpha * magnitude
            rows.append(dict(
                method=method, gradient_formula=FORMULAS[method], mode=mode,
                nominal_lr=float(eta), diagnostic_coefficient=alpha,
                update_gradient_norm=magnitude, update_sha256=tensor_sha(u),
                base_gradient_sha256=base['gradient_sha256'],
                post_update_gradient_sha256=after['gradient_sha256'],
                nominal_parameter_step=nominal, actual_parameter_step=actual,
                target_g0_nominal_parameter_step=reference_nominal,
                reference_g0_actual_parameter_step=reference_actual,
                relative_error_to_nominal_target=abs(actual-reference_nominal)/reference_nominal
                    if reference_nominal > EPS else None,
                relative_error_to_actual_g0_step=abs(actual-reference_actual)/reference_actual
                    if reference_actual > EPS else None,
                effective_BA_plus_FC_step=after['distance'].get('effective'),
                matching_scope='trainable_parameter_coordinates_only; NOT BA/FC effective-map length',
                zero_direction=magnitude <= EPS,
                cosine_base_and_real_post_update_gradient=cosine(g0, after['gradient']),
                R_grad_actual=norm(after['gradient']-g0)/actual if actual > EPS else None,
                probe_ce_before=base['loss'], probe_ce_after=after['loss'],
                probe_ce_change=after['loss']-base['loss'],
                current_train_holdout_current20_ce_before=current_before,
                current_train_holdout_current20_ce_after=current_after,
                current_train_holdout_current20_ce_change=current_after-current_before,
                current_train_holdout_seen40_ce_before=current_seen_before,
                current_train_holdout_seen40_ce_after=current_seen_after,
                current_train_holdout_seen40_ce_change=current_seen_after-current_seen_before,
                old_train_holdout_seen40_ce_before=old_before,
                old_train_holdout_seen40_ce_after=old_after,
                old_train_holdout_seen40_ce_change=old_after-old_before,
                old_train_holdout_old20_ce_before=old20_before,
                old_train_holdout_old20_ce_after=old20_after,
                old_train_holdout_old20_ce_change=old20_after-old20_before,
                no_training_commit=True, old_data_only_in_offline_no_grad_loss=True,
                not_used_for_training_or_selection=True))
    checks = engine.entry.check()
    return dict(rows=rows, gradient_reads=engine.gradient_reads,
                loss_reads=engine.loss_reads, restore_count=engine.restore_count,
                checks=checks, shared_g0_g1_g2_g3=True,
                probe_dtype='native_float32', reduction_dtype='float64',
                probe_gradient_hashes={name: read['gradient_sha256'] for name, read in
                    zip(('g0', 'g1', 'g2', 'g3'), (base, read1, read2, read3))},
                probe_actual_distances={name: read['distance'] for name, read in
                    zip(('g0', 'g1', 'g2', 'g3'), (base, read1, read2, read3))},
                response_direction_valid=norm(response) > EPS,
                expected_gradient_reads=20, expected_loss_reads=68,
                committed_updates=0, same_entry_R0_for_all_reads=True,
                no_R1_advance=True, current_holdout_scope='disjoint_Task1_TRAIN',
                old_holdout_scope='Task0_TRAIN_no_grad_only_no_selection')


def _count_context_reads(context):
    return sum(context.get(key) is not None for key in ('g0', 'g1', 'g2', 'g3'))


def native_cost_benchmark(model, batch, geometry, eta, rho=.05, radius=1.,
                          warmup=3, timed=5, verification=2, seed=20261010,
                          step_class=None, expected_candidates=None):
    """True native skip-probe compute+update; independent same-state trials.

    Restores, CUDA synchronization before start, and equality checks after stop
    are outside timing. CUDA synchronization after the actual step is included
    in wall time. No CE evaluation or snapshot serialization is timed.
    """
    if step_class is None:
        from upward_residual_step import UpwardResidualStep
        step_class = UpwardResidualStep
    params = [p for p in model.parameters() if p.requires_grad]
    entry = State(model)
    originals = [p.detach().clone() for p in params]
    device = params[0].device
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    def closure(inputs, targets):
        output = model(inputs)
        logits = output['logits'] if isinstance(output, dict) else output
        return F.cross_entropy(logits[:, 20:40], targets-20), logits
    # Explicit local generator does not consume shared model RNG.
    generator = torch.Generator().manual_seed(seed)
    orders = [torch.randperm(len(DIRECTIONS), generator=generator).tolist()
              for _ in range(warmup + timed + verification)]
    trials, checks = [], []
    started = time.time()
    try:
        for round_index, order in enumerate(orders):
            phase = ('warmup' if round_index < warmup else
                     'timed' if round_index < warmup+timed else 'verification')
            for method_index in order:
                method = DIRECTIONS[method_index]
                entry.restore()
                optimizer = torch.optim.SGD(params, lr=eta, momentum=0., weight_decay=0.)
                algorithm = step_class(model, params, geometry, config=dict(
                    direction=method, rho=rho, radius=radius, eps=EPS, backend='native'))
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                    allocated_before = torch.cuda.memory_allocated(device)
                else:
                    allocated_before = None
                sync()
                start = time.perf_counter()
                output = algorithm.compute(*batch, closure, eta)
                algorithm.apply_update(eta, optimizer)
                sync()
                elapsed = time.perf_counter()-start
                count = _count_context_reads(output['context'])
                if count != READS[method]:
                    raise AssertionError(f'{method}: expected {READS[method]} actual ordinary gradient reads; got {count}')
                candidate_checks = {}
                if phase == 'verification' and expected_candidates is not None:
                    u = expected_candidates[method]
                    actual_u = torch.cat([v.reshape(-1) for v in output['U']])
                    if not torch.equal(actual_u, u):
                        # Native expression arithmetic must agree with offline construction.
                        delta = norm(actual_u-u)
                        raise AssertionError(f'Native/offline {method} direction mismatch {delta:g}')
                    expected = []
                    offset = 0
                    for p, initial in zip(params, originals):
                        n = p.numel()
                        want = initial.clone().add_(u[offset:offset+n].reshape_as(initial), alpha=-eta)
                        expected.append(torch.equal(p, want)); offset += n
                    if not all(expected):
                        raise AssertionError('Native optimizer and counterfactual parameter update differ')
                    candidate_checks = dict(native_and_offline_U_exact=True,
                                            native_optimizer_delta_exact=True)
                row = dict(method=method, round_index=round_index, phase=phase,
                           ordinary_gradient_reads=count, seconds=elapsed,
                           native_skips_unused_probes=True,
                           peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device)
                               if device.type == 'cuda' else None,
                           peak_memory_extra_above_entry_bytes=torch.cuda.max_memory_allocated(device)-allocated_before
                               if device.type == 'cuda' else None,
                           verification=candidate_checks)
                trials.append(row)
                # Drop all step-owned tensors before the following method.
                del algorithm, output, optimizer
                entry.restore(); checks.append(entry.check())
        rows = []
        for method in DIRECTIONS:
            values = [row for row in trials if row['method']==method and row['phase']=='timed']
            durations = [row['seconds'] for row in values]
            rows.append(dict(method=method, ordinary_gradient_reads_per_update=READS[method],
                timed_updates=len(durations), mean_seconds=statistics.mean(durations),
                median_seconds=statistics.median(durations),
                sample_sd_seconds=statistics.stdev(durations) if len(durations)>1 else 0.,
                max_peak_memory_allocated_bytes=max((r['peak_memory_allocated_bytes'] or 0) for r in values),
                max_peak_memory_extra_above_entry_bytes=max((r['peak_memory_extra_above_entry_bytes'] or 0) for r in values)))
        return dict(rows=rows, trials=trials, randomized_orders=orders,
            seed=seed, warmup_trials_per_method=warmup, timed_trials_per_method=timed,
            validation_trials_per_method=verification,
            trials_per_method=warmup+timed+verification,
            temporary_optimizer_updates=len(trials), committed_training_updates=0,
            elapsed_seconds=time.time()-started,
            source='same-state_microbenchmark_NOT_formal_training_cost',
            all_trials_share_same_snapshot_and_R0=True,
            cuda_synchronized=device.type=='cuda',
            external_snapshot_restores_excluded_from_timing=True,
            mandatory_internal_state_guards_included=True,
            forward_backward_native_update_included=True,
            no_holdout_or_old_task_evaluation_timed=True,
            no_microbenchmark_update_survives=True,
            checks=dict(all_entry_restores=all(all(c.values()) for c in checks),
                        all_probe_counts_equal_native_minimum=True,
                        no_extra_probe_to_equalize_training_budget=True))
    finally:
        entry.restore(); entry.check()


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if
                isinstance(value, (list, dict)) else value for key, value in row.items()})


def _resolve_path(path, relative_to):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (relative_to/p).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--source')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
    suite_path = Path(cli.suite).resolve()
    suite = json.loads(suite_path.read_text())
    source = _resolve_path(cli.source or suite['source'], suite_path.parent)
    split = _resolve_path(suite.get('dev_split', suite.get('dev_split_path')), suite_path.parent)
    seed = cli.seed if cli.seed is not None else int(suite.get('seed', 2))
    if 'upward_audit_refs_by_seed' in suite:
        key = ('upward_audit_smoke_refs_by_seed' if cli.smoke and
               'upward_audit_smoke_refs_by_seed' in suite else 'upward_audit_refs_by_seed')
        refs = suite[key][str(seed)]
        trajectories = [dict(direction=d, reference_output_dir=refs[d]) for d in ('g0', 'g1', 'g2')]
    else:
        trajectories = suite['trajectories']
    directions = [row['direction'] for row in trajectories]
    if directions != ['g0', 'g1', 'g2']:
        raise AssertionError('Preregistered diagnostics require distinct SGD, SAM, g2 trajectories in that order')
    if len({row['reference_output_dir'] for row in trajectories}) != 3:
        raise AssertionError('SGD/SAM/g2 diagnostics require independently trained trajectory sources')
    if cli.smoke:
        trajectories = trajectories[:1]
        directions = ['g0']
    epochs = [1] if cli.smoke else [1, 10, 50]
    probe_batches = 1 if cli.smoke else 2
    benchmark_cfg = suite.get('benchmark', {})
    warmup = 1 if cli.smoke else int(benchmark_cfg.get('warmup', 3))
    timed = 1 if cli.smoke else int(benchmark_cfg.get('timed', 5))
    verification = 1 if cli.smoke else int(benchmark_cfg.get('verification', 2))
    if cli.dry_run:
        print(json.dumps(dict(status='valid', source=str(source), dev_split=str(split),
            seed=seed, trajectories=trajectories, epochs=epochs, candidates=list(DIRECTIONS),
            fixed_train_probe_batches=probe_batches,
            fixed_current_holdout_batches=1, fixed_old_train_holdout_batches=1,
            virtual_update_modes=['native', 'nominal_parameter_length_match'],
            batch_size=128, rho=.05, radius=1.,
            gradient_reads_per_probe_batch=20, loss_reads_per_probe_batch=68,
            committed_training_updates=0,
            benchmark=dict(warmup=warmup, timed=timed, verification=verification,
                           native_reads=READS), old_data_gradient_reads=0)))
        return
    out = Path(cli.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Offline audit output must be fresh')
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Deterministic CUBLAS workspace required')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(source)); os.chdir(source)
    from upward_fork_train import load_snapshot
    from probe_study_data import build_data_manager, current_task_train_dataset
    config_path = _resolve_path(trajectories[0]['reference_output_dir'], suite_path.parent)/'effective_config.json'
    config = json.loads(config_path.read_text())
    dm = build_data_manager(str(source), config, str(split))
    current_ds, current_ids = current_task_train_dataset(dm, 1, config)
    old_ds, old_ids = current_task_train_dataset(dm, 0, config)
    pb, current_holdout, receipts = deterministic_batches(current_ds, current_ids,
        probe_batches=probe_batches, seed=20261010)
    # One old holdout is materialized from TRAIN. The extra returned batch is
    # discarded, and no old example is ever sent to a backward closure.
    _, old_holdout, old_receipts = deterministic_batches(old_ds, old_ids,
        probe_batches=0, seed=20261011)
    old_receipt = {**old_receipts[0], 'role': 'offline_old_Task0_TRAIN_holdout_no_grad'}
    chosen_current = {i for row in receipts for i in row['raw_train_indices']}
    chosen_old = set(old_receipt['raw_train_indices'])
    if chosen_current & chosen_old or (chosen_current | chosen_old) & set(dm.study_dev_indices):
        raise AssertionError('Offline TRAIN probe/holdout leakage or overlap')
    device = torch.device('cuda:0')
    pb = [tuple(v.to(device) for v in b) for b in pb]
    current_holdout = tuple(v.to(device) for v in current_holdout)
    old_holdout = tuple(v.to(device) for v in old_holdout)
    result = dict(schema='UpwardEightOfflineAudit_v1', status='running', smoke=cli.smoke,
        suite=str(suite_path), suite_sha256=file_sha(suite_path), audit_sha256=file_sha(__file__),
        source=str(source), seed=seed, dev_split_sha256=file_sha(split), snapshots=[],
        source_trajectories=directions, epochs=epochs, candidates=list(DIRECTIONS),
        data_receipts=receipts+[old_receipt], committed_training_updates=0,
        no_dev_or_test_examples_used=True, old_examples_only_offline_no_grad=True,
        model_selection_eligible=False, benchmarks_separate_from_training_purity=True,
        match_criterion='nominal LoRA factor+FC parameter length; actual error explicitly recorded; NOT BA matching')
    atomic_json(out/'final_summary.json', result)
    metrics = out/'metrics.jsonl'
    flat, benchmarks = [], []
    started = time.time()
    try:
        for trajectory in trajectories:
            reference = _resolve_path(trajectory['reference_output_dir'], suite_path.parent)
            summary_path = reference/'final_summary.json'
            summary = json.loads(summary_path.read_text())
            if summary.get('status') != 'completed' or bool(summary.get('smoke', False)) != cli.smoke:
                raise AssertionError('Diagnostics require completed, budget-matched training trajectories')
            for epoch in epochs:
                snapshot = reference/'snapshots'/f'epoch_{epoch:03d}.pt'
                model, payload = load_snapshot(snapshot, device=str(device), restore_rng=True)
                if payload['task'] != 1 or payload['class_slice'] != [20, 40]:
                    raise AssertionError('Incorrect diagnostic task snapshot')
                if payload['dev_split_sha256'] != file_sha(split):
                    raise AssertionError('Diagnostic/snapshot split mismatch')
                payload_seed = payload['config'].get('seed')
                payload_seed = payload_seed[0] if isinstance(payload_seed, list) and len(payload_seed)==1 else payload_seed
                if payload_seed != seed:
                    raise AssertionError('Diagnostic snapshot belongs to the wrong seed')
                if payload['config'].get('gam_grad_rho') != .05 or payload['config'].get('gam_grad_norm_rho') != 1.:
                    raise AssertionError('Offline diagnostic requires registered upward rho=.05,r=1 trajectory')
                config_direction = payload['config'].get('upward_direction', payload['config'].get('direction'))
                if config_direction is not None and config_direction != trajectory['direction']:
                    raise AssertionError('Wrong source trajectory for diagnostic snapshot')
                entry_rng = rng_get()
                geometry = EffectiveStep.from_model(model)
                eta = float(payload['current_optimizer_lr'])
                def closure(inputs, targets):
                    output = model(inputs)
                    logits = output['logits'] if isinstance(output, dict) else output
                    return F.cross_entropy(logits[:, 20:40], targets-20), logits
                snap = dict(trajectory=trajectory['direction'], epoch=epoch,
                    source_output_dir=str(reference), source_summary_sha256=file_sha(summary_path),
                    source_snapshot=str(snapshot), source_snapshot_sha256=file_sha(snapshot),
                    full_model_parameter_sha256=payload['full_model_parameter_sha256'],
                    frozen_origin_sha256=payload['frozen_origin_sha256'],
                    current_optimizer_lr=eta, model_training_mode=model.training, batches=[])
                benchmark_candidates = None
                for index, batch in enumerate(pb):
                    rng_set(entry_rng)
                    engine = UpwardLocalEngine(model, closure, geometry)
                    row = audit_eight_batch(engine, batch, current_holdout, old_holdout, eta)
                    if row['gradient_reads'] != 20 or row['loss_reads'] != 68:
                        raise AssertionError('Preregistered offline read budget differs')
                    row['probe_batch_index'] = index
                    snap['batches'].append(row)
                    for candidate in row['rows']:
                        item = dict(trajectory=trajectory['direction'], epoch=epoch,
                                    probe_batch_index=index, **candidate)
                        flat.append(item)
                        with metrics.open('a') as handle:
                            handle.write(json.dumps(item, allow_nan=False)+'\n')
                    if index == 0:
                        g0 = engine.gradient(batch)['gradient']
                        g1 = engine.gradient(batch, ((g0, .05/(engine.native_norm(g0)+EPS)),))['gradient']
                        s = g1-g0
                        outer = ((s, 1./(engine.native_norm(s)+EPS)),)
                        g2 = engine.gradient(batch, outer)['gradient']
                        g3 = engine.gradient(batch, outer+((g2, .05/(engine.native_norm(g2)+EPS)),))['gradient']
                        benchmark_candidates = construct_candidates(g0, g1, g2, g3)
                rng_set(entry_rng)
                bench = native_cost_benchmark(model, pb[0], geometry, eta,
                    warmup=warmup, timed=timed, verification=verification,
                    seed=20261010+epoch, expected_candidates=benchmark_candidates)
                snap['native_cost_microbenchmark'] = bench
                for row in bench['rows']:
                    benchmarks.append(dict(trajectory=trajectory['direction'], epoch=epoch, **row))
                if not rng_equal(entry_rng, rng_get()):
                    raise AssertionError('Offline diagnostics consumed source RNG')
                result['snapshots'].append(snap)
                atomic_json(out/'final_summary.json', result)
                del geometry, model, benchmark_candidates
                torch.cuda.empty_cache()
        batches = [b for s in result['snapshots'] for b in s['batches']]
        cost_trials = [trial for snapshot in result['snapshots']
                       for trial in snapshot['native_cost_microbenchmark']['trials']]
        local_gradient_reads = sum(b['gradient_reads'] for b in batches)+4*len(result['snapshots'])
        microbenchmark_gradient_reads = sum(trial['ordinary_gradient_reads'] for trial in cost_trials)
        result.update(status='passed', elapsed_seconds=time.time()-started,
            checked_snapshots=len(result['snapshots']), checked_probe_batches=len(batches),
            candidate_virtual_updates=len(flat), benchmark_methods=len(benchmarks),
            gradient_reads=local_gradient_reads,
            offline_candidate_gradient_reads=local_gradient_reads,
            microbenchmark_gradient_reads=microbenchmark_gradient_reads,
            total_ordinary_current_train_gradient_reads=local_gradient_reads+microbenchmark_gradient_reads,
            microbenchmark_temporary_sgd_updates=len(cost_trials),
            microbenchmark_timed_sgd_updates=sum(trial['phase']=='timed' for trial in cost_trials),
            microbenchmark_untimed_warmup_sgd_updates=sum(trial['phase']=='warmup' for trial in cost_trials),
            microbenchmark_untimed_verification_sgd_updates=sum(trial['phase']=='verification' for trial in cost_trials),
            old_task_gradient_reads=0,
            holdout_loss_reads=sum(b['loss_reads'] for b in batches),
            old_task_no_grad_holdout_loss_reads=34*len(batches),
            current_task_no_grad_holdout_loss_reads=34*len(batches),
            checks=dict(all_candidate_state_restores=all(all(b['checks'].values()) for b in batches),
                all_benchmark_state_restores=all(s['native_cost_microbenchmark']['checks']['all_entry_restores'] for s in result['snapshots']),
                all_candidates_from_same_four_gradients=True,
                expected_source_trajectory_count=1 if cli.smoke else 3,
                all_registered_source_trajectories_used=len(result['snapshots']) == (1 if cli.smoke else 9),
                no_committed_training_update=True, no_old_task_backward=True,
                no_dev_test_access=True, all_probes_replay_source_R0=True,
                actual_parameter_match_error_recorded=True, no_effective_BA_matching_claim=True))
        _write_csv(out/'local_updates.csv', flat)
        _write_csv(out/'native_cost_microbenchmark.csv', benchmarks)
        atomic_json(out/'local_updates.json', flat)
        atomic_json(out/'native_cost_microbenchmark.json', benchmarks)
        (out/'AUDIT_REPORT.md').write_text(
            '# Eight-direction common-state audit\n\n'
            f"Status: {result['status']}. {len(result['snapshots'])} trajectory snapshots, "
            f"{len(batches)} fixed current TRAIN minibatches, {len(flat)} virtual updates.\n\n"
            'Snapshots come independently from SGD, SAM, and g2 trajectories. Within each '
            'snapshot/batch all eight updates use the same four gradient vectors and R0. '
            'Native and nominal parameter-length matched steps are diagnostic counterfactuals, '
            'not trained algorithm results. Actual FP32 parameter distance and its matching '
            'error are reported separately from effective unscaled LoRA BA + FC displacement.\n\n'
            'Current and old holdouts come only from the original TRAIN subset, excluding '
            'development and test. Old examples are used only by offline no-grad CE and never '
            'enter updates, gating, or model/hyperparameter selection.\n\n'
            'Native timing uses genuinely skipped unused probes. Each trial starts at the same '
            'snapshot/R0 and is restored. It includes only native compute and SGD update; '
            'snapshot loading, restoration, checks, and holdout evaluation are excluded. '
            'Formal training costs must be reported from training counters separately.\n')
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        atomic_json(out/'final_summary.json', result)


if __name__ == '__main__':
    main()
