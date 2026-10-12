"""Matched 3x2 historical-importance experiment with explicit S+A anchors.

The Kronecker representation uses gradient-matrix partial-trace moments,
not activation KFAC. The already validated KFAC penalty/history are reused.
"""
from __future__ import annotations

import hashlib
import math
import time

import torch

from models_LoRAbasedCL.as1_normfisher_fishershape import Learner as _FS
from models_LoRAbasedCL.as1_normfisher_fishershape_kfac import Learner as _KFAC
from utils.fisher_complementarity import estimate_importance
from utils.kfac_ewc import KFACEWCState
from utils.lora_anchor_average import ABFactorAverager
from utils.mechanism_eval import as_bool


class Learner(_KFAC):
    def __init__(self, args):
        args = dict(args)
        args['resume_require_fisher_state'] = True
        args['checkpoint_require_nonempty_fisher'] = True
        args['ewc_preserve_fisher_rng'] = True
        # Bypass the activation-KFAC mode whitelist, but retain its state logic.
        _FS.__init__(self, args)
        self._importance_representation = args.get('importance_representation', 'diagonal')
        if self._importance_representation not in {'diagonal', 'kronecker'}:
            raise ValueError('importance_representation must be diagonal or kronecker')
        if self._fisher_mode not in {'batch_empirical', 'sample_empirical', 'true_sample'}:
            raise ValueError('Unsupported complementarity computation')
        if len(self._multiple_gpus) != 1 or torch.distributed.is_initialized():
            raise ValueError('Complementarity validation requires one process/device')
        if self._ewc_eta != 0 or self._ewc_fisher_damping_beta != 0 or self._dual_ascent:
            raise ValueError('This protocol requires eta=beta=0 and fixed lambda')
        if not self._normalize_fisher or self._normalize_accumulated_fisher:
            raise ValueError('This protocol normalizes each task, not accumulated history')
        if not math.isfinite(self._ewc_lambda) or self._ewc_lambda < 0:
            raise ValueError('ewc_lambda must be finite and nonnegative')
        if self._ewc_gamma != 1:
            raise ValueError('This protocol fixes gamma=1')
        if as_bool(args.get('endpoint_eval', False)) or self._compute_full_fisher:
            raise ValueError('Legacy endpoint/full-Fisher diagnostics are disabled')
        if as_bool(args.get('learn_alpha', False)) or self._ewc_skip_task_fisher:
            raise ValueError('Fixed LoRA scaling and task consolidation are required')
        self._anchor_average_enabled = as_bool(args.get('anchor_average_enabled', False))
        self._anchor_average_omega = float(args.get('anchor_average_omega', .5))
        self._anchor_average_start_epoch = int(args.get('anchor_average_start_epoch', 3))
        if not 0 <= self._anchor_average_omega <= 1:
            raise ValueError('anchor_average_omega outside [0,1]')
        self._kfac_history = KFACEWCState()
        self._pending_kfac = None
        self._kfac_last_metrics = {}
        self._phase_records = []
        self._complementarity_metrics = {}
        self._importance_metadata = {}

    def _kfac_protocol(self):
        return {
            'fisher_mode': self._fisher_mode,
            'representation': self._importance_representation,
            'parameter_space': 'unscaled_delta_w_ba',
            'kronecker_approximation': 'gradient_moment_partial_trace_v1',
            'normalize_task': True, 'normalize_accumulated': False,
            'normalization_eps': self._fisher_norm_eps, 'gamma': 1.0, 'damping_beta': 0.0,
            'mc_samples': self._true_fisher_mc_samples,
            'anchor': 'ab_factor_endpoint_mean_mix' if self._anchor_average_enabled else 'endpoint',
            'average_omega': self._anchor_average_omega if self._anchor_average_enabled else 0.0,
            'average_start_epoch': self._anchor_average_start_epoch if self._anchor_average_enabled else None,
        }

    def _start_phase(self):
        if self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)
            torch.cuda.reset_peak_memory_stats(self._device)
        return time.perf_counter()

    def _end_phase(self, phase, started, **extra):
        if self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)
        record = {'task': self._cur_task, 'phase': phase, 'seconds': time.perf_counter() - started, **extra}
        if self._device.type == 'cuda':
            record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(self._device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(self._device))
        self._phase_records.append(record)
        return record

    def _compute_delta_fisher(self, loader, model, class_offset):
        started = self._start_phase()
        seed_arg = self.args.get('seed', [0])
        seed = int(seed_arg[0] if isinstance(seed_arg, (list, tuple)) else seed_arg)
        payload, metadata = estimate_importance(
            learner=self, model=model, loader=loader, mode=self._fisher_mode,
            representation=self._importance_representation, class_offset=class_offset,
            seed=seed + 100003 * (self._cur_task + 1),
            mc_samples=self._true_fisher_mc_samples, max_batches=self._ewc_max_batches,
        )
        self._importance_metadata = metadata
        if self._importance_representation == 'kronecker':
            factors = {key: {name: value[name].float() for name in ('G', 'C')} for key, value in payload.items()}
            self._pending_kfac = KFACEWCState.from_estimate(factors, normalize=True, beta=0, eps=self._fisher_norm_eps)
            result = self._pending_kfac.diagonal()
        else:
            payload = {key: value.float() for key, value in payload.items()}
            trace = sum(float(value.double().sum()) for value in payload.values())
            if not math.isfinite(trace) or trace <= self._fisher_norm_eps:
                raise ValueError(f'Invalid task importance trace: {trace}')
            result = {key: value / trace for key, value in payload.items()}
        self._last_fisher_estimation_metrics = {
            'fisher_mode': self._fisher_mode,
            'importance_representation': self._importance_representation,
            'fisher_estimator_family': 'gradient_moment',
            'fisher_num_samples': metadata['sample_count'],
            'fisher_gradient_draws': metadata['gradient_draws'],
            'fisher_raw_trace': metadata['raw_trace'],
        }
        self._last_fisher_norm_metrics = {'fisher_normalized_trace': self._fisher_trace(result)}
        self._complementarity_metrics['importance'] = metadata
        self._complementarity_metrics['fisher_phase'] = self._end_phase('fisher', started)
        self._complementarity_metrics['fisher_anchor_sha256'] = self._hash_tensor_items(
            (n, p) for n, p in model.named_parameters() if p.requires_grad)
        return result

    def _accumulate_fisher(self, old, new):
        started = self._start_phase()
        if self._importance_representation == 'kronecker':
            result = _KFAC._accumulate_fisher(self, old, new)
        else:
            result = _FS._accumulate_fisher(self, old, new)
            self._fisher_past_trace_mass = sum(float(v.double().sum()) for v in result.values())
        self._complementarity_metrics['merge_phase'] = self._end_phase('merge', started)
        return result

    def _past_fisher_penalty(self, model):
        if self._importance_representation == 'kronecker':
            return _KFAC._past_fisher_penalty(self, model)
        return _FS._past_fisher_penalty(self, model)

    def _compute_delta_drift_metrics(self):
        if self._importance_representation == 'kronecker':
            return _KFAC._compute_delta_drift_metrics(self)
        return _FS._compute_delta_drift_metrics(self)

    def _run_regularized_train(self, train_loader, test_loader, optimizer, scheduler, *, epochs, class_offset):
        del test_loader  # End-of-task dev/test matrices are collected by the runner.
        self._snapshot_task_reference()
        self._epoch_dynamics_metrics = []
        self._complementarity_metrics = {'epochs': [], 'protocol': self._kfac_protocol()}
        model = self._unwrap_network()
        self._complementarity_metrics['expanded_head_sha256'] = self._hash_tensor_items(
            (n, p) for n, p in model.named_parameters() if n.startswith('fc.'))
        self._complementarity_metrics['expanded_new_head_sha256'] = self._hash_tensor_items(
            (n, p[class_offset:]) for n, p in model.named_parameters() if n.startswith('fc.') and p.ndim > 0)
        selected_epochs = list(range(self._anchor_average_start_epoch, epochs + 1))
        if self._anchor_average_enabled and len(selected_epochs) < 2:
            raise ValueError('S+A requires at least two selected epoch endpoints')
        averager = ABFactorAverager(selected_epochs) if self._anchor_average_enabled else None
        for epoch in range(1, epochs + 1):
            self._network.train()
            started = self._start_phase()
            order = hashlib.sha256()
            count, loss_sum, correct = 0, 0.0, 0
            for batch in train_loader:
                if len(batch) != 3:
                    raise ValueError('Paired training requires sample indices')
                indices, inputs, targets = batch
                ids = torch.as_tensor(indices).detach().cpu().to(torch.int64).contiguous()
                order.update(ids.numpy().tobytes())
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, loss = self._step_batch(optimizer, scheduler, inputs, targets, class_offset)
                scalar = float(loss)
                if not math.isfinite(scalar):
                    raise FloatingPointError('Nonfinite training loss')
                loss_sum += scalar * len(targets)
                correct += int((logits[:, class_offset:self._total_classes].argmax(1) == targets - class_offset).sum())
                count += len(targets)
            record = self._end_phase('train', started, epoch=epoch, samples=count,
                                     learning_rates=[float(group['lr']) for group in optimizer.param_groups],
                                     order_sha256=order.hexdigest(), loss=loss_sum/max(count, 1),
                                     task_local_accuracy=100*correct/max(count, 1))
            self._complementarity_metrics['epochs'].append(record)
            if scheduler is not None:
                scheduler.step()
            if averager is not None and epoch in selected_epochs:
                started = self._start_phase()
                averager.record(model, epoch)
                self._end_phase('anchor_record', started, epoch=epoch)
            self._log(f'[Complementarity][Epoch] {record}')
        started = self._start_phase()
        average = averager.apply(model, omega=self._anchor_average_omega) if averager else {'enabled': False, 'merge_applied': False}
        self._complementarity_metrics['average'] = average
        self._end_phase('anchor_apply', started)
        self._complementarity_metrics['retained_model_sha256'] = self._hash_tensor_items(
            (n, p) for n, p in model.named_parameters() if p.requires_grad)
        self._log(f'[Complementarity][Anchor] {average}')

    def after_task(self):
        _FS.after_task(self)
        self._last_mechanism_metrics.update(method_type='fs_sa_complementarity' if self._anchor_average_enabled else 'ewc_complementarity',
                                            importance_representation=self._importance_representation)
        if self._complementarity_metrics.get('fisher_anchor_sha256') != self._complementarity_metrics.get('retained_model_sha256'):
            raise RuntimeError('Fisher was not estimated at the actual retained anchor')
        self._complementarity_metrics['mechanism'] = dict(self._last_mechanism_metrics)
        self._log(f'[Complementarity][TaskEnd] {self._last_mechanism_metrics}')

    def _extra_fisher_checkpoint_state(self):
        extra = _KFAC._extra_fisher_checkpoint_state(self) if self._importance_representation == 'kronecker' else {}
        extra['complementarity_state'] = {
            'version': 1, 'protocol': self._kfac_protocol(), 'task': self._cur_task,
            'anchor': {key: value.detach().cpu().clone() for key, value in self._iter_delta_terms(self._unwrap_network())},
            'metrics': self._complementarity_metrics,
        }
        return extra

    def _restore_extra_fisher_checkpoint_state(self, state):
        extra = state.get('complementarity_state', {})
        if extra.get('version') != 1 or extra.get('protocol') != self._kfac_protocol() or extra.get('task') != self._cur_task:
            raise ValueError('Missing or mismatched complementarity checkpoint protocol')
        current = dict(self._iter_delta_terms(self._unwrap_network()))
        anchors = extra.get('anchor', {})
        if set(current) != set(anchors):
            raise ValueError('Checkpoint anchor keys mismatch')
        for key, value in anchors.items():
            if not torch.is_tensor(value) or value.shape != current[key].shape or not torch.allclose(value, current[key].detach().cpu(), rtol=1e-5, atol=0):
                raise ValueError('Checkpoint anchor differs from retained model')
        if self._importance_representation == 'kronecker':
            _KFAC._restore_extra_fisher_checkpoint_state(self, state)
        else:
            if 'kfac_ewc_state' in state:
                raise ValueError('Cannot load Kronecker state as diagonal')
            self._delta_reference = anchors
        self._complementarity_metrics = extra.get('metrics', {})

    def _load_auxiliary_task_state(self, checkpoint_dir, task_idx):
        if self._importance_representation == 'kronecker':
            _KFAC._load_auxiliary_task_state(self, checkpoint_dir, task_idx)
        else:
            _FS._load_auxiliary_task_state(self, checkpoint_dir, task_idx)
