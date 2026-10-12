"""Native GAM/SGD with CE-weighted factor averaging BEFORE Fisher estimation."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from models_LoRAbasedCL.as1_normfisher_gam_intask_damp_weightmergethenfisher import Learner as SourceLearner
from optimer_PerturabtionType.gam import GAM
from scripts.imagenetr_gam_ewc_1x1.learner import additive_step


def factor_hash(items):
    h = hashlib.sha256()
    for key, tensor in sorted(items):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Learner(SourceLearner):
    def __init__(self, args):
        super().__init__(args)
        self._ewc_fisher_damping_beta = float(args['ewc_fisher_damping_alpha'])
        self._table_diagnostic_loader = None
        self._table_diagnostic_task = None
        self._table_merge_done_task = None
        self._table_step_counts = {'ce_backward_count': 0, 'updates': 0}

    def _save_experiment_identity(self, ckpt_dir, task_idx, fisher_state):
        super()._save_experiment_identity(ckpt_dir, task_idx, fisher_state)
        path = Path(ckpt_dir) / f'experiment_identity_{task_idx}.json'
        identity = json.loads(path.read_text())
        identity['lr'] = float(self.args.get('init_lr' if task_idx == 0 else 'lrate', self.args.get('init_lr', 0.)))
        identity['epochs'] = int(self.args.get('init_epoch' if task_idx == 0 else 'epochs', self.args.get('init_epoch', 0)))
        path.write_text(json.dumps(identity, indent=2) + '\n')

    def _build_optimizer(self, params, stage):
        params = list(params)
        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
        base = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self._last_optimizer_class = 'SGD'
        if self.args['optimizer_type'] == 'sgd':
            return base
        self._last_optimizer_class = 'GAM'
        gam_args = SimpleNamespace(grad_beta_1=1., grad_beta_2=1., grad_beta_3=1.,
                                   grad_gamma=float(self.args['gam_grad_gamma']), use_sgv=False)
        return GAM(params, base_optimizer=base, model=self._network,
                   grad_rho=float(self.args['gam_grad_rho']),
                   grad_norm_rho=float(self.args['gam_grad_norm_rho']),
                   adaptive=False, args=gam_args)

    def _step_batch_with_log(self, optimizer, inputs, targets, class_offset):
        logits, loss, stats = additive_step(
            self._network, optimizer, inputs, targets, class_offset,
            lambda: self._past_fisher_penalty(self._unwrap_network()), self._task_loss)
        self._table_step_counts['ce_backward_count'] += stats['ce_backward_count']
        self._table_step_counts['updates'] += 1
        self._record_mechanism_loss_stats(stats['clean_ce'], stats['ewc_penalty'])
        self._table_step_counts['ewc_penalty_max'] = max(
            self._table_step_counts.get('ewc_penalty_max', 0.), stats['ewc_penalty'])
        return logits, loss, {'ce_clean': stats['clean_ce'], 'gam': stats['clean_ce'],
                             'ewc': stats['ewc_penalty'], 'delta_w_penalty': 0.,
                             'ce_backward_count': stats['ce_backward_count']}

    def _get_current_fisher_loader(self, fallback=None):
        # CE quality and batch Fisher see the SAME full current training split
        # with deterministic preprocessing and a separate loader RNG.
        if self._table_diagnostic_task != self._cur_task:
            dataset = self._data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source='train', mode='test')
            generator = torch.Generator().manual_seed(17001 + self._cur_task)
            self._table_diagnostic_loader = DataLoader(
                dataset, batch_size=int(self.args['batch_size']), shuffle=False,
                num_workers=int(self.args['eval_num_workers']), generator=generator)
            self._table_diagnostic_task = self._cur_task
        return self._table_diagnostic_loader

    def _should_capture_in_task_weight(self, epoch_idx):
        return bool(self._in_task_weight_merge and int(epoch_idx) in self.args['snapshot_epochs'])

    def _trajectory_dir(self):
        path = Path(self.args['table_run_dir']) / 'trajectory' / f'task_{self._cur_task:02d}'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _capture_in_task_weight_snapshot(self, train_loader, model, optimizer, epoch_idx, class_offset):
        count = super()._capture_in_task_weight_snapshot(
            train_loader, model, optimizer, epoch_idx, class_offset)
        epoch, factors = self._in_task_weight_snapshots[-1]
        metrics = self._in_task_snapshot_metrics[-1]
        if not np.isfinite(metrics['snapshot_train_ce']):
            raise FloatingPointError('Nonfinite snapshot CE; refusing fallback to uniform weights.')
        torch.save({'epoch': epoch, 'factors': factors,
                    'head': {k: v.detach().cpu().clone() for k, v in model.fc.state_dict().items()},
                    'quality': metrics, 'frozen_backbone_checkpoint': self.args['frozen_backbone_checkpoint']},
                   self._trajectory_dir() / f'epoch_{epoch:02d}.pt')
        return count

    def _apply_in_task_weight_merge(self, model):
        expected = self.args['snapshot_epochs'] if self._in_task_weight_merge else []
        actual = [epoch for epoch, _ in self._in_task_weight_snapshots]
        if actual != expected:
            raise RuntimeError(f'Snapshot epochs {actual} do not match required {expected}')
        before = self._snapshot_lora_factor_weights(model)
        torch.save({'factors': before, 'head': {k: v.detach().cpu().clone()
                                              for k, v in model.fc.state_dict().items()}},
                   self._trajectory_dir() / 'pre_merge_endpoint.pt')
        selected = self._select_late_weight_snapshot_entries()
        weights = self._path_checkpoint_weights(selected)
        metrics = super()._apply_in_task_weight_merge(model)
        self._table_merge_done_task = self._cur_task
        receipt = {'task': self._cur_task, 'enabled': bool(self._in_task_weight_merge),
                   'coordinate': 'LoRA_A_B_factors; classifier excluded',
                   'epochs': actual, 'weights': weights,
                   'ce': [m['snapshot_train_ce'] for m in self._in_task_snapshot_metrics],
                   'weight_formula': 'softmax(-beta*(CE-mean(CE))/(population_std(CE)+eps))',
                   'beta': self._in_task_path_beta, 'omega': self._in_task_weight_merge_omega,
                   'pre_factor_sha256': factor_hash(before.items()),
                   'post_factor_sha256': factor_hash(self._iter_lora_factor_weights(model)),
                   'metrics': metrics, 'step_counts': self._table_step_counts}
        (self._trajectory_dir() / 'merge.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return metrics

    def _compute_delta_fisher(self, loader, model, class_offset):
        if self._table_merge_done_task != self._cur_task:
            raise RuntimeError('Current Fisher must be computed AFTER task-end weight merge.')
        result = super()._compute_delta_fisher(loader, model, class_offset)
        receipt = {'task': self._cur_task, 'source': 'post_merge_current_train',
                   'batch_size': self.args['batch_size'], 'estimator': 'batch_gradient_square_proxy',
                   'raw_trace': self._fisher_new_task_raw_trace,
                   'normalized_trace': self._fisher_trace(result),
                   'factor_sha256': factor_hash(self._iter_lora_factor_weights(model)),
                   'estimation_metrics': self._last_fisher_estimation_metrics}
        (self._trajectory_dir() / 'current_fisher.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return result
