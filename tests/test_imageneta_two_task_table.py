"""CPU scientific checks for the ImageNet-A two-task, six-row protocol.

Run with CopyFlatCV dependencies installed:
  CUDA_VISIBLE_DEVICES='' /data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python \
      -m unittest discover -s tests -p test_imageneta_two_task_table.py -v

Small source LoRA modules exercise actual production methods. No dataset,
pretrained model download, GPU allocation, or experimental training is used.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import test_imagenetr_gam_ewc_1x1 as shared

SPEC = importlib.util.spec_from_file_location(
    'imageneta_two_task_test_learner', ROOT / 'scripts/imageneta_two_task_table/learner.py')
production = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(production)


class NativeStepProtocolSanity(shared.NativeGAMEWCSanity):
    """Run the eight independent GAM/EWC/lifecycle checks with the new module."""
    def setUp(self):
        super().setUp()
        replacement = patch.object(shared, 'production', production)
        replacement.start()
        self.addCleanup(replacement.stop)


class WeightedTrajectorySanity(unittest.TestCase):
    def setUp(self):
        cuda_patch = patch.object(torch.cuda, 'is_available', return_value=False)
        cuda_patch.start()
        self.addCleanup(cuda_patch.stop)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def fixture(self):
        torch.manual_seed(37)
        model = shared.TinySourceLoRANet()
        model.update_fc(2)
        learner = object.__new__(production.Learner)
        learner.__dict__.update({
            'args': {'snapshot_epochs': [20, 40], 'init_cls': 2, 'increment': 2,
                     'task_increments': [2, 2], 'filepath': self.directory.name + '/',
                     'table_run_dir': self.directory.name, 'batch_size': 4,
                     'frozen_backbone_checkpoint': 'cpu_synthetic_no_external_checkpoint'},
            '_network': model, '_device': torch.device('cpu'),
            '_is_main_process': False, '_cur_task': 0, '_known_classes': 0,
            '_total_classes': 2, '_snapshot_epochs': [20, 40],
            '_in_task_weight_merge': True, '_in_task_weight_merge_start_task': 0,
            '_in_task_weight_checkpoint_interval': 20,
            '_in_task_weight_merge_omega': 0.3, '_in_task_fisher_omega': 0.3,
            '_in_task_path_weighted': True, '_in_task_path_beta': 3.0,
            '_in_task_path_loss_eps': 1e-12, '_in_task_path_start_ratio': 0.0,
            '_in_task_total_epochs': 50, '_in_task_weight_snapshot_tensors': 0,
            '_in_task_lora_snapshot_tensors': 0, '_in_task_weight_snapshots': [],
            '_in_task_snapshot_metrics': [], '_in_task_path_weight_last_metrics': {},
            '_in_task_weight_merge_last_metrics': {},
            '_weight_merge_then_fisher_last_metrics': {},
            '_posterior_state_save': False, '_mechanism_anchor_eval': False,
            '_mechanism_future_surrogate_eval': False, '_posterior_fisher_endpoint': {},
            '_posterior_fisher_merge': {}, '_normalize_fisher': True,
            '_normalize_accumulated_fisher': True, '_fisher_norm_eps': 1e-12,
            '_ewc_fisher_damping_alpha': 0.2, '_ewc_fisher_damping_beta': 0.2,
            '_ewc_gamma': 0.9, '_fisher_past_trace_mass': 0.0,
            '_last_fisher_norm_metrics': {}, '_last_fisher_estimation_metrics': {},
            '_last_mechanism_metrics': {}, '_fisher_past_delta': {},
            '_in_task_path_loss_max_batches': None, '_task_endpoint_weight_merge': False,
            '_task_endpoint_weight_merge_omega': 0.0,
            '_mechanism_eval': False, '_mechanism_eps': 1e-12,
            '_lambda_ewc': 2000.0, '_ewc_lambda': 2000.0,
            '_delta_w_penalty_lambda': 0.0, '_dual_ascent': False,
            '_ewc_skip_task_fisher': False, '_ewc_preserve_fisher_rng': False,
            '_compute_full_fisher': False,
            '_table_merge_done_task': None,
            '_table_step_counts': {'ce_backward_count': 0, 'updates': 0},
        })
        learner._snapshot_task_reference = lambda: None
        learner._log_task_start_state = lambda: None
        learner._current_fisher_drift_value = lambda: 0.0
        learner._get_current_fisher_loader = lambda loader: loader
        return learner, model

    @staticmethod
    def independent_weights(losses):
        mean = sum(losses) / len(losses)
        std = math.sqrt(sum((loss-mean)**2 for loss in losses) / len(losses))
        if std <= 1e-12:
            return [1.0/len(losses)] * len(losses)
        values = [math.exp(-3.0*(loss-mean)/(std+1e-12)) for loss in losses]
        return [value/sum(values) for value in values]

    def prepare_snapshots(self, learner, model):
        states = []
        for value in (0.2, 0.6, 1.1):
            with torch.no_grad():
                for index, (_, weight) in enumerate(learner._iter_lora_factor_weights(model)):
                    weight.fill_(value + 0.09*index)
            states.append(learner._snapshot_lora_factor_weights(model))
        learner._in_task_weight_snapshots = [(20, states[0]), (40, states[1])]
        learner._in_task_snapshot_metrics = [{'snapshot_train_ce': 0.4},
                                             {'snapshot_train_ce': 0.2}]
        learner._in_task_weight_snapshot_tensors = sum(len(s) for s in states[:2])
        weights = self.independent_weights([0.4, 0.2])
        expected = {key: 0.7*states[2][key] + 0.3*(weights[0]*states[0][key]
                    + weights[1]*states[1][key]) for key in states[2]}
        return states, expected

    def test_snapshot_schedule_is_exactly_epochs_20_and_40(self):
        learner, _ = self.fixture()
        self.assertEqual([epoch for epoch in range(1, 51)
                          if learner._should_capture_in_task_weight(epoch)], [20, 40])

    def test_missing_snapshot_and_fisher_before_merge_are_rejected(self):
        learner, model = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'Snapshot epochs'):
            learner._apply_in_task_weight_merge(model)
        with self.assertRaisesRegex(RuntimeError, 'AFTER task-end weight merge'):
            learner._compute_delta_fisher(None, model, 0)

    def test_production_optimizer_and_step_log_use_one_or_four_ce_calls(self):
        for optimizer_name, expected_calls in [('sgd', 1), ('gam', 4)]:
            with self.subTest(optimizer=optimizer_name):
                learner, _ = self.fixture()
                model = shared.TinyClassifier()
                learner._network = model
                learner.args.update(optimizer_type=optimizer_name, gam_grad_gamma=0.03,
                                    gam_grad_rho=0.05, gam_grad_norm_rho=0.5)
                learner._resolve_optimizer_hyper = lambda stage: (0.02, 0.0, 0.0)
                learner._past_fisher_penalty = lambda model: model.weight.new_zeros(())
                optimizer = learner._build_optimizer(model.parameters(), 'init')
                _, _, metrics = learner._step_batch_with_log(
                    optimizer, torch.ones(2, 3, dtype=torch.float64),
                    torch.tensor([0, 1]), class_offset=0)
                self.assertEqual(metrics['ce_backward_count'], expected_calls)
                self.assertEqual(learner._table_step_counts['updates'], 1)
                self.assertEqual(learner._table_step_counts['ce_backward_count'], expected_calls)
                self.assertEqual(learner._table_step_counts['ewc_penalty_max'], 0.0)
                self.assertEqual(len(model.probe_weights), expected_calls)
                self.assertEqual(learner._last_optimizer_class,
                                 'SGD' if optimizer_name == 'sgd' else 'GAM')

    def test_checkpoint_identity_records_each_tasks_actual_schedule(self):
        learner, _ = self.fixture()
        # Unequal synthetic epoch budgets detect accidental reuse of init_epoch.
        learner.args.update(init_lr=0.04, lrate=0.02, init_epoch=50, epochs=37)
        fisher_state = {'fisher_past_delta': {'q': torch.ones(2, 2)/4}}
        for task_idx, expected_lr, expected_epochs in [(0, 0.04, 50), (1, 0.02, 37)]:
            learner._cur_task = task_idx
            learner._save_experiment_identity(self.directory.name, task_idx, fisher_state)
            identity = json.loads((Path(self.directory.name) /
                                   f'experiment_identity_{task_idx}.json').read_text())
            self.assertEqual(identity['lr'], expected_lr)
            self.assertEqual(identity['epochs'], expected_epochs)

    def test_ce_weights_use_standardized_loss_and_handle_constant_loss(self):
        learner, _ = self.fixture()
        entries = [(0, 20, {}, {'snapshot_train_ce': 0.4}),
                   (1, 40, {}, {'snapshot_train_ce': 0.2})]
        weights = learner._path_checkpoint_weights(entries)
        expected = self.independent_weights([0.4, 0.2])
        torch.testing.assert_close(torch.tensor(weights), torch.tensor(expected))
        self.assertAlmostEqual(weights[1], 0.997527376843, places=10)
        self.assertGreater(abs(weights[1] - 1/(1+math.exp(-0.6))), 0.3)
        for entry in entries:
            entry[3]['snapshot_train_ce'] = 0.2
        self.assertEqual(learner._path_checkpoint_weights(entries), [0.5, 0.5])

    def test_factor_mean_then_endpoint_mix_preserves_base_and_head(self):
        learner, model = self.fixture()
        states, expected = self.prepare_snapshots(learner, model)
        factor_keys = set(expected)
        frozen = {key: value.clone() for key, value in model.state_dict().items()
                  if key not in factor_keys}
        report = learner._apply_in_task_weight_merge(model)
        self.assertTrue(report['weight_merge_applied'])
        self.assertEqual(report['in_task_path_selected_snapshots'], 2)
        for key, weight in learner._iter_lora_factor_weights(model):
            torch.testing.assert_close(weight, expected[key])
        # Aliased module registration can duplicate factor keys in state_dict.
        factor_ptrs = {weight.data_ptr() for _, weight in learner._iter_lora_factor_weights(model)}
        for key, value in model.state_dict().items():
            if value.data_ptr() not in factor_ptrs:
                self.assertTrue(torch.equal(value, frozen[key]), key)
        module_name, _ = next(learner._iter_delta_modules(model))
        akey, bkey = (f'{module_name}.linear_a_q.weight', f'{module_name}.linear_b_q.weight')
        actual_ba = expected[bkey] @ expected[akey]
        w = self.independent_weights([0.4, 0.2])
        mean_of_products = 0.7*(states[2][bkey] @ states[2][akey]) + 0.3*(
            w[0]*(states[0][bkey] @ states[0][akey]) +
            w[1]*(states[1][bkey] @ states[1][akey]))
        self.assertGreater(torch.linalg.vector_norm(actual_ba-mean_of_products).item(), 0.1)

    def test_snapshot_ce_is_sample_weighted_clean_task_local_ce(self):
        learner, _ = self.fixture()
        model = shared.TinyClassifier()
        model.train()
        xs = torch.tensor([[0.7, -1.2, 0.4], [1.1, 0.3, -0.8],
                           [-0.2, 0.9, 1.3]], dtype=torch.float64)
        ys = torch.tensor([1, 2, 1])
        loader = [(xs[:2], ys[:2]), (xs[2:], ys[2:])]
        learner._past_fisher_penalty = lambda *_: self.fail('Snapshot CE included EWC')
        metrics = learner._estimate_snapshot_train_metrics(loader, model, class_offset=1)
        expected = F.cross_entropy(F.linear(xs, model.weight)[:, 1:], ys-1).item()
        self.assertAlmostEqual(metrics['snapshot_train_ce'], expected, places=13)
        self.assertEqual(metrics['snapshot_train_samples'], 3)
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_task_and_accumulated_fisher_normalization_then_damping(self):
        learner, model = self.fixture()
        self.prepare_snapshots(learner, model)
        raw = {'q': torch.tensor([[1.0, 3.0]], dtype=torch.float64),
               'v': torch.tensor([[2.0, 4.0]], dtype=torch.float64)}
        learner._compute_raw_delta_fisher = lambda *args: {k: v.clone() for k, v in raw.items()}
        learner._apply_in_task_weight_merge(model)
        current = learner._compute_delta_fisher(None, model, 0)
        for key in raw:
            torch.testing.assert_close(current[key], raw[key]/10, atol=1e-10, rtol=1e-7)
        omega0 = learner._accumulate_fisher({}, current)
        for key in raw:
            torch.testing.assert_close(omega0[key], 0.8*raw[key]/10+0.2/4, atol=1e-9, rtol=1e-6)
        learner._cur_task = 1
        second = {'q': torch.tensor([[0.4, 0.2]], dtype=torch.float64),
                  'v': torch.tensor([[0.1, 0.3]], dtype=torch.float64)}
        omega1 = learner._accumulate_fisher(omega0, second)
        for key in raw:
            expected = 0.8*(0.9*omega0[key]+second[key])/1.9 + 0.2/4
            torch.testing.assert_close(omega1[key], expected, atol=1e-8, rtol=1e-6)
        self.assertAlmostEqual(sum(v.sum().item() for v in omega1.values()), 1.0, places=6)
        self.assertAlmostEqual(learner._fisher_past_trace_mass, 1.9, places=6)

    def test_training_loop_merges_before_single_task_end_fisher_call(self):
        learner, model = self.fixture()
        step = [0]
        observed = []

        def fake_training_step(optimizer, inputs, targets, class_offset):
            step[0] += 1
            with torch.no_grad():
                for index, (_, weight) in enumerate(learner._iter_lora_factor_weights(model)):
                    weight.fill_(step[0]/50 + 0.02*index)
            return torch.zeros(1, 2), 0.0, {'ce_clean': 0.0}

        def snapshot_metrics(**kwargs):
            return {'snapshot_train_ce': 1-step[0]/100,
                    'snapshot_train_samples': 1, 'snapshot_train_acc': 0,
                    'snapshot_train_batches': 1}

        def raw_fisher(loader, model, class_offset):
            observed.append(learner._snapshot_lora_factor_weights(model))
            return {'q': torch.tensor([[1.0, 3.0]]), 'v': torch.tensor([[2.0, 4.0]])}

        learner._step_batch_with_log = fake_training_step
        learner._estimate_snapshot_train_metrics = snapshot_metrics
        learner._compute_raw_delta_fisher = raw_fisher
        learner.train_loader = [(torch.zeros(1, 3), torch.zeros(1, dtype=torch.long))]
        learner._run_regularized_train(learner.train_loader, None, None, None,
                                       epochs=50, class_offset=0)
        self.assertEqual(step[0], 50)
        self.assertEqual([epoch for epoch, _ in learner._in_task_weight_snapshots], [20, 40])
        self.assertEqual(observed, [])  # No Fisher snapshot passes at 20/40.
        states = [state for _, state in learner._in_task_weight_snapshots]
        w = self.independent_weights([0.8, 0.6])
        expected = {}
        for index, (key, weight) in enumerate(learner._iter_lora_factor_weights(model)):
            end = torch.full_like(weight, 1+0.02*index)
            expected[key] = 0.7*end + 0.3*(w[0]*states[0][key]+w[1]*states[1][key])
            torch.testing.assert_close(weight, expected[key])
        learner.after_task()
        self.assertEqual(len(observed), 1)
        for key in expected:
            torch.testing.assert_close(observed[0][key], expected[key])
        self.assertEqual(learner._known_classes, 2)
        self.assertEqual(learner._in_task_fisher_last_metrics['current_fisher_source'],
                         'post_weight_merge')


if __name__ == '__main__':
    unittest.main()
