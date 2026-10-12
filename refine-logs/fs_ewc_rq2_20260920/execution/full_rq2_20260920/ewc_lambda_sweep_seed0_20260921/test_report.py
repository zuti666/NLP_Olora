"""CPU report checks: terminal-state semantics, pairing, telemetry, BA geometry."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('raw_sweep_report', Path(__file__).with_name('report.py'))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


class ReportTests(unittest.TestCase):
    def test_selection_uses_finite_completed_points_and_fixed_tradeoff(self):
        def row(lam, a21, a22, outcome='complete'):
            return dict(ewc_lambda=lam, a21=a21, a22=a22, outcome=outcome,
                        average_accuracy=(a21 + a22) / 2 if a21 is not None else None)
        rows = [row(0, 50, 80), row(1, 51, 79), row(10, 55, 74), row(30, 70, 50),
                row(60, None, None, 'diverged'), row(100, None, None, None)]
        thresholds = dict(retention_gain_pp=1, max_new_task_drop_pp=5, nondecreasing_average_gain_pp=0)
        result = report.summarize(rows, thresholds)
        self.assertEqual(result['best_old_retention'], [30])
        self.assertEqual(result['best_average'], [0, 1])
        self.assertEqual(result['observable_retention_lambdas'], [1, 10, 30])
        self.assertEqual(result['balanced_retention_lambdas'], [1])
        self.assertEqual(result['observable_retention_lambdas_observed_span'], [1, 30])
        self.assertNotIn('delta_a21_pp', rows[-2])

    def test_no_complete_receipt_until_all_14_observed_and_fresh_baseline_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = report.read_json(Path(__file__).with_name('sweep_spec.json'))
            write(root / 'sweep_spec.json', spec)
            rows = {lam: dict(ewc_lambda=lam, status='observed', outcome='diverged',
                             **{k: None for k in report.METRICS}) for lam in report.GRID}
            def validate(_root, lam, *_args):
                return copy.deepcopy(rows[lam]), None
            with patch.object(report, 'validate_prepared', return_value={'payload_sha256': 'raw'}), \
                 patch.object(report, 'validate_trial', side_effect=validate):
                data = report.build_report(root, root / 'report', allow_partial=True)
                self.assertEqual(data['observed_lambda_points'], 14)
                self.assertEqual(data['status'], 'partial')
                self.assertFalse((root / 'report/COMPLETE.json').exists())
                rows[0].update(outcome='complete', a11=87, a21=50, a22=80, average_accuracy=65)
                rows[100].update(status='pending', outcome=None)
                with self.assertRaisesRegex(RuntimeError, '13/14'):
                    report.build_report(root, root / 'report')
                self.assertFalse((root / 'report/COMPLETE.json').exists())
                rows[100].update(status='observed', outcome='diverged')
                data = report.build_report(root, root / 'report')
                self.assertEqual(data['status'], 'complete')
                receipt = report.read_json(root / 'report/COMPLETE.json')
                self.assertEqual(receipt['completed_lambda_points'], 1)
                self.assertEqual(receipt['diverged_lambda_points'], 13)
                self.assertEqual(receipt['observed_lambda_points'], 14)
                rows[1].update(status='pending', outcome=None)
                report.build_report(root, root / 'report', allow_partial=True)
                self.assertFalse((root / 'report/COMPLETE.json').exists())

    def test_native_matrix_nan_placeholder_is_not_a_formal_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.json'
            write(path, {'matrix': [[87.3, math.nan], [80., 78.]]})
            with self.assertRaisesRegex(ValueError, 'Nonfinite JSON'):
                report.read_json(path)
            matrix = report.read_json(path, metric_matrices=True)['matrix']
            self.assertEqual(report.numeric(matrix[1][0], 'a21'), 80)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                report.numeric(matrix[0][1], 'unobserved cell')

    def test_actual_gradient_schedule_and_diverged_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            rows = []
            for step in range(400):
                sampled = step == 1 or step % 8 == 0
                rows.append(dict(step=step, task=1, ewc_lambda=10, parameters_finite_after_step=True,
                                 ce_clean=2., ewc=1., loss=3., grad_combined_norm=4.,
                                 gradient_components_sampled=sampled, grad_clean_norm=2., grad_fisher_norm=1.,
                                 fisher_to_clean_norm_ratio=1 / (2 + 1e-12)))
            lines(run / 'training_batch_metrics.jsonl', rows)
            validation = dict(losses=dict(train_clean_loss_mean=2., train_fisher_loss_mean=1.),
                              gradients=dict(grad_clean_norm_mean=2., grad_fisher_norm_mean=1.,
                                             fisher_to_clean_norm_ratio_mean=1 / (2 + 1e-12), component_sample_count=51))
            result = report.telemetry(run, 10, True, validation)
            self.assertEqual(result['gradient_component_samples'], 51)
            self.assertEqual(result['P_over_CE'], .5)
            lines(run / 'training_batch_metrics.jsonl', rows[:3])
            result = report.telemetry(run, 10, False, {})
            self.assertEqual(result['gradient_component_samples'], 2)
            self.assertEqual(result['observed_finite_steps'], 3)
            self.assertEqual(result['telemetry_scope'], 'finite prefix before divergence')
            with self.assertRaisesRegex(ValueError, '400 finite'):
                report.telemetry(run, 10, True, validation)

    def test_partial_loader_validates_first_pixels_but_does_not_claim_full_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, reference = root / 'trial', root / 'reference/rng_receipts.jsonl'
            base = dict(task=1, shuffle=True, generator_seed=3, num_workers=4,
                        dataset_ids_sha256='dataset', sample_count=997)
            common = [dict(event='head_initialized', task=1, old_classes=20, total_classes=40, task_seed=4, new_head_sha256='head'),
                      dict(event='task_rng_reset', task=1, task_seed=4, rng_protocol='protocol', optimizer_type='sgd'),
                      dict(base, event='loader_created')]
            ref_iter = dict(base, event='loader_iteration', iteration=0, samples=997, batches=8,
                            indices_sha256='whole_epoch', first_batch_pixels_targets_sha256='pixels')
            lines(reference, common + [ref_iter])
            write(reference.parent / 'COMPLETE.json', {'validation': {'rng_receipts_sha256': report.sha(reference)}})
            partial = dict(ref_iter, samples=128, batches=1, indices_sha256='only_prefix')
            lines(run / 'rng_receipts.jsonl', common + [partial])
            result = report.validate_pairing(run, False, report.Assets(), reference)
            self.assertEqual(result['full_passes'], 0)
            self.assertEqual(result['partial_passes'], 1)
            partial['first_batch_pixels_targets_sha256'] = 'wrong'
            lines(run / 'rng_receipts.jsonl', common + [partial])
            with self.assertRaisesRegex(ValueError, 'pairing mismatch'):
                report.validate_pairing(run, False, report.Assets(), reference)

    def test_qv_geometry_excludes_new_classifier_and_uses_raw_weights(self):
        import torch
        torch.set_num_threads(4)
        old = {'fc.weight': torch.zeros(20, 768), 'fc.bias': torch.zeros(20)}
        new = {'fc.weight': torch.zeros(40, 768), 'fc.bias': torch.zeros(40)}
        new['fc.weight'][20:] = 1e6
        fisher = {}
        for layer in range(12):
            for role in ('q', 'v'):
                prefix = f'backbone.lora_vit.blocks.{layer}.attn.qkv'
                a, b = prefix + f'.linear_a_{role}.weight', prefix + f'.linear_b_{role}.weight'
                old[a], old[b] = torch.zeros(10, 768), torch.zeros(768, 10)
                old[a][0, 0] = 1
                new[a], new[b] = old[a].clone(), old[b].clone()
                fisher[prefix + '.' + role] = torch.full((768, 768), 3.)
        new['backbone.lora_vit.blocks.0.attn.qkv.linear_b_q.weight'][0, 0] = 2
        result = report.drift_from_states(old, new, fisher)
        self.assertEqual(result['d_QV_cont'], 2)
        self.assertEqual(result['raw_fisher_weighted_drift_squared'], 12)
        new['fc.weight'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'Old classifier changed'):
            report.drift_from_states(old, new, fisher)


if __name__ == '__main__':
    unittest.main()
