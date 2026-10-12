"""CPU regression tests for evidence rejection and complete-three-seed reporting."""
import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('fixed_lambda_report', Path(__file__).with_name('report.py'))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def rows(self):
        return [dict(arm=arm, seed=seed, trial_id=f'{arm}_seed{seed}', status='complete', ewc_lambda=20000.,
                     **{k: float(seed + 1) for k in report.METRICS}) for arm in report.ARMS for seed in report.SEEDS]

    def test_sample_sd_only_with_all_three_valid_seeds(self):
        rows = self.rows()
        full = report.summarize(rows)
        self.assertEqual(full[0]['statistics']['a21'], dict(mean=2., sample_sd=1., n=3))
        for state in ('pending', 'diverged', 'invalid', 'failed'):
            with self.subTest(state=state):
                bad = copy.deepcopy(rows)
                bad[1]['status'] = state
                data = report.summarize(bad)
                self.assertEqual(data[0]['n_complete'], 2)
                self.assertTrue(all(s['mean'] is None and s['sample_sd'] is None for s in data[0]['statistics'].values()))
                self.assertEqual(data[1]['statistics']['a21']['sample_sd'], 1.)

    def test_duplicate_missing_and_nonfinite_seed_cells_rejected(self):
        rows = self.rows()
        for bad in (rows[:-1], rows + [rows[0]], [rows[0]] + rows[:-1]):
            with self.assertRaisesRegex(ValueError, 'exactly 12'):
                report.summarize(bad)
        for value in (float('nan'), float('inf'), True):
            bad = copy.deepcopy(rows)
            bad[1]['a21'] = value
            with self.assertRaises(ValueError):
                report.summarize(bad)

    def test_cil_lower_triangle_and_parent_are_required(self):
        metrics = {'cnn': {'matrices': {'final': [[80., float('nan')], [75., 65.]],
                                        't01': [[80., float('nan')], [75., 65.]]}}}
        validation = {'accuracy_CIL': dict(a11=80., a21=75., a22=65.)}
        result = report.read_matrix(metrics, validation, 80.)
        self.assertEqual((result['BWT'], result['A2']), (-5., 70.))
        for field, value in (([1, 0], float('nan')), ([1, 1], 101.), ([0, 0], 81.)):
            bad = copy.deepcopy(metrics)
            for kind in ('final', 't01'):
                bad['cnn']['matrices'][kind][field[0]][field[1]] = value
            with self.assertRaises(ValueError):
                report.read_matrix(bad, validation, 80.)
        bad = copy.deepcopy(metrics)
        bad['cnn']['matrices']['t01'][1][0] = 74.
        with self.assertRaisesRegex(ValueError, 'disagree'):
            report.read_matrix(bad, validation, 80.)

    def telemetry_fixture(self, optimizer='gam', count=400):
        source = f'actual_native_{optimizer}_base_optimizer_gradients_immediately_before_step'
        component = ('native_clean_and_fisher_grad_caches; gam_direction=actual_combined-EWC; float64_norms'
                     if optimizer == 'gam' else 'native_EWC_autograd; clean=actual_combined-EWC; float64_norms')
        rows = []
        for step in range(count):
            row = dict(step=step, task=1, ewc_lambda=20000., parameters_finite_after_step=True,
                       ce_clean=2., ewc=.5, loss=2.5, grad_combined_norm=4., gradient_source=source,
                       native_ewc_evaluations=1, single_ewc_injection_verified=True,
                       gradient_components_sampled=step == 1 or step % 8 == 0)
            if row['gradient_components_sampled']:
                row.update(grad_clean_norm=2., grad_fisher_norm=.2, fisher_to_clean_norm_ratio=.2 / (2. + 1e-12),
                           component_source=component, grad_gam_direction_norm=3.)
            rows.append(row)
        valid = dict(losses={'train_clean_loss_mean': 2., 'train_fisher_loss_mean': .5},
                     gradients=dict(source=source, component_source=component,
                                    component_sample_count=sum(r['gradient_components_sampled'] for r in rows),
                                    grad_clean_norm_mean=2., grad_fisher_norm_mean=.2,
                                    fisher_to_clean_norm_ratio_mean=.2 / (2. + 1e-12), grad_gam_direction_norm_mean=3.,
                                    single_ewc_injection_verified=True, step1_ewc=.5, step1_grad_fisher_norm=.2))
        return rows, valid

    def test_gam_shaping_residual_not_called_clean_ce(self):
        rows, valid = self.telemetry_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'training_batch_metrics.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            data = report.telemetry(run, 'gam', True, valid)
            self.assertEqual(data['grad_reference_norm_mean'], 2.)
            self.assertEqual(data['grad_gam_direction_norm_mean'], 3.)
            self.assertEqual(data['gradient_reference'], 'native GAM clean_grad_cache')
            self.assertNotIn('grad_CE_norm_mean', data)
            bad = copy.deepcopy(valid)
            bad['gradients']['source'] = 'clean_CE_gradient'
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                report.telemetry(run, 'gam', True, bad)

    def test_batch_source_finiteness_scaling_and_step1_gates(self):
        rows, valid = self.telemetry_fixture('sgd')
        mutations = [lambda r: r[2].update(gradient_source='actual_native_gam_base_optimizer_gradients_immediately_before_step'),
                     lambda r: r[2].update(native_ewc_evaluations=2),
                     lambda r: r[1].update(component_source='fabricated'),
                     lambda r: r[5].update(loss=float('nan')),
                     lambda r: r[5].update(loss=3.),
                     lambda r: r[1].update(grad_fisher_norm=0., fisher_to_clean_norm_ratio=0.),
                     lambda r: r[1].update(fisher_to_clean_norm_ratio=.9),
                     lambda r: r.pop()]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for mutate in mutations:
                bad = copy.deepcopy(rows)
                mutate(bad)
                (run / 'training_batch_metrics.jsonl').write_text('\n'.join(map(json.dumps, bad)))
                with self.assertRaises(ValueError):
                    report.telemetry(run, 'sgd', True, valid)

    def test_divergent_telemetry_is_only_finite_prefix(self):
        rows, _ = self.telemetry_fixture(count=3)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'training_batch_metrics.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            data = report.telemetry(run, 'gam', False, {})
            self.assertEqual(data['observed_finite_steps'], 3)
            self.assertEqual(data['telemetry_scope'], 'finite prefix before divergence')

    def test_prepared_identity_rejects_wrong_seed_method_and_hash(self):
        prepared = dict(payload_sha256='a', receipt_sha256='b', parent_checkpoint_sha256='c',
                        payload_path='/tmp/payload', receipt_path='/tmp/receipt', receipt={'seed': 2, 'anchor_method': 'gam'})
        identity = dict(status='verified', execution_ready=True, fisher_policy=report.FISHER_POLICY,
                        path='/tmp/payload', receipt_path='/tmp/receipt', payload_sha256='a', receipt_sha256='b',
                        parent_checkpoint_sha256='c', seed=2, anchor_method='gam')
        report.prepared_identity(identity, prepared)
        for key, value in [('seed', 0), ('anchor_method', 'sgd'), ('payload_sha256', 'x'),
                           ('parent_checkpoint_sha256', 'x'), ('path', '/tmp/other'), ('execution_ready', False)]:
            bad = dict(identity, **{key: value})
            with self.assertRaises(ValueError):
                report.prepared_identity(bad, prepared)

    @contextmanager
    def report_fixture(self, states=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trials = [dict(trial_id=f'{arm}_seed{seed}', arm=arm, seed=seed,
                           parent_id=f'{report.ARMS[arm][0]}_seed{seed}', run_dir=str(root / 'training' / f'{arm}_seed{seed}'))
                      for arm in report.ARMS for seed in report.SEEDS]
            spec = dict(trials=trials, parents=[dict(id=f'{method}_seed{seed}') for method in ('sgd', 'gam') for seed in report.SEEDS])
            selection = dict(selection_method='explicit_user_fixed_common_lambda')
            for name in ('experiment_spec.json', 'SELECTION.json'):
                (root / name).write_text('{}')
            rows = {r['trial_id']: r for r in self.rows()}
            for name, state in (states or {}).items():
                rows[name]['status'] = state
            with patch.object(report, 'HERE', root), patch.object(report, 'validate_spec', return_value=(spec, selection)), \
                 patch.object(report, 'old_helpers'), patch.object(report, 'validate_anchor', return_value={}), \
                 patch.object(report, 'validate_prepared', return_value={}), \
                 patch.object(report, 'validate_trial', side_effect=lambda _root, t, *_: copy.deepcopy(rows[t['trial_id']])):
                yield root

    def test_all_12_complete_outputs_and_sample_sd(self):
        with self.report_fixture() as root:
            data = report.build_report(root, root / 'report')
            self.assertEqual(data['completed_trials'], 12)
            self.assertEqual(data['validated_seeds'], [0, 1, 2])
            receipt = report.read_json(root / 'report/COMPLETE.json')
            self.assertEqual(receipt['ewc_lambda'], 20000.)
            for name, digest in receipt['outputs_sha256'].items():
                self.assertEqual(report.sha(root / 'report' / name), digest)
            self.assertIn('不是扫描最优值', (root / 'report/RESULTS.md').read_text())
            self.assertIn('not an automatically selected optimum', (root / 'report/table.tex').read_text())
            self.assertFalse(data['matched_controls']['causal_protection_gain_available'])

    def test_partial_divergent_invalid_results_never_complete(self):
        states = {'ewclora_seed1': 'diverged', 'fs_ewclora_seed2': 'invalid', 'task1_shaping_seed0': 'pending'}
        with self.report_fixture(states) as root:
            output = root / 'report'
            output.mkdir()
            (output / 'COMPLETE.json').write_text('{"status":"complete"}')
            with self.assertRaisesRegex(RuntimeError, '9/12'):
                report.build_report(root, output)
            self.assertFalse((output / 'COMPLETE.json').exists())
            data = report.build_report(root, output, True)
            self.assertEqual(data['status'], 'partial')
            self.assertEqual(data['diverged_trials'], 1)
            self.assertEqual(data['invalid_trials'], 1)
            self.assertEqual(data['pending_trials'], 1)
            self.assertEqual(data['validated_seeds'], [])
            self.assertIsNone(data['summary'][0]['statistics']['a11']['mean'])

    def test_extra_training_cell_and_unsafe_output_rejected(self):
        with self.report_fixture() as root:
            (root / 'training/unexpected_seed3').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Unexpected formal'):
                report.build_report(root, root / 'report')
            for folder in ('training', 'smoke', 'prepared_fisher', 'campaign', 'cpu_validation'):
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    report.build_report(root, root / folder / 'bad')

    def test_changed_evidence_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.json'
            path.write_text('{}')
            assets = report.Assets()
            digest = assets.hash(path)
            path.write_text('{"altered":true}')
            with self.assertRaisesRegex(ValueError, 'changed during collection'):
                assets.verify(path, digest, 'input')
            path.write_text('{"metric":NaN}')
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                report.read_json(path)

    def test_unmodified_native_terminal_absence_remains_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = dict(trial_id='task1_shaping_seed2', arm='task1_shaping', seed=2,
                         anchor_method='gam', future_optimizer='sgd')
            row = report.validate_trial(root, trial, None, None, report.Assets(), None)
            self.assertEqual(row['status'], 'pending')
            run = root / 'training' / trial['trial_id']
            run.mkdir(parents=True)
            (run / 'COMPLETE.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'validated parent/raw Fisher'):
                report.validate_trial(root, trial, None, None, report.Assets(), None)
            (run / 'DIVERGED.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                report.validate_trial(root, trial, None, None, report.Assets(), None)


if __name__ == '__main__':
    unittest.main()
