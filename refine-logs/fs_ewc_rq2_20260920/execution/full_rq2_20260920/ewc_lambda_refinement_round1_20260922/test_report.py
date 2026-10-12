"""CPU-only refinement collector tests; no training, GPU query, or initial writes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('refinement_report', Path(__file__).with_name('report.py'))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class RefinementReportTests(unittest.TestCase):
    def specification(self):
        return report.read_json(Path(__file__).with_name('refinement_spec.json'))

    def test_float_grid_and_safe_unique_trial_ids(self):
        spec = self.specification()
        trials = report.validate_spec(spec)
        self.assertEqual(len(trials), 6)
        self.assertTrue(all(t['ewc_lambda'] != int(t['ewc_lambda']) for t in trials))
        for mutate in (lambda s: s['trials'][0].update(trial_id='../escape'),
                       lambda s: s['trials'][1].update(trial_id=s['trials'][0]['trial_id']),
                       lambda s: s['trials'][0].update(ewc_lambda=13000.)):
            bad = copy.deepcopy(spec)
            mutate(bad)
            with self.assertRaises(ValueError):
                report.validate_spec(bad)

    def test_pending_float_lambda_uses_exact_trial_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial = self.specification()['trials'][0]
            row, cfg = report.validate_refinement_trial(Path(tmp), trial, None, report.old.Assets(), {})
            self.assertEqual(row['ewc_lambda'], trial['ewc_lambda'])
            self.assertEqual(Path(row['run_dir']).name, trial['trial_id'])
            self.assertEqual(row['status'], 'pending')
            self.assertIsNone(row['outcome'])
            self.assertIsNone(cfg)

    def test_qualification_requires_step1_gradient_and_separates_geometry(self):
        def row(lam, old, new, step1, drift):
            return dict(ewc_lambda=lam, outcome='complete', a21=old, a22=new,
                        average_accuracy=(old + new) / 2, P_mean=1 if lam else 0,
                        grad_P_norm_mean=2, grad_P_norm_step1=step1, batch_loss_equals_CE_plus_P=True,
                        d_QV_cont=drift, raw_fisher_weighted_drift_squared=drift)
        rows = [row(0, 80, 70, 0, 1), row(10, 81, 69, 0, .5), row(20, 82, 70, 1, 1.1), row(30, 81, 70, 1, .4)]
        result = report.select_points(rows, self.specification()['efficacy_thresholds'])
        self.assertNotIn(10, result['qualified_lambdas'])
        self.assertEqual(result['qualified_lambdas'], [20, 30])
        self.assertEqual(result['ranked_qualified_lambdas'], [20, 30])
        self.assertEqual(result['geometry_review_required_lambdas'], [20])
        self.assertFalse(result['lambda_locked'])
        self.assertAlmostEqual(rows[-1]['delta_raw_fisher_weighted_drift_squared'], -.6)

    def test_six_observed_without_qualified_point_is_complete_round_not_success_claim(self):
        original = report.read_json(report.INITIAL / 'report/results.json')
        initial_rows = copy.deepcopy(original['individual'])
        for row in initial_rows:
            row.update(source='initial_sweep', source_round=0, trial_id=Path(row['run_dir']).name,
                       grad_P_norm_step1=0 if row['ewc_lambda'] == 0 else 1,
                       batch_loss_equals_CE_plus_P=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.specification()
            spec['input_fingerprints'] = {}
            (root / 'refinement_spec.json').write_text(json.dumps(spec))
            pending = False
            def new_trial(_root, trial, *_args):
                is_pending = pending and trial['trial_id'] == spec['trials'][-1]['trial_id']
                return dict(ewc_lambda=trial['ewc_lambda'], status='pending' if is_pending else 'observed',
                            outcome=None if is_pending else 'diverged',
                            **{k: None for k in report.METRICS}), None
            before = report.sha(report.INITIAL / 'report/COMPLETE.json')
            with patch.object(report, 'validate_initial', return_value=(initial_rows, original['prepared_fisher'],
                                                                       original['shared_scientific_config'], {})), \
                 patch.object(report, 'validate_refinement_trial', side_effect=new_trial), \
                 patch.object(report, 'mechanism_evidence'):
                data = report.build_report(root, root / 'report')
                self.assertEqual(data['total_observed_points'], 20)
                self.assertEqual(data['round_status'], 'observed')
                self.assertEqual(data['status'], 'refinement_required')
                receipt = report.read_json(root / 'report/COMPLETE.json')
                self.assertEqual(receipt['status'], 'complete')
                self.assertEqual(receipt['observed_refinement_points'], 6)
                self.assertEqual(receipt['scientific_status'], 'refinement_required')
                self.assertFalse(receipt['additional_round_launched'])
                pending = True
                with self.assertRaisesRegex(RuntimeError, '5/6 observed'):
                    report.build_report(root, root / 'report')
                self.assertFalse((root / 'report/COMPLETE.json').exists())
            self.assertEqual(report.sha(report.INITIAL / 'report/COMPLETE.json'), before)


if __name__ == '__main__':
    unittest.main()
