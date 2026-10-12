"""Round2 CPU collector regression tests; fixtures never mutate frozen sources."""
import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('refinement_round2_report', Path(__file__).with_name('report.py'))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class RefinementReportTests(unittest.TestCase):
    def specification(self):
        return report.read_json(Path(__file__).with_name('refinement_spec.json'))

    def history(self):
        original = report.read_json(report.PRIOR / 'report/results.json')
        return (copy.deepcopy(original['individual']), original['prepared_fisher'],
                original['shared_scientific_config'], {})

    @contextmanager
    def collection_fixture(self, *, pending=False, duplicate=False, config=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.specification()
            spec['input_fingerprints'] = {}
            (root / 'refinement_spec.json').write_text(json.dumps(spec))
            def new_trial(_root, trial, *_args):
                is_pending = pending and trial['trial_id'] == spec['trials'][-1]['trial_id']
                lam = 10000 if duplicate and trial['trial_id'] == spec['trials'][0]['trial_id'] else trial['ewc_lambda']
                return dict(ewc_lambda=lam, status='pending' if is_pending else 'observed',
                            outcome=None if is_pending else 'diverged',
                            run_dir=str(root / 'training' / trial['trial_id']),
                            **{k: None for k in report.METRICS}), config
            with patch.object(report, 'validate_history', side_effect=lambda *_: self.history()), \
                 patch.object(report, 'validate_refinement_trial', side_effect=new_trial), \
                 patch.object(report, 'mechanism_evidence'):
                yield root

    def test_exact_round2_grid_and_frozen_history_required(self):
        spec = self.specification()
        self.assertEqual(len(report.validate_spec(spec)), 6)
        mutations = [lambda s: s.update(round=1),
                     lambda s: s['trials'][0].update(trial_id='../escape'),
                     lambda s: s['trials'][1].update(trial_id=s['trials'][0]['trial_id']),
                     lambda s: s['trials'][0].update(ewc_lambda=3000),
                     lambda s: s['trials'][1].update(ewc_lambda=s['trials'][0]['ewc_lambda']),
                     lambda s: s['trials'][0].update(interval=[10000, 30000]),
                     lambda s: s['tested_lambda_union'].pop(),
                     lambda s: s.update(prior_refinement_rounds=[]),
                     lambda s: s['prior_refinement_rounds'][0].update(complete_sha256='0' * 64),
                     lambda s: s['efficacy_thresholds'].update(retention_gain_pp=.3)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                bad = copy.deepcopy(spec)
                mutate(bad)
                with self.assertRaises(ValueError):
                    report.validate_spec(bad)

    def test_pending_trial_needs_observed_wrapper_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial = self.specification()['trials'][0]
            root = Path(tmp)
            run = root / 'training' / trial['trial_id']
            run.mkdir(parents=True)
            (run / 'COMPLETE.json').write_text('{}')
            row, cfg = report.validate_refinement_trial(root, trial, None, report.old.Assets(), {})
            self.assertEqual(row['ewc_lambda'], trial['ewc_lambda'])
            self.assertEqual(Path(row['run_dir']).name, trial['trial_id'])
            self.assertEqual(row['status'], 'pending')
            self.assertIsNone(row['outcome'])
            self.assertIsNone(cfg)
            (run / 'FAILED.json').write_text('{}')
            row, _ = report.validate_refinement_trial(root, trial, None, report.old.Assets(), {})
            self.assertEqual(row['status'], 'failed')

    def test_round2_validator_rejects_unmodified_round1_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.specification()
            (root / 'refinement_spec.json').write_text(json.dumps(spec))
            trial = spec['trials'][0]
            run = root / 'training' / trial['trial_id']
            run.mkdir(parents=True)
            receipt = dict(trial_id=trial['trial_id'], round=1, spec=str(root / 'refinement_spec.json'))
            (run / 'TRIAL_RESULT.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'round/spec mismatch'):
                report.validate_refinement_trial(root, trial, {}, report.old.Assets(), {})

    def test_union_preserves_negative_and_divergent_history(self):
        historical = self.history()[0]
        before = {str(path): report.sha(path) for path in (report.INITIAL / 'report/COMPLETE.json',
                                                         report.PRIOR / 'report/COMPLETE.json')}
        with self.collection_fixture() as root:
            data = report.build_report(root, root / 'report')
            self.assertEqual(data['total_observed_points'], 26)
            self.assertEqual(len({row['ewc_lambda'] for row in data['individual']}), 26)
            self.assertEqual([sum(row['source_round'] == n for row in data['individual']) for n in range(3)], [14, 6, 6])
            self.assertEqual([row for row in data['individual'] if row['source_round'] < 2], historical)
            self.assertEqual(data['completed_lambda_points'], 17)
            self.assertEqual(data['diverged_lambda_points'], 9)
            self.assertEqual(data['selection']['qualified_lambdas'], [])
            self.assertEqual(data['scientific_status'], 'refinement_required')
            complete = report.read_json(root / 'report/COMPLETE.json')
            self.assertEqual(complete['status'], 'complete')
            self.assertEqual(complete['round'], 2)
            self.assertEqual(complete['observed_refinement_points'], 6)
            self.assertEqual(complete['total_observed_points'], 26)
            self.assertEqual(complete['scientific_status'], 'refinement_required')
            self.assertFalse(complete['additional_round_launched'])
            self.assertFalse(complete['gpu_evaluation_performed'])
            self.assertFalse((root / 'SELECTION.json').exists())
            for name, digest in complete['outputs_sha256'].items():
                self.assertEqual(report.sha(root / 'report' / name), digest)
            md = (root / 'report/RESULTS.md').read_text()
            self.assertIn('第 2 轮', md)
            self.assertIn('26/26', md)
        self.assertEqual({path: report.sha(path) for path in before}, before)

    def test_missing_trial_cannot_create_or_retain_complete(self):
        with self.collection_fixture(pending=True) as root:
            out = root / 'report'
            out.mkdir()
            (out / 'COMPLETE.json').write_text('{"status":"complete"}')
            with self.assertRaisesRegex(RuntimeError, '5/6 observed'):
                report.build_report(root, out)
            self.assertFalse((out / 'COMPLETE.json').exists())
            data = report.build_report(root, out, allow_partial=True)
            self.assertEqual(data['round_status'], 'partial')
            self.assertEqual(data['scientific_status'], 'pending_round')
            self.assertEqual(data['total_observed_points'], 25)
            self.assertEqual(data['missing_refinement_trials'], ['refine2_06_seed0'])
            self.assertFalse((out / 'COMPLETE.json').exists())

    def test_duplicate_combined_lambda_is_rejected(self):
        with self.collection_fixture(duplicate=True) as root:
            with self.assertRaisesRegex(ValueError, '26 unique points'):
                report.build_report(root, root / 'report')
            self.assertFalse((root / 'report/COMPLETE.json').exists())

    def test_shared_scientific_config_change_is_rejected(self):
        with self.collection_fixture(config={'lrate': .03}) as root:
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                report.build_report(root, root / 'report')

    def test_bad_frozen_source_is_rejected_before_native_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)
            (fake / 'report').mkdir()
            for name in ('report.py', 'refinement_spec.json', 'report/COMPLETE.json',
                         'report/RESULTS.md', 'report/results.csv', 'report/results.json'):
                shutil.copyfile(report.PRIOR / name, fake / name)
            (fake / 'report/results.json').write_text('{}')
            with patch.object(report, 'PRIOR', fake), \
                 patch.object(report, 'validate_initial', side_effect=AssertionError('must fail before initial loading')):
                with self.assertRaisesRegex(ValueError, 'Frozen round1 report output changed'):
                    report.validate_history(report.EvidenceAssets(), {})
            (fake / 'report.py').write_text('# changed helper')
            with patch.object(report, 'PRIOR', fake):
                with self.assertRaisesRegex(ValueError, 'Frozen round1 report helper changed'):
                    report.load_prior_helpers()

    def test_historical_sources_cannot_be_output_roots(self):
        for root in (report.INITIAL, report.PRIOR, report.PRIOR / 'nested'):
            with self.assertRaisesRegex(ValueError, 'historical sources are immutable'):
                report.build_report(root, root / 'report')

    def test_evidence_changed_after_verification_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'receipt.json'
            path.write_text('{}')
            assets = report.EvidenceAssets()
            digest = assets.hash(path)
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'Evidence changed during collection'):
                assets.verify(path, digest, 'test')

    def test_qualification_preserves_step1_and_three_accuracy_thresholds(self):
        def row(lam, old, new, step1=1, drift=.5, loss=True):
            return dict(ewc_lambda=lam, outcome='complete', a21=old, a22=new,
                        average_accuracy=(old + new) / 2, P_mean=1 if lam else 0,
                        grad_P_norm_mean=2, grad_P_norm_step1=step1, batch_loss_equals_CE_plus_P=loss,
                        d_QV_cont=drift, raw_fisher_weighted_drift_squared=drift)
        rows = [row(0, 80, 70, 0, 1), row(10, 81, 70, 0), row(20, 82, 70, drift=1.1),
                row(30, 81, 70), row(40, 80.99, 72), row(50, 87, 64.99),
                row(60, 81, 68.99), row(70, 82, 70, loss=False)]
        result = report.select_points(rows, self.specification()['efficacy_thresholds'])
        self.assertEqual(result['qualified_lambdas'], [20, 30])
        self.assertEqual(result['ranked_qualified_lambdas'], [20, 30])
        self.assertEqual(result['geometry_review_required_lambdas'], [20])
        self.assertFalse(result['lambda_locked'])


if __name__ == '__main__':
    unittest.main()
