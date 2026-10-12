"""CPU-only regression checks for historical subset stage-path extension."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PATH = Path(__file__).with_name('diagnostic_stage.py')
SPEC = importlib.util.spec_from_file_location('rq2_stage_test_target', PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class StagePathTests(unittest.TestCase):
    def fake_runner(self, scope='subset256', drift=2., damage=.4, q=4.):
        runner = module.StageRunner.__new__(module.StageRunner)
        runner.args = SimpleNamespace(method='sgd', path_task=4, job='path')
        runner.scope_label, runner.scope = module.path_scope({'path_fisher_scope': scope})
        runner.y = {'subset': list(range(256)), 'train': list(range(1259))}
        name = 'sgd_t04'
        runner.drift = {name: drift}
        runner.deltas = {name: [torch.tensor([drift])]}
        runner.meta = {'actual_direction_sha256': {name: 'direction'}}
        runner.config = {}
        runner.protocol = 'test-protocol'
        runner.calls = []
        runner.result = None
        def row(s):
            return {'metrics': {'train': {'loss': 2. + .2*s, 'accuracy': 50., 'samples': 1259},
                                'test': {'loss': 3. + .1*s, 'accuracy': 40., 'samples': 307}},
                    'fisher_by_scope': {runner.scope: {'q': {name: q}, 'raw_trace': 2.,
                                                       'subset_loss': 1. + damage*s,
                                                       'samples': len(runner.y[runner.scope])}}}
        def measure(state, delta=None, fraction=0., fisher_scopes=(), directions=None, **kw):
            runner.calls.append(tuple(fisher_scopes))
            return row(fraction)
        runner.measure_full = measure
        runner.origin_full = lambda scopes, directions: measure('r0', fisher_scopes=scopes)
        runner.endpoint_reference = lambda task: (runner.deltas[name],
            {s: {'local': m} for s, m in row(1.)['metrics'].items()}, {'passed': True})
        runner.finish = lambda result: setattr(runner, 'result', result)
        return runner

    def test_scope_is_explicit_and_256_is_preserved(self):
        self.assertEqual(module.path_scope({}), ('subset256', 'subset'))
        self.assertEqual(module.path_scope({'path_fisher_scope': 'full_train'}), ('full_train', 'train'))
        for config in ({'fisher_samples': 1259}, {'path_fisher_scope': 'train'}):
            with self.assertRaises(ValueError):
                module.path_scope(config)

    def test_subset_path_uses_scope_for_every_node_and_bound(self):
        runner = self.fake_runner()
        runner.path()
        out = runner.result
        self.assertTrue(all(scopes == ('subset',) for scopes in runner.calls))
        self.assertEqual((out['samples'], out['fisher_scope']), (256, 'subset256'))
        self.assertEqual(out['selected_nodes'], 9)
        self.assertEqual(out['integral'], 4.)
        self.assertEqual(out['S_path'], 1.)
        self.assertAlmostEqual(out['D_fisher_scope'], .4)
        self.assertAlmostEqual(out['D_train'], .2)
        self.assertAlmostEqual(out['D_test'], .1)
        self.assertEqual(out['endpoint_metrics']['train']['samples'], 1259)
        self.assertEqual(out['path'][-1]['fisher_scope_loss_for_bound'], 1.4)
        self.assertTrue(out['quadrature_converged'] and out['numerical_gate_passed'])

    def test_optional_full_train_scope(self):
        runner = self.fake_runner(scope='full_train')
        runner.path()
        self.assertTrue(all(scopes == ('train',) for scopes in runner.calls))
        self.assertEqual(runner.result['samples'], 1259)
        self.assertEqual(runner.result['fisher_scope'], 'full_train')

    def test_zero_displacement_has_no_defined_normalized_sensitivity(self):
        runner = self.fake_runner(drift=0., damage=0., q=0.)
        runner.path()
        self.assertEqual(runner.result['integral'], 0.)
        self.assertIsNone(runner.result['S_path'])
        self.assertIsNone(runner.result['directional_to_trace_bound_ratio'])
        self.assertEqual(module.integrate_simpson([0.] * 9), 0.)
        self.assertEqual(module.norm_sq([torch.tensor([3., 4.])]), 25.)
        self.assertAlmostEqual(module.integrate_simpson([(i/8)**3 for i in range(9)]), .25)

    def test_failure_refines_to_33_and_does_not_finish(self):
        runner = self.fake_runner(q=.04, damage=.3)
        with tempfile.TemporaryDirectory() as folder:
            runner.directory = Path(folder)
            with self.assertRaises(RuntimeError):
                runner.path()
            failure = json.loads((runner.directory/'numerical_gate_failure.json').read_text())
            self.assertEqual(failure['selected_nodes'], 33)
            self.assertFalse(failure['numerical_gate_passed'])
            self.assertEqual(failure['status'], 'needs_review')
            self.assertIsNone(runner.result)
            self.assertFalse((runner.directory/'summary.json').exists())

    def test_cli_accepts_all_requested_tasks_and_rejects_task0(self):
        parser = module.argument_parser()
        for task in range(1, 10):
            args = parser.parse_args(['--config', 'unused', '--method', 'sgd', '--path-task', str(task)])
            self.assertEqual(args.path_task, task)
            self.assertEqual(module.required_checkpoints('sgd', 'path', task), {'sgd': (0, task)})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(['--config', 'unused', '--method', 'gam', '--path-task', '0'])

    def test_endpoint_failure_restores_exact_anchor(self):
        runner = module.StageRunner.__new__(module.StageRunner)
        runner.params = [torch.nn.Parameter(torch.tensor([8.]))]
        runner.anchor = [torch.tensor([2.])]
        with patch.object(module.FullRunner, 'endpoint_reference', side_effect=ValueError('bad endpoint')):
            with self.assertRaises(ValueError):
                runner.endpoint_reference(4)
        self.assertTrue(torch.equal(runner.params[0], runner.anchor[0]))

    def test_real_legacy_receipt_and_mutated_subset_rejection(self):
        root = module.EXECUTION/'results_seed0_recovery1'
        data = json.loads((root/'sgd/data_manifest.json').read_text())
        meta = json.loads((root/'sgd/path/manifest.json').read_text())
        receipt = module.legacy_compatibility({}, 'sgd', 4, data, meta)
        self.assertTrue(receipt['passed'])
        altered = copy.deepcopy(data)
        altered['subset_ids'][0] = 'different-example'
        with self.assertRaisesRegex(AssertionError, 'data/subset'):
            module.legacy_compatibility({}, 'sgd', 4, altered, meta)
        altered_meta = copy.deepcopy(meta)
        altered_meta['head_sha256'] = 'different-head'
        with self.assertRaisesRegex(AssertionError, 'head_sha256'):
            module.legacy_compatibility({}, 'sgd', 4, data, altered_meta)


if __name__ == '__main__':
    unittest.main()
