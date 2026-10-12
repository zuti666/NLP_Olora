"""CPU checks for scope separation, forward-only sweeps, and numerical gates."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PATH = Path(__file__).with_name('diagnostic_full.py')
SPEC = importlib.util.spec_from_file_location('rq2_full_test_target', PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class MemoryJournal:
    def __init__(self):
        self.rows = {}

    def add(self, row):
        self.rows[row['state_id']] = row
        return row


class DiagnosticsTests(unittest.TestCase):
    def fake_runner(self):
        runner = module.FullRunner.__new__(module.FullRunner)
        runner.args = SimpleNamespace(method='sgd')
        runner.seed = 2
        runner.device = torch.device('cpu')
        runner.params = [torch.nn.Parameter(torch.tensor([2.0]))]
        runner.anchor = [torch.tensor([2.0])]
        runner.journal = MemoryJournal()
        runner.meta = {'coordinate_sha256': 'coordinates'}
        runner.data = {'manifest_sha256': 'samples'}
        runner.checksums = {'sgd': {'0': 'anchor'}}
        runner.evaluate = lambda split: {'loss': float(runner.params[0].detach().square()),
                                         'accuracy': 50., 'samples': 3}
        return runner

    def test_dense_point_is_forward_only_and_restores(self):
        runner = self.fake_runner()
        with patch.object(module, 'fisher', side_effect=AssertionError('Dense path must not calculate Fisher')):
            row = runner.measure_full('perturbed', [torch.tensor([1.])], 3., kind='dense')
        self.assertEqual(row['metrics']['train']['loss'], 25.)
        self.assertEqual(row['effective_weight_drift'], 3.)
        self.assertEqual(row['fisher_by_scope'], {})
        self.assertTrue(row['anchor_restored_exactly'])
        self.assertTrue(torch.equal(runner.params[0], runner.anchor[0]))

    def test_failure_restores_anchor_without_committing(self):
        runner = self.fake_runner()
        runner.evaluate = lambda split: {'loss': float('nan'), 'accuracy': 0., 'samples': 3}
        with self.assertRaises(ValueError):
            runner.measure_full('bad', [torch.tensor([1.])], 3.)
        self.assertTrue(torch.equal(runner.params[0], runner.anchor[0]))
        self.assertEqual(runner.journal.rows, {})

    def test_anchor_jobs_do_not_require_future_or_other_method(self):
        self.assertEqual(module.required_checkpoints('gam', 'dense', 9), {'gam': (0,)})
        self.assertEqual(module.required_checkpoints('sgd', 'path', 5), {'sgd': (0, 5)})
        self.assertEqual(module.required_checkpoints('sgd', 'bridge_full', 9), {'sgd': (0, 9), 'gam': (0, 9)})

    def test_protocol_cardinalities(self):
        # Old pilot 'radii' and 'directions' cannot silently shrink full jobs.
        config = {'radii': [0, 1], 'directions': 1}
        for job, expected in [('dense', 8001), ('sparse_train', 97),
                              ('sparse_test', 49), ('sparse_train_full', 13)]:
            radii, indices = module.grid(job, config)
            self.assertEqual(1 + (len(radii) - 1) * len(indices), expected)

    def test_simpson_exact_cubic_and_bound_gate(self):
        values = [(i / 8) ** 3 for i in range(9)]
        self.assertAlmostEqual(module.integrate_simpson(values), .25)
        gate = module.quadrature_gate(.25, .25, .4, values, {})
        self.assertTrue(gate['passed'])
        gate = module.quadrature_gate(.25, .25, .6, values, {})
        self.assertTrue(gate['stable'])
        self.assertFalse(gate['bound_passed'])
        self.assertFalse(gate['passed'])

    def test_stability_uses_absolute_plus_symmetric_relative_tolerance(self):
        gate = module.quadrature_gate(1e-12, 2e-12, 0., [1., 1., 1.], {})
        self.assertTrue(gate['stable'])
        gate = module.quadrature_gate(1., 1.1, 0., [1., 1., 1.], {})
        self.assertFalse(gate['stable'])
        gate = module.quadrature_gate(1., 1., 0., [1., 10., 1.], {})
        self.assertTrue(gate['sampled_peak_needs_resolution_check'])
        self.assertTrue(gate['request_refinement'])
        self.assertTrue(gate['passed'])  # a physical peak alone is not failure


if __name__ == '__main__':
    unittest.main()
