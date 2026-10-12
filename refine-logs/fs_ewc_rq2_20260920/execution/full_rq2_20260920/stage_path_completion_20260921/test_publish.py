"""Numerical guards for the CPU stage-path publisher; no GPU or fake files."""
import copy
import math
import unittest

import publish


class NumericalGuards(unittest.TestCase):
    def test_nested_simpson_cubic(self):
        # Analytic polynomial integral is exact on every supported nested grid.
        for n in (5, 9, 17, 33):
            values = [(i/(n-1))**3 + 2*(i/(n-1)) + 3 for i in range(n)]
            self.assertAlmostEqual(publish.simpson(values), 4.25, places=13)
            if n > 5:
                self.assertAlmostEqual(publish.simpson(values[::2]), 4.25, places=13)

    def test_reject_nonfinite_integrand(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(AssertionError):
                publish.simpson([1, 2, value, 4, 5])
            with self.assertRaises(AssertionError):
                publish.finite_tree({'nested': [{'bad': value}]})

    def test_gate_refines_unstable_integral(self):
        cfg = dict(quadrature_atol=1e-8, quadrature_rtol=.05,
                   bound_rtol=.02, path_peak_neighbor_factor=4.)
        gate = publish.recompute_gate(1., 2., .1, [2.] * 9, cfg)
        self.assertFalse(gate['passed'])
        self.assertTrue(gate['request_refinement'])

    def test_gate_refines_hidden_sampled_peak(self):
        cfg = dict(quadrature_atol=1e-8, quadrature_rtol=.05,
                   bound_rtol=.02, path_peak_neighbor_factor=4.)
        gate = publish.recompute_gate(1., 1., .1, [1., 1., 1., 1., 8., 1., 1., 1., 1.], cfg)
        self.assertTrue(gate['passed'])
        self.assertTrue(gate['request_refinement'])
        self.assertTrue(gate['sampled_peak_needs_resolution_check'])

    def test_only_output_directory_is_nonscientific(self):
        original = {'output_dir': '/old', 'fisher_samples': 256, 'workers': 4}
        moved = dict(original, output_dir='/new')
        self.assertEqual(publish.scientific_config(original), publish.scientific_config(moved))
        moved['fisher_samples'] = 1259
        self.assertNotEqual(publish.scientific_config(original), publish.scientific_config(moved))

    def test_reject_changed_real_summary_sensitivity(self):
        # Read genuine original records; alter only a private in-memory copy.
        sources, checkpoints = [], {}
        base = publish.BASE / 'formal_results/seed0/sgd'
        summary, meta, rows, raw = publish.load_job(base/'trajectory_fisher', 0, 'sgd', sources, checkpoints)
        trajectory = publish.extract_trajectory(0, 'sgd', summary, meta, rows, raw)
        endpoint = next(r for r in trajectory if r['task'] == 1)
        summary, meta, rows, raw = publish.load_job(base/'path_t01', 0, 'sgd', sources, checkpoints,
                                                   trajectory_meta=meta)
        valid = publish.extract_path(0, 'sgd', 1, summary, meta, rows, raw, endpoint)
        self.assertEqual(len(valid[1]), 9)
        corrupt = copy.deepcopy(summary)
        corrupt['S_path'] *= 2
        with self.assertRaises(AssertionError):
            publish.extract_path(0, 'sgd', 1, corrupt, meta, rows, raw, endpoint)


if __name__ == '__main__':
    unittest.main()
