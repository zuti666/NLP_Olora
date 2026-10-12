"""Independent analytic, no-commit, and true skip-probe timing checks."""
import math
import unittest

import torch
from torch import nn

from effective_step import EffectiveStep
from state_transaction import rng_equal, rng_get
from upward_local_audit import (DIRECTIONS, READS, UpwardLocalEngine,
    audit_eight_batch, construct_candidates, native_cost_benchmark, norm)
from upward_residual_step import UpwardResidualStep


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([[.3, .2], [-.1, .4]]))
        self.b = nn.Parameter(torch.tensor([[.4, .2], [.3, -.2]]))
        self.fc = nn.Linear(2, 40)
        self.drop = nn.Dropout(.25)
        self.register_buffer('counter', torch.tensor(7.))
        self.forward_cache = {'initial': torch.tensor([2.])}

    def vector(self):
        return torch.cat([p.reshape(-1) for p in self.parameters()])

    def forward(self, x):
        self.counter.add_(1)
        self.forward_cache = {'last': x.detach().clone()}
        return self.fc(self.drop(x @ self.a.T @ self.b.T))


class UpwardLocalAuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = Toy()
        self.geometry = EffectiveStep([(self.model.a, self.model.b)],
            list(self.model.fc.parameters()), list(self.model.parameters()))
        self.batch = (torch.tensor([[1., 2.], [-1., .5]]), torch.tensor([20, 21]))
        self.current = (torch.tensor([[.7, -1.], [.2, .4]]), torch.tensor([22, 23]))
        self.old = (torch.tensor([[.4, .1], [-.3, 1.]]), torch.tensor([0, 1]))

    def quadratic(self):
        diagonal = torch.linspace(1., 3., self.model.vector().numel())
        def closure(x, y):
            theta = self.model.vector()
            logits = theta[:40].unsqueeze(0).repeat(x.shape[0], 1)
            return .5*(diagonal*theta.square()).sum(), logits
        return diagonal, closure

    def test_candidates_are_three_point_sweep_and_terminal_pair(self):
        g0, g1 = torch.tensor([1., 2.]), torch.tensor([3., 4.])
        g2, g3 = torch.tensor([5., 6.]), torch.tensor([7., 8.])
        values = construct_candidates(g0, g1, g2, g3)
        s = g1-g0
        self.assertEqual(tuple(values), DIRECTIONS)
        self.assertTrue(torch.equal(values['g2_minus_s'], g2-s))
        self.assertTrue(torch.equal(values['g2_plus_s'], g2+s))
        self.assertTrue(torch.equal(values['g3'], g2+(g3-g2)))
        self.assertTrue(torch.equal(values['g0_plus_d'], g0+(g3-g2)))
        self.assertTrue(torch.equal(values['g1_plus_d'], g1+(g3-g2)))

    def test_native_and_matched_true_update_response_on_quadratic(self):
        diagonal, closure = self.quadratic()
        original_rng = rng_get()
        original = self.model.vector().detach().clone()
        engine = UpwardLocalEngine(self.model, closure, self.geometry)
        result = audit_eight_batch(engine, self.batch, self.current, self.old, .02)
        self.assertEqual(result['gradient_reads'], 20)
        self.assertEqual(result['loss_reads'], 68)
        self.assertEqual(len(result['rows']), 16)
        self.assertEqual(result['committed_updates'], 0)
        self.assertTrue(torch.equal(original, self.model.vector()))
        self.assertTrue(rng_equal(original_rng, rng_get()))
        self.assertTrue(all(result['checks'].values()))
        for row in result['rows']:
            self.assertIsNotNone(row['R_grad_actual'])
            self.assertGreaterEqual(row['R_grad_actual'], float(diagonal.min())-.0001)
            self.assertLessEqual(row['R_grad_actual'], float(diagonal.max())+.0001)
            self.assertIn('NOT BA/FC', row['matching_scope'])
            self.assertTrue(row['old_data_only_in_offline_no_grad_loss'])
            if row['mode'] == 'nominal_parameter_length_match':
                self.assertLess(row['relative_error_to_nominal_target'], 1e-5)
        self.assertTrue(torch.equal(self.model.counter, torch.tensor(7.)))
        self.assertEqual(set(self.model.forward_cache), {'initial'})

    def test_native_microbenchmark_really_skips_probes_and_restores(self):
        def closure(x, y):
            logits = self.model(x)
            return nn.functional.cross_entropy(logits[:, 20:40], y-20), logits
        engine = UpwardLocalEngine(self.model, closure, self.geometry)
        g0 = engine.gradient(self.batch)['gradient']
        g1 = engine.gradient(self.batch, ((g0, .05/(engine.native_norm(g0)+1e-12)),))['gradient']
        s = g1-g0
        outer = ((s, 1./(engine.native_norm(s)+1e-12)),)
        g2 = engine.gradient(self.batch, outer)['gradient']
        g3 = engine.gradient(self.batch, outer+((g2, .05/(engine.native_norm(g2)+1e-12)),))['gradient']
        candidates = construct_candidates(g0, g1, g2, g3)
        original = self.model.vector().detach().clone()
        original_rng = rng_get()
        result = native_cost_benchmark(self.model, self.batch, self.geometry, .02,
            warmup=1, timed=1, verification=1, expected_candidates=candidates,
            step_class=UpwardResidualStep)
        self.assertEqual(len(result['trials']), 24)
        self.assertEqual(result['committed_training_updates'], 0)
        self.assertEqual(result['trials_per_method'], 3)
        self.assertTrue(torch.equal(original, self.model.vector()))
        self.assertTrue(rng_equal(original_rng, rng_get()))
        self.assertTrue(all(result['checks'].values()))
        self.assertFalse(result['cuda_synchronized'])
        for row in result['trials']:
            self.assertEqual(row['ordinary_gradient_reads'], READS[row['method']])
            if row['phase'] == 'verification':
                self.assertTrue(all(row['verification'].values()))
        for row in result['rows']:
            self.assertEqual(row['timed_updates'], 1)
            self.assertGreater(row['mean_seconds'], 0.)

    def test_failure_restores_whole_entry(self):
        def failure(x, y):
            self.model.counter.add_(4)
            self.model.forward_cache = {'failure': torch.rand(3)}
            raise RuntimeError('intentional diagnostic failure')
        engine = UpwardLocalEngine(self.model, failure, self.geometry)
        original_rng = rng_get()
        with self.assertRaisesRegex(RuntimeError, 'intentional diagnostic failure'):
            audit_eight_batch(engine, self.batch, self.current, self.old, .02)
        self.assertTrue(all(engine.entry.check().values()))
        self.assertTrue(rng_equal(original_rng, rng_get()))

    def test_zero_first_response_is_explicit_no_outer_motion(self):
        def closure(x, y):
            theta = self.model.vector()
            logits = theta[:40].unsqueeze(0).repeat(x.shape[0], 1)
            return theta.sum(), logits
        engine = UpwardLocalEngine(self.model, closure, self.geometry)
        result = audit_eight_batch(engine, self.batch, self.current, self.old, .02)
        self.assertFalse(result['response_direction_valid'])
        self.assertEqual(result['probe_actual_distances']['g2']['parameter_norm'], 0.)
        self.assertTrue(all(row['R_grad_actual'] == 0. for row in result['rows']))
        self.assertTrue(all(result['checks'].values()))


if __name__ == '__main__':
    unittest.main()
