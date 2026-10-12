#!/usr/bin/env python3
"""CPU algebra and scientific-gate tests; no data/model/GPU training."""
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import curvature_diagnostics as extension
CAMPAIGN = Path(__file__).resolve().parent.parent / 'imageneta-response-state-chain-gpu456-20261011T195426Z'
extension.imports(CAMPAIGN)
import torch
from state_transaction import State
from offline_core import DiagnosticEngine, norm


@contextlib.contextmanager
def unchanged_backend(model, backend):
    yield


class Toy(torch.nn.Module):
    def __init__(self, theta, loss):
        super().__init__()
        self.theta = torch.nn.Parameter(theta.clone())
        self.frozen = torch.nn.Parameter(torch.ones(2, dtype=theta.dtype), requires_grad=False)
        self.cache_tensor = torch.ones(2, dtype=theta.dtype)
        self.loss_definition = loss

    def loss(self, inputs, targets):
        return self.loss_definition(self.theta)


class ToyGeometry:
    @staticmethod
    def from_model(model):
        return SimpleNamespace(params=[model.theta])


def make_context(theta, loss):
    model = Toy(theta, loss)
    return SimpleNamespace(model=model, geometry=ToyGeometry.from_model(model),
                           loss_fn=model.loss, unbound_loss=lambda m, x, y: m.loss(x, y),
                           State=State, backend_context=unchanged_backend)


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.x = torch.zeros(8, 1, dtype=torch.float64)
        self.y = torch.zeros(8, dtype=torch.long)

    def test_indefinite_quadratic_exact_original_extended_and_correction_scales(self):
        H = torch.diag(torch.tensor([-1., 2., 4.], dtype=torch.float64))
        context = make_context(torch.tensor([1., 0., 0.], dtype=torch.float64),
                               lambda theta: .5 * theta @ H @ theta)
        result = extension.diagnostic_cell(context, self.x, self.y, .02)
        self.assertTrue(result['fp32']['original_gate_accepted'])
        self.assertTrue(result['fp32']['extended_gate_accepted'])
        self.assertLess(result['fp32']['frozen_H_quadratic_form'], 0.)
        self.assertEqual(len(result['fp32']['finite_difference']['H_g0']['exploratory']), 7)
        self.assertEqual(len(result['fp32']['finite_difference']['H_g0']['original']), 3)
        for row in result['scales']:
            self.assertAlmostEqual(row['cosine'], 1., places=12)
            self.assertLess(row['relative_error'], 1e-10)
        self.assertAlmostEqual(result['scales'][1]['delta_native_norm'] * 2,
                               result['scales'][0]['delta_native_norm'], places=12)
        self.assertEqual(result['submitted_training_updates'], 0)
        self.assertTrue(all(result['restoration'].values()))

    def test_original_failure_retained_when_predeclared_small_grid_converges(self):
        # d^2/(3 theta^2) central-FD error: all original distances are too large;
        # the predeclared small points converge. Linear term keeps g0/q nonzero.
        context = make_context(torch.tensor([.001], dtype=torch.float64),
                               lambda theta: (theta + .25 * theta ** 4).sum())
        engine = DiagnosticEngine(context, 2)
        _, q, p, gate = extension.curvature_gate(engine, self.x, self.y)
        self.assertFalse(gate['original_gate_accepted'])
        self.assertTrue(gate['extended_gate_accepted'])
        self.assertIsNotNone(q)
        self.assertIsNotNone(p)
        for product in gate['finite_difference'].values():
            self.assertFalse(product['original_passed'])
            self.assertTrue(product['exploratory_passed'])
            self.assertEqual([r['distance'] for r in product['original']], [.001, .003, .01])
        engine.close()

    def test_first_order_reference_failure_prohibits_hvp_and_comparison(self):
        context = make_context(torch.ones(2, dtype=torch.float64), lambda theta: .5 * theta.square().sum())
        engine = DiagnosticEngine(context, 2)
        wrong_reference = (torch.full_like(context.model.theta, 2.),)
        _, q, p, gate = extension.curvature_gate(engine, self.x, self.y, wrong_reference)
        self.assertFalse(gate['accepted'])
        self.assertEqual(gate['status'], 'first_order_reference_disagreement')
        self.assertIsNone(q)
        self.assertIsNone(p)
        self.assertEqual(engine.hvp_calls, 0)
        engine.close()

    def test_detached_second_vector_matches_dense_H_squared(self):
        H = torch.tensor([[2., 1.], [1., -3.]], dtype=torch.float64)
        theta = torch.tensor([.2, -.7], dtype=torch.float64)
        context = make_context(theta, lambda t: .5 * t @ H @ t)
        engine = DiagnosticEngine(context, 3)
        g0, q, p, gate = extension.curvature_gate(engine, self.x, self.y)
        torch.testing.assert_close(q[0], H @ g0[0], rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(p[0], H @ H @ g0[0], rtol=1e-13, atol=1e-13)
        self.assertFalse(q[0].requires_grad)
        self.assertFalse(p[0].requires_grad)
        self.assertTrue(gate['accepted'])
        engine.close()

    def test_normalized_FD_direction_rescaling_restores_unscaled_HVP(self):
        H = torch.tensor([[2., 1.], [1., 3.]], dtype=torch.float64)
        context = make_context(torch.ones(2, dtype=torch.float64), lambda t: .5 * t @ H @ t)
        engine = DiagnosticEngine(context, 2)
        vector = (torch.tensor([.3, -.8], dtype=torch.float64),)
        small = engine.finite_difference_hvp(self.x, self.y, vector, 1e-4)
        big = engine.finite_difference_hvp(self.x, self.y, extension.scaled(vector, 7.), 1e-4)
        torch.testing.assert_close(big[0], 7 * small[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(small[0], H @ vector[0], rtol=1e-11, atol=1e-11)
        engine.close()

    def test_degenerate_gradient_and_zero_H_retained_without_fake_cosine(self):
        for loss, status in ((lambda t: (t * 0.).sum(), 'degenerate_gradient'),
                             (lambda t: t.sum(), 'degenerate_curvature')):
            context = make_context(torch.ones(2, dtype=torch.float64), loss)
            result = extension.diagnostic_cell(context, self.x, self.y, .02)
            self.assertEqual(result['fp32']['status'], status)
            self.assertFalse(result['candidate_comparisons_permitted'])
            self.assertEqual(result['scales'], [])
            self.assertTrue(all(result['restoration'].values()))

    def test_nonfinite_gradient_is_scientific_outcome_and_state_restored(self):
        context = make_context(torch.ones(2, dtype=torch.float64), lambda t: (t * float('nan')).sum())
        context.model.theta.grad = torch.full_like(context.model.theta, 3.)
        state = State(context.model)
        result = extension.diagnostic_cell(context, self.x, self.y, .02)
        self.assertEqual(result['status'], 'inconclusive_nonfinite_numerics')
        self.assertFalse(result['candidate_comparisons_permitted'])
        self.assertTrue(all(state.check().values()))

    def test_FP64_clone_preserves_BASE_and_reference_not_native_model(self):
        context = make_context(torch.tensor([.125, -.375], dtype=torch.float32),
                               lambda t: .5 * t.square().sum())
        original = State(context.model)
        clone = extension.float64_context(context, ToyGeometry)
        self.assertIsNot(clone.model, context.model)
        self.assertEqual(clone.model.cache_tensor.dtype, torch.float64)
        engine = DiagnosticEngine(clone, 4)
        native32 = (context.model.theta.detach().clone(),)
        _, q, p, gate = extension.curvature_gate(engine, self.x, self.y, native32)
        self.assertTrue(gate['first_order']['accepted'])
        self.assertTrue(gate['accepted'])
        torch.testing.assert_close(q[0], native32[0].double(), rtol=0., atol=0.)
        torch.testing.assert_close(p[0], native32[0].double(), rtol=0., atol=0.)
        engine.close()
        self.assertTrue(all(original.check().values()))

    def test_adjacent_rule_rejects_separated_passes_and_nonfinite(self):
        rows = [dict(degenerate=False, relative_error=.01),
                dict(degenerate=False, relative_error=.2),
                dict(degenerate=False, relative_error=.01)]
        self.assertFalse(extension.two_adjacent(rows, .08))
        rows[1]['relative_error'] = .02
        self.assertTrue(extension.two_adjacent(rows, .08))
        rows[1]['relative_error'] = float('nan')
        self.assertFalse(extension.two_adjacent(rows, .08))


if __name__ == '__main__':
    unittest.main(verbosity=2)
