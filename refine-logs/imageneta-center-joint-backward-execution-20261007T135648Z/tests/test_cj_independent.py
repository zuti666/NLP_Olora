"""Independent nonlinear analytic, transactional, and frozen-regression gates.

The oracle reconstructs locations and gradients without importing engine
normalization or combination helpers. It is a CPU integrity gate, not evidence
of model accuracy. Run with CUDA_VISIBLE_DEVICES='' and the pinned Python.
"""
import copy
import importlib.util
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'trainer'))
from audit_core import rng_get, rng_set, rng_equal
from direction_step import FourProbeStep, direction_name
from effective_step import EffectiveStep
from state_transaction import State

OLD = HERE.parent / 'imageneta-second-probe-sign-deterministic-20261007T071400Z' / 'trainer' / 'direction_step.py'
spec = importlib.util.spec_from_file_location('frozen_old_second_probe', OLD)
old_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_module)


def flat(values):
    return torch.cat([value.reshape(-1) for value in values])


class AnalyticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([[-.31, .12], [.25, -.09]], dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.tensor([[.42, -.28], [.17, .53]], dtype=torch.float64))
        self.fc = torch.nn.Linear(2, 2, bias=True, dtype=torch.float64)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[.15, -.36], [.46, .07]], dtype=torch.float64))
            self.fc.bias.copy_(torch.tensor([-.22, .39], dtype=torch.float64))
        self.frozen = torch.nn.Parameter(torch.ones(2, dtype=torch.float64), requires_grad=False)
        self.register_buffer('counter', torch.tensor(7., dtype=torch.float64))
        self.dropout = torch.nn.Dropout(.25)
        self.forward_cache = {'entry': torch.tensor([2.], dtype=torch.float64)}
        self._register_delta_hook = False
        self.delta_w_q_new_grad = None
        self.external_delta_w_q = None


def case(model=None, quadratic=False, zero=False):
    model = AnalyticModel() if model is None else model
    params = tuple(p for p in model.parameters() if p.requires_grad)
    geometry = EffectiveStep([(model.a, model.b)], tuple(model.fc.parameters()), params)
    n = sum(p.numel() for p in params)
    axis = torch.linspace(-.4, .6, n, dtype=torch.float64)
    H = torch.diag(torch.linspace(.7, 2.1, n, dtype=torch.float64)) + .31 * torch.outer(axis, axis)
    linear = torch.linspace(.19, -.23, n, dtype=torch.float64)
    calls = []

    def loss(theta):
        if zero:
            return theta.sum() * 0.
        out = .5 * theta @ H @ theta - linear @ theta
        if not quadratic:
            out = out + .08 * theta.pow(4).sum() + .07 * (1 - theta.cos()).sum() + .017 * (axis @ theta).pow(4)
        return out

    def derivative(theta):
        if zero:
            return torch.zeros_like(theta)
        out = H @ theta - linear
        if not quadratic:
            out = out + .32 * theta.pow(3) + .07 * theta.sin() + .068 * axis * (axis @ theta).pow(3)
        return out

    def closure(_inputs, _targets):
        theta = flat(params)
        calls.append(theta.detach().clone())
        return loss(theta), theta[:2].reshape(1, 2)

    return model, params, geometry, derivative, closure, calls


def config(first, second, coefficient=1, rho=.05, radius=1.):
    return dict(first_location_estimator=first, second_residual_estimator=second,
                residual_coefficient=coefficient, update_base='g0',
                direction=direction_name(first, second, coefficient, 'g0'),
                branch='P' if first == 'forward' else 'C', rho=rho, radius=radius,
                backend='native', eps=1e-12)


def analytic_oracle(theta, gradient, first, second, coefficient, rho=.05, radius=1.):
    def unit(vector):
        # Deliberately independent from direction_step.normalized.
        return vector / (torch.linalg.vector_norm(vector) + 1e-12)
    g0 = gradient(theta)
    point1 = theta + (1 if first == 'forward' else -1) * rho * unit(g0)
    g1 = gradient(point1)
    q = g1 - g0 if first == 'forward' else g0 - g1
    point2 = theta + radius * unit(q)
    g2 = gradient(point2)
    plus = point2 + rho * unit(g2)
    minus = point2 - rho * unit(g2)
    gplus, gminus = gradient(plus), gradient(minus)
    if second == 'centered':
        residual = .5 * (gplus - gminus)
        points = [theta, point1, point2, plus, minus]
    elif second == 'forward':
        residual = gplus - g2
        points = [theta, point1, point2, plus]
    else:
        residual = g2 - gminus
        points = [theta, point1, point2, minus]
    return dict(points=points, U=g0+coefficient*residual, residual=residual,
                q=q, even=.5*(gplus+gminus-2*g2), g0=g0)


class IndependentAcceptance(unittest.TestCase):
    def assert_close(self, actual, expected, atol=2e-11, rtol=2e-11):
        actual = flat(actual) if isinstance(actual, (tuple, list)) else actual
        expected = flat(expected) if isinstance(expected, (tuple, list)) else expected
        self.assertTrue(torch.allclose(actual, expected, atol=atol, rtol=rtol),
                        'maximum error ' + str(float((actual-expected).abs().max())))

    def test_nonlinear_all_four_oracles_budget_and_actual_sgd(self):
        for first, second, coefficient in [('forward', 'centered', 1), ('forward', 'centered', -1),
                                           ('backward_corrected', 'forward', 1), ('backward_corrected', 'backward_corrected', 1)]:
            with self.subTest(first=first, second=second, coefficient=coefficient):
                model, params, geometry, gradient, closure, calls = case()
                theta = flat(params).detach().clone()
                reference = analytic_oracle(theta, gradient, first, second, coefficient)
                engine = FourProbeStep(model, params, geometry, config(first, second, coefficient))
                optimizer = torch.optim.SGD(params, lr=.04, momentum=0, weight_decay=0)
                with patch('torch.autograd.grad', wraps=torch.autograd.grad) as probes:
                    result = engine.compute(None, None, closure, .04)
                expected_budget = 5 if second == 'centered' else 4
                self.assertEqual(probes.call_count, expected_budget)
                self.assertEqual(result['record']['ordinary_gradient_calls'], expected_budget)
                self.assertTrue(all(call.kwargs['create_graph'] is False for call in probes.call_args_list))
                self.assertEqual(len(calls), expected_budget)
                for observed, expected in zip(calls, reference['points']):
                    self.assert_close(observed, expected)
                self.assert_close(result['U'], reference['U'])
                self.assert_close(flat(params), theta)
                self.assertGreater(float(reference['even'].norm()), 1e-5)
                if second == 'centered':
                    # Reject both hidden 1/rho scaling and retained even response.
                    self.assertGreater(float((reference['U'] - (reference['g0']+coefficient*reference['residual']/.05)).norm()), .01)
                    self.assertGreater(float((reference['U'] - (reference['g0']+coefficient*(reference['residual']+reference['even']))).norm()), 1e-5)
                if first == second == 'backward_corrected':
                    # The exact call list proves no ascent-side local reading.
                    self.assertLess(float((calls[1]-theta) @ gradient(theta)), 0.)
                    self.assertLess(float((calls[3]-calls[2]) @ gradient(calls[2])), 0.)
                report = engine.apply_update(optimizer=optimizer)
                self.assertTrue(report['natural_sgd_exact'])
                self.assert_close(flat(params), theta-.04*reference['U'])
                self.assertEqual(len(optimizer.state), 0)

    def test_quadratic_first_estimators_and_positive_responses_coincide(self):
        updates, qs, outer_points = [], [], []
        for first, second in [('forward', 'forward'), ('forward', 'backward_corrected'),
                              ('backward_corrected', 'forward'), ('backward_corrected', 'backward_corrected'),
                              ('forward', 'centered')]:
            model, params, geometry, gradient, closure, calls = case(quadratic=True)
            theta = flat(params).detach().clone()
            reference = analytic_oracle(theta, gradient, first, second, 1)
            result = FourProbeStep(model, params, geometry, config(first, second)).compute(None, None, closure, .04)
            self.assert_close(result['U'], reference['U'])
            self.assertLess(float(reference['even'].norm()), 1e-12)
            updates.append(flat(result['U'])); qs.append(reference['q']); outer_points.append(calls[2])
        for values in (updates, qs, outer_points):
            for value in values[1:]:
                self.assert_close(value, values[0])

    def test_zero_rho_and_zero_gradient_are_finite_degenerations(self):
        for rho, zero in [(0., False), (.05, True)]:
            for first, second, coefficient in [('forward', 'centered', 1), ('forward', 'centered', -1),
                                               ('backward_corrected', 'forward', 1), ('backward_corrected', 'backward_corrected', 1)]:
                model, params, geometry, gradient, closure, calls = case(zero=zero)
                theta = flat(params).detach().clone()
                result = FourProbeStep(model, params, geometry, config(first, second, coefficient, rho=rho)).compute(None, None, closure, .04)
                self.assertTrue(torch.isfinite(flat(result['U'])).all())
                self.assert_close(result['U'], gradient(theta))
                for point in calls:
                    self.assert_close(point, theta)

    def stochastic_closure(self, model, params, draws):
        def closure(_inputs, _targets):
            theta = flat(params)
            mask = model.dropout(torch.ones_like(theta))
            draws.append((random.random(), np.random.rand(), torch.rand(3), mask.detach().clone()))
            model.counter.add_(1)
            model.dropout.training = False
            model.forward_cache['entry'].add_(11)
            model.forward_cache['new'] = torch.ones(2)
            model.created_cache = [torch.ones(2)]
            model.delta_w_q_new_grad = torch.ones(3)
            model.external_delta_w_q = torch.ones(2)
            model._register_delta_hook = True
            value = theta * mask
            return .5*value.square().sum()+.08*value.pow(4).sum(), theta[:2].reshape(1,2)
        return closure

    def test_rng_replay_and_complete_nonupdate_state_restore(self):
        for first, second in [('forward', 'centered'), ('backward_corrected', 'backward_corrected')]:
            model, params, geometry, *_ = case()
            for index, param in enumerate(params):
                param.grad = torch.full_like(param, index+.5)
            state = State(model)
            reference_draws = []
            self.stochastic_closure(model, params, reference_draws)(None, None)
            expected_rng1 = rng_get()
            state.restore()
            draws = []
            result = FourProbeStep(model, params, geometry, config(first, second)).compute(
                None, None, self.stochastic_closure(model, params, draws), .04)
            for value in draws[1:]:
                self.assertEqual(value[:2], draws[0][:2])
                self.assertTrue(torch.equal(value[2], draws[0][2]))
                self.assertTrue(torch.equal(value[3], draws[0][3]))
            self.assertTrue(rng_equal(rng_get(), expected_rng1))
            for param, original_grad in state.grads:
                param.grad = None if original_grad is None else original_grad.clone()
            rng_set(state.rng)
            self.assertTrue(all(state.check().values()))
            self.assertFalse(hasattr(model, 'created_cache'))

    def test_each_probe_exception_restores_original_full_state(self):
        for first, second, budget in [('forward', 'centered', 5), ('backward_corrected', 'forward', 4),
                                      ('backward_corrected', 'backward_corrected', 4)]:
            for fail_at in range(1, budget+1):
                model, params, geometry, *_ = case()
                for param in params:
                    param.grad = torch.full_like(param, 7.)
                state, draws = State(model), []
                stochastic = self.stochastic_closure(model, params, draws)
                def failure(inputs, targets):
                    out = stochastic(inputs, targets)
                    if len(draws) == fail_at:
                        raise RuntimeError('independent intentional failure')
                    return out
                engine = FourProbeStep(model, params, geometry, config(first, second))
                with self.assertRaisesRegex(RuntimeError, 'independent intentional'):
                    engine.compute(None, None, failure, .04)
                self.assertTrue(all(state.check().values()))

    def test_ff_fb_frozen_old_module_regression(self):
        for second, sign in [('forward', 1), ('backward_corrected', -1)]:
            model = AnalyticModel()
            old_model = copy.deepcopy(model)
            new_model, new_params, new_geometry, _, new_closure, new_calls = case(model)
            old_model, old_params, old_geometry, _, old_closure, old_calls = case(old_model)
            entry_rng = rng_get()
            old_engine = old_module.FourProbeStep(old_model, old_params, old_geometry,
                dict(branch='P', direction=old_module.direction_name('g0', sign, sign),
                     rho=.05, radius=1., first_probe_sign=1, second_probe_sign=sign,
                     second_difference_sign=sign, update_base='g0', second_difference_weight=1., backend='native'))
            old_result = old_engine.compute(None, None, old_closure, .04)
            old_rng = rng_get()
            rng_set(entry_rng)
            new_engine = FourProbeStep(new_model, new_params, new_geometry, config('forward', second))
            new_result = new_engine.compute(None, None, new_closure, .04)
            self.assertTrue(rng_equal(rng_get(), old_rng))
            self.assertTrue(torch.equal(flat(new_result['U']), flat(old_result['U'])))
            for new_point, old_point in zip(new_calls, old_calls):
                self.assertTrue(torch.equal(new_point, old_point))
            for key in ('g0', 'g1', 'g2', 'g3'):
                self.assertTrue(torch.equal(flat(new_result['context']['grads'][key]), flat(old_result['context']['grads'][key])))
            old_engine.apply_update(optimizer=torch.optim.SGD(old_params, lr=.04))
            new_engine.apply_update(optimizer=torch.optim.SGD(new_params, lr=.04))
            self.assertTrue(torch.equal(flat(new_params), flat(old_params)))

    def test_optimizer_exception_restores_full_preprobe_entry_state(self):
        model, params, geometry, _, closure, _ = case()
        for index, param in enumerate(params):
            param.grad = torch.full_like(param, index+.75)
        entry = State(model)
        engine = FourProbeStep(model, params, geometry, config('forward', 'centered'))
        optimizer = torch.optim.SGD(params, lr=.04)
        native_step = optimizer.step
        engine.compute(None, None, closure, .04)
        def fail_after_step(*args, **kwargs):
            native_step(*args, **kwargs)
            model.counter.add_(9)
            model.created_cache = [torch.ones(1)]
            raise RuntimeError('intentional optimizer exception')
        with patch.object(optimizer, 'step', side_effect=fail_after_step):
            with self.assertRaisesRegex(RuntimeError, 'intentional optimizer exception'):
                engine.apply_update(optimizer=optimizer)
        self.assertTrue(all(entry.check().values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
