"""CPU correctness tests for the historical/future response algorithm."""
import json
import math
import random
import unittest

import numpy as np
import torch

from history_step import HistoryResponseStep, _digest
from state_transaction import State, rng_get, rng_set, rng_equal


class VectorModel(torch.nn.Module):
    def __init__(self, values=(.37, -.41, .53)):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, x):
        return self.w


class PlainGeometry:
    def __init__(self, model):
        self.params = list(model.parameters())

    def capture(self):
        return tuple(p.detach().clone() for p in self.params)

    def actual_delta(self, before):
        squared = sum((p.detach().double() - b.double()).square().sum() for p, b in zip(self.params, before))
        return {'parameter_norm': math.sqrt(float(squared)), 'joint_norm': math.sqrt(float(squared))}


def make(model=None, direction='history_fused'):
    model = model or VectorModel()
    params = list(model.parameters())
    stepper = HistoryResponseStep(model, params, PlainGeometry(model),
                                 {'direction': direction, 'history_weight': .5 if direction == 'history_fused' else 0.,
                                  'eps': 1e-12, 'backend': 'native'})
    optimizer = torch.optim.SGD(params, lr=.07)
    return model, stepper, optimizer


Q = torch.tensor([[3., .7, -.2], [.7, 2., .3], [-.2, .3, 1.]], dtype=torch.float32)
B = torch.tensor([-.2, .1, .4], dtype=torch.float32)


def quadratic(model):
    def closure(x, y):
        w = model.w
        return .5 * torch.dot(w, Q @ w) + torch.dot(B, w), w.unsqueeze(0)
    return closure


class HistoryStepCPU(unittest.TestCase):
    def test_spd_quadratic_two_steps_and_raw_cache(self):
        # Positive definite, genuinely coupled coordinates; the second LR differs.
        self.assertTrue(bool((torch.linalg.eigvalsh(Q) > 0).all()))
        model, stepper, opt = make()
        previous_g = previous_delta = None
        for index, eta in enumerate((.07, .023)):
            opt.param_groups[0]['lr'] = eta
            before = model.w.detach().clone()
            out = stepper.compute(None, None, quadratic(model), eta)
            ctx = out['context']
            g, gf = ctx['g'][0], ctx['gf'][0]
            torch.testing.assert_close(g, Q @ before + B)
            torch.testing.assert_close(gf, Q @ ctx['future'][0] + B)
            if index == 0:
                expected = 2 * g - gf
                self.assertFalse(out['record']['history_used'])
            else:
                ell_b = float(previous_delta.double().norm())
                ell_f = eta * float(g.double().norm())
                expected = g + .5 * (g - gf) + ell_f / (2 * ell_b) * (previous_g - g)
                self.assertTrue(out['record']['history_used'])
                self.assertEqual(out['record']['history_cache_raw_gradient_sha256'], _digest((previous_g,)))
                self.assertEqual(out['record']['history_cache_actual_displacement_sha256'], _digest((previous_delta,)))
                self.assertAlmostEqual(out['record']['ell_b'], ell_b, places=14)
            torch.testing.assert_close(out['U'][0], expected, rtol=1e-6, atol=1e-7)
            self.assertTrue(torch.equal(model.w, before))
            update = stepper.apply_update(eta, optimizer=opt)
            self.assertTrue(update['natural_sgd_exact'])
            torch.testing.assert_close(model.w, before.clone().add_(out['U'][0], alpha=-eta), rtol=0, atol=0)
            previous_g = g.clone()
            previous_delta = model.w.detach() - before
            self.assertTrue(torch.equal(stepper._history['raw_gradient'][0], previous_g))
            self.assertFalse(torch.equal(stepper._history['raw_gradient'][0], out['U'][0]))
            self.assertTrue(torch.equal(stepper._history['displacement'][0], previous_delta))
            # Actual subtraction contains FP32 rounding; caching -eta*U is wrong.
            self.assertFalse(torch.equal(previous_delta, -eta * out['U'][0]))
            self.assertEqual(update['sgd_updates'], 1)
            json.dumps(out['record'], allow_nan=False)

    def test_first_step_equals_future_only_exactly(self):
        model_a, step_a, opt_a = make()
        model_b, step_b, opt_b = make(direction='future_only')
        out_a = step_a.compute(None, None, quadratic(model_a), .07)
        out_b = step_b.compute(None, None, quadratic(model_b), .07)
        self.assertTrue(torch.equal(out_a['U'][0], out_b['U'][0]))
        step_a.apply_update(.07, opt_a)
        step_b.apply_update(.07, opt_b)
        self.assertTrue(torch.equal(model_a.w, model_b.w))
        self.assertIsNone(step_b._history)

    def test_linear_changed_minibatch_not_a_hessian(self):
        model, stepper, opt = make()
        coefficients = (torch.tensor([.7, -.3, .1]), torch.tensor([-.2, .5, .4]))
        for index, coef in enumerate(coefficients):
            def closure(x, y):
                return torch.dot(model.w, coef), model.w.unsqueeze(0)
            out = stepper.compute(None, None, closure, .07)
            self.assertTrue(torch.equal(out['context']['g'][0], out['context']['gf'][0]))
            self.assertEqual(out['record']['gradient_norms']['future_response'], 0.)
            if index:
                self.assertTrue(out['record']['history_used'])
                self.assertGreater(out['record']['gradient_norms']['history_response'], 0.)
                self.assertGreater(out['record']['history_component_relative_norm'], 0.)
                self.assertFalse(torch.equal(out['U'][0], coef))
                # Loss is linear (Hessian zero), but minibatch turnover makes v nonzero.
                self.assertIsNone(out['record']['response_cosine'])
                self.assertIsNotNone(out['record']['u_v_cosine'])
            stepper.apply_update(.07, opt)

    def test_full_transaction_and_rng_replay(self):
        class StochasticModel(VectorModel):
            def __init__(self):
                super().__init__()
                self.register_buffer('forward_count', torch.tensor(7))
                self.forward_cache = {'value': torch.tensor([9.])}
                self.sub = torch.nn.Dropout(.5)
            def forward(self, x):
                self.forward_count.add_(1)
                self.forward_cache['value'].add_(1)
                self.new_cache = torch.randn(2)
                self.training = not self.training
                random.random(); np.random.rand()
                return self.sub(self.w)
        torch.manual_seed(117); random.seed(117); np.random.seed(117)
        model = StochasticModel()
        model.w.grad = torch.tensor([.2, .3, .4])
        model, stepper, opt = make(model)
        initial = State(model)
        seen = []
        def closure(x, y):
            seen.append(rng_get())
            z = model(x)
            return z.square().sum(), z.unsqueeze(0)
        # Independently establish the one base-forward random-state advance.
        closure(None, None)
        expected_R1 = rng_get()
        initial.restore(); initial.check(); seen.clear()
        out = stepper.compute(None, None, closure, .07)
        self.assertEqual(len(seen), 2)
        self.assertTrue(rng_equal(seen[0], seen[1]))
        self.assertTrue(rng_equal(rng_get(), expected_R1))
        self.assertTrue(torch.equal(model.forward_count, torch.tensor(7)))
        self.assertTrue(torch.equal(model.forward_cache['value'], torch.tensor([9.])))
        self.assertFalse(hasattr(model, 'new_cache'))
        self.assertTrue(torch.equal(model.w.grad, torch.tensor([.2, .3, .4])))
        self.assertTrue(model.training)
        self.assertTrue(model.sub.training)
        self.assertTrue(all(out['record']['state_restore_checks'].values()))
        self.assertTrue(rng_equal(out['context']['rng0'], initial.rng))
        self.assertTrue(rng_equal(out['context']['rng1'], expected_R1))
        stepper.apply_update(.07, opt)

    def test_zero_gradient_and_unrepresentable_point_fallback(self):
        model, stepper, opt = make()
        def zero_closure(x, y):
            return (model.w * 0.).sum(), model.w.unsqueeze(0)
        out = stepper.compute(None, None, zero_closure, .07)
        self.assertEqual(out['record']['gradient_calls'], 1)
        self.assertTrue(out['record']['fallback'])
        self.assertEqual(out['record']['fallback_count'], 1)
        self.assertIsNone(out['context']['gf'])
        before = model.w.detach().clone()
        stepper.apply_update(.07, opt)
        self.assertTrue(torch.equal(model.w, before))
        self.assertIsNone(stepper._history)
        # A nonzero mathematical displacement may still round to the base point.
        huge = VectorModel(values=(1e10, 1e10, 1e10))
        huge, huge_step, huge_opt = make(huge)
        def linear(x, y):
            return huge.w.sum(), huge.w.unsqueeze(0)
        out = huge_step.compute(None, None, linear, .07)
        self.assertGreater(out['record']['ell_f'], huge_step.eps)
        self.assertEqual(out['record']['gradient_calls'], 1)
        self.assertEqual(out['record']['fallback_reason'], 'predicted_point_bitwise_indistinguishable_in_FP32')
        update = huge_step.apply_update(.07, huge_opt)
        self.assertFalse(update['history_cache_valid_after_update'])
        self.assertEqual(update['actual_parameter_step_norm'], 0.)

    def test_optimizer_constraints_and_once_only(self):
        for kwargs in ({'momentum': .9}, {'weight_decay': .1}, {'maximize': True}, {'foreach': True}):
            model, stepper, _ = make()
            out = stepper.compute(None, None, quadratic(model), .07)
            before = model.w.detach().clone()
            wrong = torch.optim.SGD(model.parameters(), lr=.07, **kwargs)
            with self.assertRaises(ValueError):
                stepper.apply_update(.07, wrong)
            self.assertTrue(torch.equal(model.w, before))
        model, stepper, opt = make()
        stepper.compute(None, None, quadratic(model), .07)
        with self.assertRaises(RuntimeError):
            stepper.compute(None, None, quadratic(model), .07)
        with self.assertRaises(ValueError):
            stepper.apply_update(.08, opt)
        opt.state[model.w]['momentum_buffer'] = torch.zeros_like(model.w)
        with self.assertRaises(ValueError):
            stepper.apply_update(.07, opt)
        opt.state.clear()
        stepper.apply_update(.07, opt)
        with self.assertRaises(RuntimeError):
            stepper.apply_update(.07, opt)

    def test_reset_and_task_layout_and_continuity(self):
        model, stepper, opt = make()
        stepper.compute(None, None, quadratic(model), .07)
        stepper.apply_update(.07, opt)
        stepper.reset_history('task_boundary')
        out = stepper.compute(None, None, quadratic(model), .07)
        self.assertFalse(out['record']['history_used'])
        stepper.apply_update(.07, opt)
        with torch.no_grad():
            model.w.add_(.1)
        out = stepper.compute(None, None, quadratic(model), .07)
        self.assertFalse(out['record']['history_used'])
        self.assertEqual(out['record']['history_reason'], 'parameter_continuity_changed')
        stepper.apply_update(.07, opt)
        fresh = HistoryResponseStep(model, list(model.parameters()), PlainGeometry(model),
                                    {'direction': 'history_fused', 'history_weight': .5})
        self.assertIsNone(fresh._history)
        model.w = torch.nn.Parameter(torch.zeros(4))
        with self.assertRaises(ValueError):
            stepper.compute(None, None, quadratic(model), .07)
        self.assertIsNone(stepper._history)

    def test_portable_checkpoint_and_identical_continuation(self):
        import io
        model, stepper, opt = make()
        out = stepper.compute(None, None, quadratic(model), .07)
        with self.assertRaises(RuntimeError):
            stepper.state_dict()
        stepper.apply_update(.07, opt)
        saved = stepper.state_dict()
        self.assertTrue(saved['history']['valid'])
        self.assertEqual(saved['history']['layout'][0]['name'], 'w')
        self.assertEqual(saved['history']['raw_gradient'][0].device.type, 'cpu')
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        reloaded = torch.load(stream, weights_only=True)
        model2, step2, opt2 = make()
        with self.assertRaises(ValueError):
            step2.load_state_dict(reloaded)
        model2.load_state_dict(model.state_dict())
        step2.load_state_dict(reloaded)
        self.assertEqual(step2.real_updates, stepper.real_updates)
        self.assertEqual(step2._history['layout'], step2._layout)
        self.assertNotEqual(step2._history['layout'], stepper._layout)
        eta = .023
        opt.param_groups[0]['lr'] = opt2.param_groups[0]['lr'] = eta
        a = stepper.compute(None, None, quadratic(model), eta)
        b = step2.compute(None, None, quadratic(model2), eta)
        self.assertTrue(torch.equal(a['U'][0], b['U'][0]))
        self.assertEqual(a['record']['history_cache_raw_gradient_sha256'], b['record']['history_cache_raw_gradient_sha256'])
        expected_cos = float(torch.dot(a['context']['g'][0].double(), -stepper._history['displacement'][0].double()) /
                             (a['context']['g'][0].double().norm() * stepper._history['displacement'][0].double().norm()))
        self.assertAlmostEqual(a['record']['u_v_cosine'], expected_cos, places=14)
        self.assertEqual(a['record']['ordinary_gradient_calls'], 2)
        stepper.apply_update(eta, opt)
        step2.apply_update(eta, opt2)
        self.assertTrue(torch.equal(model.w, model2.w))
        self.assertEqual(stepper.state_dict()['history']['displacement_sha256'], step2.state_dict()['history']['displacement_sha256'])

    def test_observational_post_diagnostics_preserve_history_and_R1(self):
        model, stepper, opt = make()
        def closure(x, y):
            random.random(); np.random.rand(); torch.rand(2)
            return quadratic(model)(x, y)
        stepper.compute(None, None, closure, .07)
        stepper.apply_update(.07, opt)
        prior_hashes = (stepper._history['raw_gradient_sha256'], stepper._history['displacement_sha256'])
        out = stepper.compute(None, None, closure, .07)
        post = State(model)
        try:
            rng_set(out['context']['rng0'])
            with torch.no_grad():
                model.w.copy_(out['context']['future'][0])
            closure(None, None)
            model.eval()
        finally:
            post.restore()
        self.assertTrue(all(post.check().values()))
        self.assertTrue(rng_equal(rng_get(), out['context']['rng1']))
        self.assertEqual(prior_hashes, (stepper._history['raw_gradient_sha256'], stepper._history['displacement_sha256']))
        stepper.apply_update(.07, opt)

    def test_exception_restores_all_entry_state(self):
        model, stepper, opt = make()
        before = State(model)
        def bad(x, y):
            random.random(); torch.rand(5)
            raise RuntimeError('intentional closure failure')
        with self.assertRaises(RuntimeError):
            stepper.compute(None, None, bad, .07)
        self.assertTrue(all(before.check().values()))
        self.assertIsNone(stepper._pending)
        out = stepper.compute(None, None, quadratic(model), .07)
        self.assertEqual(out['record']['gradient_calls'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
