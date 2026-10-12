"""CPU mathematical, state-transaction, cache-source and checkpoint tests.

These tests use coupled quadratics and changing batches, not ImageNet scores.
Run with CUDA_VISIBLE_DEVICES='' in the registered existing environment.
"""
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import unittest

import numpy as np
import torch

from response_step import ResponseStep, METHODS, _digest, _norm
from state_transaction import State, rng_equal, rng_get, rng_set


class VectorModel(torch.nn.Module):
    def __init__(self, values=(.37, -.41, .53)):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, x):
        return self.w


class PlainGeometry:
    def __init__(self, model):
        self.params = [p for p in model.parameters() if p.requires_grad]

    def capture(self):
        return tuple(p.detach().clone() for p in self.params)

    def actual_delta(self, before):
        n = _norm(tuple(p.detach() - b for p, b in zip(self.params, before)))
        return {'parameter_norm': n, 'joint_norm': n}


Q = torch.tensor([[3., .7, -.2], [.7, 2., .3], [-.2, .3, 1.]], dtype=torch.float32)
B = torch.tensor([-.2, .1, .4], dtype=torch.float32)


def quadratic(model, b=B):
    def closure(x, y):
        w = model.w
        return .5 * torch.dot(w, Q @ w) + torch.dot(b, w), w.unsqueeze(0)
    return closure


def make(method='history_fusion', model=None, eta=.07):
    model = model or VectorModel()
    params = [p for p in model.parameters() if p.requires_grad]
    config = {'method': method, 'history_weight': .5 if method.startswith('history_') else 0.,
              'eps': 1e-12, 'backend': 'native', 'radius_mode': 'sgd_step', 'radius_scale': 1}
    stepper = ResponseStep(model, params, PlainGeometry(model), config)
    return model, stepper, torch.optim.SGD(params, lr=eta)


def algebra_gradient(w, b):
    # Independent analytic gradient; autograd may associate the scalar loss's
    # matrix products differently in FP32, so formula oracle uses close checks.
    return Q @ w + b


class ResponseCPU(unittest.TestCase):
    def test_all_four_multistep_coupled_quadratic_oracle(self):
        self.assertTrue(bool((torch.linalg.eigvalsh(Q) > 0).all()))
        for method in METHODS:
            with self.subTest(method=method):
                model, stepper, opt = make(method)
                previous_vector = previous_delta = None
                for index, eta in enumerate((.07, .023, .051, .011)):
                    opt.param_groups[0]['lr'] = eta
                    b = B + torch.tensor([.03, -.04, .02]) * index
                    base = model.w.detach().clone()
                    out = stepper.compute(None, None, quadratic(model, b), eta)
                    g0 = out['context']['g0'][0]
                    torch.testing.assert_close(g0, algebra_gradient(base, b))
                    h = eta * float(g0.double().norm())
                    self.assertAlmostEqual(out['record']['h'], h, places=14)
                    if method == 'sgd':
                        expected = g0
                        self.assertIsNone(out['context']['g1'])
                        self.assertIsNone(out['context']['g2'])
                        self.assertEqual(out['record']['ordinary_gradient_calls'], 1)
                    else:
                        future = base.clone().add_(g0, alpha=-eta)
                        self.assertTrue(torch.equal(out['context']['future'][0], future))
                        g1 = out['context']['g1'][0]
                        torch.testing.assert_close(g1, algebra_gradient(future, b))
                        a = g0 - g1
                        if previous_vector is not None and method.startswith('history_'):
                            ell = float(previous_delta.double().norm())
                            response = .5 * a + h / (2. * ell) * (previous_vector - g0)
                            self.assertTrue(out['record']['history_used'])
                            self.assertEqual(out['record']['history_incoming_vector_sha256'], _digest((previous_vector,)))
                            self.assertEqual(out['record']['history_cache_actual_displacement_sha256'], _digest((previous_delta,)))
                            self.assertAlmostEqual(out['record']['ell'], ell, places=14)
                        else:
                            response = a
                            self.assertFalse(out['record']['history_used'])
                        if method == 'history_fusion':
                            expected = (g0 + .5 * a + h / (2. * float(previous_delta.double().norm())) * (previous_vector - g0)
                                        if previous_vector is not None else 2. * g0 - g1)
                            self.assertEqual(out['record']['ordinary_gradient_calls'], 2)
                        else:
                            point = base.clone().add_(response, alpha=h / float(response.double().norm()))
                            torch.testing.assert_close(out['context']['relocation'][0], point, rtol=1e-6, atol=1e-7)
                            expected = algebra_gradient(point, b)
                            self.assertEqual(out['record']['ordinary_gradient_calls'], 3)
                    torch.testing.assert_close(out['U'][0], expected, rtol=2e-6, atol=2e-7)
                    self.assertTrue(torch.equal(model.w, base))
                    update = stepper.apply_update(eta, opt)
                    self.assertTrue(torch.equal(model.w, base.clone().add_(out['U'][0], alpha=-eta)))
                    self.assertEqual(update['sgd_updates'], 1)
                    previous_delta = model.w.detach() - base
                    if method.startswith('history_'):
                        previous_vector = g0.clone() if method == 'history_fusion' else out['U'][0].clone()
                        self.assertTrue(torch.equal(stepper._history['vector'][0], previous_vector))
                    else:
                        self.assertIsNone(stepper._history)
                    json.dumps(out['record'], allow_nan=False)

    def test_history_g2_first_update_bitwise_equals_current_g2(self):
        model_a, a, oa = make('current_g2')
        model_b, b, ob = make('history_g2')
        xa = a.compute(None, None, quadratic(model_a), .07)
        xb = b.compute(None, None, quadratic(model_b), .07)
        for key in ('g0', 'g1', 'g2', 'U', 'relocation'):
            self.assertTrue(torch.equal(xa['context'][key][0], xb['context'][key][0]))
        a.apply_update(.07, oa); b.apply_update(.07, ob)
        self.assertTrue(torch.equal(model_a.w, model_b.w))
        self.assertIsNone(a._history)
        self.assertEqual(b._history['cache_source'], 'final')

    def test_final_cache_full_amplitude_not_base_unit_or_step(self):
        model, stepper, opt = make('history_g2')
        base = model.w.detach().clone()
        out = stepper.compute(None, None, quadratic(model), .07)
        stepper.apply_update(.07, opt)
        v = stepper._history['vector'][0]
        self.assertTrue(torch.equal(v, out['U'][0]))
        self.assertTrue(torch.equal(v, out['context']['g2'][0]))
        self.assertFalse(torch.equal(v, out['context']['g0'][0]))
        self.assertFalse(torch.equal(v, out['U'][0] / out['U'][0].norm()))
        self.assertFalse(torch.equal(v, - .07 * out['U'][0]))
        self.assertTrue(torch.equal(stepper._history['displacement'][0], model.w.detach() - base))
        self.assertEqual(out['record']['history_cached_vector_sha256'], out['record']['final_gradient_sha256'])
        self.assertNotAlmostEqual(float(v.norm()), 1., places=4)

    def test_actual_displacement_not_ideal_fp32_step(self):
        for method in ('history_fusion', 'history_g2'):
            model, stepper, opt = make(method)
            base = model.w.detach().clone()
            out = stepper.compute(None, None, quadratic(model), .07)
            stepper.apply_update(.07, opt)
            actual = model.w.detach() - base
            self.assertTrue(torch.equal(stepper._history['displacement'][0], actual))
            self.assertFalse(torch.equal(actual, -.07 * out['U'][0]))
            self.assertAlmostEqual(stepper._history['length'], float(actual.double().norm()), places=14)

    def test_relocation_anchored_at_base_not_predictor(self):
        model, stepper, _ = make('current_g2')
        out = stepper.compute(None, None, quadratic(model), .07)
        c = out['context']
        a, h = c['response'][0], out['record']['h']
        right = c['base'][0].clone().add_(a, alpha=h / float(a.double().norm()))
        wrong = c['future'][0].clone().add_(a, alpha=h / float(a.double().norm()))
        self.assertTrue(torch.equal(c['relocation'][0], right))
        self.assertFalse(torch.equal(c['relocation'][0], wrong))
        self.assertEqual(out['record']['relocation_origin'], 'base')

    def test_prior_hrf_multistep_bitwise_equivalence(self):
        prior = Path(__file__).resolve().parents[2] / 'imageneta-history-response-execution-20261008T175403Z' / 'trainer/history_step.py'
        spec = importlib.util.spec_from_file_location('prior_history_step_exact_reference', prior)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        model, new, opt = make('history_fusion')
        old_model = VectorModel()
        old = module.HistoryResponseStep(old_model, list(old_model.parameters()), PlainGeometry(old_model),
                                        {'direction': 'history_fused', 'history_weight': .5, 'eps': 1e-12, 'backend': 'native'})
        old_opt = torch.optim.SGD(old_model.parameters(), lr=.07)
        for index, eta in enumerate((.07, .023, .041, .013, .055)):
            b = B + index * torch.tensor([.041, -.019, .026])
            opt.param_groups[0]['lr'] = old_opt.param_groups[0]['lr'] = eta
            x = new.compute(None, None, quadratic(model, b), eta)
            y = old.compute(None, None, quadratic(old_model, b), eta)
            for key in ('base', 'g', 'gf', 'U', 'future'):
                self.assertTrue(torch.equal(x['context'][key][0], y['context'][key][0]), key)
            self.assertEqual(x['record']['history_used'], y['record']['history_used'])
            self.assertEqual(x['record']['history_coefficient'], y['record']['history_coefficient'])
            ux = new.apply_update(eta, opt); uy = old.apply_update(eta, old_opt)
            self.assertTrue(torch.equal(model.w, old_model.w))
            self.assertEqual(ux['actual_step_displacement_sha256'], uy['actual_step_displacement_sha256'])
            self.assertTrue(torch.equal(new._history['vector'][0], old._history['raw_gradient'][0]))
            self.assertTrue(torch.equal(new._history['displacement'][0], old._history['displacement'][0]))

    def test_linear_batch_change_has_history_without_hessian(self):
        for method in ('history_fusion', 'history_g2'):
            model, stepper, opt = make(method)
            for index, coef in enumerate((torch.tensor([.7, -.3, .1]), torch.tensor([-.2, .5, .4]))):
                def closure(x, y):
                    return torch.dot(model.w, coef), model.w.unsqueeze(0)
                out = stepper.compute(None, None, closure, .07)
                self.assertTrue(torch.equal(out['context']['g0'][0], out['context']['g1'][0]))
                if index:
                    self.assertTrue(out['record']['history_used'])
                    self.assertGreater(out['record']['gradient_norms']['history_response'], 0.)
                    if method == 'history_g2':
                        self.assertEqual(out['record']['ordinary_gradient_calls'], 3)
                stepper.apply_update(.07, opt)

    def test_transaction_restores_all_and_same_R0_all_probes(self):
        class Stochastic(VectorModel):
            def __init__(self):
                super().__init__()
                self.register_buffer('forward_count', torch.tensor(7))
                self.forward_cache = {'value': torch.tensor([9.])}
                self.dropout = torch.nn.Dropout(.2)
            def forward(self, x):
                self.forward_count.add_(1)
                self.forward_cache['value'].add_(1)
                self.new_cache = torch.randn(2)
                random.random(); np.random.rand()
                return self.dropout(self.w)
        for method in METHODS:
            torch.manual_seed(117); random.seed(117); np.random.seed(117)
            model, stepper, opt = make(method, Stochastic())
            model.w.grad = torch.tensor([.2, .3, .4])
            initial = State(model)
            seen = []
            def closure(x, y):
                seen.append(rng_get())
                z = model(x)
                return z.square().sum(), z.unsqueeze(0)
            closure(None, None); expected_R1 = rng_get()
            initial.restore(); initial.check(); seen.clear()
            out = stepper.compute(None, None, closure, .07)
            self.assertEqual(len(seen), out['record']['ordinary_gradient_calls'])
            self.assertTrue(all(rng_equal(seen[0], item) for item in seen))
            self.assertTrue(rng_equal(rng_get(), expected_R1))
            self.assertTrue(torch.equal(model.forward_count, torch.tensor(7)))
            self.assertTrue(torch.equal(model.forward_cache['value'], torch.tensor([9.])))
            self.assertFalse(hasattr(model, 'new_cache'))
            self.assertTrue(torch.equal(model.w.grad, torch.tensor([.2, .3, .4])))
            self.assertTrue(all(out['record']['state_restore_checks'].values()))
            stepper.apply_update(.07, opt)

    def test_gradients_unused_are_zero_global_trainable_scope(self):
        class Split(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor([.5, -.4]))
                self.b = torch.nn.Parameter(torch.tensor([.7]))
                self.unused = torch.nn.Parameter(torch.tensor([.2]))
                self.frozen = torch.nn.Parameter(torch.tensor([9.]), requires_grad=False)
        model, stepper, opt = make('current_g2', Split())
        def closure(x, y):
            return 2. * model.a.square().sum() + .5 * model.b.square().sum(), model.a.unsqueeze(0)
        out = stepper.compute(None, None, closure, .07)
        self.assertTrue(torch.equal(out['context']['g0'][2], torch.zeros(1)))
        self.assertAlmostEqual(out['record']['h'], .07 * math.sqrt(4*.5**2*4 + 4*.4**2*4 + .7**2), places=6)
        frozen = model.frozen.detach().clone()
        stepper.apply_update(.07, opt)
        self.assertTrue(torch.equal(model.frozen, frozen))

    def test_zero_gradient_fallback_each_method(self):
        for method in METHODS:
            model, stepper, opt = make(method)
            def zero(x, y): return (model.w * 0.).sum(), model.w.unsqueeze(0)
            out = stepper.compute(None, None, zero, .07)
            self.assertEqual(out['record']['ordinary_gradient_calls'], 1)
            self.assertEqual(out['record']['fallback'], method != 'sgd')
            base = model.w.detach().clone()
            stepper.apply_update(.07, opt)
            self.assertTrue(torch.equal(model.w, base))
            self.assertIsNone(stepper._history)

    def test_inner_FP32_unrepresentable_fallback(self):
        for method in ('current_g2', 'history_fusion', 'history_g2'):
            model, stepper, opt = make(method, VectorModel((1e10, 1e10, 1e10)))
            def linear(x, y): return model.w.sum(), model.w.unsqueeze(0)
            out = stepper.compute(None, None, linear, .07)
            self.assertEqual(out['record']['ordinary_gradient_calls'], 1)
            self.assertEqual(out['record']['fallback_reason'], 'predicted_point_bitwise_indistinguishable_in_FP32')
            stepper.apply_update(.07, opt)
            self.assertIsNone(stepper._history)

    def test_outer_FP32_unrepresentable_fallback(self):
        model, stepper, opt = make('current_g2', VectorModel((1e10, .5)))
        def closure(x, y):
            return (model.w[0] - 1e10) * model.w[1] + model.w[1], model.w.unsqueeze(0)
        out = stepper.compute(None, None, closure, .07)
        self.assertTrue(out['record']['predicted_point_FP32_distinguishable'])
        self.assertFalse(out['record']['relocated_point_FP32_distinguishable'])
        self.assertEqual(out['record']['ordinary_gradient_calls'], 2)
        self.assertEqual(out['record']['fallback_reason'], 'relocated_point_bitwise_indistinguishable_in_FP32')
        self.assertTrue(torch.equal(out['U'][0], out['context']['g0'][0]))
        stepper.apply_update(.07, opt)

    def test_blended_zero_uses_current_response_before_g0(self):
        model, stepper, opt = make('history_g2', VectorModel((1.,)), eta=.5)
        delta = (torch.tensor([-.5]),)
        vector = (torch.tensor([.5]),)
        stepper._history = {'vector': vector, 'displacement': delta, 'length': .5,
                            'vector_sha256': _digest(vector), 'displacement_sha256': _digest(delta),
                            'after_sha256': _digest(stepper.params), 'layout': stepper._layout, 'cache_source': 'final'}
        def closure(x, y): return .5 * model.w.square().sum(), model.w.unsqueeze(0)
        out = stepper.compute(None, None, closure, .5)
        self.assertEqual(out['record']['response_fallback_reason'], 'blended_response_zero_try_current_response')
        self.assertIsNone(out['record']['fallback_reason'])
        self.assertEqual(out['record']['ordinary_gradient_calls'], 3)
        self.assertFalse(out['record']['history_used'])
        self.assertTrue(torch.equal(out['context']['relocation'][0], torch.tensor([1.5])))
        self.assertTrue(torch.equal(out['U'][0], torch.tensor([1.5])))
        stepper.apply_update(.5, opt)

    def test_fallback_caches_actually_submitted_g0(self):
        model, stepper, opt = make('history_g2')
        coef = torch.tensor([.7, -.3, .1])
        def linear(x, y): return torch.dot(model.w, coef), model.w.unsqueeze(0)
        out = stepper.compute(None, None, linear, .07)
        self.assertEqual(out['record']['ordinary_gradient_calls'], 2)
        self.assertIsNone(out['context']['g2'])
        stepper.apply_update(.07, opt)
        self.assertTrue(torch.equal(stepper._history['vector'][0], coef))
        self.assertEqual(out['record']['history_cached_vector_sha256'], out['record']['final_gradient_sha256'])

    def test_history_source_and_method_swap_rejected(self):
        model, stepper, opt = make('history_g2')
        stepper.compute(None, None, quadratic(model), .07); stepper.apply_update(.07, opt)
        saved = stepper.state_dict()
        model2, other, _ = make('history_fusion'); model2.load_state_dict(model.state_dict())
        with self.assertRaises(ValueError): other.load_state_dict(saved)
        corrupted = copy.deepcopy(saved); corrupted['history']['cache_source'] = 'base'
        with self.assertRaises(ValueError): stepper.load_state_dict(corrupted)

    def test_portable_checkpoint_identical_continuation_both_history_methods(self):
        for method in ('history_fusion', 'history_g2'):
            model, stepper, opt = make(method)
            stepper.compute(None, None, quadratic(model), .07)
            with self.assertRaises(RuntimeError): stepper.state_dict()
            stepper.apply_update(.07, opt)
            stream = io.BytesIO(); torch.save(stepper.state_dict(), stream); stream.seek(0)
            saved = torch.load(stream, weights_only=True)
            model2, step2, opt2 = make(method)
            with self.assertRaises(ValueError): step2.load_state_dict(saved)
            model2.load_state_dict(model.state_dict()); step2.load_state_dict(saved)
            self.assertNotEqual(step2._history['layout'], stepper._history['layout'])
            self.assertEqual(step2._history['vector'][0].device.type, 'cpu')
            opt.param_groups[0]['lr'] = opt2.param_groups[0]['lr'] = .023
            x = stepper.compute(None, None, quadratic(model), .023)
            y = step2.compute(None, None, quadratic(model2), .023)
            self.assertTrue(torch.equal(x['U'][0], y['U'][0]))
            self.assertEqual(x['record']['history_incoming_vector_sha256'], y['record']['history_incoming_vector_sha256'])
            stepper.apply_update(.023, opt); step2.apply_update(.023, opt2)
            self.assertTrue(torch.equal(model.w, model2.w))

    def test_checkpoint_corrupt_tensor_length_layout_and_current_point(self):
        model, stepper, opt = make('history_g2')
        stepper.compute(None, None, quadratic(model), .07); stepper.apply_update(.07, opt)
        saved = stepper.state_dict()
        for alteration in ('tensor', 'length', 'layout', 'point'):
            bad = copy.deepcopy(saved)
            if alteration == 'tensor': bad['history']['vector'][0].add_(.1)
            elif alteration == 'length': bad['history']['length'] *= 2
            elif alteration == 'layout': bad['history']['layout'][0]['name'] = 'wrong'
            else: bad['history']['after_sha256'] = '0' * 64
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                stepper.load_state_dict(bad)
        self.assertEqual(stepper._history['vector_sha256'], saved['history']['vector_sha256'])

    def test_task_reset_layout_and_parameter_continuity(self):
        model, stepper, opt = make('history_g2')
        stepper.compute(None, None, quadratic(model), .07); stepper.apply_update(.07, opt)
        stepper.reset_history('task_boundary')
        x = stepper.compute(None, None, quadratic(model), .07)
        self.assertFalse(x['record']['history_used']); stepper.apply_update(.07, opt)
        with torch.no_grad(): model.w.add_(.1)
        x = stepper.compute(None, None, quadratic(model), .07)
        self.assertFalse(x['record']['history_used'])
        self.assertEqual(x['record']['history_reason'], 'parameter_continuity_changed')
        stepper.apply_update(.07, opt)
        model.w = torch.nn.Parameter(torch.zeros(4))
        with self.assertRaises(ValueError): stepper.compute(None, None, quadratic(model), .07)
        self.assertIsNone(stepper._history)

    def test_exception_restores_entry_and_preserves_prior_cache(self):
        model, stepper, opt = make('history_g2')
        stepper.compute(None, None, quadratic(model), .07); stepper.apply_update(.07, opt)
        saved = stepper.state_dict(); before = State(model)
        count = 0
        def bad(x, y):
            nonlocal count
            count += 1
            random.random(); np.random.rand(); torch.rand(2)
            if count == 2: raise RuntimeError('second probe deliberate failure')
            return quadratic(model)(x, y)
        with self.assertRaises(RuntimeError): stepper.compute(None, None, bad, .07)
        self.assertTrue(all(before.check().values()))
        self.assertIsNone(stepper._pending)
        self.assertEqual(stepper.real_updates, saved['real_updates'])
        self.assertEqual(stepper._history['vector_sha256'], saved['history']['vector_sha256'])
        self.assertEqual(stepper._history['displacement_sha256'], saved['history']['displacement_sha256'])
        out = stepper.compute(None, None, quadratic(model), .07)
        self.assertTrue(out['record']['history_used'])

    def test_optimizer_requirements_and_one_real_update(self):
        for kwargs in ({'momentum': .9}, {'weight_decay': .1}, {'maximize': True}, {'foreach': True}):
            model, stepper, _ = make('history_g2')
            stepper.compute(None, None, quadratic(model), .07)
            base = model.w.detach().clone()
            wrong = torch.optim.SGD(model.parameters(), lr=.07, **kwargs)
            with self.assertRaises(ValueError): stepper.apply_update(.07, wrong)
            self.assertTrue(torch.equal(model.w, base))
            self.assertIsNone(stepper._history)
        model, stepper, opt = make('sgd')
        stepper.compute(None, None, quadratic(model), .07)
        with self.assertRaises(RuntimeError): stepper.compute(None, None, quadratic(model), .07)
        with self.assertRaises(RuntimeError): stepper.reset_history()
        with self.assertRaises(ValueError): stepper.apply_update(.08, opt)
        opt.state[model.w]['momentum_buffer'] = torch.zeros_like(model.w)
        with self.assertRaises(ValueError): stepper.apply_update(.07, opt)
        opt.state.clear(); stepper.apply_update(.07, opt)
        with self.assertRaises(RuntimeError): stepper.apply_update(.07, opt)

    def test_optimizer_failure_rolls_back_before_cache_commit(self):
        model, stepper, opt = make('history_g2')
        stepper.compute(None, None, quadratic(model), .07); stepper.apply_update(.07, opt)
        saved = stepper.state_dict()
        stepper.compute(None, None, quadratic(model), .07)
        before = State(model)
        original_step = opt.step
        def fail():
            with torch.no_grad(): model.w.add_(.1)
            raise RuntimeError('optimizer interruption')
        opt.step = fail
        with self.assertRaises(RuntimeError): stepper.apply_update(.07, opt)
        self.assertTrue(all(before.check().values()))
        self.assertEqual(stepper._history['vector_sha256'], saved['history']['vector_sha256'])
        self.assertEqual(stepper.real_updates, saved['real_updates'])
        opt.step = original_step; stepper.apply_update(.07, opt)

    def test_passive_post_loss_preserves_committed_state_and_history(self):
        model, stepper, opt = make('history_g2')
        def closure(x, y):
            random.random(); np.random.rand(); torch.rand(2)
            return quadratic(model)(x, y)
        out = stepper.compute(None, None, closure, .07); stepper.apply_update(.07, opt)
        saved = stepper.state_dict(); committed = State(model)
        try:
            rng_set(out['context']['rng0'])
            with torch.no_grad(): closure(None, None)
            model.eval()
        finally: committed.restore()
        self.assertTrue(all(committed.check().values()))
        self.assertTrue(rng_equal(rng_get(), out['context']['rng1']))
        self.assertEqual(stepper._history['vector_sha256'], saved['history']['vector_sha256'])

    def test_illegal_configs_and_trainable_scope_rejected(self):
        model = VectorModel()
        for cfg in ({'method': 'unknown'}, {'method': 'sgd', 'history_weight': .5},
                    {'method': 'history_g2', 'eps': 0}, {'method': 'current_g2', 'backend': 'math'},
                    {'method': 'history_g2', 'radius_mode': 'fixed'}, {'method': 'history_g2', 'radius_scale': 2}):
            with self.subTest(config=cfg), self.assertRaises(ValueError):
                ResponseStep(model, list(model.parameters()), None, cfg)
        with self.assertRaises(ValueError): ResponseStep(model, [], None, {'method': 'sgd'})
        model.w.data = model.w.data.double()
        with self.assertRaises(ValueError): ResponseStep(model, list(model.parameters()), None, {'method': 'sgd'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
