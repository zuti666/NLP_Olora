"""Independent algebra, stochastic transaction, SGD and protocol rejection tests."""
import copy
import json
import math
import random
import unittest

import numpy as np
import torch

from independent_response_step import IndependentResponseStep
from response_step import ResponseStep, _digest, _norm
from state_transaction import State, rng_equal, rng_get, rng_set
from test_response_step_cpu import VectorModel, PlainGeometry, Q, B, quadratic


VARIANTS = {'fixed_inner': ('fixed_l2', 'sgd_step'),
            'fixed_outer': ('sgd_step', 'fixed_l2'),
            'fixed_both': ('fixed_l2', 'fixed_l2'),
            'coupled_reference': ('sgd_step', 'sgd_step')}


def cfg(variant='fixed_both', inner=.05, outer=.05):
    im, om = VARIANTS[variant]
    return {'method': 'current_g2', 'history_weight': 0., 'backend': 'native',
            'radius_mode': 'independent_inner_outer_v1', 'radius_scale': 1.,
            'inner_probe_radius_mode': im, 'outer_probe_radius_mode': om,
            'fixed_inner_probe_radius': inner if im == 'fixed_l2' else None,
            'fixed_outer_probe_radius': outer if om == 'fixed_l2' else None,
            'radius_variant': variant}


def make(variant='fixed_both', model=None, eta=.04, **kwargs):
    model = model or VectorModel()
    step = IndependentResponseStep(model, list(model.parameters()), PlainGeometry(model), cfg(variant, **kwargs))
    return model, step, torch.optim.SGD(model.parameters(), lr=eta)


class IndependentGeometryCPU(unittest.TestCase):
    def test_three_registered_variants_coupled_quadratic_positions_and_final_update(self):
        for variant in ('fixed_inner', 'fixed_outer', 'fixed_both'):
            with self.subTest(variant=variant):
                model, step, opt = make(variant)
                for index, eta in enumerate((.04, .011)):
                    opt.param_groups[0]['lr'] = eta
                    base = model.w.detach().clone()
                    b = B + index * torch.tensor([.02, -.03, .04])
                    g0 = Q @ base + b
                    h = eta * float(g0.double().norm())
                    inner = .05 if VARIANTS[variant][0] == 'fixed_l2' else h
                    outer = .05 if VARIANTS[variant][1] == 'fixed_l2' else h
                    # Oracle is the analytic quadratic, without invoking the
                    # implementation's response helper or saved autograd g1/g2.
                    theta1 = base - inner * g0 / float(g0.double().norm())
                    g1 = Q @ theta1 + b
                    a = g0 - g1
                    theta2 = base + outer * a / float(a.double().norm())
                    g2 = Q @ theta2 + b
                    out = step.compute(None, None, quadratic(model, b), eta)
                    for actual, expected in ((out['context']['future'][0], theta1),
                                             (out['context']['relocation'][0], theta2),
                                             (out['U'][0], g2)):
                        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
                    self.assertEqual(out['record']['ordinary_gradient_calls'], 3)
                    self.assertEqual(out['record']['relocation_origin'], 'base')
                    self.assertFalse(out['record']['history_used'])
                    self.assertAlmostEqual(out['record']['inner_probe_radius_theoretical'], inner, places=7)
                    self.assertAlmostEqual(out['record']['outer_probe_radius_theoretical'], outer, places=7)
                    self.assertAlmostEqual(out['record']['inner_probe_radius_actual'], inner, places=7)
                    self.assertAlmostEqual(out['record']['outer_probe_radius_actual'], outer, places=7)
                    self.assertTrue(torch.equal(model.w, base))
                    step.apply_update(eta, opt)
                    self.assertTrue(torch.equal(model.w, base.clone().add_(out['U'][0], alpha=-eta)))
                    self.assertIsNone(step._history)
                    json.dumps(out['record'], allow_nan=False)

    def test_fixed_both_probe_and_U_invariant_to_learning_rate_but_real_update_is_not(self):
        ma, a, oa = make(eta=.04)
        mb, b, ob = make(eta=.007)
        base = ma.w.detach().clone()
        r0 = rng_get()
        xa = a.compute(None, None, quadratic(ma), .04)
        rng_set(r0)
        xb = b.compute(None, None, quadratic(mb), .007)
        for key in ('g0', 'g1', 'g2', 'future', 'relocation', 'U', 'response'):
            self.assertTrue(torch.equal(xa['context'][key][0], xb['context'][key][0]), key)
        self.assertNotEqual(xa['record']['h_ref'], xb['record']['h_ref'])
        a.apply_update(.04, oa); b.apply_update(.007, ob)
        self.assertFalse(torch.equal(ma.w, mb.w))
        torch.testing.assert_close(ma.w.detach() - base, (mb.w.detach() - base) * (.04 / .007), rtol=2e-5, atol=4e-8)

    def test_fixed_inner_does_not_decouple_outer_and_fixed_outer_does_not_decouple_inner(self):
        for variant in ('fixed_inner', 'fixed_outer'):
            ma, a, _ = make(variant, eta=.04)
            mb, b, _ = make(variant, eta=.007)
            xa = a.compute(None, None, quadratic(ma), .04)
            xb = b.compute(None, None, quadratic(mb), .007)
            fixed = 'inner' if variant == 'fixed_inner' else 'outer'
            coupled = 'outer' if fixed == 'inner' else 'inner'
            self.assertEqual(xa['record'][fixed + '_probe_radius_theoretical'], xb['record'][fixed + '_probe_radius_theoretical'])
            self.assertNotEqual(xa['record'][coupled + '_probe_radius_theoretical'], xb['record'][coupled + '_probe_radius_theoretical'])
            if variant == 'fixed_inner':
                self.assertTrue(torch.equal(xa['context']['future'][0], xb['context']['future'][0]))
                self.assertFalse(torch.equal(xa['context']['relocation'][0], xb['context']['relocation'][0]))

    def test_fixed_inner_not_rejected_by_tiny_reference_eta(self):
        model, step, _ = make('fixed_both', eta=1e-18)
        out = step.compute(None, None, quadratic(model), 1e-18)
        self.assertLess(out['record']['h_ref'], step.eps)
        self.assertEqual(out['record']['ordinary_gradient_calls'], 3)
        self.assertFalse(out['record']['fallback'])
        self.assertEqual(out['record']['inner_probe_radius_theoretical'], .05)
        self.assertEqual(out['record']['outer_probe_radius_theoretical'], .05)

    def test_zero_lr_still_probes_fixed_both_then_ordinary_zero_update(self):
        model, step, opt = make('fixed_both', eta=0.)
        before = model.w.detach().clone()
        out = step.compute(None, None, quadratic(model), 0.)
        self.assertEqual(out['record']['ordinary_gradient_calls'], 3)
        proof = step.apply_update(0., opt)
        self.assertTrue(torch.equal(model.w, before))
        self.assertEqual(proof['actual_parameter_step_norm'], 0.)

    def test_zero_gradient_is_one_call_undefined_direction_fallback(self):
        model, step, _ = make()
        def zero(x, y):
            return 0. * model.w.sum(), model.w.unsqueeze(0)
        out = step.compute(None, None, zero, .04)
        self.assertEqual(out['record']['ordinary_gradient_calls'], 1)
        self.assertEqual(out['record']['fallback_reason'], 'base_gradient_length_at_or_below_eps')
        self.assertTrue(torch.equal(out['U'][0], torch.zeros_like(model.w)))

    def test_linear_loss_is_two_call_zero_response_fallback(self):
        model, step, _ = make()
        def linear(x, y):
            return torch.dot(model.w, torch.tensor([.7, -.3, .2])), model.w.unsqueeze(0)
        out = step.compute(None, None, linear, .04)
        self.assertEqual(out['record']['ordinary_gradient_calls'], 2)
        self.assertEqual(out['record']['fallback_reason'], 'current_response_length_at_or_below_eps')
        self.assertTrue(torch.equal(out['U'][0], out['context']['g0'][0]))

    def test_FP32_indistinguishable_inner_or_outer_truthful_fallback_budget(self):
        model, step, _ = make(model=VectorModel((1e10, -1e10, 1e10)))
        out = step.compute(None, None, quadratic(model), .04)
        self.assertEqual(out['record']['ordinary_gradient_calls'], 1)
        self.assertEqual(out['record']['fallback_reason'], 'predicted_point_bitwise_indistinguishable_in_FP32')
        self.assertEqual(out['record']['inner_probe_radius_actual'], 0.)
        model2, step2, _ = make(model=VectorModel((1e3, -1e3, 1e3)), outer=1e-7)
        out2 = step2.compute(None, None, quadratic(model2), .04)
        self.assertEqual(out2['record']['ordinary_gradient_calls'], 2)
        self.assertEqual(out2['record']['fallback_reason'], 'relocated_point_bitwise_indistinguishable_in_FP32')
        self.assertEqual(out2['record']['outer_probe_radius_actual'], 0.)

    def test_outer_coupled_zero_eta_does_not_issue_spurious_probe(self):
        model, step, _ = make('fixed_inner', eta=0.)
        out = step.compute(None, None, quadratic(model), 0.)
        self.assertEqual(out['record']['ordinary_gradient_calls'], 2)
        self.assertEqual(out['record']['fallback_reason'], 'outer_probe_length_at_or_below_eps')

    def test_coupled_passthrough_12_steps_dropout_variable_lr_bitwise_parent(self):
        class Stochastic(VectorModel):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(.2)
                self.register_buffer('count', torch.tensor(3))
                self.forward_cache = {'value': torch.tensor([7.])}
            def forward(self, x):
                self.count.add_(1); self.forward_cache['value'].add_(1)
                random.random(); np.random.rand()
                return self.dropout(self.w)
        torch.manual_seed(55); random.seed(55); np.random.seed(55)
        ma, a, oa = make('coupled_reference', model=Stochastic())
        mb = copy.deepcopy(ma)
        b = ResponseStep(mb, list(mb.parameters()), PlainGeometry(mb),
                         {'method': 'current_g2', 'history_weight': 0., 'radius_mode': 'sgd_step', 'radius_scale': 1.})
        ob = torch.optim.SGD(mb.parameters(), lr=.04)
        def loss(model, target):
            def closure(x, y):
                w = model(x)
                return .5 * torch.dot(w, Q @ w) + torch.dot(target, w), w.unsqueeze(0)
            return closure
        for index in range(12):
            eta = (.04, .007, .02, .003)[index % 4]
            oa.param_groups[0]['lr'] = ob.param_groups[0]['lr'] = eta
            target = B + index * torch.tensor([.01, -.02, .015])
            r0 = rng_get()
            xa = a.compute(None, None, loss(ma, target), eta); ra = rng_get()
            rng_set(r0)
            xb = b.compute(None, None, loss(mb, target), eta)
            self.assertTrue(rng_equal(ra, rng_get()))
            for key in ('base', 'future', 'relocation', 'g0', 'g1', 'g2', 'U'):
                va, vb = xa['context'][key], xb['context'][key]
                if va is None:
                    self.assertIsNone(vb, key)
                else:
                    self.assertTrue(all(torch.equal(x, y) for x, y in zip(va, vb)), key)
            pa = a.apply_update(eta, oa); pb = b.apply_update(eta, ob)
            self.assertTrue(torch.equal(ma.w, mb.w))
            self.assertEqual(pa['actual_step_displacement_sha256'], pb['actual_step_displacement_sha256'])
            self.assertEqual(int(ma.count), 3)
            self.assertTrue(torch.equal(ma.forward_cache['value'], torch.tensor([7.])))

    def test_fixed_transaction_probe_rng_replay_and_only_base_R1_commit(self):
        class Stochastic(VectorModel):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(.3)
                self.register_buffer('count', torch.tensor(4))
                self.forward_cache = {'count': torch.tensor([8.])}
            def forward(self, x):
                self.count.add_(1); self.forward_cache['count'].add_(1)
                self.new_cache = torch.randn(2)
                return self.dropout(self.w)
        for variant in ('fixed_inner', 'fixed_outer', 'fixed_both'):
            torch.manual_seed(106)
            model, step, _ = make(variant, model=Stochastic())
            model.w.grad = torch.tensor([1., 2., 3.])
            entry = State(model)
            masks = []
            def closure(x, y):
                w = model(x)
                masks.append(w.detach() != 0.)
                return .5 * torch.dot(w, Q @ w), w.unsqueeze(0)
            out = step.compute(None, None, closure, .04)
            self.assertEqual(out['record']['ordinary_gradient_calls'], 3)
            self.assertTrue(all(torch.equal(mask, masks[0]) for mask in masks))
            self.assertTrue(rng_equal(rng_get(), out['context']['rng1']))
            committed_rng = rng_get()
            rng_set(entry.rng); self.assertTrue(all(entry.check().values())); rng_set(committed_rng)
            self.assertFalse(hasattr(model, 'new_cache'))

    def test_exception_restores_entry_state_rng_and_no_pending_update(self):
        for bad_probe in (1, 2, 3):
            model, step, _ = make()
            entry = State(model)
            counter = [0]
            def closure(x, y):
                counter[0] += 1; torch.rand(1)
                if counter[0] == bad_probe:
                    raise RuntimeError('deliberate probe failure')
                return quadratic(model)(x, y)
            with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                step.compute(None, None, closure, .04)
            self.assertTrue(all(entry.check().values()))
            self.assertIsNone(step._pending)
            self.assertEqual(step.real_updates, 0)

    def test_invalid_registered_modes_radii_and_history_are_rejected(self):
        mutations = ({'method': 'history_g2'}, {'history_weight': .5}, {'radius_scale': 2.},
                     {'inner_probe_radius_mode': 'unknown'}, {'fixed_inner_probe_radius': None},
                     {'fixed_inner_probe_radius': 0.}, {'fixed_inner_probe_radius': -1.},
                     {'fixed_outer_probe_radius': float('nan')}, {'fixed_outer_probe_radius': float('inf')},
                     {'fixed_outer_probe_radius': True}, {'radius_variant': 'fixed_outer'})
        for mutation in mutations:
            model = VectorModel()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                IndependentResponseStep(model, list(model.parameters()), PlainGeometry(model), dict(cfg(), **mutation))
        model = VectorModel()
        with self.assertRaises(ValueError):
            IndependentResponseStep(model, list(model.parameters()), PlainGeometry(model),
                                    dict(cfg('fixed_outer'), fixed_inner_probe_radius=.05))

    def test_explicit_checkpoint_roundtrip_and_geometry_history_format_rejections(self):
        model, step, opt = make()
        step.compute(None, None, quadratic(model), .04); step.apply_update(.04, opt)
        state = step.state_dict()
        self.assertEqual(state['format'], 'IndependentResponseStep_v1')
        self.assertIsNone(state['history'])
        model2 = copy.deepcopy(model)
        _, step2, _ = make(model=model2)
        step2.load_state_dict(state)
        self.assertEqual(step2.real_updates, 1)
        for mutation in ({'format': 'ResponseStep_v1'}, {'history': {'valid': True}},
                         {'independent_radius_config': dict(state['independent_radius_config'], fixed_outer_probe_radius=.1)},
                         {'method': 'history_g2'}, {'cache_source': 'final'}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                step2.load_state_dict(dict(state, **mutation))


if __name__ == '__main__':
    unittest.main(verbosity=2)
