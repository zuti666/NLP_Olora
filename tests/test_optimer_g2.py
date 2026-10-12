"""CPU correctness tests for the native NLP Current-G2 optimizer adapter."""
import copy
import json
import math
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from optimer_g2 import CurrentG2, _bf16_weight_cast_proxy


class Vector(torch.nn.Module):
    def __init__(self, values=(1., -2.), dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(values, dtype=dtype))


class CurrentG2Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        np.random.seed(101)
        random.seed(101)

    def _quadratic(self, inner_mode='sgd_step', outer_mode='sgd_step', first_ready=False):
        model = Vector(dtype=torch.float64)
        H = torch.tensor([2., 4.], dtype=torch.float64)
        base = torch.optim.SGD(model.parameters(), lr=.1)
        opt = CurrentG2(model.parameters(), base, model,
                        inner_radius_mode=inner_mode, outer_radius_mode=outer_mode,
                        inner_radius=.05 if inner_mode == 'fixed_l2' else None,
                        outer_radius=.08 if outer_mode == 'fixed_l2' else None,
                        require_fp32_trainable=False)
        old = model.weight.detach().clone()
        g0 = H * old
        q = .1 * float(g0.norm()) if inner_mode == 'sgd_step' else .05
        R = .1 * float(g0.norm()) if outer_mode == 'sgd_step' else .08
        inner = old - q * g0 / g0.norm()
        g1 = H * inner
        response = g0 - g1
        outer = old + R * response / response.norm()
        g2 = H * outer
        points = []
        if first_ready:
            (0.5 * (H * model.weight.pow(2)).sum()).backward()

        def closure():
            points.append(model.weight.detach().clone())
            opt.zero_grad()
            loss = 0.5 * (H * model.weight.pow(2)).sum()
            loss.backward()
            return model.weight.detach().clone(), loss.detach()

        opt.step(closure, first_grad_ready=first_ready)
        torch.testing.assert_close(model.weight, old - .1 * g2, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(points[-2], inner, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(points[-1], outer, rtol=1e-13, atol=1e-13)
        self.assertEqual(len(points), 2 if first_ready else 3)
        self.assertEqual(opt.last_stats['g2_ce_gradient_phases'], 3)
        self.assertEqual(opt.last_stats['g2_base_optimizer_steps'], 1)
        self.assertEqual(opt.last_stats['g2_final_direction'], 'g2')
        self.assertEqual(opt.last_stats['g2_relocation_origin'], 'base')
        self.assertAlmostEqual(opt.last_stats['g2_outer_radius_actual'], R, places=12)
        stats = opt.last_stats
        self.assertAlmostEqual(stats['g2_cos_g0_g1'], float(torch.dot(g0, g1) / (g0.norm() * g1.norm())), places=13)
        self.assertAlmostEqual(stats['g2_cos_g0_g2'], float(torch.dot(g0, g2) / (g0.norm() * g2.norm())), places=13)
        self.assertAlmostEqual(stats['g2_g2_minus_g0_norm'], float((g2 - g0).norm()), places=12)
        self.assertAlmostEqual(stats['g2_g2_minus_g0_relative_norm'], float((g2 - g0).norm() / g0.norm()), places=12)
        self.assertAlmostEqual(stats['g2_response_relative_norm'], float(response.norm() / g0.norm()), places=12)
        self.assertAlmostEqual(stats['g2_actual_update_over_h_ref'], float(g2.norm() / g0.norm()), places=12)
        self.assertAlmostEqual(stats['g2_actual_update_over_inner_radius'], .1 * float(g2.norm()) / q, places=12)
        self.assertAlmostEqual(stats['g2_actual_update_over_outer_radius'], .1 * float(g2.norm()) / R, places=12)
        self.assertEqual(stats['g2_gradient_clip_scale'], 1.)
        json.dumps(stats, allow_nan=False)
        return opt

    def test_quadratic_coupled_both_paths(self):
        self._quadratic(first_ready=False)
        self._quadratic(first_ready=True)

    def test_quadratic_independent_radius_combinations(self):
        for inner, outer in [('fixed_l2', 'sgd_step'), ('sgd_step', 'fixed_l2'),
                             ('fixed_l2', 'fixed_l2')]:
            with self.subTest(inner=inner, outer=outer):
                self._quadratic(inner, outer, True)

    def test_sgd_twelve_steps_bitwise_match_original_cv_current_g2(self):
        reference_dir = ROOT / 'refine-logs/imageneta-response-four-methods-gpu567-20261009T103449Z/trainer'
        sys.path.insert(0, str(reference_dir))
        from response_step import ResponseStep

        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)
                self.dropout = torch.nn.Dropout(.3)
            def forward(self, x):
                return self.linear(self.dropout(x))

        reference = Network()
        adapted = copy.deepcopy(reference)
        data = torch.tensor([[.3, -1., .7], [1., .5, -.2]])
        target = torch.tensor([[.2, .3], [-.4, .1]])
        rb = torch.optim.SGD(reference.parameters(), lr=.1, foreach=False)
        ab = torch.optim.SGD(adapted.parameters(), lr=.1, foreach=False)
        original = ResponseStep(reference, reference.parameters(), None,
                                dict(method='current_g2', history_weight=0., backend='native'))
        translated = CurrentG2(adapted.parameters(), ab, adapted)
        rng = torch.get_rng_state().clone()
        for iteration in range(12):
            eta = .08 * (.92 ** iteration)
            rb.param_groups[0]['lr'] = eta
            ab.param_groups[0]['lr'] = eta
            torch.set_rng_state(rng)
            def reference_closure(inputs, targets):
                logits = reference(inputs)
                return (logits - targets).square().mean(), logits
            original.compute(data, target, reference_closure, eta)
            # torch2.1 stores {'momentum_buffer': None} even with momentum=0;
            # the frozen CV torch2.4 validator rejects any nonempty state.
            # Delete only these empty placeholders, never actual moments.
            for param in reference.parameters():
                self.assertTrue(all(v is None for v in rb.state[param].values()))
                rb.state[param].clear()
            original.apply_update(eta, rb)
            reference_rng = torch.get_rng_state().clone()
            torch.set_rng_state(rng)
            def translated_closure():
                translated.zero_grad()
                logits = adapted(data)
                loss = (logits - target).square().mean()
                loss.backward()
                return logits.detach(), loss.detach()
            translated.step(translated_closure)
            self.assertTrue(torch.equal(torch.get_rng_state(), reference_rng))
            for expected, actual in zip(reference.parameters(), adapted.parameters()):
                self.assertTrue(torch.equal(expected, actual), (iteration, expected, actual))
            rng = reference_rng

    def test_probe_rng_replayed_and_only_base_advances_standalone(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.5))
        opt = CurrentG2(model.parameters(), torch.optim.SGD(model.parameters(), lr=.1), model)
        x = torch.tensor([[1., 2., 3.], [-.1, .7, .4]])
        r0 = torch.get_rng_state().clone()
        py0 = random.getstate()
        np0 = np.random.get_state()
        starts = []
        ends = []
        def closure():
            starts.append(torch.get_rng_state().clone())
            random.random()
            np.random.random()
            opt.zero_grad()
            out = model(x)
            loss = out.square().mean()
            loss.backward()
            ends.append(torch.get_rng_state().clone())
            return out.detach(), loss.detach()
        opt.step(closure)
        self.assertEqual(len(starts), 3)
        self.assertTrue(all(torch.equal(s, r0) for s in starts))
        self.assertTrue(torch.equal(torch.get_rng_state(), ends[0]))
        python_after = random.getstate()
        numpy_after = np.random.get_state()
        random.setstate(py0)
        random.random()
        self.assertEqual(random.getstate(), python_after)
        np.random.set_state(np0)
        np.random.random()
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_after[1]))

    def test_first_ready_per_microbatch_replay_preserves_afterbase_rng(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.3))
        opt = CurrentG2(model.parameters(), torch.optim.SGD(model.parameters(), lr=.1), model)
        batches = [torch.tensor([[1., 2., 3.]]), torch.tensor([[-.1, .7, .4]])]
        cached = []
        for x in batches:
            cached.append(torch.get_rng_state().clone())
            (model(x).square().mean() / len(batches)).backward()
        r1 = torch.get_rng_state().clone()
        probe_starts = []
        def closure():
            opt.zero_grad()
            total = torch.tensor(0.)
            for x, rng in zip(batches, cached):
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(rng)
                    probe_starts.append(torch.get_rng_state().clone())
                    loss = model(x).square().mean() / len(batches)
                    loss.backward()
                    total += loss.detach()
            return None, total
        opt.step(closure, first_grad_ready=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), r1))
        self.assertEqual(len(probe_starts), 4)
        self.assertTrue(all(torch.equal(s, cached[i % 2]) for i, s in enumerate(probe_starts)))

    def test_exception_restores_weights_flags_grads_buffers_cache_and_rng(self):
        model = Vector()
        model.frozen = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)
        model.register_buffer('visits', torch.tensor(0))
        model.task_cache = {'tensor': torch.tensor([4.])}
        opt = CurrentG2([model.weight], torch.optim.SGD([model.weight], lr=.1), model)
        model.weight.square().sum().backward()
        old = model.weight.detach().clone()
        old_grad = model.weight.grad.clone()
        rng = torch.get_rng_state().clone()
        def closure():
            model.visits.add_(1)
            model.task_cache['tensor'].add_(3)
            model.new_cache = [torch.tensor([9.])]
            model.eval()
            model.frozen.requires_grad_(True)
            model.weight.requires_grad_(False)
            torch.rand(3)
            raise RuntimeError('deliberate probe failure')
        with self.assertRaisesRegex(RuntimeError, 'deliberate probe failure'):
            opt.step(closure, first_grad_ready=True)
        self.assertTrue(torch.equal(model.weight, old))
        self.assertTrue(torch.equal(model.weight.grad, old_grad))
        self.assertTrue(model.weight.requires_grad)
        self.assertFalse(model.frozen.requires_grad)
        self.assertTrue(model.training)
        self.assertEqual(int(model.visits), 0)
        self.assertTrue(torch.equal(model.task_cache['tensor'], torch.tensor([4.])))
        self.assertFalse(hasattr(model, 'new_cache'))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(opt.real_updates, 0)

    def test_unused_parameter_stays_none_for_adamw(self):
        model = Vector()
        model.unused = torch.nn.Parameter(torch.tensor([2.]))
        base = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.2)
        opt = CurrentG2(model.parameters(), base, model)
        def closure():
            opt.zero_grad()
            loss = model.weight.square().sum()
            loss.backward()
            return None, loss.detach()
        unused = model.unused.detach().clone()
        opt.step(closure)
        self.assertIsNone(model.unused.grad)
        self.assertTrue(torch.equal(model.unused, unused))
        self.assertNotIn(model.unused, base.state)

    def test_zero_response_falls_back_to_clean_gradient(self):
        model = Vector()
        base = torch.optim.SGD(model.parameters(), lr=.1)
        opt = CurrentG2(model.parameters(), base, model)
        old = model.weight.detach().clone()
        def closure():
            opt.zero_grad()
            loss = model.weight.sum()
            loss.backward()
            return None, loss.detach()
        opt.step(closure)
        torch.testing.assert_close(model.weight, old - .1)
        self.assertEqual(opt.last_stats['g2_fallback_reason'], 'response_at_or_below_eps')
        self.assertEqual(opt.last_stats['g2_ce_gradient_phases'], 2)
        self.assertEqual(opt.last_stats['g2_final_direction'], 'g0_fallback')
        self.assertAlmostEqual(opt.last_stats['g2_cos_g0_g1'], 1., places=14)
        self.assertEqual(opt.last_stats['g2_response_relative_norm'], 0.)
        self.assertIsNone(opt.last_stats['g2_cos_g0_g2'])
        self.assertIsNone(opt.last_stats['g2_g2_minus_g0_norm'])
        self.assertIsNone(opt.last_stats['g2_g2_minus_g0_relative_norm'])
        self.assertIsNone(opt.last_stats['g2_actual_update_over_outer_radius'])
        json.dumps(opt.last_stats, allow_nan=False)

    def test_zero_gradient_fallback_does_not_replay(self):
        model = Vector(values=(0., 0.))
        opt = CurrentG2(model.parameters(), torch.optim.SGD(model.parameters(), lr=.1), model)
        model.weight.square().sum().backward()
        opt.step(lambda: self.fail('zero-gradient fallback must not replay'), first_grad_ready=True)
        self.assertEqual(opt.last_stats['g2_ce_gradient_phases'], 1)
        self.assertEqual(opt.last_stats['g2_fallback_reason'], 'base_gradient_at_or_below_eps')
        for field in ('g2_cos_g0_g1', 'g2_cos_g0_g2', 'g2_g2_minus_g0_relative_norm',
                      'g2_response_relative_norm', 'g2_gradient_clip_scale',
                      'g2_actual_update_over_h_ref', 'g2_actual_update_over_inner_radius',
                      'g2_actual_update_over_outer_radius'):
            self.assertIsNone(opt.last_stats[field], field)
        json.dumps(opt.last_stats, allow_nan=False)

    def test_fp32_indistinguishable_coupled_predictor_falls_back(self):
        model = Vector(values=(1e7, -1e7))
        opt = CurrentG2(model.parameters(), torch.optim.SGD(model.parameters(), lr=1e-8), model)
        model.weight.square().sum().backward()
        opt.step(lambda: self.fail('indistinguishable point must not replay'), first_grad_ready=True)
        self.assertIn('bitwise_indistinguishable', opt.last_stats['g2_fallback_reason'])
        self.assertEqual(opt.last_stats['g2_ce_gradient_phases'], 1)

    def test_fixed_inner_remains_active_at_tiny_learning_rate(self):
        model = Vector()
        opt = CurrentG2(model.parameters(), torch.optim.SGD(model.parameters(), lr=1e-20), model,
                        inner_radius_mode='fixed_l2', outer_radius_mode='fixed_l2',
                        inner_radius=.05, outer_radius=.05)
        def closure():
            opt.zero_grad()
            loss = model.weight.square().sum()
            loss.backward()
            return None, loss.detach()
        opt.step(closure)
        self.assertIsNone(opt.last_stats['g2_fallback_reason'])
        self.assertEqual(opt.last_stats['g2_ce_gradient_phases'], 3)
        self.assertGreater(opt.last_stats['g2_inner_radius_actual'], .04999)
        self.assertLess(opt.last_stats['g2_h_ref'], opt.eps)

    def test_fixed_both_same_point_probes_independent_of_lr(self):
        points_by_lr = []
        for lr in (.01, .2):
            model = Vector()
            opt = CurrentG2(model.parameters(), torch.optim.SGD(model.parameters(), lr=lr), model,
                            inner_radius_mode='fixed_l2', outer_radius_mode='fixed_l2',
                            inner_radius=.05, outer_radius=.05)
            points = []
            def closure():
                points.append(model.weight.detach().clone())
                opt.zero_grad()
                loss = model.weight.square().sum()
                loss.backward()
                return None, loss.detach()
            opt.step(closure)
            points_by_lr.append((points, model.weight.detach().clone()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(points_by_lr[0][0], points_by_lr[1][0])))
        self.assertFalse(torch.equal(points_by_lr[0][1], points_by_lr[1][1]))

    def test_adamw_commit_matches_one_manual_step_and_roundtrip(self):
        model = Vector()
        base = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.02, eps=1e-8)
        opt = CurrentG2(model.parameters(), base, model)
        def closure():
            opt.zero_grad()
            loss = .5 * (torch.tensor([2., 4.]) * model.weight.square()).sum()
            loss.backward()
            return None, loss.detach()
        for _ in range(3):
            before = model.weight.detach().clone()
            base_before = copy.deepcopy(base.state_dict())
            opt.step(closure)
            manual = Vector()
            manual.weight.data.copy_(before)
            mb = torch.optim.AdamW(manual.parameters(), lr=.01, weight_decay=.02, eps=1e-8)
            mb.load_state_dict(base_before)
            manual.weight.grad = model.weight.grad.clone()
            mb.step()
            self.assertTrue(torch.equal(model.weight, manual.weight))
        state = copy.deepcopy(opt.state_dict())
        restored = Vector()
        restored.weight.data.copy_(model.weight)
        rb = torch.optim.AdamW(restored.parameters(), lr=.5, weight_decay=.02, eps=1e-8)
        ropt = CurrentG2(restored.parameters(), rb, restored)
        ropt.load_state_dict(state)
        def restored_closure():
            ropt.zero_grad()
            loss = .5 * (torch.tensor([2., 4.]) * restored.weight.square()).sum()
            loss.backward()
            return None, loss.detach()
        opt.step(closure)
        ropt.step(restored_closure)
        self.assertTrue(torch.equal(model.weight, restored.weight))
        self.assertEqual(ropt.real_updates, opt.real_updates)
        self.assertFalse(opt.last_stats['g2_update_is_sgd'])
        self.assertEqual(opt.last_stats['g2_commit_semantics'], 'predictive_sgd_probe_then_adamw_commit')
        bad = copy.deepcopy(state)
        bad['_current_g2']['config']['eps'] = .5
        with self.assertRaisesRegex(ValueError, 'algorithm/config/layout mismatch'):
            ropt.load_state_dict(bad)

    def test_transformers_adamw_commit_supported(self):
        from transformers.optimization import AdamW
        model = Vector()
        base = AdamW(model.parameters(), lr=.01, no_deprecation_warning=True)
        opt = CurrentG2(model.parameters(), base, model)
        def closure():
            opt.zero_grad()
            loss = model.weight.square().sum()
            loss.backward()
            return None, loss.detach()
        opt.step(closure)
        self.assertEqual(opt.last_stats['g2_commit_optimizer_module'], 'transformers.optimization')
        self.assertFalse(opt.last_stats['g2_update_is_sgd'])

    def test_transformers_adamw_moments_roundtrip(self):
        from transformers.optimization import AdamW
        model = Vector()
        base = AdamW(model.parameters(), lr=.005, eps=1e-8, weight_decay=.02,
                     no_deprecation_warning=True)
        opt = CurrentG2(model.parameters(), base, model)
        def closure():
            opt.zero_grad()
            loss = model.weight.square().sum()
            loss.backward()
            return None, loss.detach()
        for _ in range(3):
            opt.step(closure)
        restored = Vector()
        restored.weight.data.copy_(model.weight)
        rb = AdamW(restored.parameters(), lr=.005, eps=1e-8, weight_decay=.02,
                   no_deprecation_warning=True)
        ropt = CurrentG2(restored.parameters(), rb, restored)
        ropt.load_state_dict(copy.deepcopy(opt.state_dict()))
        def restored_closure():
            ropt.zero_grad()
            loss = restored.weight.square().sum()
            loss.backward()
            return None, loss.detach()
        opt.step(closure)
        ropt.step(restored_closure)
        self.assertTrue(torch.equal(model.weight, restored.weight))
        self.assertTrue(torch.equal(base.state[model.weight]['exp_avg'],
                                    rb.state[restored.weight]['exp_avg']))
        self.assertTrue(torch.equal(base.state[model.weight]['exp_avg_sq'],
                                    rb.state[restored.weight]['exp_avg_sq']))
        self.assertEqual(base.state[model.weight]['step'], 4)

    def test_cuda_rng_snapshot_never_initializes_other_visible_devices(self):
        from optimer_g2 import _rng_get, _rng_set
        with patch('torch.cuda.is_initialized', return_value=True), \
             patch('torch.cuda.current_device', return_value=3), \
             patch('torch.cuda.get_rng_state', return_value=torch.get_rng_state().clone()) as get, \
             patch('torch.cuda.set_rng_state') as setter, \
             patch('torch.cuda.get_rng_state_all', side_effect=AssertionError('all-device contexts')), \
             patch('torch.cuda.set_rng_state_all', side_effect=AssertionError('all-device restore')):
            snapshot = _rng_get()
            self.assertEqual(snapshot[3][0], 3)
            get.assert_called_once_with(3)
            _rng_set(snapshot)
            self.assertEqual(setter.call_args.kwargs['device'], 3)

    def test_pure_g2_has_no_regularization_hook_and_clip_is_explicit(self):
        model = Vector()
        opt = CurrentG2(model.parameters(), torch.optim.SGD(model.parameters(), lr=.1), model)
        self.assertFalse(hasattr(opt, 'set_direction_projector'))
        self.assertFalse(hasattr(opt, '_direction_projector'))
        def closure():
            opt.zero_grad()
            loss = model.weight.square().sum()
            loss.backward()
            return None, loss.detach()
        opt.step(closure, clip_grad_norm=.7)
        self.assertFalse(opt.last_stats['g2_regularization_hook_used'])
        self.assertEqual(opt.last_stats['g2_projection_calls'], 0)
        self.assertTrue(opt.last_stats['g2_final_clip_enabled'])
        self.assertAlmostEqual(opt.last_stats['g2_submitted_postclip_norm'], .7, places=6)
        self.assertAlmostEqual(opt.last_stats['g2_actual_update_norm'], .07, places=6)
        self.assertAlmostEqual(opt.last_stats['g2_gradient_clip_scale'],
                               .7 / opt.last_stats['g2_submitted_preclip_norm'], places=6)

    def test_gradient_metrics_include_missing_gradients_as_zero_across_all_parameters(self):
        model = Vector(values=(1.,))
        model.branch = torch.nn.Parameter(torch.tensor([2.]))
        opt = CurrentG2(model.parameters(), torch.optim.SGD(model.parameters(), lr=.1), model,
                        inner_radius_mode='fixed_l2', outer_radius_mode='fixed_l2',
                        inner_radius=.1, outer_radius=.2)
        gradients = []
        def closure():
            opt.zero_grad()
            loss = .5 * model.weight.square().sum()
            if float(model.weight.detach()) < .95:
                loss = loss + .5 * model.branch.square().sum()
            loss.backward()
            gradients.append(torch.cat([torch.zeros_like(p) if p.grad is None else p.grad.clone()
                                        for p in model.parameters()]).double())
            return None, loss.detach()
        opt.step(closure)
        g0, g1, g2 = gradients
        self.assertEqual(float(g0[1]), 0.)
        self.assertGreater(float(g1[1]), 0.)
        self.assertEqual(float(g2[1]), 0.)
        self.assertIsNone(model.branch.grad)
        self.assertAlmostEqual(opt.last_stats['g2_cos_g0_g1'], float(torch.dot(g0, g1) / (g0.norm() * g1.norm())), places=12)
        self.assertAlmostEqual(opt.last_stats['g2_response_relative_norm'], float((g0 - g1).norm() / g0.norm()), places=7)
        self.assertAlmostEqual(opt.last_stats['g2_g2_minus_g0_relative_norm'], float((g2 - g0).norm() / g0.norm()), places=7)

    def test_computed_zero_g2_has_null_cosine_and_unit_relative_difference(self):
        model = Vector(values=(1.,))
        opt = CurrentG2(model.parameters(), torch.optim.SGD(model.parameters(), lr=.1), model,
                        inner_radius_mode='fixed_l2', outer_radius_mode='fixed_l2',
                        inner_radius=.5, outer_radius=.5)
        def closure():
            opt.zero_grad()
            # g(1)=.25, g(.5)=0, g(1.5)=0: the inner response
            # is nonzero while the actually computed outer gradient is zero.
            loss = (-model.weight.pow(3).sum() / 3 + model.weight.square().sum()
                    - .75 * model.weight.sum())
            loss.backward()
            return None, loss.detach()
        opt.step(closure)
        self.assertIsNone(opt.last_stats['g2_fallback_reason'])
        self.assertEqual(opt.last_stats['g2_g2_norm'], 0.)
        self.assertIsNone(opt.last_stats['g2_cos_g0_g2'])
        self.assertEqual(opt.last_stats['g2_g2_minus_g0_relative_norm'], 1.)
        self.assertIsNone(opt.last_stats['g2_gradient_clip_scale'])
        json.dumps(opt.last_stats, allow_nan=False)

    def test_bf16_proxy_uses_actual_dtype_and_whole_vector_cast_displacement(self):
        original = (torch.tensor([1., 2.]), torch.tensor([0.]))
        point = (torch.tensor([1.0001, 2.03125]), torch.tensor([.125]))
        saved = [value.clone() for value in (*original, *point)]
        proxy = _bf16_weight_cast_proxy(original, point)
        intended = torch.cat([new.double() - old.double() for old, new in zip(original, point)])
        cast_delta = torch.cat([new.to(torch.bfloat16).double() - old.to(torch.bfloat16).double()
                                for old, new in zip(original, point)])
        self.assertEqual(proxy['source_parameter_dtypes'], ['torch.float32'])
        self.assertEqual(proxy['total_coordinates'], 3)
        self.assertEqual(proxy['changed_coordinates'], 2)
        self.assertEqual(proxy['changed_fraction'], 2 / 3)
        self.assertAlmostEqual(proxy['intended_displacement_norm'], float(intended.norm()), places=13)
        self.assertAlmostEqual(proxy['represented_displacement_norm'], float(cast_delta.norm()), places=13)
        self.assertAlmostEqual(proxy['cosine_represented_vs_intended'],
                               float(torch.dot(intended, cast_delta) / (intended.norm() * cast_delta.norm())), places=13)
        self.assertTrue(all(torch.equal(actual, previous) for actual, previous in zip((*original, *point), saved)))
        for dtype in (torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                result = _bf16_weight_cast_proxy(tuple(v.to(dtype) for v in original),
                                                 tuple(v.to(dtype) for v in point))
                self.assertEqual(result['source_parameter_dtypes'], [str(dtype)])
                json.dumps(result, allow_nan=False)

    def test_bf16_proxy_zero_and_nonfinite_cast_never_emit_nan(self):
        original = (torch.tensor([1., 2.]),)
        zero = _bf16_weight_cast_proxy(original, original)
        self.assertEqual(zero['changed_coordinates'], 0)
        self.assertEqual(zero['represented_displacement_norm'], 0.)
        self.assertIsNone(zero['represented_over_intended_norm'])
        self.assertIsNone(zero['cosine_represented_vs_intended'])
        huge = (torch.tensor([torch.finfo(torch.float32).max]),)
        overflow = _bf16_weight_cast_proxy(huge, huge)
        self.assertEqual(overflow['status'], 'nonfinite_bf16_cast')
        self.assertIsNone(overflow['represented_displacement_norm'])
        json.dumps(zero, allow_nan=False)
        json.dumps(overflow, allow_nan=False)

    def test_tiny_bf16_forward_can_lose_fp32_distinguishable_probe(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        opt = CurrentG2(model.parameters(), torch.optim.SGD(model.parameters(), lr=1e-4), model,
                        quantization_diagnostics_interval=1)
        data = torch.tensor([[1., -.5]])
        def closure():
            opt.zero_grad()
            with torch.autocast('cpu', dtype=torch.bfloat16):
                prediction = model(data)
            self.assertEqual(prediction.dtype, torch.bfloat16)
            loss = .5 * prediction.float().square().sum()
            loss.backward()
            return prediction.detach(), loss.detach()
        opt.step(closure)
        stats = opt.last_stats
        self.assertTrue(stats['g2_inner_distinguishable'])
        self.assertEqual(stats['g2_fallback_reason'], 'response_at_or_below_eps')
        self.assertEqual(stats['g2_response_relative_norm'], 0.)
        inner = stats['g2_quantization_proxy']['inner']
        self.assertGreater(inner['intended_displacement_norm'], 0.)
        self.assertEqual(inner['changed_coordinates'], 0)
        self.assertEqual(inner['represented_displacement_norm'], 0.)
        self.assertIsNone(inner['cosine_represented_vs_intended'])
        self.assertFalse(stats['g2_quantization_proxy']['proves_forward_visibility'])
        self.assertIsNone(stats['g2_quantization_proxy']['outer'])
        json.dumps(stats, allow_nan=False)

    def test_proxy_sampling_does_not_change_adamw_trajectory_or_rng(self):
        initial = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.25))
        plain, observed = copy.deepcopy(initial), copy.deepcopy(initial)
        plain_opt = CurrentG2(plain.parameters(), torch.optim.AdamW(plain.parameters(), lr=.01), plain,
                              inner_radius_mode='fixed_l2', outer_radius_mode='fixed_l2',
                              inner_radius=.05, outer_radius=.05)
        observed_opt = CurrentG2(observed.parameters(), torch.optim.AdamW(observed.parameters(), lr=.01), observed,
                                 inner_radius_mode='fixed_l2', outer_radius_mode='fixed_l2',
                                 inner_radius=.05, outer_radius=.05, quantization_diagnostics_interval=2)
        data = torch.tensor([[.25, -1.], [1., -.5]])
        start = torch.get_rng_state()
        for step in range(4):
            def run(model, optimizer):
                def closure():
                    optimizer.zero_grad()
                    loss = model(data).square().sum()
                    loss.backward()
                    return None, loss.detach()
                return optimizer.step(closure)
            torch.set_rng_state(start)
            run(plain, plain_opt)
            after = torch.get_rng_state()
            torch.set_rng_state(start)
            run(observed, observed_opt)
            self.assertTrue(torch.equal(torch.get_rng_state(), after))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(plain.parameters(), observed.parameters())))
            self.assertEqual(observed_opt.last_stats['g2_quantization_proxy']['sampled'], step % 2 == 0)
            self.assertFalse(plain_opt.last_stats['g2_quantization_proxy']['sampled'])
            self.assertEqual(observed_opt.last_stats['g2_base_optimizer_steps'], 1)
            self.assertEqual(observed_opt.last_stats['g2_ce_gradient_phases'], 3)
            start = after
        # Diagnostic cadence is observational: old algorithm checkpoints
        # remain loadable and do not override the requested new cadence.
        observed_opt.load_state_dict(plain_opt.state_dict())
        self.assertEqual(observed_opt.quantization_diagnostics_interval, 2)

    def test_no_duplicate_ddp_reduce_when_trainer_owns_phases(self):
        model = Vector()
        opt = CurrentG2(model.parameters(), torch.optim.SGD(model.parameters(), lr=.1), model,
                        closure_gradients_reduced=True)
        model.weight.square().sum().backward()
        def closure():
            opt.zero_grad()
            loss = model.weight.square().sum()
            loss.backward()
            return None, loss.detach()
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.all_reduce', side_effect=AssertionError('duplicate reduction')):
            opt.step(closure, first_grad_ready=True)
        self.assertEqual(opt.last_stats['g2_core_explicit_mean_reductions'], 0)

    def test_standalone_core_reduces_each_gradient_phase_once(self):
        model = Vector()
        opt = CurrentG2(model.parameters(), torch.optim.SGD(model.parameters(), lr=.1), model,
                        closure_gradients_reduced=False)
        def closure():
            opt.zero_grad()
            loss = model.weight.square().sum()
            loss.backward()
            return None, loss.detach()
        def fake_reduce(value, op=None):
            value.mul_(2)
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_world_size', return_value=2), \
             patch('torch.distributed.all_reduce', side_effect=fake_reduce):
            opt.step(closure)
        self.assertEqual(opt.last_stats['g2_core_explicit_mean_reductions'], 3)
        self.assertEqual(opt.last_stats['g2_ce_gradient_phases'], 3)

    def test_nonfinite_probe_and_invalid_config_are_rejected(self):
        model = Vector()
        base = torch.optim.SGD(model.parameters(), lr=.1)
        for kw in [dict(inner_radius_mode='other'), dict(eps=float('nan')),
                   dict(inner_radius_mode='fixed_l2'), dict(inner_radius=.05),
                   dict(outer_radius_mode='fixed_l2', outer_radius=-1.), dict(grad_reduce='sum'),
                   dict(quantization_diagnostics_interval=-1),
                   dict(quantization_diagnostics_interval=True),
                   dict(quantization_diagnostics_interval=2.5)]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                CurrentG2(model.parameters(), base, model, **kw)
        with self.assertRaisesRegex(ValueError, 'actual FP32'):
            m = Vector(dtype=torch.bfloat16)
            CurrentG2(m.parameters(), torch.optim.SGD(m.parameters(), lr=.1), m)
        opt = CurrentG2(model.parameters(), base, model)
        original = model.weight.detach().clone()
        model.weight.square().sum().backward()
        def bad():
            opt.zero_grad()
            loss = model.weight.sum() * float('nan')
            loss.backward()
            return None, loss.detach()
        with self.assertRaises(FloatingPointError):
            opt.step(bad, first_grad_ready=True)
        self.assertTrue(torch.equal(model.weight, original))


if __name__ == '__main__':
    unittest.main()
