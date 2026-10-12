"""Independent oracles and transaction checks for two-point NLP directions."""
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
from transformers.optimization import AdamW as HFAdamW

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from optimer_reverse_probe import TwoPointProbe


class Vector(torch.nn.Module):
    def __init__(self, values=(1., -2.), dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(values, dtype=dtype))


def assert_state_equal(testcase, left, right):
    if torch.is_tensor(left):
        testcase.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        testcase.assertEqual(set(left), set(right))
        for key in left:
            assert_state_equal(testcase, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        testcase.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_state_equal(testcase, a, b)
    else:
        testcase.assertEqual(left, right)


class TwoPointProbeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(193)
        np.random.seed(193)
        random.seed(193)

    def make(self, mode='reverse', values=(1., -2.), dtype=torch.float32, **options):
        model = Vector(values, dtype)
        base = torch.optim.SGD(model.parameters(), lr=.07)
        optimizer = TwoPointProbe(model.parameters(), base, model, mode=mode,
                                  require_fp32_trainable=dtype == torch.float32, **options)
        return model, base, optimizer

    @staticmethod
    def closure(model, optimizer, points=None, h=None):
        def evaluate():
            if points is not None:
                points.append(model.weight.detach().clone())
            optimizer.zero_grad()
            loss = .5 * (model.weight.square() * (h if h is not None else 1)).sum()
            loss.backward()
            return model.weight.detach().clone(), loss.detach()
        return evaluate

    def test_independent_quadratic_oracle_all_modes_and_first_ready(self):
        for mode in ('reverse', 'sam', 'sgd'):
            for first_ready in (False, True):
                with self.subTest(mode=mode, first_ready=first_ready):
                    model, _, opt = self.make(mode, dtype=torch.float64, radius=.37, eps=.11)
                    h = torch.tensor([2., 5.], dtype=torch.float64)
                    original = model.weight.detach().clone()
                    g0 = h * original
                    sign = -1 if mode == 'reverse' else 1
                    point = original + sign * .37 * g0 / (g0.norm() + .11)
                    g1 = h * point
                    expected = {'reverse': 2 * g0 - g1, 'sam': g1, 'sgd': g0}[mode]
                    if first_ready:
                        (.5 * (model.weight.square() * h).sum()).backward()
                    points = []
                    opt.step(self.closure(model, opt, points, h), first_grad_ready=first_ready)
                    torch.testing.assert_close(model.weight, original - .07 * expected, rtol=1e-14, atol=1e-14)
                    self.assertEqual(len(points), (0 if first_ready else 1) + (mode != 'sgd'))
                    if mode != 'sgd':
                        torch.testing.assert_close(points[-1], point, rtol=1e-14, atol=1e-14)
                    stats = opt.last_stats
                    self.assertEqual(stats['probe_ce_gradient_phases'], 1 if mode == 'sgd' else 2)
                    self.assertEqual(stats['probe_replay_closure_calls'], 0 if mode == 'sgd' else 1)
                    self.assertEqual(stats['probe_base_optimizer_steps'], 1)
                    self.assertTrue(stats['probe_restore_base_exact'])
                    self.assertEqual(stats['probe_outer_probe_calls'], 0)
                    self.assertEqual(stats['probe_g2_calls'], 0)
                    self.assertEqual(stats['probe_g3_calls'], 0)
                    self.assertEqual(stats['probe_schema'], 'TwoPointProbe_v1')
                    self.assertAlmostEqual(stats['probe_submitted_minus_g0_norm'], float((expected-g0).norm()), places=13)
                    self.assertAlmostEqual(stats['probe_gradient_clip_scale'], 1.)
                    json.dumps(stats, allow_nan=False)

    def test_reverse_sam_share_first_order_and_differ_at_second_order(self):
        # Nonquadratic independent analytic gradients, not the implementation's
        # finite difference. Their difference is O(rho**2), not zero at finite rho.
        def submitted(mode, radius):
            model, _, opt = self.make(mode, values=(.9, -.7), dtype=torch.float64, radius=radius)
            before = model.weight.detach().clone()
            def closure():
                opt.zero_grad()
                loss = model.weight.exp().sum() + .25 * model.weight.pow(4).sum()
                loss.backward()
                return None, loss.detach()
            opt.step(closure)
            return (before - model.weight.detach()) / .07
        difference_big = (submitted('reverse', .02) - submitted('sam', .02)).norm()
        difference_small = (submitted('reverse', .01) - submitted('sam', .01)).norm()
        self.assertGreater(float(difference_big), 1e-6)
        self.assertAlmostEqual(float(difference_big/difference_small), 4., delta=.002)

    def test_one_hf_adamw_commit_at_restored_base_and_single_final_clip(self):
        model = Vector()
        reference = copy.deepcopy(model)
        base = HFAdamW(model.parameters(), lr=.001, eps=1e-6, weight_decay=0, no_deprecation_warning=True)
        refbase = HFAdamW(reference.parameters(), lr=.001, eps=1e-6, weight_decay=0, no_deprecation_warning=True)
        opt = TwoPointProbe(model.parameters(), base, model, radius=.05)
        h = torch.tensor([2., 5.])
        for step in range(4):
            before = reference.weight.detach().clone()
            g0 = h * before
            point = before - .05 * g0 / (float(g0.double().square().sum().sqrt()) + 1e-12)
            final = 2*g0 - h*point
            reference.weight.grad = final.clone()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), .6, error_if_nonfinite=True)
            original_step = base.step
            commits = []
            def counted_commit():
                commits.append(model.weight.detach().clone())
                return original_step()
            with patch.object(base, 'step', side_effect=counted_commit), patch(
                    'torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clipping:
                opt.step(self.closure(model, opt, h=h), clip_grad_norm=.6)
            self.assertEqual(clipping.call_count, 1)
            self.assertEqual(len(commits), 1)
            self.assertTrue(torch.equal(commits[0], before))
            refbase.step()
            self.assertTrue(torch.equal(model.weight, reference.weight))
            assert_state_equal(self, base.state_dict(), refbase.state_dict())
            self.assertEqual(opt.real_updates, step+1)
            self.assertLess(opt.last_stats['probe_gradient_clip_scale'], 1.)

    def test_probe_radius_independent_of_commit_learning_rate(self):
        points = []
        for eta in (.1, .001):
            model, base, opt = self.make(radius=.05)
            base.param_groups[0]['lr'] = eta
            seen = []
            opt.step(self.closure(model, opt, seen))
            points.append(seen[-1])
        self.assertTrue(torch.equal(*points))

    def test_zero_radius_still_replays_and_is_plain_clean_gradient(self):
        for mode in ('reverse', 'sam'):
            model, _, opt = self.make(mode, radius=0)
            before = model.weight.detach().clone()
            points = []
            opt.step(self.closure(model, opt, points))
            self.assertEqual(len(points), 2)
            self.assertTrue(torch.equal(points[0], points[1]))
            torch.testing.assert_close(model.weight, before * .93)
            self.assertEqual(opt.last_stats['probe_radius_actual'], 0)
            self.assertEqual(opt.last_stats['probe_response_norm'], 0)
            self.assertIsNone(opt.last_stats['probe_fallback_reason'])
            self.assertIsNone(opt.last_stats['probe_actual_update_over_probe_radius'])

    def test_zero_gradient_is_finite_same_point_probe(self):
        model, _, opt = self.make(values=(0., 0.))
        opt.step(self.closure(model, opt))
        self.assertEqual(opt.last_stats['probe_ce_gradient_phases'], 2)
        self.assertEqual(opt.last_stats['probe_actual_update_norm'], 0)
        self.assertIsNone(opt.last_stats['probe_cos_g0_g1'])
        self.assertIsNone(opt.last_stats['probe_response_relative_norm'])
        json.dumps(opt.last_stats, allow_nan=False)

    def test_missing_in_both_phases_stays_none_and_has_no_adamw_state(self):
        model = Vector()
        model.unused = torch.nn.Parameter(torch.tensor([2.]))
        base = HFAdamW(model.parameters(), lr=.01, weight_decay=.1, no_deprecation_warning=True)
        opt = TwoPointProbe(model.parameters(), base, model)
        opt.step(self.closure(model, opt))
        self.assertIsNone(model.unused.grad)
        self.assertEqual(float(model.unused), 2.)
        self.assertNotIn(model.unused, base.state)

    def test_reverse_union_grad_presence_preserves_base_only_coordinate(self):
        model = Vector(values=(1., 2.))
        model.extra = torch.nn.Parameter(torch.tensor([3.]))
        base = torch.optim.SGD(model.parameters(), lr=.1)
        opt = TwoPointProbe(model.parameters(), base, model)
        calls = 0
        def closure():
            nonlocal calls
            calls += 1
            opt.zero_grad()
            loss = model.weight.square().sum()
            if calls == 1:
                loss = loss + model.extra.square().sum()
            loss.backward()
            return None, loss.detach()
        opt.step(closure)
        self.assertIsNotNone(model.extra.grad)
        self.assertAlmostEqual(float(model.extra.grad), 12.)
        self.assertAlmostEqual(float(model.extra), 1.8, places=6)

    def test_standalone_replays_python_numpy_and_dropout_rng_only_base_advances(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.5))
        base = torch.optim.SGD(model.parameters(), lr=.1)
        opt = TwoPointProbe(model.parameters(), base, model)
        x = torch.tensor([[1., 2., 3.], [-.1, .7, .4]])
        starts, ends = [], []
        initial = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        def closure():
            starts.append((random.getstate(), np.random.get_state(), torch.get_rng_state().clone()))
            random.random()
            np.random.random()
            opt.zero_grad()
            out = model(x)
            loss = out.square().mean()
            loss.backward()
            ends.append((random.getstate(), np.random.get_state(), torch.get_rng_state().clone()))
            return out.detach(), loss.detach()
        opt.step(closure)
        self.assertEqual(len(starts), 2)
        for start in starts:
            self.assertEqual(start[0], initial[0])
            self.assertTrue(np.array_equal(start[1][1], initial[1][1]))
            self.assertTrue(torch.equal(start[2], initial[2]))
        self.assertEqual(random.getstate(), ends[0][0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], ends[0][1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), ends[0][2]))

    def test_first_ready_microbatch_replay_does_not_advance_outer_rng(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.3))
        opt = TwoPointProbe(model.parameters(), torch.optim.SGD(model.parameters(), lr=.1), model)
        batches = [torch.tensor([[1., 2., 3.]]), torch.tensor([[-.1, .7, .4]])]
        cached = []
        for x in batches:
            cached.append(torch.get_rng_state().clone())
            (model(x).square().mean()/2).backward()
        afterbase = torch.get_rng_state().clone()
        starts = []
        def closure():
            opt.zero_grad()
            loss_sum = torch.tensor(0.)
            for x, rng in zip(batches, cached):
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(rng)
                    starts.append(torch.get_rng_state().clone())
                    loss = model(x).square().mean()/2
                    loss.backward()
                    loss_sum += loss.detach()
            return None, loss_sum
        opt.step(closure, first_grad_ready=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), afterbase))
        self.assertEqual(len(starts), 2)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(starts, cached)))

    def test_probe_failure_restores_trainables_grads_rng_buffers_flags_and_cache(self):
        model, base, opt = self.make()
        model.register_buffer('visits', torch.tensor(0))
        model.task_cache = {'x': torch.tensor([5.])}
        model.weight.square().sum().backward()
        before, grad = model.weight.detach().clone(), model.weight.grad.clone()
        initial_rng = torch.get_rng_state().clone()
        initial_state = copy.deepcopy(base.state_dict())
        def closure():
            model.visits.add_(1)
            model.task_cache['x'].add_(1)
            model.new_cache = 'probe'
            model.eval()
            model.weight.requires_grad_(False)
            torch.rand(3)
            raise RuntimeError('injected failure')
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            opt.step(closure, first_grad_ready=True)
        self.assertTrue(torch.equal(model.weight, before))
        self.assertTrue(torch.equal(model.weight.grad, grad))
        self.assertTrue(torch.equal(torch.get_rng_state(), initial_rng))
        self.assertEqual(int(model.visits), 0)
        self.assertEqual(float(model.task_cache['x']), 5.)
        self.assertTrue(model.training)
        self.assertTrue(model.weight.requires_grad)
        self.assertFalse(hasattr(model, 'new_cache'))
        assert_state_equal(self, base.state_dict(), initial_state)
        self.assertEqual(opt.real_updates, 0)

    def test_commit_failure_rolls_back_adamw_moments_and_base_parameters(self):
        model = Vector()
        base = HFAdamW(model.parameters(), lr=.001, no_deprecation_warning=True)
        opt = TwoPointProbe(model.parameters(), base, model)
        opt.step(self.closure(model, opt))
        old_params = model.weight.detach().clone()
        old_state = copy.deepcopy(base.state_dict())
        old_grad = model.weight.grad.clone()
        old_rng = torch.get_rng_state().clone()
        old_stats = copy.deepcopy(opt.last_stats)
        original_step = base.step
        def failing_step():
            original_step()
            torch.rand(2)
            raise RuntimeError('commit failed after mutating')
        with patch.object(base, 'step', side_effect=failing_step):
            with self.assertRaisesRegex(RuntimeError, 'commit failed'):
                opt.step(self.closure(model, opt))
        self.assertTrue(torch.equal(model.weight, old_params))
        self.assertTrue(torch.equal(model.weight.grad, old_grad))
        self.assertTrue(torch.equal(torch.get_rng_state(), old_rng))
        assert_state_equal(self, base.state_dict(), old_state)
        assert_state_equal(self, opt.last_stats, old_stats)
        self.assertEqual(opt.real_updates, 1)

    def test_nonfinite_commit_rolls_back_instead_of_counting_real_update(self):
        model, base, opt = self.make()
        before = model.weight.detach().clone()
        def overflowing_step():
            with torch.no_grad():
                model.weight.fill_(float('inf'))
        with patch.object(base, 'step', side_effect=overflowing_step):
            with self.assertRaises(FloatingPointError):
                opt.step(self.closure(model, opt))
        self.assertTrue(torch.equal(model.weight, before))
        self.assertEqual(opt.real_updates, 0)

    def test_nonfinite_or_absent_gradient_rejected_before_commit(self):
        for failure in ('nonfinite', 'absent', 'loss'):
            with self.subTest(failure=failure):
                model, base, opt = self.make()
                before = model.weight.detach().clone()
                def closure():
                    opt.zero_grad()
                    if failure == 'nonfinite':
                        model.weight.grad = torch.full_like(model.weight, float('inf'))
                    return None, float('nan') if failure == 'loss' else 1.
                with patch.object(base, 'step') as commit:
                    with self.assertRaises((RuntimeError, FloatingPointError)):
                        opt.step(closure)
                self.assertFalse(commit.called)
                self.assertTrue(torch.equal(model.weight, before))

    def test_frozen_mutation_rejected_with_no_commit(self):
        model, base, opt = self.make()
        model.frozen = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)
        # Reconstruct transaction layout to include frozen immutability guard.
        def closure():
            with torch.no_grad():
                model.frozen.add_(1)
            opt.zero_grad()
            loss = model.weight.square().sum()
            loss.backward()
            return None, loss.detach()
        with patch.object(base, 'step') as commit:
            with self.assertRaisesRegex(AssertionError, 'frozen parameter'):
                opt.step(closure)
        self.assertFalse(commit.called)
        self.assertEqual(opt.real_updates, 0)

    def test_closure_cannot_mutate_trainables(self):
        model, _, opt = self.make()
        before = model.weight.detach().clone()
        def closure():
            opt.zero_grad()
            loss = model.weight.square().sum()
            loss.backward()
            with torch.no_grad():
                model.weight.add_(1)
            return None, loss.detach()
        with self.assertRaisesRegex(AssertionError, 'mutated trainable'):
            opt.step(closure)
        self.assertTrue(torch.equal(model.weight, before))

    def test_state_dict_resume_and_cross_mode_rejection(self):
        model = Vector()
        base = HFAdamW(model.parameters(), lr=.001, no_deprecation_warning=True)
        opt = TwoPointProbe(model.parameters(), base, model, quantization_diagnostics_interval=20)
        opt.step(self.closure(model, opt))
        state = copy.deepcopy(opt.state_dict())
        copied = copy.deepcopy(model)
        copied_base = HFAdamW(copied.parameters(), lr=.001, no_deprecation_warning=True)
        resumed = TwoPointProbe(copied.parameters(), copied_base, copied, quantization_diagnostics_interval=0)
        resumed.load_state_dict(state)
        self.assertEqual(resumed.real_updates, 1)
        for _ in range(3):
            opt.step(self.closure(model, opt))
            resumed.step(self.closure(copied, resumed))
            self.assertTrue(torch.equal(model.weight, copied.weight))
            assert_state_equal(self, base.state_dict(), copied_base.state_dict())
        other = TwoPointProbe(copied.parameters(), copied_base, copied, mode='sam')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            other.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, 'metadata'):
            resumed.load_state_dict(base.state_dict())

    def test_bf16_autocast_forward_fp32_master_and_quantization_cadence(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Dropout(.2), torch.nn.Linear(4, 2))
        base = HFAdamW(model.parameters(), lr=.001, eps=1e-6, weight_decay=0, no_deprecation_warning=True)
        opt = TwoPointProbe(model.parameters(), base, model, quantization_diagnostics_interval=2)
        x = torch.tensor([[.3, -.2, .9], [1., .7, -.4]])
        sampled = []
        for _ in range(4):
            def closure():
                opt.zero_grad()
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    out = model(x)
                    loss = out.square().mean()
                loss.backward()
                return out.detach(), loss.detach()
            opt.step(closure, clip_grad_norm=1.)
            proxy = opt.last_stats['probe_quantization_proxy']
            sampled.append(proxy['sampled'])
            self.assertFalse(proxy['proves_forward_visibility'])
            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
            json.dumps(opt.last_stats, allow_nan=False)
        self.assertEqual(sampled, [True, False, True, False])

    def test_quantization_diagnostics_do_not_change_math_or_rng(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.3))
        copied = copy.deepcopy(model)
        opt = TwoPointProbe(model.parameters(), HFAdamW(model.parameters(), lr=.001, no_deprecation_warning=True), model)
        checked = TwoPointProbe(copied.parameters(), HFAdamW(copied.parameters(), lr=.001, no_deprecation_warning=True), copied,
                                quantization_diagnostics_interval=1)
        x = torch.tensor([[.3, -.2, .9]])
        rng = torch.get_rng_state().clone()
        for _ in range(4):
            for module, optimizer in ((model, opt), (copied, checked)):
                torch.set_rng_state(rng)
                def closure():
                    optimizer.zero_grad()
                    loss = module(x).square().mean()
                    loss.backward()
                    return None, loss.detach()
                optimizer.step(closure)
                if optimizer is opt:
                    after = torch.get_rng_state().clone()
                else:
                    self.assertTrue(torch.equal(torch.get_rng_state(), after))
            for a, b in zip(model.parameters(), copied.parameters()):
                self.assertTrue(torch.equal(a, b))
            assert_state_equal(self, opt.base_optimizer.state_dict(), checked.base_optimizer.state_dict())
            rng = after

    def test_invalid_inputs_and_pending_checkpoint_rejected(self):
        for options in ({'mode': 'g2'}, {'radius': -1}, {'radius': float('nan')},
                        {'eps': 0}, {'grad_reduce': 'sum'}, {'quantization_diagnostics_interval': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make(**options)
        model, _, opt = self.make()
        opt._busy = True
        with self.assertRaises(RuntimeError):
            opt.state_dict()
        with self.assertRaises(RuntimeError):
            opt.load_state_dict({})
        opt._busy = False
        with self.assertRaises(ValueError):
            opt.step(None)
        model.weight.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'layout changed'):
            opt.step(lambda: (None, 1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
