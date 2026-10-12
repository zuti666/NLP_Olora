"""Independent nonlinear oracles and paired-probe transaction invariants."""
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
from optimer_three_point_probe import ThreePointProbe


class Vector(torch.nn.Module):
    def __init__(self, values=(.9, -.7), dtype=torch.float32):
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


class ThreePointProbeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(619)
        np.random.seed(619)
        random.seed(619)

    def make(self, mode='central', dtype=torch.float32, values=(.9, -.7), **kwargs):
        model = Vector(values, dtype)
        base = torch.optim.SGD(model.parameters(), lr=.07)
        opt = ThreePointProbe(model.parameters(), base, model, mode=mode,
                              require_fp32_trainable=dtype == torch.float32, **kwargs)
        return model, base, opt

    @staticmethod
    def closure(model, optimizer, points=None, quadratic=False):
        def evaluate():
            if points is not None:
                points.append(model.weight.detach().clone())
            optimizer.zero_grad()
            loss = (.5 * (model.weight.square()*model.weight.new_tensor([2., 5.])).sum()
                    if quadratic else model.weight.exp().sum()+.25*model.weight.pow(4).sum())
            loss.backward()
            return model.weight.detach().clone(), loss.detach()
        return evaluate

    @staticmethod
    def analytic_gradient(point):
        return point.exp()+point.pow(3)

    def test_nonlinear_independent_oracle_all_four_modes_and_first_ready(self):
        for mode in ThreePointProbe.MODES:
            for first_ready in (False, True):
                with self.subTest(mode=mode, first_ready=first_ready):
                    model, _, opt = self.make(mode, dtype=torch.float64, radius=.37, eps=.11)
                    original = model.weight.detach().clone()
                    g0 = self.analytic_gradient(original)
                    plus = original+.37*g0/(g0.norm()+.11)
                    minus = original-.37*g0/(g0.norm()+.11)
                    gp, gm = self.analytic_gradient(plus), self.analytic_gradient(minus)
                    final = {'g1_plus': gp, 'backward_reflected': 2*g0-gm,
                             'central': g0+.5*(gp-gm), 'g0': g0}[mode]
                    if first_ready:
                        self.closure(model, opt)()
                    points = []
                    opt.step(self.closure(model, opt, points), first_grad_ready=first_ready)
                    torch.testing.assert_close(model.weight, original-.07*final, rtol=1e-14, atol=1e-14)
                    torch.testing.assert_close(points[-2], plus, rtol=1e-14, atol=1e-14)
                    torch.testing.assert_close(points[-1], minus, rtol=1e-14, atol=1e-14)
                    self.assertEqual(len(points), 2 if first_ready else 3)
                    stats = opt.last_stats
                    self.assertEqual(stats['probe_schema'], 'ThreePointProbe_v1')
                    self.assertEqual(stats['probe_protocol'], 'paired_three_point')
                    self.assertEqual(stats['probe_signs'], [1, -1])
                    self.assertEqual(stats['probe_ce_gradient_phases'], 3)
                    self.assertEqual(stats['probe_replay_closure_calls'], 2)
                    self.assertEqual(stats['probe_base_optimizer_steps'], 1)
                    self.assertEqual(stats['probe_mode'], mode)
                    self.assertTrue(stats['probe_restore_base_exact'])
                    for key in ('probe_outer_probe_calls', 'probe_g2_calls', 'probe_g3_calls',
                                'probe_fisher_probe_calls', 'probe_projection_calls', 'probe_hvp_calls'):
                        self.assertEqual(stats[key], 0)
                    self.assertAlmostEqual(stats['probe_symmetric_response_norm'], float((.5*(gp-gm)).norm()), places=13)
                    self.assertAlmostEqual(stats['probe_even_response_norm'], float((.5*(gp+gm-2*g0)).norm()), places=13)
                    json.dumps(stats, allow_nan=False)

    def test_central_is_mean_of_forward_and_reflected_raw_directions(self):
        directions = {}
        stats = {}
        for mode in ('g1_plus', 'backward_reflected', 'central'):
            model, _, opt = self.make(mode, dtype=torch.float64, radius=.2)
            before = model.weight.detach().clone()
            opt.step(self.closure(model, opt))
            directions[mode] = (before-model.weight.detach())/.07
            stats[mode] = opt.last_stats
        torch.testing.assert_close(directions['central'],
                                   .5*(directions['g1_plus']+directions['backward_reflected']),
                                   rtol=1e-13, atol=1e-13)
        self.assertGreater(stats['central']['probe_even_response_norm'], 1e-4)

    def test_central_cancels_even_curvature_error_before_adamw(self):
        # Analytic Hu is exp(theta)*u+3*theta**2*u. For the central
        # gradient correction its error is O(rho**3), versus O(rho**2)
        # for either side. This tests cancellation, not a training claim.
        errors = []
        for radius in (.02, .01):
            model, _, opt = self.make('central', dtype=torch.float64, radius=radius)
            original = model.weight.detach().clone()
            g0 = self.analytic_gradient(original)
            u = g0/(g0.norm()+1e-12)
            first_order = g0+radius*(original.exp()+3*original.square())*u
            opt.step(self.closure(model, opt))
            submitted = (original-model.weight.detach())/.07
            errors.append(float((submitted-first_order).norm()))
        self.assertGreater(errors[0], 1e-8)
        self.assertAlmostEqual(errors[0]/errors[1], 8., delta=.002)

    def test_one_hf_adamw_commit_restored_base_and_one_final_clip_all_modes(self):
        for mode in ThreePointProbe.MODES:
            with self.subTest(mode=mode):
                model, reference = Vector(), Vector()
                base = HFAdamW(model.parameters(), lr=.001, eps=1e-6, weight_decay=0,
                               no_deprecation_warning=True)
                refbase = HFAdamW(reference.parameters(), lr=.001, eps=1e-6, weight_decay=0,
                                  no_deprecation_warning=True)
                opt = ThreePointProbe(model.parameters(), base, model, mode=mode)
                h = torch.tensor([2., 5.])
                for step in range(3):
                    original = reference.weight.detach().clone()
                    g0 = original*h
                    scale = .05/(float(g0.double().square().sum().sqrt())+1e-12)
                    gp = original.clone().add_(g0, alpha=scale)*h
                    gm = original.clone().add_(g0, alpha=-scale)*h
                    final = {'g1_plus':gp, 'backward_reflected':2*g0-gm,
                             'central':g0+.5*(gp-gm), 'g0':g0}[mode]
                    reference.weight.grad = final.clone()
                    torch.nn.utils.clip_grad_norm_(reference.parameters(), .6, error_if_nonfinite=True)
                    real_step, commits = base.step, []
                    def counted_commit():
                        commits.append(model.weight.detach().clone())
                        return real_step()
                    with patch.object(base, 'step', side_effect=counted_commit), patch(
                            'torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
                        opt.step(self.closure(model, opt, quadratic=True), clip_grad_norm=.6)
                    self.assertEqual(clip.call_count, 1)
                    self.assertEqual(len(commits), 1)
                    self.assertTrue(torch.equal(commits[0], original))
                    refbase.step()
                    self.assertTrue(torch.equal(model.weight, reference.weight))
                    assert_state_equal(self, base.state_dict(), refbase.state_dict())
                    self.assertEqual(opt.real_updates, step+1)

    def test_fixed_probe_radius_and_points_independent_of_commit_lr_and_mode(self):
        all_points = []
        for mode in ThreePointProbe.MODES:
            for lr in (.04, .0001):
                model, base, opt = self.make(mode, radius=.05)
                base.param_groups[0]['lr'] = lr
                points = []
                opt.step(self.closure(model, opt, points))
                all_points.append(points)
                self.assertAlmostEqual(opt.last_stats['probe_plus_radius_actual'], .05, places=7)
                self.assertAlmostEqual(opt.last_stats['probe_minus_radius_actual'], .05, places=7)
        for points in all_points[1:]:
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(all_points[0], points)))

    def test_zero_radius_keeps_three_phases_and_reduces_to_g0_all_modes(self):
        for mode in ThreePointProbe.MODES:
            model, _, opt = self.make(mode, radius=0)
            original = model.weight.detach().clone()
            points = []
            opt.step(self.closure(model, opt, points, quadratic=True))
            self.assertEqual(len(points), 3)
            self.assertTrue(all(torch.equal(p, original) for p in points))
            torch.testing.assert_close(model.weight, original-.07*original*original.new_tensor([2., 5.]))
            self.assertEqual(opt.last_stats['probe_ce_gradient_phases'], 3)
            self.assertEqual(opt.last_stats['probe_side_difference_norm'], 0)
            self.assertIsNone(opt.last_stats['probe_fallback_reason'])
            self.assertIsNone(opt.last_stats['probe_actual_update_over_probe_radius'])

    def test_zero_gradient_keeps_three_finite_phases(self):
        for mode in ThreePointProbe.MODES:
            model, _, opt = self.make(mode, values=(0., 0.))
            points = []
            opt.step(self.closure(model, opt, points, quadratic=True))
            self.assertEqual(len(points), 3)
            self.assertEqual(opt.last_stats['probe_actual_update_norm'], 0)
            self.assertFalse(opt.last_stats['probe_plus_distinguishable'])
            self.assertIsNone(opt.last_stats['probe_cos_gplus_gminus'])
            json.dumps(opt.last_stats, allow_nan=False)

    def test_gradient_presence_masks_follow_selected_expression(self):
        expected = {'g1_plus':(False,True,False), 'backward_reflected':(True,False,True),
                    'central':(True,True,True), 'g0':(True,False,False)}
        for mode in ThreePointProbe.MODES:
            with self.subTest(mode=mode):
                model = Vector()
                for name in ('baseonly', 'plusonly', 'minusonly', 'unused'):
                    model.register_parameter(name, torch.nn.Parameter(torch.tensor([3.])))
                base = HFAdamW(model.parameters(), lr=.001, weight_decay=.1, no_deprecation_warning=True)
                opt = ThreePointProbe(model.parameters(), base, model, mode=mode)
                calls = 0
                def closure():
                    nonlocal calls
                    opt.zero_grad()
                    name = ('baseonly','plusonly','minusonly')[calls]
                    calls += 1
                    loss = model.weight.square().sum()+getattr(model, name).square().sum()
                    loss.backward()
                    return None, loss.detach()
                opt.step(closure)
                self.assertEqual(calls, 3)
                for name, present in zip(('baseonly','plusonly','minusonly'), expected[mode]):
                    parameter = getattr(model, name)
                    self.assertEqual(parameter.grad is not None, present)
                    self.assertEqual(parameter in base.state, present)
                    if not present:
                        self.assertEqual(float(parameter), 3.)
                self.assertIsNone(model.unused.grad)
                self.assertNotIn(model.unused, base.state)
                self.assertEqual(float(model.unused), 3.)

    def test_same_python_numpy_dropout_rng_all_three_phases_only_clean_advances(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.5))
        opt = ThreePointProbe(model.parameters(), torch.optim.SGD(model.parameters(), lr=.1), model)
        x = torch.tensor([[1., 2., 3.], [-.1, .7, .4]])
        initial = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        starts, ends = [], []
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
        self.assertEqual(len(starts), 3)
        for start in starts:
            self.assertEqual(start[0], initial[0])
            self.assertTrue(np.array_equal(start[1][1], initial[1][1]))
            self.assertTrue(torch.equal(start[2], initial[2]))
        self.assertEqual(random.getstate(), ends[0][0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], ends[0][1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), ends[0][2]))

    def test_first_ready_replays_same_microbatches_twice_without_outer_rng_advance(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.3))
        opt = ThreePointProbe(model.parameters(), torch.optim.SGD(model.parameters(), lr=.1), model)
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
        self.assertEqual(len(starts), 4)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(starts,cached+cached)))

    def test_both_sides_restore_base_buffers_and_controlled_cache(self):
        model, _, opt = self.make()
        model.register_buffer('visits', torch.tensor(0))
        model.task_cache = {'x': torch.tensor([5.])}
        snapshots = []
        def closure():
            snapshots.append((int(model.visits), float(model.task_cache['x'])))
            model.visits.add_(1)
            model.task_cache['x'].add_(2)
            return self.closure(model, opt)()
        opt.step(closure)
        self.assertEqual(snapshots, [(0,5.),(0,5.),(0,5.)])
        self.assertEqual(int(model.visits), 0)
        self.assertEqual(float(model.task_cache['x']), 5.)

    def test_second_side_failure_rolls_back_whole_transaction(self):
        model = Vector()
        model.register_buffer('visits', torch.tensor(0))
        model.task_cache = {'x':torch.tensor([5.])}
        base = HFAdamW(model.parameters(), lr=.001, no_deprecation_warning=True)
        opt = ThreePointProbe(model.parameters(), base, model)
        opt.step(self.closure(model, opt))
        before, grad, state = model.weight.detach().clone(), model.weight.grad.clone(), copy.deepcopy(base.state_dict())
        rng, stats = torch.get_rng_state().clone(), copy.deepcopy(opt.last_stats)
        calls = 0
        def closure():
            nonlocal calls
            calls += 1
            if calls == 3:
                model.visits.add_(1)
                model.task_cache['x'].add_(1)
                model.new_cache = 'mutated'
                model.eval()
                model.weight.requires_grad_(False)
                torch.rand(3)
                raise RuntimeError('minus failed')
            return self.closure(model, opt)()
        with patch.object(base, 'step') as commit:
            with self.assertRaisesRegex(RuntimeError, 'minus failed'):
                opt.step(closure)
        self.assertFalse(commit.called)
        self.assertEqual(calls, 3)
        self.assertTrue(torch.equal(model.weight, before))
        self.assertTrue(torch.equal(model.weight.grad, grad))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(model.training)
        self.assertTrue(model.weight.requires_grad)
        self.assertFalse(hasattr(model,'new_cache'))
        self.assertEqual(int(model.visits), 0)
        self.assertEqual(float(model.task_cache['x']), 5.)
        assert_state_equal(self, base.state_dict(), state)
        assert_state_equal(self, opt.last_stats, stats)
        self.assertEqual(opt.real_updates, 1)

    def test_failed_partial_adamw_commit_restores_moments_params_grad_rng(self):
        model = Vector()
        base = HFAdamW(model.parameters(), lr=.001, no_deprecation_warning=True)
        opt = ThreePointProbe(model.parameters(), base, model)
        opt.step(self.closure(model,opt))
        before, grad = model.weight.detach().clone(), model.weight.grad.clone()
        state, stats, rng = copy.deepcopy(base.state_dict()), copy.deepcopy(opt.last_stats), torch.get_rng_state().clone()
        real_step = base.step
        def failed_commit():
            real_step()
            torch.rand(3)
            raise RuntimeError('partial commit failed')
        with patch.object(base,'step',side_effect=failed_commit):
            with self.assertRaisesRegex(RuntimeError,'partial commit failed'):
                opt.step(self.closure(model,opt))
        self.assertTrue(torch.equal(model.weight,before))
        self.assertTrue(torch.equal(model.weight.grad,grad))
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        assert_state_equal(self,base.state_dict(),state)
        assert_state_equal(self,opt.last_stats,stats)
        self.assertEqual(opt.real_updates,1)

    def test_nonfinite_second_side_rejected_no_commit(self):
        model, base, opt = self.make()
        before = model.weight.detach().clone()
        calls = 0
        def closure():
            nonlocal calls
            calls += 1
            if calls == 3:
                opt.zero_grad()
                model.weight.grad = torch.full_like(model.weight,float('inf'))
                return None, 1.
            return self.closure(model,opt)()
        with patch.object(base,'step') as commit:
            with self.assertRaises(FloatingPointError):
                opt.step(closure)
        self.assertFalse(commit.called)
        self.assertTrue(torch.equal(model.weight,before))
        self.assertEqual(opt.real_updates,0)

    def test_checkpoint_roundtrip_all_modes_and_cross_protocol_rejection(self):
        for mode in ThreePointProbe.MODES:
            with self.subTest(mode=mode):
                model = Vector()
                base = HFAdamW(model.parameters(),lr=.001,no_deprecation_warning=True)
                opt = ThreePointProbe(model.parameters(),base,model,mode=mode,
                                      quantization_diagnostics_interval=20)
                opt.step(self.closure(model,opt))
                state = copy.deepcopy(opt.state_dict())
                self.assertIn('_three_point_probe',state)
                self.assertNotIn('_two_point_probe',state)
                cloned = copy.deepcopy(model)
                cloned_base = HFAdamW(cloned.parameters(),lr=.001,no_deprecation_warning=True)
                resumed = ThreePointProbe(cloned.parameters(),cloned_base,cloned,mode=mode)
                resumed.load_state_dict(state)
                self.assertEqual(resumed.real_updates,1)
                for _ in range(2):
                    opt.step(self.closure(model,opt))
                    resumed.step(self.closure(cloned,resumed))
                    self.assertTrue(torch.equal(model.weight,cloned.weight))
                    assert_state_equal(self,base.state_dict(),cloned_base.state_dict())
                other_mode = next(name for name in ThreePointProbe.MODES if name != mode)
                other = ThreePointProbe(cloned.parameters(),cloned_base,cloned,mode=other_mode)
                with self.assertRaisesRegex(ValueError,'mismatch'):
                    other.load_state_dict(state)
                with self.assertRaisesRegex(ValueError,'metadata'):
                    resumed.load_state_dict(base.state_dict())

    def test_bf16_forward_fp32_masters_per_side_proxy_cadence_all_modes(self):
        for mode in ThreePointProbe.MODES:
            model = torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Tanh(),torch.nn.Dropout(.2),torch.nn.Linear(4,2))
            base = HFAdamW(model.parameters(),lr=.001,eps=1e-6,weight_decay=0,no_deprecation_warning=True)
            opt = ThreePointProbe(model.parameters(),base,model,mode=mode,quantization_diagnostics_interval=2)
            x = torch.tensor([[.3,-.2,.9],[1.,.7,-.4]])
            sampled = []
            for _ in range(3):
                def closure():
                    opt.zero_grad()
                    with torch.autocast('cpu',dtype=torch.bfloat16):
                        out=model(x)
                        loss=out.square().mean()
                    loss.backward()
                    return out.detach(),loss.detach()
                opt.step(closure,clip_grad_norm=1.)
                proxy=opt.last_stats['probe_quantization_proxy']
                sampled.append(proxy['sampled'])
                self.assertFalse(proxy['proves_forward_visibility'])
                self.assertEqual(proxy['plus'] is not None,proxy['sampled'])
                self.assertEqual(proxy['minus'] is not None,proxy['sampled'])
                self.assertTrue(all(p.dtype==torch.float32 for p in model.parameters()))
                json.dumps(opt.last_stats,allow_nan=False)
            self.assertEqual(sampled,[True,False,True])

    def test_quantization_proxy_is_observational_math_and_rng_unchanged(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.Dropout(.3))
        cloned=copy.deepcopy(model)
        opt=ThreePointProbe(model.parameters(),HFAdamW(model.parameters(),lr=.001,no_deprecation_warning=True),model)
        checked=ThreePointProbe(cloned.parameters(),HFAdamW(cloned.parameters(),lr=.001,no_deprecation_warning=True),cloned,
                                quantization_diagnostics_interval=1)
        x=torch.tensor([[.3,-.2,.9]])
        rng=torch.get_rng_state().clone()
        for _ in range(3):
            for module,optimizer in ((model,opt),(cloned,checked)):
                torch.set_rng_state(rng)
                def closure():
                    optimizer.zero_grad()
                    loss=module(x).square().mean()
                    loss.backward()
                    return None,loss.detach()
                optimizer.step(closure)
                if optimizer is opt:
                    after=torch.get_rng_state().clone()
                else:
                    self.assertTrue(torch.equal(torch.get_rng_state(),after))
            for a,b in zip(model.parameters(),cloned.parameters()):
                self.assertTrue(torch.equal(a,b))
            assert_state_equal(self,opt.base_optimizer.state_dict(),checked.base_optimizer.state_dict())
            rng=after

    def test_invalid_options_mutated_closure_and_pending_checkpoint_rejected(self):
        for kwargs in ({'mode':'central_difference'}, {'radius':-1}, {'radius':math.inf},
                       {'eps':0}, {'grad_reduce':'sum'}, {'quantization_diagnostics_interval':True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)
        model, _, opt=self.make()
        opt._busy=True
        with self.assertRaises(RuntimeError):
            opt.state_dict()
        with self.assertRaises(RuntimeError):
            opt.load_state_dict({})
        opt._busy=False
        before=model.weight.detach().clone()
        def closure():
            result=self.closure(model,opt)()
            with torch.no_grad():
                model.weight.add_(1)
            return result
        with self.assertRaisesRegex(AssertionError,'mutated trainable'):
            opt.step(closure)
        self.assertTrue(torch.equal(model.weight,before))


if __name__=='__main__':
    unittest.main(verbosity=2)
