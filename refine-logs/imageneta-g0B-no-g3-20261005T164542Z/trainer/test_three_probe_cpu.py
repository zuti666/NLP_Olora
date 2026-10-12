"""Independent analytic and stochastic acceptance tests; CUDA is disabled."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from audit_core import rng_equal, rng_get, rng_set
from direction_step import ThreeProbeStep
from effective_step import EffectiveStep
from state_transaction import State
from three_probe_train import Controller, resolve_config

LEGACY = Path('/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/trainer')
for name, path in [('response_hvp', LEGACY/'response_hvp.py'), ('legacy_four_probe', LEGACY/'direction_step.py')]:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
FourProbeStep = sys.modules['legacy_four_probe'].FourProbeStep


def flat(values):
    return torch.cat([v.reshape(-1) for v in values])


class Toy(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        gen = torch.Generator().manual_seed(913)
        self.a = torch.nn.Parameter(torch.randn(2, 3, generator=gen, dtype=dtype)*.2)
        self.b = torch.nn.Parameter(torch.randn(4, 2, generator=gen, dtype=dtype)*.2)
        self.fc = torch.nn.Linear(4, 3, dtype=dtype)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(3, 4, generator=gen, dtype=dtype)*.2)
            self.fc.bias.copy_(torch.randn(3, generator=gen, dtype=dtype)*.2)
        self.frozen = torch.nn.Parameter(torch.ones(2, dtype=dtype), requires_grad=False)
        self.register_buffer('probe_counter', torch.zeros((), dtype=dtype))
        self.dropout = torch.nn.Dropout(.4)
        self._register_delta_hook = False
        self.delta_w_q_new_grad = None
        self.external_delta_w_q = None
        self.forward_cache = {'entry': torch.ones(1, dtype=dtype)}


def make_case(dtype=torch.float64, reverse=False):
    model = Toy(dtype)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    if reverse:
        params = params[::-1]
    geometry = EffectiveStep([(model.a, model.b)], tuple(model.fc.parameters()), params)
    n = sum(p.numel() for p in params)
    gen = torch.Generator().manual_seed(177)
    matrix = torch.randn(n, n, generator=gen, dtype=dtype)
    H = matrix.T@matrix/n + torch.diag(torch.linspace(.3, 2., n, dtype=dtype))
    linear = torch.linspace(-.2, .4, n, dtype=dtype)
    calls = []

    def closure(_x, _y):
        theta = flat(params)
        calls.append(theta.detach().clone())
        return .5*theta@(H@theta)-linear@theta, theta[:3].unsqueeze(0)

    return model, params, geometry, H, linear, closure, calls


class Acceptance(unittest.TestCase):
    def assert_close(self, actual, expected, atol=1e-11, rtol=1e-11):
        actual = flat(actual) if isinstance(actual, (list, tuple)) else actual
        expected = flat(expected) if isinstance(expected, (list, tuple)) else expected
        self.assertTrue(torch.allclose(actual, expected, atol=atol, rtol=rtol),
                        'max error='+str(float((actual-expected).abs().max())))

    def test_analytic_quadratic_locations_projection_sum_and_natural_step(self):
        for radius in (.25, 1.):
            for branch, direction in (('P', 'g0_plus_B'), ('C', 'A')):
                for reverse in (False, True):
                    model, params, geom, H, b, closure, calls = make_case(reverse=reverse)
                    base = tuple(p.detach().clone() for p in params)
                    theta = flat(base)
                    g0 = H@theta-b
                    sign = 1 if branch == 'P' else -1
                    point1 = theta + sign*.05*g0/(torch.linalg.vector_norm(g0)+1e-12)
                    g1 = H@point1-b
                    q = sign*(g1-g0)
                    point2 = theta + radius*q/(torch.linalg.vector_norm(q)+1e-12)
                    g2 = H@point2-b
                    alpha = torch.dot(g1, g2-g1)/(torch.dot(g1, g1)+1e-12)
                    expected = g0+g2-alpha*g1 if branch == 'P' else g0+g2-g1
                    engine = ThreeProbeStep(model, params, geom, dict(branch=branch, direction=direction,
                                                                      rho=.05, radius=radius))
                    optimizer = torch.optim.SGD(params, lr=.04)
                    original_grad = torch.autograd.grad
                    original_step = optimizer.step
                    with patch('torch.autograd.grad', wraps=original_grad) as gradients, \
                            patch.object(optimizer, 'step', wraps=original_step) as updates:
                        result = engine.compute(None, None, closure, .04)
                        self.assertEqual(gradients.call_count, 3)
                        self.assertTrue(all(c.kwargs['create_graph'] is False for c in gradients.call_args_list))
                        self.assertEqual(updates.call_count, 0)
                        self.assertEqual(len(calls), 3)
                        self.assert_close(calls[0], theta)
                        self.assert_close(calls[1], point1)
                        self.assert_close(calls[2], point2)
                        self.assert_close(result['U'], expected)
                        self.assertTrue(all(torch.equal(p, v) for p, v in zip(params, base)))
                        self.assertEqual(result['record']['ordinary_gradient_calls'], 3)
                        self.assertEqual(result['record']['radius'], radius)
                        self.assertEqual(result['record']['g3_gradient_calls'], 0)
                        self.assertEqual(result['record']['opposite_g1_gradient_calls'], 0)
                        if branch == 'P':
                            projection = result['record']['projection']
                            self.assertAlmostEqual(projection['alpha'], float(alpha), places=12)
                            self.assertLess(projection['projection_relative_error'], 1e-12)
                            # Geometric property: the added B residual preserves
                            # the component of g1 along its own direction.
                            residual = flat(result['U'])-g0-g1
                            self.assertLess(abs(float(torch.dot(g1, residual))), 1e-11)
                            offsets = np.cumsum([0]+[p.numel() for p in params])
                            local = torch.cat([g0[l:r]+g2[l:r]-g1[l:r]*(
                                torch.dot(g1[l:r], g2[l:r]-g1[l:r])/(torch.dot(g1[l:r], g1[l:r])+1e-12))
                                for l, r in zip(offsets[:-1], offsets[1:])])
                            self.assertGreater(float(torch.linalg.vector_norm(local-expected)), 1e-3)
                        else:
                            self.assertIsNone(result['record']['projection'])
                        update = engine.apply_update(optimizer=optimizer)
                        self.assertEqual(updates.call_count, 1)
                        self.assertEqual(update['sgd_updates'], 1)
                        self.assertTrue(update['natural_sgd_exact'])
                        self.assert_close(tuple(p.detach() for p in params), theta-.04*expected)
                        self.assertEqual(len(optimizer.state), 0)

    def stochastic(self, model, params, draws):
        def closure(_x, _y):
            theta = flat(params)
            mask = model.dropout(torch.ones_like(theta))
            draws.append((random.random(), float(np.random.rand()), torch.rand(5), mask.detach().clone()))
            model.probe_counter.add_(1)
            model.dropout.training = False
            model.delta_w_q_new_grad = torch.ones(2)
            model.external_delta_w_q = torch.ones(3)
            model._register_delta_hook = True
            model.forward_cache['entry'].add_(2)
            model.forward_cache['created'] = torch.ones(2)
            model.created_cache = [torch.ones(1)]
            # Nonlinear curvature distinguishes positive and negative first probes.
            loss = ((theta*mask).square()+.2*(theta*mask).pow(4)).sum()
            return loss, theta[:3].unsqueeze(0)
        return closure

    def test_rng_replay_full_state_rollback_and_only_one_rng_advance(self):
        model, params, geom, *_ = make_case()
        model.train()
        state = State(model)
        reference_draws = []
        self.stochastic(model, params, reference_draws)(None, None)
        expected_rng1 = rng_get()
        state.restore()
        draws = []
        engine = ThreeProbeStep(model, params, geom, dict(branch='P', direction='g0_plus_B', rho=.05, radius=1.))
        out = engine.compute(None, None, self.stochastic(model, params, draws), .04)
        self.assertEqual(len(draws), 3)
        for draw in draws[1:]:
            self.assertEqual(draw[:2], draws[0][:2])
            self.assertTrue(torch.equal(draw[2], draws[0][2]))
            self.assertTrue(torch.equal(draw[3], draws[0][3]))
        self.assertTrue(rng_equal(rng_get(), out['context']['rng1']))
        self.assertTrue(rng_equal(rng_get(), expected_rng1))
        self.assertTrue(all(torch.equal(p.grad, u) for p, u in zip(params, out['U'])))
        for p, grad in state.grads:
            p.grad = None if grad is None else grad.clone()
        rng_set(state.rng)
        self.assertTrue(all(state.check().values()))
        self.assertFalse(hasattr(model, 'created_cache'))
        engine.discard()

    def test_three_probe_ac_equivalent_to_legacy_four_probe_stochastic_protocol(self):
        for radius in (.25, 1.):
            model, params, geom, *_ = make_case()
            state = State(model)
            fresh_draws = []
            fresh = ThreeProbeStep(model, params, geom, dict(branch='C', direction='A', rho=.05, radius=radius))
            new = fresh.compute(None, None, self.stochastic(model, params, fresh_draws), .04)
            new_rng = rng_get()
            state.restore()
            legacy_draws = []
            legacy = FourProbeStep(model, params, geom, dict(branch='C', direction='A', rho=.05, radius=radius))
            old = legacy.compute(None, None, self.stochastic(model, params, legacy_draws), .04)
            self.assertEqual(len(fresh_draws), 3)
            self.assertEqual(len(legacy_draws), 4)
            self.assert_close(new['U'], old['U'], atol=1e-10, rtol=1e-10)
            self.assert_close(new['context']['points']['phi2'], old['context']['points']['phi2'], atol=1e-10, rtol=1e-10)
            self.assert_close(new['context']['grads']['g1'], old['context']['grads']['g1_minus'], atol=1e-10, rtol=1e-10)
            self.assertTrue(rng_equal(new_rng, rng_get()))
            self.assertTrue(all(torch.equal(p, v) for p, v in state.trainable))
            self.assertTrue(new['record']['base_buffers_exact'])
            self.assertTrue(old['record']['base_buffers_exact'])
            fresh.discard()
            legacy.discard()
            state.restore()

    def test_exception_restores_entry_parameters_gradients_buffers_caches_and_rng(self):
        for failure_probe in (1, 2, 3):
            model, params, geom, *_ = make_case()
            for p in params:
                p.grad = torch.ones_like(p)*3
            state = State(model)
            draws = []
            stochastic = self.stochastic(model, params, draws)
            def failing(x, y):
                result = stochastic(x, y)
                if len(draws) == failure_probe:
                    raise RuntimeError('intentional probe failure')
                return result
            engine = ThreeProbeStep(model, params, geom, dict(branch='P', direction='g0_plus_B', rho=.05, radius=.25))
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                engine.compute(None, None, failing, .04)
            self.assertTrue(all(state.check().values()))
            with self.assertRaisesRegex(RuntimeError, 'compute must precede'):
                engine.apply_update()

    def test_update_exception_rolls_back_and_duplicate_step_is_rejected(self):
        model, params, geom, _, _, closure, _ = make_case()
        base = tuple(p.detach().clone() for p in params)
        engine = ThreeProbeStep(model, params, geom, dict(branch='C', direction='A', rho=.05, radius=1.))
        optimizer = torch.optim.SGD(params, lr=.04)
        native = optimizer.step
        def failing_step(*args, **kwargs):
            native(*args, **kwargs)
            raise RuntimeError('intentional SGD failure')
        engine.compute(None, None, closure, .04)
        with patch.object(optimizer, 'step', side_effect=failing_step):
            with self.assertRaisesRegex(RuntimeError, 'intentional SGD'):
                engine.apply_update(optimizer=optimizer)
        self.assertTrue(all(torch.equal(p, v) for p, v in zip(params, base)))
        with self.assertRaisesRegex(RuntimeError, 'compute must precede'):
            engine.apply_update(optimizer=optimizer)

    def test_effective_config_radius_reaches_actual_controller_stepper(self):
        raw = json.loads((LEGACY.parent/'configs/S1-09_A_C_natural_s2.json').read_text())
        for radius in (.25, 1.):
            for branch, direction in (('P', 'g0_plus_B'), ('C', 'A')):
                with tempfile.TemporaryDirectory() as output:
                    cli = SimpleNamespace(max_tasks=1, max_epochs=1, max_batches=1, output_dir=output)
                    candidate = copy.deepcopy(raw)
                    candidate.update(gam_grad_norm_rho=radius, probe_branch=branch, update_direction=direction)
                    cfg = resolve_config(candidate, cli)
                    self.assertEqual(cfg['gam_grad_norm_rho'], radius)
                    cfg['seed'] = 2  # The native trainer resolves its single seed before factory call.
                    model, params, geom, _, _, closure, calls = make_case()
                    dummy = lambda *args, **kwargs: None
                    learner = SimpleNamespace(_unwrap_network=lambda: model, _run_regularized_train=dummy,
                                              incremental_train=dummy, eval_task=dummy, after_task=dummy)
                    controller = Controller(learner, cfg, cli)
                    controller.task = 0
                    controller.old_fc = {}
                    try:
                        with patch('three_probe_train.native.install_forbidden_work_guards'), \
                             patch.object(EffectiveStep, 'from_model', return_value=geom), \
                             patch.object(controller, 'save_observation'):
                            controller.install()
                            learner._build_optimizer(params, 'init')
                        self.assertEqual(controller.stepper.radius, radius)
                        result = controller.stepper.compute(None, None, closure, .04)
                        displacement = torch.linalg.vector_norm(calls[2]-calls[0])
                        self.assertAlmostEqual(float(displacement), radius, places=9)
                        self.assertEqual(result['record']['radius'], radius)
                        controller.stepper.discard()
                    finally:
                        controller.handle.close()

    def test_fp32_projection_and_pure_sgd_guards(self):
        for radius in (.25, 1.):
            model, params, geom, _, _, closure, _ = make_case(torch.float32, reverse=True)
            engine = ThreeProbeStep(model, params, geom, dict(branch='P', direction='g0_plus_B', rho=.05, radius=radius))
            out = engine.compute(None, None, closure, .04)
            self.assertEqual(out['record']['projection']['reduction_dtype'], 'float64')
            self.assertLess(out['record']['projection']['projection_relative_error'], 1e-5)
            bad = torch.optim.SGD(params, lr=.04, momentum=.9)
            with self.assertRaisesRegex(ValueError, 'zero momentum'):
                engine.apply_update(optimizer=bad)
            good = torch.optim.SGD(params, lr=.04)
            self.assertTrue(engine.apply_update(optimizer=good)['natural_sgd_exact'])


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run acceptance with CUDA_VISIBLE_DEVICES=""')
    torch.set_num_threads(1)
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    output = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), device='cpu',
                  cuda_visible_devices='', radii=[.25, 1.],
                  directions=['P/g0_plus_B', 'C/A'],
                  legacy_AC_stochastic_equivalence_tolerance=1e-10)
    Path(__file__).with_name('CPU_GATE.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))
    raise SystemExit(0 if result.wasSuccessful() else 1)
