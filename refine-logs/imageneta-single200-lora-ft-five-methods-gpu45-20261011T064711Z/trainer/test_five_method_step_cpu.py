"""Independent mathematical, source-parity and transaction gates (CPU only)."""
import copy
import importlib
import json
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from five_method_step import FiveMethodStep, METHODS, READS, gam_decompose
from state_transaction import State, rng_equal, rng_get, rng_set


SOURCE = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')
HISTORICAL = Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-upward-eight-fork-20261010T123703Z/trainer')


class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor([.8, -.3, .4]))
        self.y = torch.nn.Parameter(torch.tensor([-.2, .7]))
        self.register_buffer('H', torch.tensor([[3., .2, 0., .1, .0],
            [.2, 2., .1, 0., .3], [0., .1, 4., .2, 0.],
            [.1, 0., .2, 1.5, .1], [0., .3, 0., .1, 2.5]]))
        self.register_buffer('linear', torch.tensor([.1, -.2, .3, -.4, .5]))

    def forward(self, inputs):
        v = torch.cat((self.x, self.y))
        return .5*v@self.H@v+self.linear@v


class Stateful(torch.nn.Module):
    def __init__(self, full_ft=True):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.eye(3), requires_grad=full_ft)
        self.a = torch.nn.Linear(3, 4)
        self.bn = torch.nn.BatchNorm1d(4)
        self.drop = torch.nn.Dropout(.35)
        self.fc = torch.nn.Linear(4, 2)
        self.register_buffer('counter', torch.zeros(()))
        self.forward_cache = None

    def forward(self, inputs):
        self.counter.add_(1.)
        value = self.drop(torch.tanh(self.bn(self.a(inputs@self.frozen))))
        self.forward_cache = value.detach().clone()
        return self.fc(value)


def closure_for(model):
    def closure(inputs, targets):
        output = model(inputs)
        loss = output if output.ndim == 0 else torch.nn.functional.cross_entropy(output, targets)
        return loss, output.reshape(1, 1) if output.ndim == 0 else output
    return closure


def split(vector, model):
    return vector[:3], vector[3:]


def oracle_shift(point, gradient, radius):
    # Independent analytic oracle: global norm across both parameter tensors.
    scale = radius/float(torch.linalg.vector_norm(gradient.double()))
    return point.clone().add_(gradient, alpha=scale)


class FiveMethodTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        np.random.seed(17)
        random.seed(17)

    def test_analytic_global_quadratic_points_native_budgets_and_real_sgd(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = Quadratic()
                params = list(model.parameters())
                base = torch.cat([p.detach() for p in params])
                grad = lambda point: model.H@point+model.linear
                g0 = grad(base)
                g1point = oracle_shift(base, g0, .05)
                gm_point = oracle_shift(base, g0, -.05)
                g1, gm = grad(g1point), grad(gm_point)
                outer = oracle_shift(base, g1-g0, 1.)
                g2 = grad(outer)
                terminal = oracle_shift(outer, g2, .05)
                g3 = grad(terminal)
                if method == 'sgd': expected = g0
                elif method == 'sam': expected = g1
                elif method == 'mirror': expected = g0+(g0-gm)
                elif method == 'g2': expected = g2
                else:
                    plus, minus = g1+g3, g0+g2
                    # Source-parity separately checks actual epsilon ordering.
                    coeff = torch.dot(plus, minus)/torch.dot(plus, plus)
                    expected = plus-.1*(minus-coeff*plus)
                stepper = FiveMethodStep(model, params, dict(method=method))
                result = stepper.compute(None, None, closure_for(model), .04)
                actual = torch.cat(result['U'])
                self.assertTrue(torch.allclose(actual, expected, atol=3e-6, rtol=3e-6))
                self.assertEqual(result['record']['ordinary_gradient_calls'], READS[method])
                self.assertEqual(result['record']['projection_calls'], int(method == 'gam'))
                self.assertTrue(all(torch.equal(p, b) for p, b in zip(params, split(base, model))))
                self.assertTrue(all(p.grad is None for p in params))
                if method in ('g2', 'gam'):
                    saved = torch.cat(result['context']['probe_points']['g2'])
                    self.assertTrue(torch.allclose(saved, outer, atol=3e-6, rtol=3e-6))
                    self.assertGreater(float(torch.linalg.vector_norm(saved-(outer+g1point-base))), .049)
                optimizer = torch.optim.SGD(params, lr=.04, momentum=0., weight_decay=0., foreach=False)
                update = stepper.apply_update(.04, optimizer)
                expect_param = tuple(p.clone().add_(u, alpha=-.04)
                                     for p, u in zip(split(base, model), result['U']))
                self.assertTrue(all(torch.equal(p, b) for p, b in zip(params, expect_param)))
                self.assertTrue(update['exact_torch_sgd_update'])
                self.assertEqual(result['record']['sgd_updates'], 1)
                self.assertTrue(all(not u.requires_grad and u.grad_fn is None for u in result['U']))

    def test_bitwise_source_gam_decomposition_including_epsilon_guards(self):
        sys.path.insert(0, str(SOURCE))
        try:
            GAM = importlib.import_module('optimer_PerturabtionType.gam').GAM
            for scale in (1., 1e-7, 0.):
                model = Quadratic()
                params = list(model.parameters())
                gradients = tuple(tuple(torch.randn_like(p)*scale for p in params) for _ in range(4))
                optimizer = torch.optim.SGD(params, lr=.04)
                source = GAM(params, optimizer, model, grad_rho=.05, grad_norm_rho=1.)
                for index, p in enumerate(params):
                    p.grad = gradients[3][index].clone()
                    for label, values in zip(('g_0', 'g_1', 'g_2'), gradients[:3]):
                        source.state[p][label] = values[index].clone()
                source.gradient_decompose(SimpleNamespace(grad_beta_1=1., grad_beta_2=1.,
                    grad_beta_3=1., grad_gamma=.1, use_sgv=False))
                ours, _ = gam_decompose(*gradients, gamma=.1, eps=1e-12)
                self.assertTrue(all(torch.equal(p.grad, v) for p, v in zip(params, ours)))
        finally:
            sys.path.remove(str(SOURCE))

    def test_historical_sgd_sam_g2_bitwise_same_native_probe_rule(self):
        sys.path.insert(0, str(HISTORICAL))
        try:
            old = importlib.import_module('upward_residual_step').UpwardResidualStep
            for method, old_method in (('sgd', 'g0'), ('sam', 'g1'), ('g2', 'g2')):
                model = Quadratic()
                old_model = copy.deepcopy(model)
                ours = FiveMethodStep(model, model.parameters(), dict(method=method))
                historical = old(old_model, list(old_model.parameters()), None,
                                 dict(direction=old_method, rho=.05, radius=1.))
                saved_rng = rng_get()
                result = ours.compute(None, None, closure_for(model), .04)
                rng_set(saved_rng)
                previous = historical.compute(None, None, closure_for(old_model), .04)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(result['U'], previous['U'])))
                self.assertEqual(result['record']['g0_sha256'], previous['record']['g0_sha256'])
                for label in result['context']['probe_points']:
                    self.assertTrue(all(torch.equal(a, b) for a, b in
                        zip(result['context']['probe_points'][label], previous['context']['probe_points'][label])))
                ours.discard()
                # Historical class has no public discard; restore its entry.
                previous['context']['entry_state'].restore()
        finally:
            sys.path.remove(str(HISTORICAL))

    def test_full_and_frozen_scopes_complete_restore_and_only_base_rng_advance(self):
        inputs, targets = torch.randn(8, 3), torch.tensor([0, 1]*4)
        for full_ft in (True, False):
            for method in METHODS:
                with self.subTest(full_ft=full_ft, method=method):
                    model = Stateful(full_ft).train()
                    # Entry includes nonempty gradients, forward caches and modes.
                    model.forward_cache = torch.randn(2, 2)
                    for p in model.parameters(): p.grad = torch.randn_like(p)
                    entry = State(model)
                    loss0, _ = closure_for(model)(inputs, targets)
                    wanted_rng1 = rng_get()
                    entry.restore()
                    stepper = FiveMethodStep(model, (p for p in model.parameters() if p.requires_grad), dict(method=method))
                    result = stepper.compute(inputs, targets, closure_for(model), .001)
                    self.assertTrue(rng_equal(rng_get(), wanted_rng1))
                    self.assertTrue(all(result['record']['state_restore_checks'].values()))
                    self.assertTrue(all(torch.equal(p, value) for p, value in entry.trainable))
                    self.assertTrue(all(torch.equal(buffer, value) for buffer, value in entry.buffers))
                    self.assertTrue(all(torch.equal(p.grad, value) for p, value in entry.grads))
                    self.assertTrue(torch.equal(model.forward_cache, entry.controls[0][1]['forward_cache']))
                    self.assertTrue(all(m.training == mode for m, mode in entry.modes))
                    if not full_ft:
                        self.assertEqual(model.frozen._version, entry.frozen[0][1])
                    optimizer = torch.optim.SGD(stepper.params, lr=.001, foreach=False)
                    stepper.apply_update(.001, optimizer)
                    self.assertTrue(rng_equal(rng_get(), wanted_rng1))
                    # Temporary BN/counter/cache side effects are never committed.
                    self.assertTrue(all(torch.equal(buffer, value) for buffer, value in entry.buffers))

    def test_exception_discard_and_bad_optimizer_are_transactional(self):
        model = Stateful().train()
        inputs, targets = torch.randn(8, 3), torch.tensor([0, 1]*4)
        entry = State(model)
        calls = 0
        def failing_closure(inputs, targets):
            nonlocal calls
            calls += 1
            answer = closure_for(model)(inputs, targets)
            if calls == 2: raise RuntimeError('Synthetic failure on virtual read')
            return answer
        stepper = FiveMethodStep(model, model.parameters(), dict(method='sam'))
        with self.assertRaisesRegex(RuntimeError, 'Synthetic failure'):
            stepper.compute(inputs, targets, failing_closure, .001)
        self.assertTrue(all(entry.check().values()))
        result = stepper.compute(inputs, targets, closure_for(model), .001)
        with self.assertRaisesRegex(ValueError, 'zero momentum'):
            stepper.apply_update(.001, torch.optim.SGD(stepper.params, lr=.001, momentum=.9))
        self.assertTrue(all(torch.equal(p, value) for p, value in entry.trainable))
        self.assertEqual(result['record']['sgd_updates'], 0)
        stepper.discard()
        self.assertTrue(all(entry.check().values()))

    def test_mirror_local_first_order_agrees_with_sam_but_is_not_raw_descent(self):
        class Fourth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(torch.tensor([1.5]))
            def forward(self, inputs): return self.x.pow(4).sum()/4.
        errors = []
        for rho in (.02, .01):
            model = Fourth()
            mirror = FiveMethodStep(model, model.parameters(), dict(method='mirror', rho=rho))
            result = mirror.compute(None, None, closure_for(model), .001)
            g0, gminus, U = result['context']['g0'][0], result['context']['g1_minus'][0], result['U'][0]
            leading = g0+rho*3.*model.x.detach().pow(2)
            errors.append(float(torch.abs(U-leading)))
            self.assertGreater(float(U), float(g0))
            self.assertLess(float(gminus), float(g0))
            self.assertTrue(torch.equal(U, g0+(g0-gminus)))
            mirror.discard()
        self.assertGreater(errors[0]/errors[1], 3.8)
        self.assertLess(errors[0]/errors[1], 4.2)

    def test_zero_gradient_and_scope_validation(self):
        class Zero(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(2))
            def forward(self, inputs): return (self.p**2).sum()
        for method in METHODS:
            model = Zero()
            stepper = FiveMethodStep(model, model.parameters(), dict(method=method))
            result = stepper.compute(None, None, closure_for(model), .001)
            self.assertTrue(torch.equal(result['U'][0], torch.zeros(2)))
            stepper.apply_update(.001, torch.optim.SGD(model.parameters(), lr=.001))
        model = Quadratic()
        with self.assertRaisesRegex(ValueError, 'exactly all'):
            FiveMethodStep(model, [model.x], dict(method='sgd'))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FiveMethodTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = dict(tests_run=result.testsRun, failures=len(result.failures),
                   errors=len(result.errors), passed=result.wasSuccessful(),
                   device='cpu', test_class='FiveMethodTests',
                   native_reads=READS, source_projection_bitwise_checked=True,
                   historical_sgd_sam_g2_bitwise_checked=True)
    destination = Path(__file__).with_name('five_method_cpu_validation.json')
    destination.write_text(json.dumps(payload, indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
