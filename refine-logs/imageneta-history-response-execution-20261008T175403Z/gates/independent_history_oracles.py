"""Independent CPU-only analytic and post-commit transaction oracles."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import unittest

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'trainer'))
import numpy as np
import torch
from history_step import HistoryResponseStep, _digest
from state_transaction import State, rng_get, rng_set, rng_equal


class Vector(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))


def make(model, weight=.5):
    direction = 'history_fused' if weight else 'future_only'
    step = HistoryResponseStep(model, list(model.parameters()), None,
                               dict(direction=direction, history_weight=weight))
    opt = torch.optim.SGD(model.parameters(), lr=.04)
    return step, opt


def inject_history(step, gradient, actual_delta):
    gp = (torch.tensor(gradient, dtype=torch.float32),)
    delta = (torch.tensor(actual_delta, dtype=torch.float32),)
    step._history = dict(raw_gradient=gp, displacement=delta,
                         length=float(delta[0].double().norm()),
                         raw_gradient_sha256=_digest(gp), displacement_sha256=_digest(delta),
                         after_sha256=_digest(step.params), layout=step._layout)


class IndependentOracles(unittest.TestCase):
    def test_nonparallel_spd_matches_independent_numpy_answers(self):
        H = np.array([[3., 1.], [1., 2.]], dtype=np.float64)
        theta = np.array([.4, -.7], dtype=np.float64)
        s = np.array([-.03, -.02], dtype=np.float64)
        eta = .04
        g = H @ theta
        gp = H @ (theta - s)
        gf = H @ (theta - eta*g)
        lf, lb = eta*np.linalg.norm(g), np.linalg.norm(s)
        analytic = g + .5*(g-gf) + lf/(2*lb)*(gp-g)
        np.testing.assert_allclose(analytic, [.5782191040240648, -.9865878428937768], atol=1e-14)
        self.assertAlmostEqual(g @ (-s)/(np.linalg.norm(g)*lb), -.12403473458920829, places=14)
        model = Vector(theta)
        step, opt = make(model)
        inject_history(step, gp, s)
        ht = torch.tensor(H, dtype=torch.float32)
        def loss(x, y):
            return .5*model.w @ ht @ model.w, model.w[None]
        out = step.compute(None, None, loss, eta)
        np.testing.assert_allclose(out['U'][0].numpy(), analytic, rtol=1e-6, atol=1e-7)
        self.assertTrue(out['record']['history_used'])
        self.assertAlmostEqual(out['record']['u_v_cosine'], -.12403473458920829, places=6)
        actual_before = model.w.detach().clone()
        step.apply_update(eta, opt)
        self.assertTrue(torch.equal(step._history['raw_gradient'][0], out['context']['g'][0]))
        self.assertTrue(torch.equal(step._history['displacement'][0], model.w.detach()-actual_before))

    def test_linear_zero_hessian_cached_batch_changes_update(self):
        model = Vector([.4, -.7])
        step, opt = make(model)
        opt.param_groups[0]['lr'] = .01
        inject_history(step, [2., 0.], [-.02, 0.])
        def loss(x, y):
            return 2*model.w[1], model.w[None]
        out = step.compute(None, None, loss, .01)
        self.assertTrue(torch.equal(out['context']['g'][0], out['context']['gf'][0]))
        torch.testing.assert_close(out['U'][0], torch.tensor([1., 1.]), rtol=1e-6, atol=1e-7)
        step.apply_update(.01, opt)

    def test_real_post_commit_diagnostic_leaves_next_trajectory_identical(self):
        class Stochastic(Vector):
            def __init__(self):
                super().__init__([.4, -.7, .2, -.5])
                self.register_buffer('calls', torch.tensor(0))
                self.forward_cache = {'counter': torch.tensor(0)}
            def forward(self):
                self.calls.add_(1)
                self.forward_cache['counter'].add_(1)
                random.random()
                np.random.rand()
                return self.w * (torch.rand_like(self.w) > .25).to(self.w)
        torch.manual_seed(71); random.seed(71); np.random.seed(71)
        model1, model2 = Stochastic(), Stochastic()
        step1, opt1 = make(model1)
        step2, opt2 = make(model2)
        def closure(model):
            def loss(x, y):
                v = model()
                return .5*v.square().sum(), v[None]
            return loss
        f1, f2 = closure(model1), closure(model2)
        for eta in [.04, .023, .015]:
            opt1.param_groups[0]['lr'] = opt2.param_groups[0]['lr'] = eta
            r0 = rng_get()
            a = step1.compute(None, None, f1, eta)
            step1.apply_update(eta, opt1)
            committed = State(model1)
            saved_history = copy.deepcopy(step1.state_dict())
            try:
                rng_set(a['context']['rng0'])
                with torch.no_grad():
                    postloss, _ = f1(None, None)
                self.assertTrue(bool(torch.isfinite(postloss)))
            finally:
                committed.restore()
            self.assertTrue(all(committed.check().values()))
            self.assertTrue(rng_equal(rng_get(), a['context']['rng1']))
            self.assertEqual(saved_history['history']['raw_gradient_sha256'], step1.state_dict()['history']['raw_gradient_sha256'])
            self.assertEqual(saved_history['history']['displacement_sha256'], step1.state_dict()['history']['displacement_sha256'])
            committed_rng = rng_get()
            rng_set(r0)
            b = step2.compute(None, None, f2, eta)
            step2.apply_update(eta, opt2)
            self.assertTrue(rng_equal(rng_get(), committed_rng))
            self.assertTrue(torch.equal(a['U'][0], b['U'][0]))
            self.assertTrue(torch.equal(model1.w, model2.w))
            self.assertTrue(torch.equal(model1.calls, model2.calls))
            self.assertEqual(step1.state_dict()['history']['displacement_sha256'], step2.state_dict()['history']['displacement_sha256'])


if __name__ == '__main__':
    assert not torch.cuda.is_available()
    unittest.main(verbosity=2)
