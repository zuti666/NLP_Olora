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
from direction_step import FourProbeStep
from effective_step import EffectiveStep
from state_transaction import State
from four_probe_train import Controller, resolve_config

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
        return .5*theta@(H@theta)-linear@theta+.08*theta.pow(4).sum()+.07*(1-theta.cos()).sum(), theta[:3].unsqueeze(0)

    return model, params, geometry, H, linear, closure, calls


class Acceptance(unittest.TestCase):
    def assert_close(self, actual, expected, atol=1e-11, rtol=1e-11):
        actual = flat(actual) if isinstance(actual, (list, tuple)) else actual
        expected = flat(expected) if isinstance(expected, (list, tuple)) else expected
        self.assertTrue(torch.allclose(actual, expected, atol=atol, rtol=rtol),
                        'max error='+str(float((actual-expected).abs().max())))

    def engine(self, model, params, geom, radius, branch='C'):
        return FourProbeStep(model, params, geom, dict(branch=branch,direction='g1_minus_plus_g2',
                                                       rho=.05,radius=radius))

    def test_nonlinear_analytic_locations_direction_and_four_pass_one_sgd(self):
        for radius in (.25, 1.):
            for branch in ('C', 'D'):
                for reverse in (False, True):
                    with self.subTest(radius=radius,branch=branch,reverse=reverse):
                        model, params, geom, H, b, closure, calls = make_case(reverse=reverse)
                        base = tuple(p.detach().clone() for p in params)
                        theta = flat(base)
                        def grad(t):
                            return H@t-b+.32*t.pow(3)+.07*t.sin()
                        def normalize(t):
                            return t/(torch.linalg.vector_norm(t)+1e-12)
                        g0 = grad(theta)
                        pointp = theta+.05*normalize(g0)
                        pointm = theta-.05*normalize(g0)
                        gp, gm = grad(pointp), grad(pointm)
                        q = g0-gm if branch == 'C' else gm-g0
                        point2 = theta+radius*normalize(q)
                        g2 = grad(point2)
                        expected = gm+g2
                        engine = self.engine(model,params,geom,radius,branch)
                        optimizer = torch.optim.SGD(params,lr=.04)
                        with patch('torch.autograd.grad',wraps=torch.autograd.grad) as gradients, \
                                patch.object(optimizer,'step',wraps=optimizer.step) as updates:
                            result = engine.compute(None,None,closure,.04)
                            self.assertEqual(gradients.call_count,4)
                            self.assertTrue(all(c.kwargs['create_graph'] is False for c in gradients.call_args_list))
                            self.assertEqual(updates.call_count,0)
                            self.assertEqual(len(calls),4)
                            for actual, reference in zip(calls,(theta,pointp,pointm,point2)):
                                self.assert_close(actual,reference)
                            self.assert_close(result['context']['grads']['g1_minus'],gm)
                            self.assert_close(result['context']['grads']['g1_plus'],gp)
                            self.assert_close(result['context']['grads']['g2'],g2)
                            self.assert_close(result['context']['q'],q)
                            self.assert_close(result['U'],expected)
                            self.assertGreater(float(torch.linalg.vector_norm(expected-.5*expected)),.1)
                            self.assertGreater(float(torch.linalg.vector_norm(expected-(g0+g2-gm))),.1)
                            self.assertGreater(float(torch.linalg.vector_norm(point2-(pointm+radius*normalize(q)))),.04)
                            self.assertTrue(all(torch.equal(p,v) for p,v in zip(params,base)))
                            record = result['record']
                            self.assertEqual(record['ordinary_gradient_calls'],4)
                            self.assertEqual(record['first_plus_gradient_calls'],1)
                            self.assertEqual(record['first_minus_gradient_calls'],1)
                            self.assertEqual(record['g3_gradient_calls'],0)
                            self.assertEqual(record['gradient_projection_calls'],0)
                            self.assertIsNone(record['projection'])
                            self.assertEqual(record['radius'],radius)
                            self.assertEqual(record['formula'],'g1_minus+g2_'+branch)
                            self.assertNotIn('phi3',record['probe_displacements'])
                            self.assertNotIn('g3',record['norms'])
                            self.assertIn('g1_minus',record['norms'])
                            update = engine.apply_update(optimizer=optimizer)
                            self.assertEqual(updates.call_count,1)
                            self.assertTrue(update['natural_sgd_exact'])
                            self.assert_close(tuple(p.detach() for p in params),theta-.04*expected)
                            self.assertEqual(len(optimizer.state),0)

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
        engine = self.engine(model,params,geom,1.)
        out = engine.compute(None, None, self.stochastic(model, params, draws), .04)
        self.assertEqual(len(draws), 4)
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

    def test_exception_restores_entry_parameters_gradients_buffers_caches_and_rng(self):
        for failure_probe in (1, 2, 3, 4):
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
            engine = self.engine(model,params,geom,.25)
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                engine.compute(None, None, failing, .04)
            self.assertTrue(all(state.check().values()))
            with self.assertRaisesRegex(RuntimeError, 'compute must precede'):
                engine.apply_update()

    def test_update_exception_rolls_back_and_duplicate_step_is_rejected(self):
        model, params, geom, _, _, closure, _ = make_case()
        base = tuple(p.detach().clone() for p in params)
        engine = self.engine(model,params,geom,1.)
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
        config_path=Path('/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/configs/S1-09_A_C_natural_s2.json')
        raw=json.loads(config_path.read_text())
        for radius in (.25,1.):
            with tempfile.TemporaryDirectory() as output:
                cli=SimpleNamespace(max_tasks=1,max_epochs=1,max_batches=1,output_dir=output)
                candidate=copy.deepcopy(raw)
                candidate.update(gam_grad_norm_rho=radius,probe_branch='D',update_direction='g1_minus_plus_g2')
                cfg=resolve_config(candidate,cli)
                self.assertEqual(cfg['gam_grad_norm_rho'],radius)
                cfg['seed']=2
                model,params,geom,_,_,closure,calls=make_case()
                dummy=lambda *args,**kwargs:None
                learner=SimpleNamespace(_unwrap_network=lambda:model,_run_regularized_train=dummy,
                                        incremental_train=dummy,eval_task=dummy,after_task=dummy)
                controller=Controller(learner,cfg,cli)
                controller.task=0;controller.old_fc={}
                try:
                    with patch('four_probe_train.native.install_forbidden_work_guards'), \
                            patch.object(EffectiveStep,'from_model',return_value=geom), \
                            patch.object(controller,'save_observation'):
                        controller.install()
                        learner._build_optimizer(params,'init')
                    self.assertEqual(controller.stepper.radius,radius)
                    result=controller.stepper.compute(None,None,closure,.04)
                    self.assertAlmostEqual(float(torch.linalg.vector_norm(calls[3]-calls[0])),radius,places=9)
                    self.assertEqual(result['record']['radius'],radius)
                    controller.stepper.discard()
                finally:
                    controller.handle.close()

    def test_fp32_natural_step_and_zero_momentum_guard(self):
        for radius in (.25,1.):
            model,params,geom,_,_,closure,_=make_case(torch.float32,reverse=True)
            engine=self.engine(model,params,geom,radius)
            out=engine.compute(None,None,closure,.04)
            self.assertTrue(all(torch.isfinite(v).all() for v in out['U']))
            bad=torch.optim.SGD(params,lr=.04,momentum=.9)
            with self.assertRaisesRegex(ValueError,'zero momentum'):
                engine.apply_update(optimizer=bad)
            good=torch.optim.SGD(params,lr=.04)
            self.assertTrue(engine.apply_update(optimizer=good)['natural_sgd_exact'])


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Run acceptance with CUDA_VISIBLE_DEVICES=""')
    torch.set_num_threads(1)
    random.seed(2);np.random.seed(2);torch.manual_seed(2)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    output=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),device='cpu',
                cuda_visible_devices='',radii=[.25,1.],branches=['C','D'],direction='g1_minus_plus_g2',
                formula='g1_minus+g2',negative_g1_verified=True,no_projection_verified=True,nonlinear_analytic_locations=True,
                four_gradient_one_sgd_verified=True,rng_replay_R0_final_R1_verified=True)
    Path(__file__).with_name('CPU_GATE.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))
    raise SystemExit(0 if result.wasSuccessful() else 1)
