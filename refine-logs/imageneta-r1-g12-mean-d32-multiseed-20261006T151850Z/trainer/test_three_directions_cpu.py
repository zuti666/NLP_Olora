"""CPU-only analytic, rollback and historical arithmetic acceptance tests."""
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

sys.dont_write_bytecode = True
import numpy as np
import torch
from audit_core import rng_equal, rng_get, rng_set
from direction_step import FourProbeStep, DIRECTIONS, FORMULAS, compose_direction
from effective_step import EffectiveStep
from state_transaction import State
from four_probe_train import Controller, resolve_config

H = Path(__file__).resolve().parent
ROOT = Path('/data/140-1/users/liying/O-LoRA/refine-logs')
OLD = {
    'g1_plus_g2': ('imageneta-positive-g012-combinations-20261005T185032Z', 'g1_plus_g2'),
    'half_g1_plus_g2': ('imageneta-half-g1-plus-g2-20261006T145451Z', 'half_g1_plus_g2'),
    'g1_plus_d32': ('imageneta-g1-plus-g3-minus-g2-20261005T170708Z', 'g1_plus_second_difference'),
}


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


def case(dtype=torch.float64, reverse=False):
    model = Toy(dtype)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    if reverse:
        params = params[::-1]
    geometry = EffectiveStep([(model.a, model.b)], tuple(model.fc.parameters()), params)
    n = sum(p.numel() for p in params)
    gen = torch.Generator().manual_seed(177)
    matrix = torch.randn(n, n, generator=gen, dtype=dtype)
    hessian = matrix.T@matrix/n + torch.diag(torch.linspace(.3, 2., n, dtype=dtype))
    linear = torch.linspace(-.2, .4, n, dtype=dtype)
    calls = []
    def closure(_x, _y):
        theta = flat(params)
        calls.append(theta.detach().clone())
        return (.5*theta@(hessian@theta)-linear@theta+.08*theta.pow(4).sum()
                +.07*(1-theta.cos()).sum()), theta[:3].unsqueeze(0)
    return model, params, geometry, hessian, linear, closure, calls


def old_module(direction):
    run, _ = OLD[direction]
    spec = importlib.util.spec_from_file_location('historical_'+direction, ROOT/run/'trainer/direction_step.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Acceptance(unittest.TestCase):
    def equal(self, a, b):
        self.assertEqual(len(a), len(b))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))

    def close(self, a, b):
        a = flat(a) if isinstance(a, (tuple, list)) else a
        b = flat(b) if isinstance(b, (tuple, list)) else b
        self.assertTrue(torch.allclose(a,b,rtol=1e-11,atol=1e-11), str(float((a-b).abs().max())))

    def engine(self, model, params, geom, direction):
        return FourProbeStep(model,params,geom,dict(branch='P',direction=direction,rho=.05,radius=1.))

    def test_independent_nonlinear_locations_formulas_and_four_gradients_one_sgd(self):
        for direction in DIRECTIONS:
            for reverse in (False,True):
                with self.subTest(direction=direction,reverse=reverse):
                    model,params,geom,h,b,closure,calls=case(reverse=reverse)
                    theta=flat(params).detach().clone()
                    grad=lambda t: h@t-b+.32*t.pow(3)+.07*t.sin()
                    normalize=lambda t: t/(torch.linalg.vector_norm(t)+1e-12)
                    g0=grad(theta);p1=theta+.05*normalize(g0);g1=grad(p1)
                    p2=theta+normalize(g1-g0);g2=grad(p2)
                    p3=p2+.05*normalize(g2);g3=grad(p3)
                    expected={'g1_plus_g2':g1+g2,'half_g1_plus_g2':.5*(g1+g2),
                              'g1_plus_d32':g1+(g3-g2)}[direction]
                    engine=self.engine(model,params,geom,direction)
                    opt=torch.optim.SGD(params,lr=.04)
                    with patch('torch.autograd.grad',wraps=torch.autograd.grad) as grads, patch.object(opt,'step',wraps=opt.step) as updates:
                        out=engine.compute(None,None,closure,.04)
                        self.assertEqual(grads.call_count,4)
                        self.assertTrue(all(call.kwargs['create_graph'] is False for call in grads.call_args_list))
                        self.assertEqual(updates.call_count,0)
                        self.assertEqual(len(calls),4)
                        for actual,reference in zip(calls,(theta,p1,p2,p3)):self.close(actual,reference)
                        for label,reference in [('g0',g0),('g1',g1),('g2',g2),('g3',g3)]:
                            self.close(out['context']['grads'][label],reference)
                        self.close(out['U'],expected)
                        self.assertEqual(out['record']['formula'],FORMULAS[direction])
                        self.assertEqual(out['record']['g3_used_in_final'],direction=='g1_plus_d32')
                        self.assertEqual(out['record']['gradient_projection_calls'],0)
                        self.assertTrue(all(out['record']['state_restoration'].values()))
                        update=engine.apply_update(optimizer=opt)
                        self.assertEqual(updates.call_count,1)
                        self.assertTrue(update['natural_sgd_exact'])
                        self.assertEqual(update['scale'],.04)
                        self.close(tuple(p.detach() for p in params),theta-.04*expected)
                        self.assertEqual(len(opt.state),0)

    def test_fp32_and_fp64_historical_sum_mean_d32_probe_direction_and_update_equivalence(self):
        for dtype in (torch.float32,torch.float64):
            for direction in DIRECTIONS:
                for reverse in (False,True):
                    with self.subTest(dtype=dtype,direction=direction,reverse=reverse):
                        model,params,geom,_,_,closure,_=case(dtype,reverse)
                        snapshot=State(model)
                        new=self.engine(model,params,geom,direction)
                        fresh=new.compute(None,None,closure,.04)
                        new_update=new.apply_update(optimizer=torch.optim.SGD(params,lr=.04))
                        new_values=tuple(p.detach().clone() for p in params)
                        snapshot.restore()
                        module=old_module(direction)
                        historical=module.FourProbeStep(model,params,geom,dict(branch='P',direction=OLD[direction][1],rho=.05,radius=1.))
                        old=historical.compute(None,None,closure,.04)
                        old_update=historical.apply_update(optimizer=torch.optim.SGD(params,lr=.04))
                        for label in ('g0','g1','g2','g3'):
                            self.equal(fresh['context']['grads'][label],old['context']['grads'][label])
                        for label in ('phi0','phi1','phi2','phi3'):
                            self.equal(fresh['context']['points'][label],old['context']['points'][label])
                        self.equal(fresh['U'],old['U']);self.equal(new_values,tuple(p.detach() for p in params))
                        self.assertEqual(new_update['scale'],old_update['scale'])
                        self.assertTrue(rng_equal(fresh['context']['rng1'],old['context']['rng1']))

    def test_same_state_all_three_share_gradients_mean_is_half_sum_without_lr_half(self):
        model,params,geom,_,_,closure,_=case(torch.float32)
        snapshot=State(model);outs={}
        for direction in DIRECTIONS:
            snapshot.restore();engine=self.engine(model,params,geom,direction)
            outs[direction]=engine.compute(None,None,closure,.04);engine.discard()
        for direction in DIRECTIONS:
            for label in ('g0','g1','g2','g3'):
                self.equal(outs[direction]['context']['grads'][label],outs['g1_plus_g2']['context']['grads'][label])
        self.equal(outs['half_g1_plus_g2']['U'],tuple(.5*x for x in outs['g1_plus_g2']['U']))
        snapshot.restore();engine=self.engine(model,params,geom,'half_g1_plus_g2')
        out=engine.compute(None,None,closure,.04);base=tuple(p.detach().clone() for p in params)
        engine.apply_update(optimizer=torch.optim.SGD(params,lr=.04))
        expected=tuple(p.clone().add_(u,alpha=-.04) for p,u in zip(base,out['U']))
        self.equal(tuple(p.detach() for p in params),expected)
        self.assertTrue(any(not torch.equal(p,q.clone().add_(u,alpha=-.02)) for p,q,u in zip(params,base,out['U'])))

    def test_g3_affects_d32_only(self):
        for direction in DIRECTIONS:
            model,params,geom,_,_,closure,_=case()
            snapshot=State(model);engine=self.engine(model,params,geom,direction)
            original=engine.compute(None,None,closure,.04);engine.discard();snapshot.restore()
            evaluate=engine._evaluate;count=[0]
            def altered(*args,**kwargs):
                result=evaluate(*args,**kwargs);count[0]+=1
                if count[0]==4:
                    loss,logits,grads,rng=result
                    return loss,logits,tuple(v+3. for v in grads),rng
                return result
            with patch.object(engine,'_evaluate',side_effect=altered):
                changed=engine.compute(None,None,closure,.04)
            if direction=='g1_plus_d32':
                self.assertGreater(float(torch.linalg.vector_norm(flat(changed['U'])-flat(original['U']))),1.)
            else:self.equal(original['U'],changed['U'])
            engine.discard()

    def stochastic(self,model,params,draws):
        def closure(_x,_y):
            theta=flat(params);mask=model.dropout(torch.ones_like(theta))
            draws.append((random.random(),float(np.random.rand()),torch.rand(5),mask.detach().clone()))
            model.probe_counter.add_(1);model.dropout.training=False
            model.delta_w_q_new_grad=torch.ones(2);model.external_delta_w_q=torch.ones(3)
            model._register_delta_hook=True;model.forward_cache['entry'].add_(2)
            model.forward_cache['created']=torch.ones(2);model.created_cache=[torch.ones(1)]
            return ((theta*mask).square()+.2*(theta*mask).pow(4)).sum(),theta[:3].unsqueeze(0)
        return closure

    def test_stochastic_rng_replay_one_advance_and_full_state_restore(self):
        for direction in DIRECTIONS:
            model,params,geom,*_=case();model.train();state=State(model)
            reference=[];self.stochastic(model,params,reference)(None,None);expected_rng1=rng_get();state.restore()
            draws=[];engine=self.engine(model,params,geom,direction)
            out=engine.compute(None,None,self.stochastic(model,params,draws),.04)
            self.assertEqual(len(draws),4)
            for draw in draws[1:]:
                self.assertEqual(draw[:2],draws[0][:2]);self.assertTrue(torch.equal(draw[2],draws[0][2]));self.assertTrue(torch.equal(draw[3],draws[0][3]))
            self.assertTrue(rng_equal(rng_get(),expected_rng1))
            for p,grad in state.grads:p.grad=None if grad is None else grad.clone()
            rng_set(state.rng);self.assertTrue(all(state.check().values()))
            self.assertFalse(hasattr(model,'created_cache'));engine.discard()

    def test_any_failed_probe_restores_gradients_parameters_buffers_caches_rng(self):
        for direction in DIRECTIONS:
            for failure in (1,2,3,4):
                model,params,geom,*_=case()
                for p in params:p.grad=torch.ones_like(p)*3
                state=State(model);draws=[];closure=self.stochastic(model,params,draws)
                def fail(x,y):
                    out=closure(x,y)
                    if len(draws)==failure:raise RuntimeError('intentional failure')
                    return out
                engine=self.engine(model,params,geom,direction)
                with self.assertRaisesRegex(RuntimeError,'intentional'):engine.compute(None,None,fail,.04)
                self.assertTrue(all(state.check().values()))
                with self.assertRaisesRegex(RuntimeError,'compute must precede'):engine.apply_update()

    def test_seed_0_1_2_config_radius_selector_and_real_controller(self):
        raw=json.loads((ROOT/'imageneta-positive-g012-combinations-20261005T185032Z/configs/formal_sum12_r1_k10_s2.json').read_text())
        for seed in (0,1,2):
            for direction in DIRECTIONS:
                with self.subTest(seed=seed,direction=direction), tempfile.TemporaryDirectory() as output:
                    candidate=copy.deepcopy(raw);candidate.update(seed=[seed],update_direction=direction)
                    cli=SimpleNamespace(max_tasks=2,max_epochs=1,max_batches=1,output_dir=output)
                    cfg=resolve_config(candidate,cli)
                    self.assertEqual(cfg['seed'],[seed]);self.assertEqual(cfg['gam_grad_norm_rho'],1.)
                    self.assertTrue(cfg['compute_g3_probe']);self.assertEqual(cfg['g3_used_in_final'],direction=='g1_plus_d32')
                    self.assertEqual(cfg['lr_multiplier'],1.);self.assertEqual(cfg['init_lr'],.04);self.assertEqual(cfg['lrate'],.02)
                    cfg['seed']=seed;model,params,geom,*_=case();dummy=lambda *a,**k:None
                    learner=SimpleNamespace(_unwrap_network=lambda:model,_run_regularized_train=dummy,incremental_train=dummy,eval_task=dummy,after_task=dummy)
                    controller=Controller(learner,cfg,cli);controller.task=0;controller.old_fc={}
                    try:
                        with patch('four_probe_train.native.install_forbidden_work_guards'),patch.object(EffectiveStep,'from_model',return_value=geom),patch.object(controller,'save_observation'):
                            controller.install();optimizer=learner._build_optimizer(params,'init')
                        self.assertEqual(controller.seed,seed);self.assertEqual(controller.stepper.direction,direction)
                        self.assertEqual(controller.stepper.radius,1.);self.assertEqual(optimizer.param_groups[0]['lr'],.04)
                    finally:controller.handle.close()
        with tempfile.TemporaryDirectory() as output:
            cli=SimpleNamespace(max_tasks=2,max_epochs=1,max_batches=1,output_dir=output)
            invalid=[dict(seed=[3]),dict(seed=[0,1]),dict(lora_rank=20),dict(gam_grad_rho=.1),dict(gam_grad_norm_rho=.25),dict(lr_multiplier=.5),dict(attention_backend='math'),dict(step_mode='matched'),dict(probe_branch='D')]
            for change in invalid:
                candidate=copy.deepcopy(raw);candidate.update(change)
                with self.assertRaises(ValueError):resolve_config(candidate,cli)

    def test_sgd_guard_duplicate_and_failed_update_rollback(self):
        model,params,geom,_,_,closure,_=case(torch.float32)
        engine=self.engine(model,params,geom,'g1_plus_d32');engine.compute(None,None,closure,.04)
        with self.assertRaisesRegex(ValueError,'zero momentum'):engine.apply_update(optimizer=torch.optim.SGD(params,lr=.04,momentum=.9))
        opt=torch.optim.SGD(params,lr=.04);self.assertTrue(engine.apply_update(optimizer=opt)['natural_sgd_exact'])
        with self.assertRaisesRegex(RuntimeError,'compute must precede'):engine.apply_update(optimizer=opt)
        base=tuple(p.detach().clone() for p in params);engine.compute(None,None,closure,.04)
        native_step=opt.step
        def fail(*a,**k):native_step(*a,**k);raise RuntimeError('intentional SGD failure')
        with patch.object(opt,'step',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'intentional'):engine.apply_update(optimizer=opt)
        self.equal(tuple(p.detach() for p in params),base)


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Acceptance requires CUDA_VISIBLE_DEVICES=""')
    torch.set_num_threads(1);random.seed(2);np.random.seed(2);torch.manual_seed(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance))
    report=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),device='cpu',directions=list(DIRECTIONS),formulas=FORMULAS,seed_config_cases=9,historical_sum_mean_d32_equivalence_dtype=['float32','float64'],rng_replay_full_restore_verified=result.wasSuccessful(),g3_uses_only_d32_verified=result.wasSuccessful(),mean_half_once_lr_original_verified=result.wasSuccessful())
    (H/'CPU_GATE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
