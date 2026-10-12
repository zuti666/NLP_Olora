"""CPU mathematical, state-transaction and real trainer integration acceptance."""
import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import numpy as np
import torch
from audit_core import rng_get, rng_equal, seed_all, digest, atomic_json
from direction_step import FullState, vector_norm
from effective_step import EffectiveStep
from odd_even_step import OddEvenStep, METHOD, BRANCH, DIRECTION, DIRECTIONS
from odd_even_train import resolve_config, Controller


class Tiny(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([[.3]], dtype=dtype))
        self.b = torch.nn.Parameter(torch.tensor([[.4]], dtype=dtype))
        self.fc = torch.nn.Linear(1,1,dtype=dtype)
        with torch.no_grad():
            self.fc.weight.fill_(.5)
            self.fc.bias.fill_(-.2)
        self.frozen = torch.nn.Parameter(torch.ones(1,dtype=dtype),requires_grad=False)
        self.register_buffer('counter',torch.zeros((),dtype=dtype))
        self.dropout = torch.nn.Dropout(.3)
        self.fused_attn = True
        self._register_delta_hook = False
        self.external_delta_w_q = None
        self.external_delta_w_v = None
        self.delta_w_q_new_grad = None
        self.forward_cache = {'nested':[torch.tensor([2.],dtype=dtype)]}


def flat(values):
    return torch.cat([v.reshape(-1) for v in values])


def case(dtype=torch.float64, reverse=False, **config):
    model = Tiny(dtype)
    ordinary = tuple(p for p in model.parameters() if p.requires_grad)
    params = tuple(reversed(ordinary)) if reverse else ordinary
    geometry = EffectiveStep([(model.a,model.b)],tuple(model.fc.parameters()),params)
    return model,params,geometry,OddEvenStep(model,params,geometry,config)


def quadratic(params):
    H = torch.diag(torch.tensor([1.,2.,3.,4.],dtype=params[0].dtype))
    b = torch.tensor([-.2,.1,-.3,.4],dtype=params[0].dtype)
    def closure(_x,_y):
        x = flat(params)
        return .5*x@(H@x)-b@x,x[:2].unsqueeze(0)
    return closure,H,b


class Acceptance(unittest.TestCase):
    def setUp(self):
        seed_all(2)
        torch.set_num_threads(1)

    def close(self,a,b,atol=1e-13,rtol=1e-12):
        self.assertTrue(torch.allclose(a,b,atol=atol,rtol=rtol),str((a-b).abs().max().item()))

    def test_quadratic_reflection_equals_analytic_positive_probe_under_three_probes(self):
        model,params,geometry,engine = case()
        closure,H,b = quadratic(params)
        out = engine.compute(None,None,closure,.04)
        c = out['context'];x = flat(c['base']);g = H@x-b;n = g/(torch.linalg.vector_norm(g)+1e-12)
        self.close(flat(c['g0']),g)
        self.close(flat(c['g1_minus']),H@(x-.05*n)-b)
        self.close(flat(c['correction']),.05*(H@n))
        self.close(flat(c['U']),g+.05*(H@n))
        self.close(flat(c['U']),H@(x+.05*n)-b)
        self.assertEqual(set(c['points']),{'phi0','phi1_plus','phi1_minus'})
        self.assertEqual(out['record']['ordinary_gradient_calls'],3)
        self.assertIsNone(out['record']['outer_radius'])
        self.assertEqual(out['record']['hvp_calls'],0)
        engine.discard()

    def test_nonlinear_reflection_and_SAM_share_first_order_but_differ_at_second_order(self):
        errors=[]
        for rho in (.02,.01,.005):
            model,params,geometry,engine = case(rho=rho)
            def grad(x):return x.pow(3)+.7*x
            def closure(_x,_y):
                x=flat(params)
                return (x.pow(4)/4+.35*x.square()).sum(),x[:2].unsqueeze(0)
            out=engine.compute(None,None,closure,.04);c=out['context'];x=flat(c['base'])
            g=grad(x);n=g/(torch.linalg.vector_norm(g)+1e-12)
            reflected=g+(g-grad(x-rho*n));sam=grad(x+rho*n)
            first_order=g+rho*(3*x.square()+.7)*n
            self.close(flat(c['U']),reflected)
            self.assertGreater(torch.linalg.vector_norm(reflected-sam).item(),1e-7)
            self.assertLess(torch.linalg.vector_norm(reflected-first_order).item(),3*rho*rho)
            self.assertLess(torch.linalg.vector_norm(sam-first_order).item(),3*rho*rho)
            errors.append(torch.linalg.vector_norm(reflected-sam).item())
            engine.discard()
        self.assertAlmostEqual(errors[0]/errors[1],4.,places=9)
        self.assertAlmostEqual(errors[1]/errors[2],4.,places=8)

    def test_FP32_requested_parentheses_exact_and_parameter_order(self):
        for reverse in (False,True):
            model,params,geometry,engine=case(dtype=torch.float32,reverse=reverse)
            closure,_,_=quadratic(params)
            out=engine.compute(None,None,closure,.04);c=out['context']
            expected=tuple(z+(z-m) for z,m in zip(c['g0'],c['g1_minus']))
            self.assertTrue(all(torch.equal(u,v) for u,v in zip(out['U'],expected)))
            optimizer=torch.optim.SGD(params,lr=.04,foreach=False)
            update=engine.apply_update(optimizer=optimizer)
            self.assertTrue(update['natural_odd_even_only'])
            self.assertNotIn('natural_g2_only',update)
            self.assertEqual(update['actual']['measurement'],'actual_factor_values_float64_Gram_unscaled_BA_plus_all_FC')
            json.dumps(dict(out['record'],**update),allow_nan=False)

    def test_rho_zero_returns_base_gradient_exactly(self):
        model,params,geometry,engine=case(rho=0)
        closure,_,_=quadratic(params)
        out=engine.compute(None,None,closure,.04)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(out['U'],out['context']['g0'])))
        self.assertEqual(out['record']['parameter_probe_displacements']['phi1_minus'],0)
        self.assertEqual(out['record']['norms']['correction'],0)
        engine.discard()

    def test_zero_base_gradient_zero_direction_and_null_cosines(self):
        model,params,geometry,engine=case()
        origins=tuple(p.detach().clone() for p in params)
        def closure(_x,_y):
            x=flat(params)-flat(origins)
            return x.square().sum(),x[:2].unsqueeze(0)
        out=engine.compute(None,None,closure,.04)
        self.assertTrue(all(value==0 for value in out['record']['norms'].values()))
        self.assertTrue(all(value is None for value in out['record']['cosines'].values()))
        update=engine.apply_update(optimizer=torch.optim.SGD(params,lr=.04))
        self.assertEqual(update['actual']['joint_norm'],0)
        self.assertTrue(all(torch.equal(p,v) for p,v in zip(params,origins)))

    def test_linear_loss_zero_correction_and_unmodified_natural_gradient(self):
        model,params,geometry,engine=case()
        def closure(_x,_y):
            x=flat(params)
            return x.sum(),x[:2].unsqueeze(0)
        out=engine.compute(None,None,closure,.04)
        self.assertEqual(out['record']['norms']['correction'],0)
        self.assertEqual(out['record']['correction_over_g0'],0)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(out['U'],out['context']['g0'])))
        engine.discard()

    def test_three_ordinary_autograd_calls_no_create_graph_and_input_identity(self):
        model,params,geometry,engine=case();closure,_,_=quadratic(params)
        x=object();y=object();seen=[]
        def tracked(a,b):
            seen.append((a,b));return closure(a,b)
        with patch.object(torch.autograd,'grad',wraps=torch.autograd.grad) as spy:
            out=engine.compute(x,y,tracked,.04)
        self.assertEqual(spy.call_count,3)
        self.assertTrue(all(call.kwargs['create_graph'] is False for call in spy.call_args_list))
        self.assertTrue(all(call.kwargs['retain_graph'] is False for call in spy.call_args_list))
        self.assertEqual(seen,[(x,y),(x,y),(x,y)])
        self.assertEqual(engine.total_closure_calls,3)
        self.assertTrue(all(not v.requires_grad for v in out['U']))
        engine.discard()

    def test_stochastic_replay_full_state_gradient_objects_and_R1_once(self):
        model,params,geometry,engine=case()
        for i,p in enumerate(model.parameters()):p.grad=torch.full_like(p,i+.5)
        entry=FullState(model);draws=[];rng1=[]
        def closure(_x,_y):
            self.assertTrue(model.training);self.assertTrue(model.dropout.training)
            self.assertFalse(model.fused_attn)
            t=flat(params);mask=model.dropout(torch.ones_like(t))
            draws.append((random.random(),float(np.random.rand()),torch.rand(5),mask.detach().clone()))
            model.counter.add_(1);model.forward_cache['nested'][0].add_(2)
            model.forward_cache['new']=torch.ones(2);model.created_cache=[torch.ones(2)]
            model.dropout.eval();model.delta_w_q_new_grad=torch.ones(2)
            rng1.append(rng_get())
            return (t*mask).square().sum(),t[:2].unsqueeze(0)
        out=engine.compute(None,None,closure,.04)
        self.assertEqual(len(draws),3)
        self.assertEqual(draws[0][:2],draws[1][:2])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(draws[0][2:],draws[1][2:])))
        self.assertEqual(draws[0][:2],draws[2][:2])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(draws[0][2:],draws[2][2:])))
        self.assertTrue(rng_equal(rng_get(),rng1[0]))
        self.assertFalse(rng_equal(entry.rng,rng_get()))
        self.assertEqual(float(model.counter),0)
        self.assertFalse(hasattr(model,'created_cache'))
        self.assertTrue(out['record']['state_restoration']['gradient_references_exact'])
        engine.discard()
        self.assertTrue(all(entry.check().values()))
        receipt=engine.verify_zero_update_replay(None,None,closure,.04)
        self.assertTrue(receipt['passed']);self.assertEqual(receipt['ordinary_gradient_calls'],6)
        self.assertEqual(receipt['closure_calls'],6);self.assertEqual(receipt['sgd_updates'],0)
        self.assertTrue(all(entry.check().values()))

    def test_probe_failure_rolls_back_trainables_gradient_refs_cache_modes_rng(self):
        model,params,geometry,engine=case()
        for p in model.parameters():p.grad=torch.ones_like(p)
        entry=FullState(model);base_closure,_,_=quadratic(params);calls=[]
        def closure(x,y):
            calls.append(1);random.random();np.random.random();torch.rand(3)
            model.counter.add_(8);model.forward_cache['nested'][0].add_(3)
            model.forward_cache={'replacement':4};model.created_cache=[2]
            model.dropout.eval()
            if len(calls)==2:
                with torch.no_grad():model.a.add_(3);model.a.grad.add_(4)
                raise RuntimeError('injected second-probe failure')
            return base_closure(x,y)
        with self.assertRaisesRegex(RuntimeError,'injected second-probe'):
            engine.compute(None,None,closure,.04)
        self.assertEqual(len(calls),2);self.assertIsNone(engine._pending)
        self.assertTrue(all(entry.check().values()))
        self.assertFalse(hasattr(model,'created_cache'))

    def test_frozen_backbone_mutation_is_forbidden_by_version_guard(self):
        model,params,geometry,engine=case();closure,_,_=quadratic(params)
        def forbidden(x,y):
            with torch.no_grad():model.frozen.add_(1)
            return closure(x,y)
        with self.assertRaisesRegex(AssertionError,'mutated a frozen parameter'):
            engine.compute(None,None,forbidden,.04)
        # Frozen tensors are protected by version counters, not backed up each
        # update. This impossible-in-registered-forward fixture is fail-stop.
        self.assertIsNone(engine._pending)

    def test_one_exact_SGD_from_theta0_R1_and_effective_BA_plus_FC(self):
        model,params,geometry,engine=case(dtype=torch.float32)
        closure,_,_=quadratic(params);out=engine.compute(None,None,closure,.04)
        c=out['context'];base=c['base'];optimizer=torch.optim.SGD(params,lr=.04,foreach=False)
        with patch.object(optimizer,'step',wraps=optimizer.step) as spy:
            update=engine.apply_update(optimizer=optimizer)
        self.assertEqual(spy.call_count,1)
        self.assertTrue(all(torch.equal(p,v.clone().add_(u,alpha=-.04)) for p,v,u in zip(params,base,c['U'])))
        self.assertTrue(rng_equal(rng_get(),c['rng1']))
        self.assertEqual(update['sgd_updates'],1);self.assertEqual(len(optimizer.state),0)
        pmap={id(p):v.double() for p,v in zip(params,base)}
        lora=(model.b.double()@model.a.double()-pmap[id(model.b)]@pmap[id(model.a)]).square().sum()
        fc=sum((p.double()-pmap[id(p)]).square().sum() for p in model.fc.parameters())
        self.assertAlmostEqual(update['actual']['joint_norm'],float((lora+fc).sqrt()),places=12)
        with self.assertRaises(RuntimeError):engine.apply_update(optimizer=optimizer)

    def test_failed_optimizer_commit_restores_model_optimizer_state_and_R0(self):
        model,params,geometry,engine=case();closure,_,_=quadratic(params)
        for p in model.parameters():p.grad=torch.ones_like(p)
        entry=FullState(model);optimizer=torch.optim.SGD(params,lr=.04)
        original=copy.deepcopy(optimizer.state_dict())
        out=engine.compute(None,None,closure,.04)
        def failed_step():
            with torch.no_grad():params[0].add_(1)
            optimizer.state[params[0]]['injected']=torch.ones(2)
            torch.rand(2)
            raise RuntimeError('injected commit failure')
        with patch.object(optimizer,'step',side_effect=failed_step):
            with self.assertRaisesRegex(RuntimeError,'commit failure'):engine.apply_update(optimizer=optimizer)
        self.assertTrue(all(entry.check().values()))
        self.assertEqual(optimizer.state_dict(),original)
        self.assertIsNone(engine._pending)

    def test_forbidden_optimizer_and_changed_base_rejected_with_rollback(self):
        for mode in ('momentum','weight_decay','adam','matched','wronglr','changed_base'):
            model,params,geometry,engine=case();closure,_,_=quadratic(params);entry=FullState(model)
            optimizer=torch.optim.Adam(params,lr=.04) if mode=='adam' else torch.optim.SGD(params,lr=.04,momentum=.9 if mode=='momentum' else 0,weight_decay=.1 if mode=='weight_decay' else 0)
            engine.compute(None,None,closure,.04)
            if mode=='changed_base':
                with torch.no_grad():params[0].add_(1)
            with self.assertRaises((ValueError,RuntimeError)):
                engine.apply_update(.02 if mode=='wronglr' else .04,matched=mode=='matched',optimizer=optimizer)
            self.assertTrue(all(entry.check().values()))

    def test_strict_production_configs_and_inactive_gamma(self):
        cli=SimpleNamespace(max_tasks=0,max_epochs=0,max_batches=0,output_dir='/tmp/odd-even-cpu-unused')
        raw=dict(probe_branch=BRANCH,update_direction=DIRECTION,step_mode='natural',seed=[2])
        for gamma in (0.,.25):
            cfg=resolve_config(dict(raw,gam_grad_gamma=gamma),cli)
            self.assertEqual(cfg['method_name'],METHOD);self.assertIsNone(cfg['outer_radius'])
            self.assertEqual(cfg['gam_grad_norm_rho'],0.);self.assertFalse(cfg['gam_grad_norm_rho_active'])
            self.assertEqual(cfg['gam_grad_gamma'],gamma);self.assertFalse(cfg['gam_grad_gamma_active'])
            self.assertEqual(cfg['gam_grad_rho'],.05);self.assertEqual(cfg['lora_rank'],10)
        conflicts=dict(probe_branch='P',update_direction='g2',step_mode='matched',attention_backend='native',seed=[0],
            gam_grad_rho=.1,gam_grad_norm_rho=.25,eps=1e-10,lora_rank=5,epochs=2,momentum=.9,weight_decay=.1,
            ewc_lambda=1.,ewc_skip_task_fisher=False,approximate_fisher_in_task=True,in_task_weight_merge=True,
            task_endpoint_weight_merge=True,use_sgv=True,amp=True,tf32=True,normalize_final_gradient=True,
            hvp=True,create_graph=True,gradient_projection=True,component_rule='us',component_scale='matched',
            flat_eval=True,nb_tasks=5,total_sessions=2,max_train_tasks=3,outer_radius=.25,r1=.25,
            method_name='g2_central_location_v1')
        for key,value in conflicts.items():
            with self.subTest(key=key),self.assertRaises(ValueError):resolve_config(dict(raw,**{key:value}),cli)
        for override in ('resume_from_task','fc_lrate','min_lr'):
            with self.subTest(key=override),self.assertRaises(ValueError):resolve_config(dict(raw,**{override:0}),cli)
        model,params,geometry,engine=case()
        for bad in (dict(branch='C'),dict(direction='g2'),dict(backend='native'),dict(radius=0),dict(outer_radius=.25),
                    dict(rho=float('nan')),dict(eps=1e-10),dict(hvp=True),dict(create_graph=True),
                    dict(normalize_final_gradient=True),dict(fisher=True),dict(weight_merge=True),dict(gradient_projection=True)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):OddEvenStep(model,params,geometry,bad)
        with self.assertRaises(ValueError):engine.compute(None,None,lambda x,y:None,.04,observer=lambda c:None)

    def test_all_seven_exact_registered_formulas_FP32_and_FP64(self):
        for dtype in (torch.float32,torch.float64):
            for direction in DIRECTIONS:
                with self.subTest(dtype=dtype,direction=direction):
                    model,params,geometry,engine=case(dtype=dtype,direction=direction)
                    def closure(_x,_y):
                        x=flat(params)
                        return (x.pow(4)/4+.35*x.square()).sum(),x[:2].unsqueeze(0)
                    out=engine.compute(None,None,closure,.04);c=out['context']
                    z,p,m=c['g0'],c['g1_plus'],c['g1_minus']
                    expected={
                        'g1_plus':tuple(v.clone() for v in p),
                        'backward_reflected':tuple(a+(a-b) for a,b in zip(z,m)),
                        'central':tuple(a+.5*(b-d) for a,b,d in zip(z,p,m)),
                        'g0':tuple(v.clone() for v in z),
                        'g1_minus':tuple(v.clone() for v in m),
                        'even_average':tuple(.5*(a+b) for a,b in zip(p,m)),
                        'even_reflected':tuple(a+(a-.5*(b+d)) for a,b,d in zip(z,p,m))}[direction]
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(out['U'],expected)))
                    self.assertEqual(out['record']['ordinary_gradient_calls'],3)
                    self.assertGreater(out['record']['norms']['e'],0.)
                    self.close(flat(c['s']),.5*(flat(p)-flat(m)),atol=1e-7,rtol=1e-7)
                    self.close(flat(c['e']),.5*((flat(p)-flat(z))+(flat(m)-flat(z))),atol=1e-7,rtol=1e-7)
                    for side,sgn in [('phi1_plus',1),('phi1_minus',-1)]:
                        displacement=flat(c['points'][side])-flat(c['base'])
                        unit=flat(z)/(torch.linalg.vector_norm(flat(z))+1e-12)
                        self.close(displacement,sgn*.05*unit,atol=1e-7,rtol=1e-7)
                    optimizer=torch.optim.SGD(params,lr=.04,foreach=False)
                    with patch.object(optimizer,'step',wraps=optimizer.step) as spy:
                        update=engine.apply_update(optimizer=optimizer)
                    self.assertEqual(spy.call_count,1)
                    self.assertTrue(all(torch.equal(a,b.clone().add_(u,alpha=-.04)) for a,b,u in zip(params,c['base'],expected)))
                    self.assertTrue(update['natural_odd_even_only'])

    def test_additional_plus_probe_preserves_old_backward_exactly(self):
        from reflected_step import ReflectedStep
        model,params,geometry,new=case(dtype=torch.float32,direction='backward_reflected')
        old=ReflectedStep(model,params,geometry,{})
        closure,_,_=quadratic(params)
        before=FullState(model)
        a=new.compute(None,None,closure,.04);new.discard()
        b=old.compute(None,None,closure,.04);old.discard()
        for name in ('g0','g1_minus','U'):
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a['context'][name],b['context'][name])))
        self.assertTrue(rng_equal(a['context']['rng1'],b['context']['rng1']))
        self.assertTrue(all(before.check().values()))

    def test_quadratic_even_zero_and_three_curvature_updates_agree(self):
        outputs=[]
        for direction in ('g1_plus','backward_reflected','central'):
            model,params,geometry,engine=case(direction=direction)
            closure,H,b=quadratic(params)
            out=engine.compute(None,None,closure,.04)
            self.assertLess(out['record']['norms']['e'],1e-15)
            outputs.append(flat(out['U']));engine.discard()
        self.close(outputs[0],outputs[1]);self.close(outputs[1],outputs[2])

    def test_all_directions_resolve_and_coefficients_rejected(self):
        cli=SimpleNamespace(max_tasks=0,max_epochs=0,max_batches=0,output_dir='/tmp/odd-even-cpu-unused')
        for direction in DIRECTIONS:
            raw=dict(probe_branch=BRANCH,update_direction=direction,step_mode='natural',seed=[2])
            cfg=resolve_config(raw,cli)
            self.assertEqual(cfg['update_direction'],direction)
            for key in ('alpha_s','alpha_e','lambda_H','correction_coefficient','coefficient','normalize_correction'):
                with self.assertRaises(ValueError):resolve_config(dict(raw,**{key:1}),cli)

    def test_real_trainer_two_tasks_three_CE_and_six_isolated_CE_each(self):
        class MiniModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.backbone=torch.nn.Module()
                self.backbone.w_As=torch.nn.ModuleList([torch.nn.Linear(2,1,bias=False)])
                self.backbone.w_Bs=torch.nn.ModuleList([torch.nn.Linear(1,2,bias=False)])
                self.backbone.wrapped_param=[];self.fc=torch.nn.Linear(2,20);self.dropout=torch.nn.Dropout(.2)
            def forward(self,x):
                h=self.backbone.w_Bs[0](self.backbone.w_As[0](x))
                return {'logits':self.fc(self.dropout(x+h))}
        class Learner:
            def __init__(self):
                self.model=MiniModel();self._cur_task=0;self._total_classes=20
                self._known_classes=20;self._ewc_skip_task_fisher=True;self._lambda_ewc=0.
            def _unwrap_network(self):return self.model
            def _task_loss(self,output,y,class_offset):
                return torch.nn.functional.cross_entropy(output[:,class_offset:class_offset+20],y-class_offset),None,None
            def _run_regularized_train(self,*a,**kw):pass
            def incremental_train(self,*a,**kw):pass
            def eval_task(self,*a,**kw):pass
            def after_task(self):pass
        with tempfile.TemporaryDirectory(prefix='odd-even-cpu-integration-') as directory:
            cli=SimpleNamespace(max_tasks=2,max_epochs=1,max_batches=1,output_dir=directory)
            cfg=resolve_config(dict(probe_branch=BRANCH,update_direction=DIRECTION,step_mode='natural',seed=[2]),cli)
            cfg['seed']=2
            learner=Learner();controller=Controller(learner,cfg,cli);controller.install()
            for task in range(2):
                controller.task=task;learner._cur_task=task;learner._total_classes=20*(task+1)
                controller.old_fc={} if task==0 else {k:v.clone() for k,v in learner.model.fc.state_dict().items()}
                if task:
                    old=learner.model.fc;learner.model.fc=torch.nn.Linear(2,40)
                    with torch.no_grad():
                        learner.model.fc.weight[:20].copy_(old.weight);learner.model.fc.bias[:20].copy_(old.bias)
                optimizer=learner._build_optimizer(list(learner.model.parameters()),'init' if task==0 else 'later')
                controller.task_start_step=controller.steps;controller.task_start_counters=dict(controller.counters)
                controller.begin_epoch(1);controller.task_batch_count=1
                x=torch.tensor([[.2,.4],[.7,-.1]]);y=torch.tensor([20*task,20*task+1])
                controller.batch_identity=dict(input_sha256='test-input',target_sha256='test-target-'+str(task),indices_sha256='test-indices',indices=[0,1],data_seed=200001+task*1000)
                before={k:v.clone() for k,v in learner.model.fc.state_dict().items()}
                with patch.object(optimizer,'step',wraps=optimizer.step) as spy:
                    learner._step_batch_with_log(optimizer,x,y,20*task)
                self.assertEqual(spy.call_count,1)
                self.assertEqual(controller.stepper.total_ordinary_gradient_calls,9)
                if task:
                    self.assertTrue(all(torch.equal(before[k][:20],v[:20]) for k,v in learner.model.fc.state_dict().items()))
                controller.write_task_invariants();learner.after_task()
            self.assertEqual(controller.counters['training_ce_gradient_passes'],6)
            self.assertEqual(controller.counters['training_forward_calls'],6)
            self.assertEqual(controller.counters['diagnostic_replay_ce_gradient_passes'],12)
            self.assertEqual(controller.counters['diagnostic_replay_forward_calls'],12)
            self.assertEqual(controller.counters['sgd_updates'],2)
            receipts=json.loads((Path(directory)/'task_invariant_receipts.json').read_text())
            self.assertEqual(len(receipts),2);self.assertTrue(all(r['passed'] for r in receipts))
            trace=[json.loads(row) for row in (Path(directory)/'step_trace.jsonl').read_text().splitlines()]
            self.assertEqual(len(trace),2)
            self.assertEqual(set(trace[0]['probe_tensor_sha256']),{'g0','g1_plus','g1_minus','plus_point','minus_point'})
            self.assertEqual(len(trace[0]['rng0_sha256']),64)
            self.assertEqual(len(trace[0]['rng1_sha256']),64)
            self.assertTrue(all(r['ordinary_gradient_calls']==3 and r['sgd_updates']==1 and r['natural_odd_even_only'] for r in trace))
            self.assertEqual([r['class_slice'] for r in trace],[[0,20],[20,40]])
            self.assertTrue(all(r['old_fc_rows_inherited_exact'] for r in controller.task_identities))
            self.assertEqual(len(controller.after_task_checks),2)
            controller.handle.close()


def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='passed' if result.wasSuccessful() else 'failed',cpu_only=True,
                cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_available=torch.cuda.is_available(),
                tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                test_names=unittest.defaultTestLoader.getTestCaseNames(Acceptance),
                scope='analytic quadratic reflection; nonlinear SAM first-order equivalence; exact FP32 parentheses; degeneracies; R0 replay/R1 once; trainable/buffer/gradient-reference/cache/mode/RNG and optimizer rollback; frozen version guard; 3 CE; seven exact formulas; single SGD; forbidden configs; real trainer CPU two-task integration',
                gpu_acceptance=False,ordinary_training_allowed=False,
                odd_even_step_sha256=digest(Path(__file__).with_name('odd_even_step.py')),
                trainer_sha256=digest(Path(__file__).with_name('odd_even_train.py')),
                tests_sha256=digest(__file__),python=sys.executable,torch_version=torch.__version__)
    atomic_json(Path(__file__).with_name('ODD_EVEN_CPU_GATE.json'),report)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
