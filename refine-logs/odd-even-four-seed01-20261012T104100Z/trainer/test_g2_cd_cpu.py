"""Independent CPU acceptance for the isolated P/C/CD outer-location method."""
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
from audit_core import rng_get, rng_equal, seed_all
from direction_step import FourProbeStep, FullState, vector_norm
from effective_step import EffectiveStep
from four_probe_train import resolve_config, Controller


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

    def forward(self,_x=None):
        return torch.cat([p.reshape(-1) for p in (self.a,self.b,self.fc.weight,self.fc.bias)])


def flat(values):
    return torch.cat([v.reshape(-1) for v in values])


def case(branch='P',dtype=torch.float64,reverse=False,**cfg):
    model = Tiny(dtype)
    ordinary = tuple(p for p in model.parameters() if p.requires_grad)
    params = tuple(reversed(ordinary)) if reverse else ordinary
    geom = EffectiveStep([(model.a,model.b)],tuple(model.fc.parameters()),params)
    engine = FourProbeStep(model,params,geom,dict(branch=branch,**cfg))
    return model,params,geom,engine


def loss_case(params,alpha=0.,linear=False,stationary=False):
    H = torch.diag(torch.tensor([1.,2.,3.,4.],dtype=params[0].dtype))
    b = torch.tensor([-.2,.1,-.3,.4],dtype=params[0].dtype)
    origin = flat(params).detach().clone()
    def closure(_x,_y):
        x=flat(params)
        if linear:
            loss=b@x
        elif stationary:
            loss=.5*(x-origin).square().sum()
        else:
            loss=.5*x@(H@x)+b@x+alpha*x.pow(3).sum()/3
        return loss,x.unsqueeze(0)
    def grad(x):
        if linear:return b
        if stationary:return x-origin
        return H@x+b+alpha*x.square()
    return closure,grad,H


class Acceptance(unittest.TestCase):
    def setUp(self):
        seed_all(2)
        torch.use_deterministic_algorithms(True,warn_only=False)

    def close(self,a,b,atol=1e-12):
        self.assertTrue(torch.allclose(a,b,rtol=1e-12,atol=atol),str((a-b).abs().max()))

    def test_quadratic_three_locations_gradients_and_g2_only(self):
        outputs=[]
        for branch in ('P','C','CD'):
            model,params,geom,engine=case(branch)
            closure,grad,H=loss_case(params)
            x=flat(params).detach().clone()
            g0=grad(x);u0=g0/(float(torch.linalg.vector_norm(g0))+1e-12)
            expectedq=.05*(H@u0)
            expectedouter=x+.25*expectedq/(float(torch.linalg.vector_norm(expectedq))+1e-12)
            out=engine.compute(None,None,closure,.04)
            c=out['context']
            self.close(flat(c['q']),expectedq)
            self.close(flat(c['outer']),expectedouter)
            self.close(flat(out['U']),grad(expectedouter))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(out['U'],c['g2'])))
            self.assertIsNone(c['grads']['g1']);self.assertIsNone(c['c21'])
            self.assertEqual(out['record']['ordinary_gradient_calls'],4)
            for field in ('hvp_calls','create_graph_calls','fisher_calls','weight_merge_calls','gradient_projection_calls'):
                self.assertEqual(out['record'][field],0)
            outputs.append((flat(c['q']),flat(c['outer']),flat(c['g2'])))
            engine.discard()
        for result in outputs[1:]:
            for observed,expected in zip(result,outputs[0]):self.close(observed,expected)

    def test_cubic_explicit_cd_and_base_outer_sign(self):
        for branch in ('P','C','CD'):
            model,params,geom,engine=case(branch)
            closure,grad,H=loss_case(params,alpha=.7)
            x=flat(params).detach().clone();g0=grad(x)
            u=g0/(float(torch.linalg.vector_norm(g0))+1e-12)
            xp=x+.05*u;xm=x-.05*u
            gp=grad(xp);gm=grad(xm)
            q={'P':gp-g0,'C':g0-gm,'CD':.5*(gp-gm)}[branch]
            outer=x+.25*q/(float(torch.linalg.vector_norm(q))+1e-12)
            out=engine.compute(None,None,closure,.04);c=out['context']
            self.close(flat(c['g0']),g0);self.close(flat(c['g1_plus']),gp);self.close(flat(c['g1_minus']),gm)
            self.close(flat(c['q']),q);self.close(flat(c['outer']),outer);self.close(flat(c['g2']),grad(outer))
            self.assertGreater(float(torch.linalg.vector_norm(flat(c['outer'])-(xp+.25*q/(float(torch.linalg.vector_norm(q))+1e-12)))),.049)
            self.assertTrue(all(torch.equal(a,.5*(p-m)) for a,p,m in zip(c['qCD'],c['g1_plus'],c['g1_minus'])))
            self.close(flat(c['qP']),flat(c['s'])+flat(c['e']))
            self.close(flat(c['qC']),flat(c['s'])-flat(c['e']))
            self.assertFalse(out['record']['low_s'])
            self.assertIsNone(out['record']['c21'])
            engine.discard()

    def test_zero_radius_preserves_g0_bitwise(self):
        for branch in ('P','C','CD'):
            model,params,geom,engine=case(branch,radius=0.)
            closure,_,_=loss_case(params,alpha=.7)
            out=engine.compute(None,None,closure,.04)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(out['context']['g0'],out['context']['g2'])))
            self.assertEqual(out['record']['parameter_probe_displacements']['phi2'],0.)
            engine.discard()

    def test_zero_q_no_fallback_and_missing_angles(self):
        for branch in ('P','C','CD'):
            model,params,geom,engine=case(branch)
            closure,_,_=loss_case(params,linear=True)
            out=engine.compute(None,None,closure,.04)
            self.assertEqual(vector_norm(out['context']['q']),0.)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(out['context']['outer'],out['context']['base'])))
            self.assertIsNone(out['record']['cosines']['qP_qC'])
            self.assertTrue(out['record']['low_s'])
            engine.discard()

    def test_zero_g0_collapses_all_probes(self):
        model,params,geom,engine=case('CD')
        closure,_,_=loss_case(params,stationary=True)
        out=engine.compute(None,None,closure,.04)
        for name in ('g0','g1_plus','g1_minus','g2','q'):
            self.assertEqual(vector_norm(out['context'][name]),0.)
        self.assertIsNone(out['record']['cosines']['g0_g2'])
        engine.discard()

    def stochastic_closure(self,model,params,draws):
        def closure(_x,_y):
            x=flat(params)
            mask=model.dropout(torch.ones_like(x))
            draw=(random.random(),float(np.random.rand()),torch.rand(3),mask.detach().clone())
            draws.append(draw)
            model.counter.add_(1)
            model.dropout.training=False
            model.forward_cache['nested'][0].add_(9)
            model.forward_cache={'replaced':torch.ones(1)}
            model.created_cache=[torch.ones(1)]
            model.delta_w_q_new_grad=torch.ones(1)
            model.external_delta_w_q=torch.ones(1)
            model._register_delta_hook=True
            return (x*mask).square().sum()+draw[0]*x.sum(),x.unsqueeze(0)
        return closure

    def test_same_random_function_and_success_advance_once(self):
        model,params,geom,engine=case('CD')
        entry=FullState(model);draws=[]
        out=engine.compute(None,None,self.stochastic_closure(model,params,draws),.04)
        self.assertEqual(len(draws),4)
        for d in draws[1:]:
            self.assertEqual(d[:2],draws[0][:2])
            self.assertTrue(torch.equal(d[2],draws[0][2]));self.assertTrue(torch.equal(d[3],draws[0][3]))
        self.assertTrue(rng_equal(rng_get(),out['context']['rng1']))
        self.assertFalse(rng_equal(rng_get(),entry.rng))
        engine.discard();self.assertTrue(all(entry.check().values()))

    def test_discard_restores_existing_gradient_and_nested_cache_identities(self):
        model,params,geom,engine=case('CD')
        for p in params:p.grad=torch.full_like(p,3.)
        model.frozen.grad=torch.full_like(model.frozen,4.)
        entry=FullState(model)
        cache=model.forward_cache;container=cache['nested'];tensor=container[0]
        engine.compute(None,None,self.stochastic_closure(model,params,[]),.04)
        engine.discard()
        self.assertTrue(all(entry.check().values()))
        self.assertIs(model.forward_cache,cache);self.assertIs(cache['nested'],container);self.assertIs(container[0],tensor)
        self.assertEqual(float(tensor),2.)

    def test_failed_compute_restores_every_entry_component(self):
        model,params,geom,engine=case('CD')
        for p in params:p.grad=torch.full_like(p,3.)
        entry=FullState(model);draws=[];wrapped=self.stochastic_closure(model,params,draws)
        def failing(x,y):
            result=wrapped(x,y)
            if len(draws)==3:raise RuntimeError('deliberate probe failure')
            return result
        with self.assertRaisesRegex(RuntimeError,'deliberate'):engine.compute(None,None,failing,.04)
        self.assertTrue(all(entry.check().values()));self.assertIsNone(engine._pending)

    def test_strict_two_repeat_replay_and_eight_diagnostic_calls(self):
        for branch in ('P','C','CD'):
            model,params,geom,engine=case(branch)
            entry=FullState(model);draws=[]
            receipt=engine.verify_zero_update_replay(None,None,self.stochastic_closure(model,params,draws),.04)
            self.assertEqual(len(draws),8);self.assertEqual(receipt['ordinary_gradient_calls'],8)
            self.assertTrue(receipt['passed']);self.assertTrue(all(receipt['checks'].values()))
            self.assertEqual(receipt['sgd_updates'],0);self.assertTrue(all(entry.check().values()))

    def test_replay_failure_stops_and_restores(self):
        model,params,geom,engine=case('CD');entry=FullState(model);calls=[0]
        closure,_,_=loss_case(params)
        def changing(x,y):
            calls[0]+=1;loss,logits=closure(x,y)
            return loss+calls[0]*flat(params).sum(),logits
        with self.assertRaisesRegex(AssertionError,'Strict first-batch'):engine.verify_zero_update_replay(None,None,changing,.04)
        self.assertTrue(all(entry.check().values()));self.assertIsNone(engine._pending)

    def test_unique_natural_sgd_exact_in_float32_and_float64(self):
        for dtype in (torch.float64,torch.float32):
            for branch in ('P','C','CD'):
                model,params,geom,engine=case(branch,dtype=dtype,reverse=True)
                closure,_,_=loss_case(params,alpha=.7)
                base=tuple(p.detach().clone() for p in params)
                optimizer=torch.optim.SGD(params,lr=.04,momentum=0,weight_decay=0,foreach=False)
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=.004)
                scheduler_state=copy.deepcopy(scheduler.state_dict())
                with patch.object(optimizer,'step',wraps=optimizer.step) as spy:
                    out=engine.compute(None,None,closure,.04)
                    post=out['context']['rng1'];update=engine.apply_update(.04,optimizer=optimizer)
                self.assertEqual(spy.call_count,1)
                self.assertTrue(all(torch.equal(p,v.clone().add_(u,alpha=-.04)) for p,v,u in zip(params,base,out['U'])))
                self.assertTrue(rng_equal(rng_get(),post));self.assertEqual(update['sgd_updates'],1)
                self.assertEqual(scheduler.state_dict(),scheduler_state);self.assertEqual(len(optimizer.state),0)
                self.assertEqual(optimizer.param_groups[0]['lr'],.04)
                self.assertGreater(update['actual']['joint_norm'],0)
                self.assertIn('lora_norm',update['actual']);self.assertIn('fc_norm',update['actual'])

    def test_failed_update_rolls_back_model_rng_grads_and_optimizer(self):
        model,params,geom,engine=case('CD')
        for p in params:p.grad=torch.full_like(p,3.)
        entry=FullState(model)
        class BadSGD(torch.optim.SGD):
            def step(self,closure=None):
                super().step(closure)
                model.counter.add_(8);model.forward_cache['nested'][0].add_(2)
                model.dropout.training=False;random.random();np.random.rand();torch.rand(2)
                self.state[params[0]]['bad']=torch.ones(2);self.param_groups[0]['lr']=.9
                raise RuntimeError('deliberate update failure')
        optimizer=BadSGD(params,lr=.04)
        old=copy.deepcopy(optimizer.state_dict())
        closure,_,_=loss_case(params)
        engine.compute(None,None,closure,.04)
        with self.assertRaisesRegex(RuntimeError,'deliberate'):engine.apply_update(.04,optimizer=optimizer)
        self.assertTrue(all(entry.check().values()));self.assertEqual(optimizer.state_dict(),old)
        self.assertIsNone(engine._pending)

    def test_invalid_update_settings_restore_entry(self):
        for mode in ('matched','momentum','wronglr','adam'):
            model,params,geom,engine=case('CD');entry=FullState(model)
            closure,_,_=loss_case(params)
            optimizer=torch.optim.Adam(params,lr=.04) if mode=='adam' else torch.optim.SGD(params,lr=.04,momentum=.9 if mode=='momentum' else 0)
            engine.compute(None,None,closure,.04)
            with self.assertRaises(ValueError):engine.apply_update(.02 if mode=='wronglr' else .04,matched=mode=='matched',optimizer=optimizer)
            self.assertTrue(all(entry.check().values()))

    def test_four_ordinary_autograd_calls_and_no_create_graph(self):
        model,params,geom,engine=case('CD');closure,_,_=loss_case(params)
        original=torch.autograd.grad
        with patch.object(torch.autograd,'grad',wraps=original) as spy:
            out=engine.compute(None,None,closure,.04)
        self.assertEqual(spy.call_count,4)
        self.assertTrue(all(call.kwargs['create_graph'] is False for call in spy.call_args_list))
        self.assertEqual(engine.total_ordinary_gradient_calls,4);engine.discard()

    def test_actual_trainer_batch_replay_and_observed_task_receipts(self):
        class MiniModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone=torch.nn.Module()
                self.backbone.w_As=torch.nn.ModuleList([torch.nn.Linear(2,1,bias=False)])
                self.backbone.w_Bs=torch.nn.ModuleList([torch.nn.Linear(1,2,bias=False)])
                self.backbone.wrapped_param=[]
                self.fc=torch.nn.Linear(2,20)
                self.dropout=torch.nn.Dropout(.2)
            def forward(self,x):
                h=self.backbone.w_Bs[0](self.backbone.w_As[0](x))
                return {"logits":self.fc(self.dropout(x+h))}
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
        for branch in ('P','C','CD'):
            seed_all(2)
            with tempfile.TemporaryDirectory(prefix='g2-cpu-integration-') as directory:
                cli=SimpleNamespace(max_tasks=2,max_epochs=1,max_batches=1,output_dir=directory)
                cfg=resolve_config(dict(probe_branch=branch,update_direction='g2',step_mode='natural',seed=[2]),cli)
                cfg['seed']=2  # frozen src.trainer passes its independent scalar seed to factory
                learner=Learner();controller=Controller(learner,cfg,cli)
                controller.install();controller.task=0;controller.old_fc={}
                optimizer=learner._build_optimizer(list(learner.model.parameters()),'init')
                controller.task_start_step=0;controller.task_start_counters=dict(controller.counters)
                controller.begin_epoch(1)
                controller.task_batch_count=2
                x=torch.tensor([[.2,.4],[.7,-.1]]);y=torch.tensor([0,1])
                controller.batch_identity=dict(input_sha256='test-input',target_sha256='test-target',indices_sha256='test-indices',indices=[0,1],data_seed=200001)
                with patch.object(optimizer,'step',wraps=optimizer.step) as spy:
                    for batch in range(2):
                        controller.batch=batch
                        learner._step_batch_with_log(optimizer,x,y,0)
                self.assertEqual(spy.call_count,2)
                self.assertEqual(controller.stepper.total_ordinary_gradient_calls,16)
                self.assertEqual(controller.counters['training_ce_gradient_passes'],8)
                self.assertEqual(controller.counters['training_forward_calls'],8)
                self.assertEqual(controller.counters['diagnostic_replay_ce_gradient_passes'],8)
                self.assertEqual(controller.counters['diagnostic_replay_forward_calls'],8)
                controller.write_task_invariants()
                receipt=json.loads((Path(directory)/'task_invariant_receipts.json').read_text())[0]
                self.assertEqual(receipt['status'],'passed');self.assertTrue(all(receipt['checks'].values()))
                replay=json.loads((Path(directory)/'replay_receipts.json').read_text())
                self.assertEqual(len(replay),1);self.assertEqual(replay[0]['forward_calls'],8)
                trace=[json.loads(row) for row in (Path(directory)/'step_trace.jsonl').read_text().splitlines()]
                self.assertEqual(len(trace),2)
                self.assertTrue(all(row['ordinary_gradient_calls']==4 and row['sgd_updates']==1 for row in trace))
                learner.after_task();self.assertEqual(len(controller.after_task_checks),1)
                controller.handle.close()

    def test_strong_configs_reject_conflicts_and_allow_three_frozen_branches(self):
        cli=SimpleNamespace(max_tasks=0,max_epochs=0,max_batches=0,output_dir='/tmp/g2_cpu_unused')
        raw=dict(probe_branch='CD',update_direction='g2',step_mode='natural',attention_backend='math',seed=[2])
        for branch in ('P','C','CD'):
            cfg=resolve_config(dict(raw,probe_branch=branch),cli)
            self.assertEqual(cfg['probe_branch'],branch);self.assertEqual(cfg['seed'],[2])
            self.assertEqual(cfg['gam_grad_rho'],.05);self.assertEqual(cfg['gam_grad_norm_rho'],.25)
            self.assertEqual(cfg['epochs'],50);self.assertTrue(cfg['deterministic_algorithms'])
        conflicts=dict(probe_branch='D',update_direction='A',step_mode='matched',attention_backend='native',seed=[0],
            gam_grad_rho=.1,gam_grad_norm_rho=1.,eps=1e-10,lora_rank=5,epochs=2,momentum=.9,weight_decay=.1,
            ewc_lambda=1.,ewc_skip_task_fisher=False,approximate_fisher_in_task=True,
            in_task_weight_merge=True,task_endpoint_weight_merge=True,use_sgv=True,amp=True,tf32=True,
            normalize_final_gradient=True,hvp=True,component_rule='us',component_scale='matched',
            in_task_path_normalize_snapshots=True,flat_eval=True,nb_tasks=5,total_sessions=2,
            max_train_tasks=3,backbone_type="vit_small_patch16_224")
        for key,value in conflicts.items():
            with self.subTest(key=key),self.assertRaises(ValueError):resolve_config(dict(raw,**{key:value}),cli)
        for override in ('resume_from_task','fc_lrate','min_lr'):
            with self.subTest(key=override),self.assertRaises(ValueError):resolve_config(dict(raw,**{override:0}),cli)
        model,params,geom,engine=case()
        for bad in (dict(branch='D'),dict(branch='CD',direction='A'),dict(backend='native'),dict(radius=float('nan')),dict(eps=1e-10),dict(hvp=True)):
            with self.assertRaises(ValueError):FourProbeStep(model,params,geom,bad)
        with self.assertRaises(ValueError):engine.compute(None,None,lambda x,y:None,.04,observer=lambda c:None)


def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    from audit_core import digest,atomic_json
    path=Path(__file__).with_name('G2_CD_CPU_GATE.json')
    report=dict(status='passed' if result.wasSuccessful() else 'failed',cpu_only=True,
                cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_available=torch.cuda.is_available(),
                tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                test_names=[name for name in unittest.defaultTestLoader.getTestCaseNames(Acceptance)],
                scope='independent quadratic/cubic signs; degeneracies; full state/RNG/cache identity rollback; strict repeat; unique SGD; invalid config; real trainer toy batch lifecycle',
                gpu_acceptance=False,ordinary_training_allowed=False,
                direction_step_sha256=digest(Path(__file__).with_name('direction_step.py')),
                trainer_sha256=digest(Path(__file__).with_name('four_probe_train.py')),
                tests_sha256=digest(__file__),python=sys.executable,torch_version=torch.__version__)
    atomic_json(path,report)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
