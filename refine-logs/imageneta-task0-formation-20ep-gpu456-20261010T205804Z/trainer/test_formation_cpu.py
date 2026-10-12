"""Independent20epoch schedule, gradient formula, state and pairing checks."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import formation_train as train
from formation_step import FormationStep,CONDITIONS,READS,active_method,cosine_lr
from state_transaction import State
import audit_core as audit

class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__();self.x=torch.nn.Parameter(torch.tensor([.8,-.6,.3]))
        self.register_buffer('H',torch.tensor([[3.,.7,.2],[.7,2.,-.1],[.2,-.1,1.]]))
        self.register_buffer('linear',torch.tensor([.2,-.3,.4]))
    def forward(self,_):return .5*self.x@self.H@self.x+self.linear@self.x

class Stateful(torch.nn.Module):
    def __init__(self):
        super().__init__();self.fc=torch.nn.Linear(3,20);self.drop=torch.nn.Dropout(.3)
        self.register_buffer('counter',torch.zeros(1));self.forward_cache={'tensor':torch.ones(1)}
    def forward(self,x):
        self.counter.add_(1.);self.forward_cache['tensor'].add_(1.)
        return self.fc(self.drop(x))

class FormationTests(unittest.TestCase):
    def test_six_schedules_native_budgets_and_epoch11_boundaries(self):
        reads=[]
        for condition in CONDITIONS:
            per_task=[[READS[active_method(condition,task,e)] for e in range(1,21)] for task in (0,1)]
            self.assertEqual(sum(per_task[1]),20)
            reads.append(sum(per_task[0])*9+sum(per_task[1])*8)
            expected=condition if not condition.endswith('then_sgd') and not condition.startswith('sgd_then') else None
            self.assertEqual(active_method(condition,1,11),'sgd')
        self.assertEqual(reads,[340,520,880,700,520,520])
        self.assertEqual(3*sum(reads),10440)
        self.assertEqual(active_method('sf_g2_then_sgd',0,10),'sf_g2')
        self.assertEqual(active_method('sf_g2_then_sgd',0,11),'sgd')
        self.assertEqual(active_method('sgd_then_sf_g2',0,10),'sgd')
        self.assertEqual(active_method('sgd_then_sf_g2',0,11),'sf_g2')
        with self.assertRaises(ValueError):active_method('sam',0,21)

    def test_four_active_rules_match_independent_quadratic_probe_oracle(self):
        for condition in CONDITIONS[:4]:
            model=Quadratic();origin=model.x.detach().clone();points=[]
            def closure(*_):
                points.append(model.x.detach().clone());loss=model(None)
                return loss,model.x.view(1,-1)
            def grad(x):return model.H@x+model.linear
            g0=grad(origin);p1=origin+.05*g0/(float(g0.double().norm())+1e-12)
            g1=grad(p1);s=g1-g0;p2=origin+s/(float(s.double().norm())+1e-12)
            g2=grad(p2);p3=p2+.05*g2/(float(g2.double().norm())+1e-12);g3=grad(p3)
            pf=origin.clone().add_(g0,alpha=-.02);gf=grad(pf);resp=g0-gf
            p2sf=origin.clone().add_(resp,alpha=.25/float(resp.double().norm()));g2sf=grad(p2sf)
            expected={'sgd':(g0,[origin]),'sam':(g1,[origin,p1]),
                'gam_residual':(g0+(g3-g2),[origin,p1,p2,p3]),'sf_g2':(g2sf,[origin,pf,p2sf])}[condition]
            step=FormationStep(model,[model.x],None,dict(formation_condition_id=condition,_formation_task=0))
            step.set_epoch(1);out=step.compute(None,None,closure,.02)
            self.assertEqual(out['record']['ordinary_gradient_calls'],READS[condition])
            torch.testing.assert_close(out['U'][0],expected[0],rtol=3e-6,atol=2e-6)
            self.assertEqual(len(points),len(expected[1]))
            for got,want in zip(points,expected[1]):torch.testing.assert_close(got,want,rtol=3e-6,atol=2e-6)
            self.assertTrue(torch.equal(model.x,origin))
            opt=torch.optim.SGD([model.x],lr=.02,momentum=0,weight_decay=0)
            commit=step.apply_update(.02,opt)
            self.assertTrue(torch.equal(model.x,origin.clone().add_(out['U'][0],alpha=-.02)))
            self.assertEqual(commit['sgd_updates'],1)
            self.assertFalse(commit['history_cache_valid_after_update'])
            self.assertIsNone(step.state_dict()['history'])

    def test_two_task20_cosine_and_switch_preserve_one_optimizer_model_and_rng(self):
        for condition in CONDITIONS:
            model=Quadratic();all_reads=0;all_updates=0
            for task in (0,1):
                initial=.04 if task==0 else .02
                opt=torch.optim.SGD([model.x],lr=initial,momentum=0,weight_decay=0)
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=initial*.1)
                step=FormationStep(model,[model.x],None,dict(formation_condition_id=condition,_formation_task=task))
                ids=(id(model),id(model.x),id(opt),id(scheduler))
                for epoch in range(1,21):
                    before=State(model);step.set_epoch(epoch);before.check()
                    self.assertEqual(ids,(id(model),id(model.x),id(opt),id(scheduler)))
                    lr=float(opt.param_groups[0]['lr']);self.assertAlmostEqual(lr,cosine_lr(initial,epoch),places=15)
                    def closure(*_):return model(None),model.x.view(1,-1)
                    out=step.compute(None,None,closure,lr);commit=step.apply_update(lr,opt)
                    all_reads+=out['record']['ordinary_gradient_calls'];all_updates+=commit['sgd_updates']
                    self.assertEqual(out['record']['active_method'],'sgd' if task==1 else active_method(condition,task,epoch))
                    scheduler.step();self.assertEqual(scheduler.last_epoch,epoch)
                self.assertAlmostEqual(opt.param_groups[0]['lr'],initial*.1,places=15)
            self.assertEqual(all_updates,40)
            self.assertEqual(all_reads,{'sgd':40,'sam':60,'gam_residual':100,'sf_g2':80,'sf_g2_then_sgd':60,'sgd_then_sf_g2':60}[condition])

    def test_first10_epochs_pair_bitwise_equal_then_only_selected_schedule_changes(self):
        for first,second in (('sgd','sgd_then_sf_g2'),('sf_g2','sf_g2_then_sgd')):
            trajectories=[]
            for condition in (first,second):
                model=Quadratic();opt=torch.optim.SGD([model.x],lr=.04,momentum=0,weight_decay=0)
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=.004)
                step=FormationStep(model,[model.x],None,dict(formation_condition_id=condition,_formation_task=0))
                rows=[]
                for epoch in range(1,21):
                    step.set_epoch(epoch)
                    def closure(*_):return model(None),model.x.view(1,-1)
                    out=step.compute(None,None,closure,opt.param_groups[0]['lr']);commit=step.apply_update(opt.param_groups[0]['lr'],opt)
                    rows.append((out['record']['g0_sha256'],out['record']['U_sha256'],commit['actual_post_parameter_sha256']))
                    scheduler.step()
                trajectories.append(rows)
            self.assertEqual(trajectories[0][:10],trajectories[1][:10])
            self.assertNotEqual(trajectories[0][10],trajectories[1][10])

    def test_every_kernel_restores_complete_state_and_commits_only_base_rng(self):
        for condition in CONDITIONS[:4]:
            torch.manual_seed(123);model=Stateful();model.train();model.fc.weight.grad=torch.ones_like(model.fc.weight)
            entry=State(model);x=torch.ones(4,3);target=torch.tensor([0,1,2,3])
            def closure(inputs,targets):
                logits=model(inputs);return torch.nn.functional.cross_entropy(logits,targets),logits
            closure(x,target);expected_rng=audit.rng_get();entry.restore();entry.check()
            ids=[id(p) for p in model.parameters()]
            step=FormationStep(model,list(model.parameters()),None,dict(formation_condition_id=condition,_formation_task=0));step.set_epoch(1)
            out=step.compute(x,target,closure,.02)
            self.assertTrue(audit.rng_equal(audit.rng_get(),expected_rng));self.assertEqual(ids,[id(p) for p in model.parameters()])
            self.assertEqual(float(model.counter),0.);self.assertEqual(float(model.forward_cache['tensor']),1.)
            self.assertTrue(torch.equal(model.fc.weight.grad,torch.ones_like(model.fc.weight)))
            for key in ('projection_calls','fisher_estimation_passes','weight_merge_updates','hvp_calls','create_graph_calls','final_norm_rescale_calls'):
                self.assertEqual(out['record'][key],0)

    def test_pending_update_cannot_switch_or_read_again(self):
        model=Quadratic();step=FormationStep(model,[model.x],None,dict(formation_condition_id='sf_g2_then_sgd',_formation_task=0))
        step.set_epoch(10)
        def closure(*_):return model(None),model.x.view(1,-1)
        step.compute(None,None,closure,.02)
        with self.assertRaises(RuntimeError):step.set_epoch(11)
        with self.assertRaises(RuntimeError):step.compute(None,None,closure,.02)

    def test_configs_accept_three_seeds_all18_and_smoke_preserves20epoch_switch(self):
        root=Path(__file__).resolve().parents[1]
        for config in sorted((root/'configs').glob('*.json')):
            raw=json.loads(config.read_text());smoke=raw['formation_smoke']
            cli=SimpleNamespace(max_batches=1 if smoke else 0,output_dir='/tmp/form20cpu',max_epochs=20,max_tasks=2)
            cfg=train.resolve_config(raw,cli)
            self.assertEqual((cfg['init_epoch'],cfg['epochs']),(20,20));self.assertEqual(cfg['seed'],raw['seed'][0])
            self.assertEqual(cfg['formation_task0_schedule'],[active_method(raw['formation_condition_id'],0,e) for e in (1,11)])
            self.assertEqual(cfg['max_train_tasks'],2);self.assertEqual(cfg['task_increments'],[20]*10)
            self.assertFalse(cfg['ewc_lambda']);self.assertTrue(cfg['ewc_skip_task_fisher'])
            self.assertFalse(cfg['in_task_weight_merge']);self.assertFalse(cfg['task_endpoint_weight_merge'])
        raw=json.loads((root/'configs/formal_sgd_s2.json').read_text())
        cli=SimpleNamespace(max_batches=0,output_dir='/tmp/form20cpu',max_epochs=20,max_tasks=2)
        train.resolve_config(raw,cli)
        cases=[({'epochs':50},'exactly20epoch'),({'formation_switch_epoch':10},'Task1 must'),
               ({'formation_task1_method':'sam'},'Task1 must'),({'gam_grad_gamma':.1},'gam_grad_gamma'),
               ({'seed':[3]},'Registered six'),({'ewc_lambda':2000},'ewc_lambda'),({'study_outer_radius':1.},'Frozen S_F')]
        for change,message in cases:
            bad=copy.deepcopy(raw);bad.update(change)
            with self.assertRaisesRegex(ValueError,message):train.resolve_config(bad,cli)

if __name__=='__main__':unittest.main(verbosity=2)
