"""Independent algebra, probe location, native cost and state transaction tests."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
import audit_core as audit
import upward_fork_train as train
from upward_residual_step import UpwardResidualStep, DIRECTIONS, READS
from response_step import _norm
from state_transaction import State

class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__();self.x=torch.nn.Parameter(torch.tensor([.8,-.6,.3]))
        self.register_buffer('H',torch.tensor([[3.,.7,.2],[.7,2.,-.1],[.2,-.1,1.]]))
        self.register_buffer('linear',torch.tensor([.2,-.3,.4]))
    def forward(self,inputs):return .5*self.x@self.H@self.x+self.linear@self.x

class Stateful(torch.nn.Module):
    def __init__(self):
        super().__init__();self.fc=torch.nn.Linear(3,20);self.drop=torch.nn.Dropout(.3)
        self.register_buffer('counter',torch.zeros(1));self.forward_cache={'tensor':torch.ones(1)}
    def forward(self,x):
        self.counter.add_(1.);self.forward_cache['tensor'].add_(1.)
        return self.fc(self.drop(x))

class UpwardTests(unittest.TestCase):
    def test_all_eight_rules_match_independent_quadratic_oracle_and_one_sgd(self):
        for direction in DIRECTIONS:
            with self.subTest(direction=direction):
                model=Quadratic();origin=model.x.detach().clone();points=[]
                def closure(*_):
                    points.append(model.x.detach().clone());loss=model(None)
                    return loss,model.x.detach().view(1,-1)
                def grad(point):return model.H@point+model.linear
                g0=grad(origin);p1=origin+.05*g0/(float(g0.double().norm())+1e-12)
                g1=grad(p1);s=g1-g0;p2=origin+1.*s/(float(s.double().norm())+1e-12)
                g2=grad(p2);p3=p2+.05*g2/(float(g2.double().norm())+1e-12);g3=grad(p3);d=g3-g2
                expected={'g0':g0,'g1':g1,'g2_minus_s':g2-s,'g2':g2,
                    'g2_plus_s':g2+s,'g0_plus_d':g0+d,'g1_plus_d':g1+d,'g3':g3}[direction]
                step=UpwardResidualStep(model,[model.x],None,dict(direction=direction,rho=.05,radius=1.))
                result=step.compute(None,None,closure,.02)
                self.assertEqual(len(points),READS[direction]);self.assertEqual(result['record']['gradient_reads'],READS[direction])
                torch.testing.assert_close(result['U'][0],expected,rtol=3e-6,atol=2e-6)
                for got,want in zip(points,[origin,p1,p2,p3]):torch.testing.assert_close(got,want,rtol=3e-6,atol=2e-6)
                self.assertTrue(torch.equal(model.x,origin))
                optimizer=torch.optim.SGD([model.x],lr=.02,momentum=0,weight_decay=0)
                applied=step.apply_update(.02,optimizer)
                self.assertTrue(torch.equal(model.x,origin.clone().add_(result['U'][0],alpha=-.02)))
                self.assertEqual(applied['sgd_updates'],1);self.assertFalse(applied['history_cache_valid_after_update'])

    def test_complete_state_restoration_and_base_only_rng_advance_for_all_budgets(self):
        for direction in DIRECTIONS:
            torch.manual_seed(123);model=Stateful();model.train()
            model.fc.weight.grad=torch.ones_like(model.fc.weight);entry=State(model)
            x=torch.ones(4,3);target=torch.tensor([0,1,2,3])
            def closure(inputs,targets):
                logits=model(inputs);return torch.nn.functional.cross_entropy(logits,targets),logits
            closure(x,target);expected_rng=audit.rng_get();entry.restore();entry.check()
            ids=[id(p) for p in model.parameters()]
            result=UpwardResidualStep(model,list(model.parameters()),None,dict(direction=direction,rho=.05,radius=1.)).compute(x,target,closure,.02)
            self.assertTrue(audit.rng_equal(audit.rng_get(),expected_rng))
            self.assertEqual(ids,[id(p) for p in model.parameters()]);self.assertEqual(float(model.counter),0.)
            self.assertEqual(float(model.forward_cache['tensor']),1.);self.assertTrue(torch.equal(model.fc.weight.grad,torch.ones_like(model.fc.weight)))
            for key in ('projection_calls','fisher_estimation_passes','weight_merge_updates','hvp_calls','create_graph_calls','final_norm_rescale_calls'):
                self.assertEqual(result['record'][key],0)

    def test_exceptions_restore_full_state_and_rng(self):
        model=Stateful();entry=State(model);n=[0]
        def closure(x,y):
            n[0]+=1;logits=model(x)
            if n[0]==2:raise RuntimeError('injected read failure')
            return logits.square().mean(),logits
        step=UpwardResidualStep(model,list(model.parameters()),None,dict(direction='g3',rho=.05,radius=1.))
        with self.assertRaisesRegex(RuntimeError,'injected'):step.compute(torch.ones(4,3),None,closure,.02)
        entry.check();self.assertIsNone(step._pending)

    def test_config_accepts_three_seeds_and_rejects_conflicts(self):
        root=Path(__file__).resolve().parents[2]
        raw=json.loads((root/'imageneta-response-four-methods-gpu567-20261009T103449Z/configs/formal_cl10_sgd_s2.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            split=Path(temporary)/'split.json';split.write_text('{}')
            raw.update(study_role='branch',study_dev_split_path=str(split),study_parent_output_dir=temporary,
                study_parent_checkpoint=temporary,study_parent_receipt=temporary)
            cli=SimpleNamespace(max_epochs=0,max_batches=0,max_tasks=0,output_dir=temporary)
            for seed in (0,1,2):
                for direction in DIRECTIONS:
                    raw.update(seed=[seed],upward_direction=direction)
                    cfg=train.resolve_config(raw,cli)
                    self.assertEqual(cfg['seed'],seed);self.assertEqual(cfg['probe_gradient_count'],READS[direction])
                    self.assertEqual(cfg['upward_direction'],direction);self.assertEqual(cfg['max_train_tasks'],2)
            raw.update(upward_direction='g2',gam_grad_norm_rho=.25)
            self.assertEqual(train.resolve_config(raw,cli)['gam_grad_norm_rho'],.25)
            raw['upward_direction']='g3'
            with self.assertRaises(ValueError):train.resolve_config(raw,cli)

    def test_complete_snapshot_preserves_mode_grad_buffer_cache_and_rng(self):
        model=Stateful();model.train();model.drop.eval();model.fc.weight.grad=torch.ones_like(model.fc.weight)
        with tempfile.TemporaryDirectory() as temporary:
            split=Path(temporary)/'split.json';split.write_text('{}')
            cfg=dict(study_dev_split_path=str(split),study_source=temporary)
            before=State(model);payload=train._complete_snapshot(model,cfg,1,10,.015,{'optimizer_state':{}})
            before.check();path=Path(temporary)/'snapshot.pt';torch.save(payload,path)
            torch.rand(8);loaded,state=train.load_snapshot(path,'cpu',restore_rng=True)
            self.assertTrue(audit.rng_equal(audit.rng_get(),payload['rng']));self.assertFalse(loaded.drop.training)
            self.assertTrue(torch.equal(loaded.fc.weight.grad,model.fc.weight.grad));self.assertTrue(torch.equal(loaded.counter,model.counter))
            self.assertTrue(torch.equal(loaded.forward_cache['tensor'],model.forward_cache['tensor']))

if __name__=='__main__':unittest.main(verbosity=2)
