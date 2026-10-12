"""Independent quadratic probes, exact factor identities, switch/RNG invariants."""
import copy,json,os
from pathlib import Path
from types import SimpleNamespace
import unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import audit_core as audit
from state_transaction import State
from chain_step import ChainStep,active_method,cosine_lr,READS
from chain_statistics import update_decomposition,factor_epoch
from effective_step import EffectiveStep
import response_chain_train as train
class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__();self.x=torch.nn.Parameter(torch.tensor([.8,-.6,.3]));self.register_buffer('H',torch.tensor([[3.,.7,.2],[.7,2.,-.1],[.2,-.1,1.]]));self.register_buffer('linear',torch.tensor([.2,-.3,.4]))
    def forward(self,x):return .5*self.x@self.H@self.x+self.linear@self.x
class FactorModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.a=torch.nn.Parameter(torch.randn(2,3,dtype=torch.float64));self.b=torch.nn.Parameter(torch.randn(4,2,dtype=torch.float64));self.fc=torch.nn.Linear(4,2,dtype=torch.float64);self.register_buffer('cachecounter',torch.zeros(1));self.forward_cache={'x':torch.ones(1)};self.fused_attn=True
class Tests(unittest.TestCase):
    def test_native_probes_against_independent_exact_oracle(self):
        for branch in ('S','R'):
            m=Quadratic();origin=m.x.detach().clone();points=[]
            def closure(*args):points.append(m.x.detach().clone());return m(None),m.x[None]
            g0=m.H@origin+m.linear
            p1=origin+.05*g0/(float(g0.double().norm())+1e-12) if branch=='S' else origin-.02*g0
            g1=m.H@p1+m.linear
            p2=origin+.25*(g0-g1)/float((g0-g1).double().norm())
            want=g1 if branch=='S' else m.H@p2+m.linear
            step=ChainStep(m,[m.x],None,dict(chain_branch=branch,chain_switch_after_epoch=25));step.set_epoch(1);r=step.compute(None,None,closure,.02)
            torch.testing.assert_close(r['U'][0],want,rtol=3e-6,atol=2e-6)
            self.assertEqual(len(points),2 if branch=='S' else 3);torch.testing.assert_close(points[1],p1,rtol=3e-6,atol=2e-6)
            if branch=='R':torch.testing.assert_close(points[2],p2,rtol=3e-6,atol=2e-6)
            self.assertTrue(torch.equal(m.x,origin));opt=torch.optim.SGD([m.x],lr=.02);commit=step.apply_update(.02,opt)
            self.assertTrue(torch.equal(m.x,origin.clone().add_(r['U'][0],alpha=-.02)));self.assertEqual(commit['sgd_updates'],1)
            for key in ('projection_calls','fisher_estimation_passes','weight_merge_updates','hvp_calls','create_graph_calls','final_norm_rescale_calls'):self.assertEqual(r['record'][key],0)
    def test_continuous_cosine50_one_optimizer_switch_and_budgets(self):
        for branch in ('S','R'):
            m=Quadratic();opt=torch.optim.SGD([m.x],lr=.02,momentum=0,weight_decay=0);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=.002);step=ChainStep(m,[m.x],None,dict(chain_branch=branch,chain_switch_after_epoch=25));ids=(id(m.x),id(opt),id(scheduler));reads=0
            for epoch in range(1,51):
                state=State(m);step.set_epoch(epoch);state.check();self.assertEqual(ids,(id(m.x),id(opt),id(scheduler)));lr=opt.param_groups[0]['lr'];self.assertAlmostEqual(lr,cosine_lr(.02,epoch),places=15)
                def closure(*args):return m(None),m.x[None]
                r=step.compute(None,None,closure,lr);reads+=r['record']['ordinary_gradient_calls'];step.apply_update(lr,opt);scheduler.step()
                self.assertEqual(step.active,('sam' if branch=='S' else 'sf_g2') if epoch<=25 else 'sgd')
            self.assertEqual(reads,75 if branch=='S' else 100);self.assertAlmostEqual(opt.param_groups[0]['lr'],.002,places=15)
            self.assertEqual(3*sum((75*8,100*8)),4200)
    def test_exact_actual_factor_update_decomposition_and_sgd_Q_first_cancellation(self):
        torch.manual_seed(4);m=FactorModel();geo=EffectiveStep([(m.a,m.b)],list(m.fc.parameters()),list(m.parameters()));before=geo.capture();a,b=before[:2];gradient_effective=torch.randn(4,3,dtype=torch.float64)
        da=-.02*b.T@gradient_effective;db=-.02*gradient_effective@a.T
        with torch.no_grad():m.a.add_(da);m.b.add_(db);m.fc.weight.add_(.02)
        r=update_decomposition(geo,before)['layers'][0]
        self.assertLess(r['delta_q_first_fro'],1e-14);self.assertLess(r['delta_q_identity_error'],1e-14)
        terms=[b@da,db@a,db@da];actual=m.b@m.a-b@a
        self.assertAlmostEqual(r['actual_delta_W_norm'],float(actual.norm()),places=13)
        self.assertAlmostEqual(r['T_A_norm'],float(terms[0].norm()),places=13);self.assertAlmostEqual(r['T_cross_norm'],float(terms[2].norm()),places=13)
        arbitrary=geo.capture()
        with torch.no_grad():m.a.add_(torch.randn_like(m.a)*.03);m.b.add_(torch.randn_like(m.b)*.03)
        r2=update_decomposition(geo,arbitrary)['layers'][0];self.assertLess(r2['delta_q_identity_error'],1e-14);self.assertGreater(r2['delta_q_first_fro'],1e-5)
        f=factor_epoch(geo,before);self.assertEqual(len(f['layers']),1);self.assertGreater(f['effective_displacement']['joint_norm'],0)
    def test_compact_runtime_restores_buffers_grads_modes_rng_and_controls(self):
        m=FactorModel();m.a.grad=torch.ones_like(m.a);saved=train.runtime_payload(m);identity=[id(p) for p in m.parameters()]
        with torch.no_grad():m.a.add_(1);m.cachecounter.add_(1)
        m.a.grad=None;m.eval();m.fused_attn=False;m.forward_cache['x'].add_(1);audit.seed_all(12)
        train.apply_runtime(m,saved)
        self.assertEqual(identity,[id(p) for p in m.parameters()]);self.assertTrue(m.training);self.assertTrue(m.fused_attn);self.assertEqual(float(m.cachecounter),0);self.assertEqual(float(m.forward_cache['x']),1);self.assertTrue(torch.equal(m.a.grad,torch.ones_like(m.a)));self.assertTrue(audit.rng_equal(audit.rng_get(),saved['rng']))
    def test_configs_restore_real_seed_and_forbid_radius_and_protocol_drift(self):
        root=Path(__file__).resolve().parents[1]
        configs=list((root/'configs').glob('*.json'));self.assertTrue(configs)
        for file in configs:
            raw=json.loads(file.read_text());smoke=raw.get('chain_smoke',False);cli=SimpleNamespace(max_epochs=2 if smoke else 0,max_batches=2 if smoke else 0,output_dir='/tmp/chaincpusmoke')
            cfg=train.resolve_config(raw,cli);seed=raw['seed'][0] if isinstance(raw['seed'],list) else raw['seed'];self.assertEqual(cfg['seed'],seed);self.assertEqual(cfg['epochs'],50);self.assertEqual(cfg['chain_switch_after_epoch'],1 if smoke else 25)
            for change in ({'lrate':.04},{'lora_rank':20},{'ewc_lambda':2000},{'in_task_weight_merge':True},{'seed':[3]}):
                bad=dict(raw,**change)
                with self.assertRaises(ValueError):train.resolve_config(bad,cli)
if __name__=='__main__':unittest.main()
