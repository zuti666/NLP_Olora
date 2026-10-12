"""Independent analytic/transaction tests for TRAIN-only offline probes."""
import math
import unittest
import torch
from torch import nn
from effective_step import EffectiveStep
from state_transaction import rng_get, rng_equal
from local_probe_audit import LocalProbeEngine, audit_fixed_batch, norm, normalized, diagnostic_decomposition


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=nn.Parameter(torch.tensor([[.3,.2],[-.1,.4]]))
        self.b=nn.Parameter(torch.tensor([[.4,.2],[.3,-.2]]))
        self.fc=nn.Linear(2,2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[.2,-.3],[.1,.2]]))
            self.fc.bias.copy_(torch.tensor([.05,-.07]))
        self.register_buffer('counter',torch.tensor(7.))
        self.drop=nn.Dropout(.4)

    def vector(self):
        return torch.cat([p.reshape(-1) for p in self.parameters()])


def quadratic(model):
    width=model.vector().numel()
    diagonal=torch.linspace(1.,3.,width)
    def closure(x,y):
        theta=model.vector()
        loss=(diagonal*theta.square()).sum()/2
        # Auxiliary differentiable logits, independent of the scalar quadratic.
        logits=theta[:4].reshape(2,2).repeat(x.shape[0]//2,1)
        return loss,logits
    return diagonal,closure


class LocalProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=Toy()
        self.batch=(torch.ones(2,2),torch.zeros(2,dtype=torch.long))
        self.heldout=(torch.ones(2,2)*2,torch.ones(2,dtype=torch.long))
        self.geometry=EffectiveStep([(self.model.a,self.model.b)],list(self.model.fc.parameters()),list(self.model.parameters()))

    def test_quadratic_inner_radius_cancels_direction_and_outer_is_linear(self):
        diagonal,closure=quadratic(self.model)
        engine=LocalProbeEngine(self.model,closure,self.geometry)
        theta=self.model.vector().detach().clone()
        base=engine.gradient(self.batch)['gradient']
        self.assertTrue(torch.allclose(base,diagonal*theta,atol=1e-7))
        u=normalized(base);directions=[]
        for a in (.025,.05,.1):
            g1=engine.gradient(self.batch,u,-a)['gradient']
            q=(base-g1)/a
            self.assertTrue(torch.allclose(q,diagonal*u,atol=3e-6))
            directions.append(normalized(base-g1))
        self.assertTrue(torch.allclose(directions[0],directions[-1],atol=2e-6))
        v=directions[1]
        for r in (.05,.25,1.):
            change=engine.gradient(self.batch,v,r)['gradient']-base
            self.assertTrue(torch.allclose(change,r*diagonal*v,atol=3e-7))
        self.assertTrue(all(engine.entry.check().values()))
        self.assertEqual(engine.committed_updates,0)

    def test_full_pointset_budget_rng_restore_and_crossed_eta_same_candidates(self):
        _,closure=quadratic(self.model)
        original_rng=rng_get()
        engine=LocalProbeEngine(self.model,closure,self.geometry)
        result=audit_fixed_batch(engine,self.batch,self.heldout,.02,[0,2])
        self.assertEqual(result['gradient_reads'],20)
        self.assertEqual(len(result['counterfactuals']),20)
        self.assertEqual(result['committed_updates'],0)
        self.assertTrue(rng_equal(original_rng,rng_get()))
        self.assertTrue(all(result['checks'].values()))
        base=engine.gradient(self.batch)['gradient']
        native_h=engine.gradient(self.batch,base,-.02)
        hrow=next(row for row in result['points'] if row['category']=='inner_response' and row['point']=='h')
        self.assertEqual(hrow['gradient_sha256'],native_h['gradient_sha256'])
        self.assertEqual(hrow['inner_placement'],'native_sgd_add_g0_minus_eta')
        for name in ('sgd','response_g2','same_radius_sam','opposite_response','same_radius_random'):
            rows=[r for r in result['counterfactuals'] if r['point']==name and r['mode']=='natural']
            self.assertEqual({r['eta'] for r in rows},{.02,.002})
            self.assertGreater(rows[0]['actual_parameter_step'],rows[1]['actual_parameter_step'])
        for row in result['counterfactuals']:
            if row['mode']=='matched_actual_parameter_length':
                self.assertTrue(row['match']['matched'])
                self.assertLessEqual(row['actual_parameter_match_relative_error'],1e-4)
        for row in result['points']:
            if row['category']=='outer_response' and 'taylor_doubling_relative_error' in row:
                self.assertLess(row['taylor_doubling_relative_error'],1e-5)

    def test_dropout_replay_buffers_gradients_modes_and_rng(self):
        def closure(x,y):
            self.model.counter.add_(1)
            z=self.model.drop(self.model.vector())
            return z.square().sum(),z[:4].reshape(2,2)
        self.model.fc.weight.grad=torch.ones_like(self.model.fc.weight)
        self.model.fc.eval()
        engine=LocalProbeEngine(self.model,closure,self.geometry)
        first=engine.gradient(self.batch)
        second=engine.gradient(self.batch)
        self.assertEqual(first['gradient_sha256'],second['gradient_sha256'])
        self.assertEqual(float(self.model.counter),7.)
        self.assertFalse(self.model.fc.training)
        self.assertTrue(torch.equal(self.model.fc.weight.grad,torch.ones_like(self.model.fc.weight)))
        self.assertTrue(all(engine.entry.check().values()))

    def test_effective_distance_matches_exact_ba_and_all_fc(self):
        _,closure=quadratic(self.model)
        engine=LocalProbeEngine(self.model,closure,self.geometry)
        before=self.geometry.capture()
        g=engine.gradient(self.batch)['gradient']
        engine._place(g,-.03)
        after=self.geometry.capture()
        expected=(after[1].double()@after[0].double()-before[1].double()@before[0].double()).square().sum()
        expected+=sum((a.double()-b.double()).square().sum() for a,b in zip(after[2:],before[2:]))
        measured=self.geometry.actual_delta(before)['joint_norm']
        self.assertAlmostEqual(measured,math.sqrt(float(expected)),places=11)
        engine._restore()

    def test_zero_response_has_explicit_invalid_flag_and_no_commit(self):
        def linear(x,y):
            theta=self.model.vector()
            return theta.sum(),theta[:4].reshape(2,2)
        engine=LocalProbeEngine(self.model,linear,self.geometry)
        result=audit_fixed_batch(engine,self.batch,self.heldout,.02,[0,2],smoke=True)
        self.assertEqual(result['gradient_reads'],8)
        inner=[r for r in result['points'] if r['category']=='inner_response']
        self.assertTrue(all(not r['response_direction_valid'] for r in inner))
        outer=[r for r in result['points'] if r['category']=='outer_response']
        self.assertTrue(all(not r['fp32_distinguishable'] for r in outer))
        self.assertEqual(result['committed_updates'],0)

    def test_exception_always_rolls_back_parameters_buffer_and_rng(self):
        def failure(x,y):
            self.model.counter.add_(3)
            torch.rand(9)
            raise RuntimeError('intentional')
        engine=LocalProbeEngine(self.model,failure,self.geometry)
        before=rng_get()
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            engine.gradient(self.batch,torch.ones(engine.dimension),.05)
        self.assertTrue(all(engine.entry.check().values()))
        self.assertTrue(rng_equal(before,rng_get()))
        self.assertEqual(engine.gradient_reads,0)

    def test_readonly_scalar_decomposition(self):
        base=torch.tensor([1.,0.]);change=torch.tensor([2.,3.])
        value=diagnostic_decomposition(change,base)
        self.assertEqual(value['parallel_to_g0_coefficient'],2.)
        self.assertEqual(value['parallel_to_g0_norm'],2.)
        self.assertEqual(value['perpendicular_to_g0_norm'],3.)
        self.assertTrue(torch.equal(change,torch.tensor([2.,3.])))


if __name__=='__main__':unittest.main()
