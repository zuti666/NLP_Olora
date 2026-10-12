"""Independent CPU algebra tests for nontrivial second-order diagnostics."""
import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'trainer'))
import torch
from effective_step import EffectiveStep,solve_positive_scale
from state_transaction import State
from offline_core import (DiagnosticEngine,correction_diagnostic,finite_lora_parts,
                          comparison,matched_effective_diagnostic,norm)


@contextlib.contextmanager
def unchanged_backend(model,name):
    yield


class Quadratic(torch.nn.Module):
    def __init__(self,H,theta):
        super().__init__();self.theta=torch.nn.Parameter(theta.clone())
        self.register_buffer('H',H.clone());self.frozen=torch.nn.Parameter(torch.ones(1,dtype=torch.float64),requires_grad=False)
    def loss(self,inputs,targets):
        return .5*self.theta@self.H@self.theta


def quadratic_context(H,theta):
    model=Quadratic(H,theta)
    return SimpleNamespace(model=model,geometry=SimpleNamespace(params=[model.theta]),
        loss_fn=model.loss,State=State,backend_context=unchanged_backend)


class OfflineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_exact_quadratic_h2_native_and_halved_scales(self):
        # Indefinite Hessian proves the named quadratic form is not an energy
        # assumed positive; both exact products remain perfectly well-defined.
        H=torch.diag(torch.tensor([-1.,2.,4.],dtype=torch.float64))
        context=quadratic_context(H,torch.tensor([1.,0.,0.],dtype=torch.float64))
        engine=DiagnosticEngine(context,2)
        x=torch.zeros(4,1,dtype=torch.float64);y=torch.zeros(4,dtype=torch.long)
        result=correction_diagnostic(engine,x,y,.02)
        self.assertTrue(result['numerical_gate']['accepted'])
        self.assertLess(result['numerical_gate']['frozen_H_quadratic_form'],0.)
        for row in result['scales']:
            self.assertAlmostEqual(row['cosine'],1.,places=12)
            self.assertLess(row['relative_error'],1e-11)
        self.assertEqual(result['submitted_training_updates'],0)
        self.assertTrue(all(result['restoration'].values()))

    def test_second_hvp_treats_q_as_fixed_and_matches_dense_hessian(self):
        raw=torch.randn(5,5,dtype=torch.float64);H=(raw+raw.T)/2
        theta=torch.randn(5,dtype=torch.float64);context=quadratic_context(H,theta)
        engine=DiagnosticEngine(context,3)
        x=torch.zeros(7,1,dtype=torch.float64);y=torch.zeros(7,dtype=torch.long)
        g=engine.gradient(x,y);q=engine.hvp(x,y,g);p=engine.hvp(x,y,q)
        torch.testing.assert_close(q[0],H@(H@theta),rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(p[0],H@H@H@theta,rtol=1e-13,atol=1e-13)
        self.assertFalse(q[0].requires_grad)
        engine.close()

    def test_finite_lora_Q_and_BA_identity_and_exact_quartic_matching(self):
        a=torch.nn.Parameter(torch.randn(3,7,dtype=torch.float64))
        b=torch.nn.Parameter(torch.randn(5,3,dtype=torch.float64))
        fc=torch.nn.Parameter(torch.randn(4,5,dtype=torch.float64))
        bias=torch.nn.Parameter(torch.randn(4,dtype=torch.float64))
        geometry=EffectiveStep([(a,b)],[fc,bias])
        gs=[torch.randn_like(p) for p in geometry.params]
        native,delta=finite_lora_parts(geometry,gs,.017)
        layer=native['layers'][0]
        self.assertLess(layer['delta_Q_identity_error'],1e-14)
        self.assertLess(layer['effective_identity_error'],1e-14)
        self.assertAlmostEqual(geometry.polynomial(gs).norm(.017),native['effective_norm'],places=12)
        other=[torch.randn_like(p) for p in geometry.params]
        solve=solve_positive_scale(geometry.polynomial(other),native['effective_norm'],initial_scale=.017)
        matched,matched_delta=finite_lora_parts(geometry,other,solve['scale'])
        self.assertLess(abs(matched['effective_norm']/native['effective_norm']-1),1e-9)
        # Full head including bias must contribute to the exact Phi norm.
        self.assertAlmostEqual(native['fc_norm']**2,.017**2*sum(float(g.square().sum()) for g in gs[2:]),places=12)

    def test_plain_BA_gradient_first_order_Q_cancellation(self):
        a=torch.nn.Parameter(torch.randn(3,7,dtype=torch.float64))
        b=torch.nn.Parameter(torch.randn(5,3,dtype=torch.float64))
        fc=torch.nn.Parameter(torch.randn(4,5,dtype=torch.float64))
        target=torch.randn(5,7,dtype=torch.float64)
        loss=.5*((b@a-target)**2).sum()
        ga,gb=torch.autograd.grad(loss,(a,b))
        geometry=EffectiveStep([(a,b)],[fc])
        out,_=finite_lora_parts(geometry,[ga,gb,torch.zeros_like(fc)],.01)
        self.assertLess(out['layers'][0]['delta_Q_first_order_norm'],2e-14)
        self.assertGreater(out['layers'][0]['delta_Q_second_order_norm'],0.)

    def test_undefined_correction_cosine_is_explicit_degenerate(self):
        zero=(torch.zeros(4,dtype=torch.float64),)
        result=comparison(zero,zero,1.)
        self.assertTrue(result['degenerate']);self.assertIsNone(result['cosine'])
        self.assertIsNone(result['relative_error'])

    def test_zero_gradient_retained_as_nonfatal_degenerate(self):
        context=quadratic_context(torch.zeros(3,3,dtype=torch.float64),torch.ones(3,dtype=torch.float64))
        result=correction_diagnostic(DiagnosticEngine(context,2),torch.zeros(4,1),torch.zeros(4,dtype=torch.long),.02)
        self.assertEqual(result['numerical_gate']['status'],'degenerate_gradient')
        self.assertFalse(result['numerical_gate']['numerical_implementation_failed'])
        self.assertEqual(len(result['scales']),2)
        self.assertTrue(all(r['degenerate'] and r['cosine'] is None for r in result['scales']))

    def test_nonzero_gradient_zero_Hessian_is_nonfatal_degenerate(self):
        context=quadratic_context(torch.zeros(3,3,dtype=torch.float64),torch.ones(3,dtype=torch.float64))
        context.loss_fn=lambda x,y: context.model.theta.sum()
        result=correction_diagnostic(DiagnosticEngine(context,2),torch.zeros(4,1),torch.zeros(4,dtype=torch.long),.02)
        self.assertEqual(result['numerical_gate']['status'],'degenerate_curvature')
        self.assertFalse(result['numerical_gate']['numerical_implementation_failed'])
        self.assertEqual(len(result['scales']),2)

    def test_parameters_gradients_buffers_modes_and_rng_restore_exact(self):
        H=torch.eye(3,dtype=torch.float64)
        context=quadratic_context(H,torch.randn(3,dtype=torch.float64))
        context.model.theta.grad=torch.randn_like(context.model.theta)
        state=State(context.model);engine=DiagnosticEngine(context,2)
        engine.native_candidates(torch.zeros(4,1),torch.zeros(4,dtype=torch.long),.02)
        self.assertTrue(all(engine.close().values()));self.assertTrue(all(state.check().values()))


if __name__=='__main__':unittest.main(verbosity=2)
