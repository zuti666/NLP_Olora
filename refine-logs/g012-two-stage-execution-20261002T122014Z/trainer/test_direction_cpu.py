"""CPU-only mathematical, state-restoration and actual-step acceptance tests."""
import json
import math
import random
import unittest
import numpy as np
import torch

from audit_core import rng_equal, rng_get, rng_set
from direction_step import FourProbeStep, normalized, vector_norm
from effective_step import EffectiveStep
from response_hvp import response_observer


class Toy(torch.nn.Module):
    def __init__(self,dtype=torch.float64):
        super().__init__()
        generator = torch.Generator().manual_seed(91)
        self.a = torch.nn.Parameter(torch.randn(2,3,generator=generator,dtype=dtype)*.2)
        self.b = torch.nn.Parameter(torch.randn(4,2,generator=generator,dtype=dtype)*.2)
        self.fc = torch.nn.Linear(4,3,dtype=dtype)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(3,4,generator=generator,dtype=dtype)*.2)
            self.fc.bias.copy_(torch.randn(3,generator=generator,dtype=dtype)*.2)
        self.frozen = torch.nn.Parameter(torch.ones(2,dtype=dtype),requires_grad=False)
        self.register_buffer("probe_counter",torch.zeros((),dtype=dtype))
        self.dropout = torch.nn.Dropout(.4)
        self._register_delta_hook = False
        self.delta_w_q_new_grad = None
        self.delta_w_v_new_grad = None
        self.external_delta_w_q = None
        self.external_delta_w_v = None
        self.forward_cache = {"entry":torch.tensor([1.],dtype=dtype)}


def flat(values):
    return torch.cat([x.reshape(-1) for x in values])


def make_case(dtype=torch.float64, reverse=False):
    model = Toy(dtype)
    original = tuple(p for p in model.parameters() if p.requires_grad)
    params = tuple(reversed(original)) if reverse else original
    geom = EffectiveStep([(model.a,model.b)],tuple(model.fc.parameters()),params)
    n = sum(p.numel() for p in params)
    generator = torch.Generator().manual_seed(25)
    matrix = torch.randn(n,n,generator=generator,dtype=dtype)
    hessian = matrix.T@matrix/n + torch.eye(n,dtype=dtype)*.4
    linear = torch.linspace(-.2,.4,n,dtype=dtype)

    def closure(_inputs,_targets):
        theta = flat(params)
        loss = .5*theta@(hessian@theta) - linear@theta
        return loss,theta[:3].unsqueeze(0)

    return model,params,geom,hessian,linear,closure


class DirectionAcceptance(unittest.TestCase):
    def assert_vectors_close(self,a,b,atol=1e-11,rtol=1e-11):
        self.assertTrue(torch.allclose(flat(a),flat(b),atol=atol,rtol=rtol),
                        "error="+str(float((flat(a)-flat(b)).abs().max())))

    def test_quadratic_signs_locations_and_final_residual_reference(self):
        model,params,geometry,H,b,closure = make_case()
        base = tuple(p.detach().clone() for p in params)
        states = {}
        for branch in ("P","D","C","R"):
            engine = FourProbeStep(model,params,geometry,{"branch":branch,"direction":"A"})
            output = engine.compute(None,None,closure,.04)
            ctx = output["context"]
            g0 = flat(ctx["grads"]["g0"])
            expected_g0 = H@flat(base)-b
            self.assertTrue(torch.allclose(g0,expected_g0,atol=1e-13,rtol=1e-13))
            cplus = flat(ctx["grads"]["g1_plus"])-g0
            cminus = flat(ctx["grads"]["g1_minus"])-g0
            expected = .05*(H@(g0/(torch.linalg.vector_norm(g0)+1e-12)))
            self.assertTrue(torch.allclose(cplus,expected,atol=1e-13,rtol=1e-12))
            self.assertTrue(torch.allclose(cminus,-expected,atol=1e-13,rtol=1e-12))
            self.assert_vectors_close(ctx["U"],tuple(x+y-z for x,y,z in zip(
                                      ctx["grads"]["g0"],ctx["grads"]["g2"],ctx["grads"]["g1"])))
            self.assertTrue(all(torch.equal(p,v) for p,v in zip(params,base)))
            self.assertEqual(output["record"]["ordinary_gradient_calls"],4)
            self.assertEqual(output["record"]["fisher_calls"],0)
            self.assertEqual(output["record"]["weight_merge_calls"],0)
            self.assertEqual(output["record"]["gradient_projection_calls"],0)
            states[branch] = ctx
            engine.discard()
        self.assert_vectors_close(states["P"]["points"]["phi2"],states["C"]["points"]["phi2"])
        self.assert_vectors_close(states["D"]["points"]["phi2"],states["R"]["points"]["phi2"])
        self.assertGreater(vector_norm(tuple(x-y for x,y in zip(states["P"]["U"],states["C"]["U"]))),1e-3)
        # d12 must be the actual snapshot difference, not (r-rho)*N(g0).
        self.assert_vectors_close(states["P"]["d12"],tuple(y-x for x,y in zip(
             states["P"]["points"]["phi1_plus"],states["P"]["points"]["phi2"])))

    def test_all_direction_selectors_and_one_actual_optimizer_step(self):
        for branch in ("P","D","C","R"):
            for direction in ("g0","g1_plus","g1_minus","g2","A","residual_only"):
                model,params,geom,_,_,closure = make_case()
                base = tuple(p.detach().clone() for p in params)
                engine = FourProbeStep(model,params,geom,{"branch":branch,"direction":direction})
                optimizer = torch.optim.SGD(params,lr=.04,momentum=0,weight_decay=0)
                out = engine.compute(None,None,closure,.04)
                ctx = out["context"]
                expected = (ctx["grads"][direction] if direction in ("g0","g1_plus","g1_minus","g2")
                            else ctx["c21"] if direction=="residual_only" else
                            tuple(x+y for x,y in zip(ctx["grads"]["g0"],ctx["c21"])))
                self.assert_vectors_close(out["U"],expected)
                update = engine.apply_update(matched=False,optimizer=optimizer)
                expected_params = tuple(v.add(u,alpha=-.04) for v,u in zip(base,expected))
                self.assertTrue(all(torch.equal(p,v) for p,v in zip(params,expected_params)))
                self.assertEqual(update["sgd_updates"],1)
                self.assertEqual(optimizer.param_groups[0]["lr"],.04)
                self.assertEqual(len(optimizer.state),0)

    def test_rng_replay_buffers_modes_and_advancement_once(self):
        model,params,geom,_,_,_ = make_case()
        model.train()
        draws = []
        initial_buffer = model.probe_counter.clone()
        initial_rng = rng_get()
        r1 = None

        def stochastic_closure(_x,_y):
            nonlocal r1
            theta = flat(params)
            mask = model.dropout(torch.ones_like(theta))
            draw = (random.random(),float(np.random.rand()),torch.rand(5),mask.detach().clone())
            draws.append(draw)
            model.probe_counter.add_(1)
            model.dropout.training = False  # intentionally mutate a mode
            model.delta_w_q_new_grad = torch.ones(2)
            model.delta_w_v_new_grad = torch.ones(3)
            model.external_delta_w_q = torch.ones(2)
            model._register_delta_hook = True
            model.forward_cache["entry"].add_(2)
            model.forward_cache["created_by_forward"] = torch.ones(2)
            model.created_cache = [torch.ones(1)]
            if len(draws)==1:
                r1 = rng_get()
            return (theta*mask).square().sum(),theta[:3].unsqueeze(0)

        engine = FourProbeStep(model,params,geom,{"direction":"A"})
        out = engine.compute(None,None,stochastic_closure,.04)
        self.assertEqual(len(draws),4)
        for draw in draws[1:]:
            self.assertEqual(draw[0],draws[0][0])
            self.assertEqual(draw[1],draws[0][1])
            self.assertTrue(torch.equal(draw[2],draws[0][2]))
            self.assertTrue(torch.equal(draw[3],draws[0][3]))
        self.assertTrue(rng_equal(rng_get(),r1))
        self.assertFalse(rng_equal(initial_rng,rng_get()))
        self.assertTrue(torch.equal(model.probe_counter,initial_buffer))
        self.assertTrue(model.dropout.training)
        self.assertIsNone(model.delta_w_q_new_grad)
        self.assertIsNone(model.delta_w_v_new_grad)
        self.assertIsNone(model.external_delta_w_q)
        self.assertFalse(model._register_delta_hook)
        self.assertTrue(torch.equal(model.forward_cache["entry"],torch.tensor([1.],dtype=torch.float64)))
        self.assertNotIn("created_by_forward",model.forward_cache)
        self.assertFalse(hasattr(model,"created_cache"))
        self.assertTrue(out["record"]["state_restoration"]["controls_caches_exact"])
        engine.discard()

    def test_exception_restores_gradients_state_and_rng(self):
        model,params,geom,_,_,closure = make_case()
        base = tuple(p.detach().clone() for p in params)
        for p in params:
            p.grad = torch.ones_like(p)*3
        initial_rng = rng_get()
        calls = 0

        def failing_closure(x,y):
            nonlocal calls
            calls += 1
            random.random()
            torch.rand(5)
            model.probe_counter.add_(2)
            if calls==3:
                raise RuntimeError("intentional closure failure")
            return closure(x,y)

        engine = FourProbeStep(model,params,geom,{"direction":"A"})
        with self.assertRaisesRegex(RuntimeError,"intentional"):
            engine.compute(None,None,failing_closure,.04)
        self.assertTrue(all(torch.equal(p,v) for p,v in zip(params,base)))
        self.assertTrue(all(torch.equal(p.grad,torch.ones_like(p)*3) for p in params))
        self.assertEqual(float(model.probe_counter),0)
        self.assertTrue(rng_equal(initial_rng,rng_get()))

    def test_matching_fp32_and_geometry_parameter_order(self):
        for direction in ("g0","g1_plus","g1_minus","g2","A","residual_only"):
            model,params,geom,_,_,closure = make_case(torch.float32,reverse=True)
            base = tuple(p.detach().clone() for p in params)
            engine = FourProbeStep(model,params,geom,{"direction":direction})
            optimizer = torch.optim.SGD(params,lr=.04)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5,eta_min=.004)
            out = engine.compute(None,None,closure,.04)
            update = engine.apply_update(matched=True,optimizer=optimizer)
            self.assertLessEqual(update["actual_match_relative_error"],1e-3)
            self.assertEqual(update["sgd_updates"],1)
            self.assertGreater(update["step_scale"],0)
            self.assertEqual(optimizer.param_groups[0]["lr"],.04)
            if direction=="g1_plus":
                self.assertTrue(update["g1_identity"])
                self.assertEqual(update["step_scale"],.04)
                expected = tuple(v.add(u,alpha=-.04) for v,u in zip(base,out["context"]["grads"]["g1_plus"]))
                self.assertTrue(all(torch.equal(p,v) for p,v in zip(params,expected)))
            scheduler.step()

    def test_degenerate_normalization_and_match_failure_no_update(self):
        model,params,geom,_,_,_ = make_case()
        base = tuple(p.detach().clone() for p in params)

        def linear_closure(_x,_y):
            theta = flat(params)
            return theta.sum(),theta[:3].unsqueeze(0)

        engine = FourProbeStep(model,params,geom,{"direction":"residual_only"})
        out = engine.compute(None,None,linear_closure,.04)
        self.assertEqual(out["record"]["norms"]["q"],0)
        self.assertEqual(out["record"]["norms"]["U"],0)
        self.assertTrue(all(torch.equal(p,v) for p,v in zip(params,base)))
        with self.assertRaises(ValueError):
            engine.apply_update(matched=True)
        self.assertTrue(all(torch.equal(p,v) for p,v in zip(params,base)))

    def test_hvp_observer_hook_is_detached_and_restored(self):
        model,params,geom,H,_,closure = make_case()
        base = tuple(p.detach().clone() for p in params)

        def observer(ctx):
            for name,point in ctx["points"].items():
                self.assertTrue(all(not x.requires_grad for x in point))
            response = H@flat(ctx["d12"])
            offset = 0
            split = []
            for p in params:
                split.append(response[offset:offset+p.numel()].reshape_as(p))
                offset += p.numel()
            expected = tuple(g+h for g,h in zip(ctx["grads"]["g0"],split))
            # A quadratic makes Sec and any local Hd identical.
            self.assert_vectors_close(expected,ctx["U"])
            with torch.no_grad():
                params[0].add_(10)
                model.probe_counter.add_(10)
                torch.rand(5)
            return expected,{"hvp_calls":1,"analytic_quadratic":True}

        engine = FourProbeStep(model,params,geom,{"direction":"hstart"})
        out = engine.compute(None,None,closure,.04,observer=observer)
        self.assertTrue(all(torch.equal(p,v) for p,v in zip(params,base)))
        self.assertEqual(float(model.probe_counter),0)
        self.assertEqual(out["record"]["observer"]["hvp_calls"],1)
        self.assertTrue(rng_equal(rng_get(),out["context"]["rng1"]))
        engine.apply_update(matched=False)

    def test_real_response_observer_hstart_hmid_quadratic(self):
        for response in ("hstart","hmid"):
            model,params,geom,_,_,closure = make_case()
            engine = FourProbeStep(model,params,geom,{"direction":response,"backend":"math"})
            out = engine.compute(None,None,closure,.04,
                                 observer=lambda context: response_observer(context,response))
            expected = tuple(x+y-z for x,y,z in zip(out["context"]["grads"]["g0"],
                                                  out["context"]["grads"]["g2"],
                                                  out["context"]["grads"]["g1_plus"]))
            self.assert_vectors_close(out["U"],expected)
            report = out["record"]["observer"]
            self.assertEqual(report["status"],"passed")
            self.assertLess(report["hvp_vs_sec"]["relative_l2"],1e-11)
            update = engine.apply_update(matched=True)
            self.assertLessEqual(update["actual_match_relative_error"],1e-3)


if __name__ == "__main__":
    torch.set_num_threads(1)
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    print(json.dumps({"device":"cpu","cuda_visible_devices":"disabled_by_launcher",
                      "tests":"four_probe_directions_restore_rng_effective_matching"}))
    unittest.main(verbosity=2)
