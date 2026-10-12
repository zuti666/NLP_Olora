"""Meaningful CPU algebra, factor geometry and native/audit transaction gates."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from direction_step import FourProbeStep, FORMULAS, expected_budget, vector_norm
from effective_step import EffectiveStep
from state_transaction import State, rng_equal, rng_get, rng_set


class Toy(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.A = torch.nn.Parameter(torch.tensor([[.7, -.2]], dtype=dtype))
        self.B = torch.nn.Parameter(torch.tensor([[.4], [-.3]], dtype=dtype))
        self.fc = torch.nn.Parameter(torch.tensor([[.2, .5], [-.4, .1]], dtype=dtype))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))
        self.forward_cache = {"entry": torch.tensor([13.], dtype=dtype)}

    def closure(self, x, y):
        self.counter.add_(1)
        self.forward_cache = {"read": torch.rand((), dtype=x.dtype)}
        random.random(); np.random.random()
        z = x @ (self.B @ self.A).T
        z = torch.nn.functional.dropout(z, .2, training=True)
        logits = z @ self.fc.T
        return torch.nn.functional.cross_entropy(logits, y), logits


def algebra():
    H = torch.tensor([[2., .3], [.3, 1.]], dtype=torch.float64)
    b = torch.tensor([.7, -.4], dtype=torch.float64)
    x1 = torch.tensor([.1, .4], dtype=torch.float64)
    x2 = torch.tensor([.8, -.2], dtype=torch.float64)
    xm = .5*(x1+x2)
    grad = lambda x: H@x+b
    e = .5*(grad(x1)+grad(x2))-grad(xm)
    assert float(e.abs().max()) < 5e-16
    # Exact mixed finite-step identity with the same frozen candidates.
    base = torch.tensor([.4, -.6], dtype=torch.float64)
    g1, g2, g3 = grad(x1), grad(x2), grad(x2+torch.tensor([.03, .02]))
    c = g3-g2
    eta = .02
    loss = lambda u: .5*(base-eta*u)@H@(base-eta*u)+b@(base-eta*u)
    interaction = (loss(g1+g3)-loss(g1+g2))-(loss(g3)-loss(g2))
    exact = eta*eta*g1@H@c
    assert abs(float(interaction-exact)) < 2e-16
    # L(A,B)=0.5(BA-y)^2: cubic factor-gradient, nonzero midpoint even response.
    def factor_grad(a, b, y=.3):
        residual = a*b-y
        return torch.tensor([residual*b, residual*a], dtype=torch.float64)
    a1, b1, a2, b2 = .2, .8, .9, -.1
    am, bm = .5*(a1+a2), .5*(b1+b2)
    factor_e = .5*(factor_grad(a1,b1)+factor_grad(a2,b2))-factor_grad(am,bm)
    assert float(torch.linalg.vector_norm(factor_e)) > .01
    assert abs(am*bm-.5*(a1*b1+a2*b2)) > .01
    return dict(quadratic_e12_zero=True, quadratic_e12_absmax=float(e.abs().max()),
                quadratic_mixed_interaction_exact=True,
                quadratic_interaction_error=abs(float(interaction-exact)),
                factor_coordinate_e12_nonzero=True,
                factor_e12_norm=float(torch.linalg.vector_norm(factor_e)),
                factor_midpoint_not_BA_average=True,
                scalar_BA_midpoint_gap=abs(am*bm-.5*(a1*b1+a2*b2)))


def run_gate():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = dict(algebra=algebra(), directions={}, failures=[])
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    x = torch.tensor([[.4, .2], [-.5, .6], [.1, -.7]], dtype=torch.float64)
    y = torch.tensor([0, 1, 0])
    for dtype in (torch.float64, torch.float32):
        for direction in FORMULAS:
            model = Toy(dtype=dtype)
            params = tuple(model.parameters())
            for index, p in enumerate(params):
                p.grad = torch.full_like(p, index+3.)
            geometry = EffectiveStep([(model.A, model.B)], [model.fc], params)
            config = dict(direction=direction, rho=.05, radius=1., eps=1e-12)
            stepper = FourProbeStep(model, params, geometry, config)
            entry = State(model)
            native = stepper.compute(x.to(dtype), y, model.closure, .02)
            expected = expected_budget(config)
            assert native["record"]["ordinary_gradient_calls"] == expected["train_ce_per_update"]
            assert all(native["record"]["state_restoration"].values())
            assert torch.equal(model.counter, torch.tensor(0))
            assert "entry" in model.forward_cache
            assert all(torch.equal(p, v) for p, v in entry.trainable)
            assert rng_equal(rng_get(), native["context"]["rng1"])
            read_labels = set(native["context"]["grads"])
            if direction == "two_midpoint":
                assert read_labels == {"g0", "g1", "gm"}
            else:
                assert "gm" not in read_labels
                assert ("g3" in read_labels) == (direction == "sum13")
            optimizer = torch.optim.SGD(params, lr=.02)
            applied_native = stepper.apply_update(.02, optimizer)
            final_native = tuple(p.detach().clone() for p in params)
            rng_native = rng_get()
            next_batch_native = torch.randperm(23)
            entry.restore(); entry.check()
            audit = stepper.compute(x.to(dtype), y, model.closure, .02, audit_all=True)
            assert audit["record"]["ordinary_gradient_calls"] == 5
            assert set(audit["context"]["grads"]) == {"g0", "g1", "g2", "gm", "g3"}
            assert all(torch.equal(a, b) for a,b in zip(native["U"],audit["U"]))
            assert rng_equal(native["context"]["rng1"],audit["context"]["rng1"])
            for point in ("phi0", "phi1", "phi2", "phim"):
                assert all(torch.equal(a,b) for a,b in zip(native["context"]["points"][point],audit["context"]["points"][point]))
            stepper.apply_update(.02, optimizer)
            assert all(torch.equal(p,v) for p,v in zip(params,final_native))
            assert rng_equal(rng_get(),rng_native)
            assert torch.equal(torch.randperm(23),next_batch_native)
            entry.restore()
            computed = stepper.compute(x.to(dtype),y,model.closure,.02)
            stepper.discard()
            entry.check()
            # Exception must also restore incoming buffers/cache/mode/grad/RNG.
            count = [0]
            def broken(a,b):
                count[0] += 1
                value = model.closure(a,b)
                if count[0] == 2:
                    model.eval()
                    model.unregistered_cache = torch.tensor(3.)
                    raise RuntimeError("intentional transactional gate")
                return value
            try:
                stepper.compute(x.to(dtype),y,broken,.02)
            except RuntimeError as error:
                assert "intentional" in str(error)
            else:
                raise AssertionError("Injected closure exception was not raised")
            entry.check()
            deltas = computed["record"]["probe_displacements"]
            assert abs(deltas["phi1"]["parameter_norm"]-.05) < 1e-5
            assert abs(deltas["phi2"]["parameter_norm"]-1.) < 1e-4
            key = str(dtype)+"/"+direction
            result["directions"][key] = dict(native_gradient_calls=expected["train_ce_per_update"],
                audit_gradient_calls=5, exact_transaction_restore=True,
                native_audit_U_exact=True, native_audit_submitted_parameters_exact=True,
                native_audit_submitted_rng_exact=True, next_batch_identity_exact=True,
                native_unused_readouts_absent=True, one_ordinary_sgd_update=applied_native["sgd_updates"] == 1,
                exception_state_restore=True, discarded_state_restore=True,
                common_points_exact=True, actual_probe_geometry=True,
                midpoint_rounding_error=computed["record"]["midpoint_rounding_parameter_error"])
    result["passed"] = True
    result["device"] = "cpu"
    result["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name("direction_step.py"))}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed": output["passed"], "direction_cases":len(output["directions"]),
                      "output":str(args.output)}, sort_keys=True))
