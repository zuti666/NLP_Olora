#!/usr/bin/env python3
"""CPU algebra witness for the existing GAM implementation; no ML training.

Uses two parameter tensors, a deterministic three-dimensional quartic loss,
float64 and lr=0. Checks actual probe coordinates, global projection, leading
2g scale, a joint-small-radius expansion and clean-EWC scaling identities.
This verifies formulas on this example; it does not explain CL performance.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(ROOT))

import torch
from optimer_PerturabtionType.gam import GAM

torch.set_num_threads(1)
DTYPE = torch.float64
EPS = 1e-12
START = torch.tensor([.7, -.4, .2], dtype=DTYPE)
A = torch.tensor([[3., .2, .1], [.2, 1.5, .3], [.1, .3, .8]], dtype=DTYPE)
LINEAR = torch.tensor([-.1, .2, .05], dtype=DTYPE)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Parameter(START[:2].clone())
        self.right = torch.nn.Parameter(START[2:].clone())

    def vector(self):
        return torch.cat([self.left, self.right])

    def forward(self):
        x = self.vector()
        return .5 * x @ A @ x + .1 * x.pow(4).sum() + LINEAR @ x


def norm(x):
    return float(torch.linalg.vector_norm(x))


def run(rho, radius, beta1=1., beta3=1., gamma=.1):
    model = Toy()
    params = list(model.parameters())
    base = torch.optim.SGD(params, lr=0., momentum=0., weight_decay=0.)
    opt = GAM(params, base_optimizer=base, model=model, grad_rho=rho,
              grad_norm_rho=radius, adaptive=False, perturb_eps=EPS,
              args=SimpleNamespace(grad_beta_1=beta1, grad_beta_2=1.,
                                   grad_beta_3=beta3, grad_gamma=gamma,
                                   use_sgv=False))
    points, gradients = [], []

    def closure():
        opt.zero_grad()
        points.append(model.vector().detach().clone())
        loss = model()
        loss.backward()
        gradients.append(torch.cat([p.grad.detach().clone() for p in params]))
        return model.vector().detach(), loss.detach()

    opt.step(closure)
    g0, g1, g2, g3 = gradients
    direction = torch.cat([p.grad.detach() for p in params])
    a, b = g0 + g2, beta1 * g1 + beta3 * g3
    cosine = (a @ b) / (a.norm() * b.norm() + EPS)
    vertical = a - cosine * a.norm() * b / (b.norm() + EPS)
    expected = b - gamma * vertical
    ascent = radius * (g1 - g0) / ((g1 - g0).norm() + EPS)
    expected_points = [START, START + rho * g0 / (g0.norm() + EPS),
                       START + ascent, START + ascent + rho * g2 / (g2.norm() + EPS)]
    result = {
        "rho": rho, "outer_radius": radius,
        "beta1": beta1, "beta2": 1., "beta3": beta3, "gamma": gamma,
        "ce_gradients": len(gradients),
        "probe_position_max_error": max(norm(x - y) for x, y in zip(points, expected_points)),
        "projection_formula_error": norm(direction - expected),
        "parameter_restore_error": norm(model.vector().detach() - START),
        "direction": direction.tolist(), "g0": g0.tolist(),
        "direction_to_clean_norm": norm(direction) / norm(g0),
    }
    assert result["ce_gradients"] == 4
    assert max(result[k] for k in ["probe_position_max_error", "projection_formula_error",
                                  "parameter_restore_error"]) < 1e-10
    return result, direction, g0


def main():
    zero, zero_d, g = run(0., 0.)
    zero["error_vs_2g"] = norm(zero_d - 2 * g)
    zero["lambda_flat_half_error_vs_1point5g"] = norm(g + .5 * (zero_d - g) - 1.5 * g)
    assert zero["error_vs_2g"] < 1e-10
    hessian = A + torch.diag(1.2 * START.square())
    u = g / g.norm()
    hg = hessian @ g
    v = hg / hg.norm()
    hu = hessian @ u
    hu_perp = hu - g * ((g @ hu) / (g @ g))
    rows = []
    for scale in [1., .5, .25, .125, .0625, .03125]:
        rho, radius = .05 * scale, .1 * scale
        row, direction, _ = run(rho, radius)
        approximation = 2 * g + 2 * rho * hu + radius * (hessian @ v) + .2 * rho * hu_perp
        row.update(scale=scale, expansion_absolute_error=norm(direction - approximation))
        rows.append(row)
    ratios = [x["expansion_absolute_error"] / y["expansion_absolute_error"]
              for x, y in zip(rows, rows[1:])]
    assert all(3.5 < ratio < 4.5 for ratio in ratios[-3:])
    original, direction, _ = run(.05, .1)
    _, half_direction, _ = run(.05, .1, beta1=.5, beta3=.5, gamma=.05)
    _, only_beta_half, _ = run(.05, .1, beta1=.5, beta3=.5, gamma=.1)
    eta, lam = .04, 2000.
    unweighted_ewc = torch.tensor([1., -2., 3.], dtype=DTYPE) * 1e-5
    target = START - eta * (.5 * direction + lam * unweighted_ewc)
    matched = START - .5 * eta * (direction + 2 * lam * unweighted_ewc)
    unmatched = START - .5 * eta * (direction + lam * unweighted_ewc)
    controls = {
        "beta1_beta3_and_gamma_half_error_vs_half_direction": norm(half_direction - .5 * direction),
        "only_beta1_beta3_half_error_vs_half_direction": norm(only_beta_half - .5 * direction),
        "half_lr_double_ewc_lambda_error_vs_half_task_only": norm(matched - target),
        "half_lr_unchanged_ewc_lambda_deviation": norm(unmatched - target),
    }
    assert controls["beta1_beta3_and_gamma_half_error_vs_half_direction"] < 1e-10
    assert controls["half_lr_double_ewc_lambda_error_vs_half_task_only"] < 1e-12
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "device": "cpu", "dtype": "float64", "optimizer_lr": 0.,
        "scope": "Deterministic toy algebra witness only; no dataset, GPU training or CL performance evidence.",
        "zero_radius": zero, "reference_radii": original, "small_radius_rows": rows,
        "successive_error_ratios": ratios, "scale_controls": controls,
    }
    target_path = Path(__file__).resolve().parent / "GAM_GEOMETRY_WITNESS.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "zero_error": zero["error_vs_2g"],
                      "expansion_error_ratios": ratios, "controls": controls,
                      "output": str(target_path)}, indent=2))


if __name__ == "__main__":
    main()
