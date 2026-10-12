"""Small synthetic checks of the actual frozen GAM and diagnostic restoration."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("paired", HERE / "paired_gam_diagnostic.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
SOURCE = "/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source"
sys.path.insert(0, SOURCE)
from optimer_PerturabtionType.gam import GAM


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_As = torch.nn.ModuleList([torch.nn.Linear(4, 2, bias=False) for _ in range(2)])
        self.w_Bs = torch.nn.ModuleList([torch.nn.Linear(2, 4, bias=False) for _ in range(2)])
        self.frozen = torch.nn.Parameter(torch.eye(4), requires_grad=False)
        self.register_buffer("constant", torch.tensor(1.0))

    def forward(self, x):
        torch.rand(1)  # Frozen production LoRA forward also consumes CPU RNG.
        return x @ self.frozen + sum(b(a(x)) for a, b in zip(self.w_As, self.w_Bs))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        return {"logits": self.fc(self.backbone(x))}


def main():
    d.seed_all(19)
    torch.set_num_threads(1)
    model = Model().double()
    geo = d.Geometry(model)
    x = torch.randn(8, 4, dtype=torch.float64)
    y = torch.tensor([2, 3] * 4)
    ex = torch.randn(8, 4, dtype=torch.float64)
    eval_batches = {0: [(ex, torch.tensor([0, 1] * 4), torch.arange(8))],
                    1: [(ex, y, torch.arange(8))]}
    ranges = {0: (0, 2), 1: (2, 4)}
    anchor = d.Snapshot(model)
    reference = d.isolated_logits(model, ex)
    before = d.eval_losses(model, eval_batches, ranges, torch.device("cpu"))
    grads, stats, shifts = d.gam_probes(model, geo, anchor, x, y, 2, 4, .05, 1, ex, reference, GAM)
    zeros, zero_stats, _ = d.gam_probes(model, geo, anchor, x, y, 2, 4, .05, 0, ex, reference, GAM)
    assert stats["closure_count"] == 4 and stats["ce_gradient_passes"] == 5
    assert torch.allclose(zeros["g3"], zeros["g1"], atol=1e-12)
    g1, g3 = grads["g1"], grads["g3"]
    a = d.dot(g3, g1) / d.dot(g1, g1)
    residual = g3 - a * g1
    assert abs(d.dot(residual, g1)) < 1e-12
    anchor.restore()
    geo.set_displacement(g1, -.02)
    target = geo.displacement_metrics()["effective_joint_norm"]
    anchor.restore()
    records = []
    directions = {"g1": g1, "g3": g3, "sum": g1 + g3, "half_sum": (g1 + g3) / 2,
                  "scale": (1 + a) * g1, "g1+residual": g1 + residual}
    for label, direction in directions.items():
        out = d.counterfactual(model, geo, anchor, direction, label, target, .02,
                               eval_batches, ranges, 1, torch.device("cpu"), before, 1e-3)
        assert out["match"]["ok"], out
        assert out["match"]["actual_relative_error"] < 1e-3
        records.append(out)
    scale = records[4]["match"]["scale"]
    one = records[0]["match"]["scale"]
    if 1 + a > 0:
        assert abs(scale * (1 + a) - one) < 1e-6
    sam, sam_gradient = d.sam_at_matched_probe(model, geo, anchor, grads["g0"],
        stats["points"][3]["parameter_norm"], x, y, 2, 4, ex, reference, .05)
    assert sam["ok"] and sam_gradient is not None
    kl, _ = d.sam_at_matched_probe(model, geo, anchor, grads["g0"],
        stats["points"][3]["parameter_norm"], x, y, 2, 4, ex, reference, .05,
        target_kl=stats["points"][3]["kl_all_seen"])
    anchor.restore()
    checks = anchor.check()
    output = {"status": "passed", "source": SOURCE, "a": a,
              "r0_identity_max_abs": float((zeros["g3"] - zeros["g1"]).abs().max()),
              "matched_candidates": len(records), "all_restore": checks,
              "radius_sam": sam, "kl_sam": kl}
    d.atomic_json(HERE / "cpu_test_report.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
