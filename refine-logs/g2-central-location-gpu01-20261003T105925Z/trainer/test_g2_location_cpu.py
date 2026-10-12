"""CPU scientific acceptance for the six-gradient offline transaction."""
import json
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from audit_core import rng_equal, rng_get, seed_all
from effective_step import EffectiveStep
from g2_location_diagnostics import FullState, RUNS, TASK_SAMPLES, audit_batch, norm, sub, validate_formal_pairing


class Toy(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        generator = torch.Generator().manual_seed(97)
        self.a = torch.nn.Parameter(torch.randn(2, 3, generator=generator, dtype=dtype)*.2)
        self.b = torch.nn.Parameter(torch.randn(4, 2, generator=generator, dtype=dtype)*.2)
        self.fc = torch.nn.Linear(4, 5, dtype=dtype)
        self.frozen = torch.nn.Parameter(torch.tensor([.25, -.3], dtype=dtype), requires_grad=False)
        self.dropout = torch.nn.Dropout(.25)
        self.register_buffer("probe_counter", torch.zeros((), dtype=dtype))
        self.forward_cache = {"tensor": torch.tensor([3.], dtype=dtype), "nested": [torch.tensor([4.], dtype=dtype)]}
        self.fused_attn = True
        self.external_delta_w_q = None
        self.external_delta_w_v = None
        self._register_delta_hook = False

    def forward(self, inputs):
        if not self.training or not self.dropout.training:
            raise AssertionError("Offline probe did not use train mode")
        if self.fused_attn:
            raise AssertionError("Offline probe did not disable fused attention")
        self.probe_counter.add_(1)
        self.forward_cache["tensor"].add_(1)
        self.forward_cache["nested"][0].add_(1)
        self.forward_cache["new"] = 9
        self.temporary_cache = torch.ones(3)
        random.random()
        np.random.random()
        features = self.dropout(inputs @ self.a.T @ self.b.T)
        return {"logits": self.fc(torch.tanh(features))}


def case(dtype=torch.float64):
    seed_all(72)
    model = Toy(dtype)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    geometry = EffectiveStep([(model.a, model.b)], tuple(model.fc.parameters()), params)
    return model, params, geometry


def flat(values):
    return torch.cat([value.reshape(-1) for value in values])


def quad_closure(params, scale=1.):
    count = sum(p.numel() for p in params)
    matrix = torch.diag(torch.linspace(.3, 1.5, count, dtype=params[0].dtype))
    bias = torch.linspace(-.2, .1, count, dtype=params[0].dtype)
    def closure():
        theta = flat(params)
        return scale*(.5*theta@(matrix@theta) - bias@theta), theta[:5].unsqueeze(0)
    return closure, matrix, bias


class OfflineAcceptance(unittest.TestCase):
    def close(self, a, b, atol=2e-12, rtol=2e-11):
        self.assertTrue(torch.allclose(flat(a), flat(b), atol=atol, rtol=rtol),
                        "max delta=" + str(float((flat(a)-flat(b)).abs().max())))

    def test_quadratic_shared_geometry_and_six_gradients(self):
        model, params, geometry = case()
        closure, hessian, bias = quad_closure(params)
        snapshot = FullState(model)
        result = audit_batch(model, None, None, 0, 5, geometry, closure=closure,
                             allow_float64_fixture=True, return_context=True)
        ctx = result.pop("_context")
        g0 = hessian@flat(ctx["base"])-bias
        self.assertTrue(torch.allclose(flat(ctx["g0"]), g0, atol=1e-14, rtol=1e-14))
        expected_q = .05*hessian@(g0/(norm(ctx["g0"])+1e-12))
        for branch in ("P", "C", "CD"):
            self.assertTrue(torch.allclose(flat(ctx["q"][branch]), expected_q, atol=2e-15, rtol=2e-13))
            expected_g2 = hessian@flat(ctx["outer"][branch])-bias
            self.assertTrue(torch.allclose(flat(ctx["g2"][branch]), expected_g2, atol=1e-14, rtol=1e-14))
        self.close(ctx["outer"]["P"], ctx["outer"]["CD"])
        self.close(ctx["outer"]["C"], ctx["outer"]["CD"])
        self.assertEqual(result["ordinary_gradient_calls"], 6)
        self.assertEqual(len(result["probes"]), 6)
        self.assertTrue(all(snapshot.check().values()))
        json.dumps(result, allow_nan=False)

    def test_nonlinear_CD_is_raw_central_not_unit_direction_average(self):
        model, params, geometry = case()
        def closure():
            theta = flat(params)
            return (theta.pow(4)/4 + .2*theta.square()).sum(), theta[:5].unsqueeze(0)
        result = audit_batch(model, None, None, 0, 5, geometry, closure=closure,
                             allow_float64_fixture=True, return_context=True)
        ctx = result["_context"]
        self.assertGreater(norm(ctx["e"]), 1e-7)
        self.close(ctx["q"]["CD"], tuple(.5*(p-m) for p, m in zip(ctx["gp"], ctx["gm"])), atol=0, rtol=0)
        self.close(ctx["q"]["P"], tuple(s+e for s, e in zip(ctx["s"], ctx["e"])))
        self.close(ctx["q"]["C"], sub(ctx["s"], ctx["e"]))
        self.assertGreater(result["same_state_contrasts"]["CD_vs_P"]["outer_point_delta"]["parameter_norm"], 1e-5)

    def test_small_q_keeps_CD_half_and_normalization_epsilon(self):
        model, params, geometry = case()
        closure, _, _ = quad_closure(params, 1e-10)
        result = audit_batch(model, None, None, 0, 5, geometry, closure=closure,
                             allow_float64_fixture=True, return_context=True)
        ctx = result["_context"]
        q = ctx["q"]["CD"]
        self.close(q, tuple(.5*(p-m) for p, m in zip(ctx["gp"], ctx["gm"])), atol=0, rtol=0)
        expected = tuple(p+.25*v/(norm(q)+1e-12) for p, v in zip(ctx["base"], q))
        self.close(ctx["outer"]["CD"], expected, atol=0, rtol=0)
        wrong = tuple(p+.25*(2*v)/(2*norm(q)+1e-12) for p, v in zip(ctx["base"], q))
        self.assertGreater(norm(sub(wrong, expected)), 1e-5)

    def test_zero_outer_radius_reads_g0_exactly(self):
        model, params, geometry = case()
        closure, _, _ = quad_closure(params)
        result = audit_batch(model, None, None, 0, 5, geometry, radius=0, closure=closure,
                             allow_float64_fixture=True, return_context=True)
        for branch in ("P", "C", "CD"):
            self.close(result["_context"]["g2"][branch], result["_context"]["g0"], atol=0, rtol=0)
            self.assertEqual(result["locations"][branch]["actual_outer_displacement"]["parameter_norm"], 0.)

    def test_constant_gradient_q_zero_angles_missing_low_signal(self):
        model, params, geometry = case()
        def closure():
            theta = flat(params)
            return theta.sum(), theta[:5].unsqueeze(0)
        result = audit_batch(model, None, None, 0, 5, geometry, closure=closure,
                             allow_float64_fixture=True, return_context=True)
        self.assertTrue(result["low_s"])
        self.assertTrue(all(value is None for value in result["q_cosines"].values()))
        for branch in ("P", "C", "CD"):
            self.assertEqual(result["locations"][branch]["q_norm"], 0.)
            self.close(result["_context"]["outer"][branch], result["_context"]["base"], atol=0, rtol=0)

    def test_zero_base_gradient_collapses_first_and_outer_points(self):
        model, params, geometry = case()
        with torch.no_grad():
            for p in params:
                p.zero_()
        def closure():
            theta = flat(params)
            return .5*theta.square().sum(), theta[:5].unsqueeze(0)
        result = audit_batch(model, None, None, 0, 5, geometry, closure=closure,
                             allow_float64_fixture=True, return_context=True)
        ctx = result["_context"]
        self.assertEqual(result["norms"]["g0"], 0.)
        for branch in ("P", "C", "CD"):
            self.assertEqual(norm(ctx["g2"][branch]), 0.)
            self.assertIsNone(result["locations"][branch]["cos_g2_g0"])
            self.close(ctx["outer"][branch], ctx["base"], atol=0, rtol=0)

    def test_FP32_current_task_CE_stochastic_replay_and_complete_state(self):
        model, params, geometry = case(torch.float32)
        model.eval()
        model.dropout.train()
        for index, p in enumerate(model.parameters()):
            p.grad = torch.full_like(p, index+.75)
        optimizer = torch.optim.SGD(params, lr=.02, momentum=.8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        for p in params:
            optimizer.state[p]["momentum_buffer"] = torch.full_like(p, .3)
        optimizer.step = lambda *a, **k: self.fail("Offline diagnostic called optimizer.step")
        scheduler.step = lambda *a, **k: self.fail("Offline diagnostic called scheduler.step")
        inputs = torch.tensor([[.3, -.5, .7], [-.2, .3, .8], [.4, -.4, .1]], dtype=torch.float32)
        targets = torch.tensor([2, 4, 3])
        snapshot = FullState(model, optimizer, scheduler)
        rng = rng_get()
        result = audit_batch(model, inputs, targets, 2, 5, geometry,
                             optimizer=optimizer, scheduler=scheduler, return_context=True)
        self.assertTrue(all(snapshot.check().values()))
        self.assertTrue(rng_equal(rng, rng_get()))
        self.assertEqual(result["ordinary_gradient_calls"], 6)
        self.assertEqual(result["class_slice"], [2, 5])
        self.assertEqual(len({p["R0_sha256"] for p in result["probes"].values()}), 1)
        self.assertEqual(len({p["R1_sha256"] for p in result["probes"].values()}), 1)
        self.assertNotEqual(result["R0_sha256"], result["R1_sha256"])
        self.assertFalse(hasattr(model, "temporary_cache"))
        # All seen FC rows participate in geometry, but old rows get zero CE gradient.
        g0 = {id(p): g for p, g in zip(params, result["_context"]["g0"])}
        self.assertTrue(torch.equal(g0[id(model.fc.weight)][:2], torch.zeros_like(model.fc.weight[:2])))
        self.assertTrue(torch.equal(g0[id(model.fc.bias)][:2], torch.zeros_like(model.fc.bias[:2])))
        # Same input/parameters/entry RNG must reproduce all six gradients bitwise.
        again = audit_batch(model, inputs, targets, 2, 5, geometry, optimizer=optimizer, scheduler=scheduler)
        self.assertEqual({k: p["gradient_sha256"] for k, p in again["probes"].items()},
                         {k: p["gradient_sha256"] for k, p in result["probes"].items()})

    def test_failure_restores_frozen_values_grad_objects_caches_modes_RNG(self):
        model, params, geometry = case()
        model.eval()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        snapshot = FullState(model)
        calls = 0
        def closure():
            nonlocal calls
            calls += 1
            model.probe_counter.add_(1)
            model.forward_cache["nested"][0].add_(7)
            model.forward_cache = {"replacement": 3}
            model.failure_cache = [1]
            random.random()
            np.random.random()
            torch.rand(1)
            if calls == 3:
                with torch.no_grad():
                    model.frozen.add_(7)
                    model.a.grad.add_(3)
                raise RuntimeError("intentional failed third probe")
            theta = flat(params)
            return .5*theta.square().sum(), theta[:5].unsqueeze(0)
        with self.assertRaisesRegex(RuntimeError, "intentional failed third probe"):
            audit_batch(model, None, None, 0, 5, geometry, closure=closure, allow_float64_fixture=True)
        self.assertEqual(calls, 3)
        self.assertTrue(all(snapshot.check().values()))
        self.assertFalse(hasattr(model, "failure_cache"))

    def test_formal_pairing_covers_all_ten_tasks_and_2550_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id, branch in zip(RUNS, ("P", "C", "CD")):
                path = root/"outputs"/run_id
                path.mkdir(parents=True)
                tasks = []
                for task in range(10):
                    tasks.append({"task": task, "seed": 2, "class_slice": [20*task, 20*(task+1)],
                                  "fc_shape": [20*(task+1), 4], "old_fc_rows_inherited_exact": True,
                                  "new_fc_rows_sha256": {"weight": "shared_rows"}, "frozen_origin_sha256": "frozen",
                                  "task_initial_parameter_sha256": "shared_initial" if task == 0 else run_id+str(task),
                                  "data_seed_rule": "fixed", "forward_rng_rule": "fixed",
                                  "probe_branch": branch, "update_direction": "g2", "attention_backend": "math"})
                (path/"task_identities.json").write_text(json.dumps(tasks))
                (path/"task_matrix.json").write_text(json.dumps({"matrix": [[42.]*(task+1) for task in range(10)]}))
                step = 0
                with open(path/"step_trace.jsonl", "w") as handle:
                    for task in range(10):
                        for epoch in range(1, 51):
                            for batch in range((TASK_SAMPLES[task]+127)//128):
                                row = {"task": task, "epoch": epoch, "batch": batch, "step_index": step,
                                       "seed": 2, "class_slice": [20*task, 20*(task+1)], "seen_classes": 20*(task+1),
                                       "identity": {"input_sha256": str(step), "target_sha256": str(step)},
                                       "probe_branch": branch, "update_direction": "g2", "step_mode": "natural", "attention_backend": "math"}
                                handle.write(json.dumps(row)+"\n")
                                step += 1
            result = validate_formal_pairing(root)
            self.assertTrue(result["formal_whole_task_matrix_identity"])
            self.assertEqual(result["paired_batch_count"], 2550)
            self.assertEqual(result["batches_per_epoch"], [10, 8, 8, 5, 4, 3, 3, 3, 4, 3])
            # A single later-task augmentation mismatch invalidates full pairing.
            target = root/"outputs"/RUNS[2]/"step_trace.jsonl"
            lines = target.read_text().splitlines()
            row = json.loads(lines[-1])
            row["identity"]["input_sha256"] = "changed_augmented_sample"
            lines[-1] = json.dumps(row)
            target.write_text("\n".join(lines)+"\n")
            with self.assertRaisesRegex(AssertionError, "paired batch/method matrix mismatch"):
                validate_formal_pairing(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
