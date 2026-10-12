import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optimer_fishershape import (  # noqa: E402
    FisherShape,
    damp_trace_normalized_fisher,
    fisher_trace,
    normalize_fisher_by_trace,
)
from optimer_sam import SAM  # noqa: E402


class _VectorModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))


class FisherShapeOptimizerTest(unittest.TestCase):
    def _run_quadratic(
        self,
        *,
        alpha=0.2,
        alpha_mode="ratio",
        sam_equivalent=False,
        projector=None,
        first_grad_ready=False,
    ):
        model = _VectorModel([1.0, -2.0])
        curvature = torch.tensor([2.0, 4.0], dtype=torch.float64)
        learning_rate = 0.1
        rho = 0.05
        base = torch.optim.SGD([model.weight], lr=learning_rate)
        optimizer = FisherShape(
            [model.weight],
            base_optimizer=base,
            model=model,
            rho=rho,
            alpha=alpha,
            alpha_mode=alpha_mode,
            sam_equivalent=sam_equivalent,
        )
        if projector is not None:
            optimizer.set_direction_projector(projector)

        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad()
            loss = 0.5 * (curvature * model.weight.pow(2)).sum()
            loss.backward()
            return None, loss.detach()

        old = model.weight.detach().clone()
        g0 = curvature * old
        s0 = rho / g0.norm()
        perturbation = s0 * g0
        g1 = curvature * (old + perturbation)
        delta_grad = g1 - g0
        if sam_equivalent:
            coefficient = 1.0
        elif alpha_mode == "ratio":
            coefficient = alpha * g0.norm() / delta_grad.norm()
        else:
            coefficient = alpha / s0
        expected_direction = g0 + coefficient * delta_grad

        if first_grad_ready:
            clean_loss = 0.5 * (curvature * model.weight.pow(2)).sum()
            clean_loss.backward()
        optimizer.step(closure=closure, first_grad_ready=first_grad_ready)
        return model, optimizer, old, expected_direction, calls

    def test_ratio_direction_has_requested_relative_norm(self):
        model, optimizer, old, expected_direction, calls = self._run_quadratic(
            alpha=0.2,
            alpha_mode="ratio",
        )
        torch.testing.assert_close(model.weight, old - 0.1 * expected_direction)
        self.assertEqual(calls, 2)
        self.assertAlmostEqual(
            optimizer.last_stats["fs_shaping_to_g0_ratio"],
            0.2,
            places=10,
        )

    def test_sam_equivalent_uses_probe_gradient(self):
        model, _, old, expected_direction, _ = self._run_quadratic(
            sam_equivalent=True,
        )
        torch.testing.assert_close(model.weight, old - 0.1 * expected_direction)

    def test_sam_equivalent_matches_sam_with_adamw(self):
        """The Trainer's first-grad-ready path must reduce to a SAM update."""

        torch.manual_seed(7)
        sam_model = torch.nn.Linear(3, 2, bias=False).double()
        fishershape_model = torch.nn.Linear(3, 2, bias=False).double()
        fishershape_model.load_state_dict(sam_model.state_dict())

        features = torch.tensor(
            [[0.2, -0.4, 0.7], [1.1, 0.3, -0.2]],
            dtype=torch.float64,
        )
        targets = torch.tensor(
            [[0.1, -0.3], [0.8, 0.2]],
            dtype=torch.float64,
        )
        adamw_kwargs = {
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            "eps": 1e-6,
            "weight_decay": 0.0,
        }
        sam_optimizer = SAM(
            sam_model.parameters(),
            base_optimizer=torch.optim.AdamW,
            rho=0.05,
            **adamw_kwargs,
        )
        fishershape_base = torch.optim.AdamW(
            fishershape_model.parameters(),
            **adamw_kwargs,
        )
        fishershape_optimizer = FisherShape(
            fishershape_model.parameters(),
            base_optimizer=fishershape_base,
            model=fishershape_model,
            rho=0.05,
            alpha=0.0,
            alpha_mode="ratio",
            sam_equivalent=True,
            adaptive=False,
        )

        def loss_for(model):
            return (model(features) - targets).pow(2).mean()

        for _ in range(3):
            sam_optimizer.zero_grad(set_to_none=True)
            loss_for(sam_model).backward()
            sam_optimizer.first_step(zero_grad=True)
            loss_for(sam_model).backward()
            sam_optimizer.second_step(zero_grad=True)

            fishershape_optimizer.zero_grad(set_to_none=True)
            loss_for(fishershape_model).backward()

            def fishershape_probe():
                fishershape_optimizer.zero_grad(set_to_none=True)
                probe_loss = loss_for(fishershape_model)
                probe_loss.backward()
                return None, probe_loss.detach()

            fishershape_optimizer.step(
                closure=fishershape_probe,
                first_grad_ready=True,
            )
            self.assertEqual(
                fishershape_optimizer.last_stats["fs_coefficient"],
                1.0,
            )

        torch.testing.assert_close(
            fishershape_model.weight,
            sam_model.weight,
            # FisherShape restores by subtracting the perturbation and forms
            # g0 + (g1 - g0), while SAM restores a saved copy and keeps g1.
            # Those algebraically equivalent paths differ by a few float64 ulps.
            rtol=1e-10,
            atol=1e-11,
        )

    def test_clean_gradient_ready_path_needs_only_one_replay(self):
        model, _, old, expected_direction, calls = self._run_quadratic(
            alpha=0.1,
            first_grad_ready=True,
        )
        torch.testing.assert_close(model.weight, old - 0.1 * expected_direction)
        self.assertEqual(calls, 1)

    def test_additive_projector_runs_at_clean_weights(self):
        extra = torch.tensor([0.3, -0.2], dtype=torch.float64)
        observed_weights = []

        def projector(param_groups):
            for group in param_groups:
                for parameter in group["params"]:
                    observed_weights.append(parameter.detach().clone())
                    parameter.grad.add_(extra)

        model, _, old, expected_direction, _ = self._run_quadratic(
            alpha=0.0,
            projector=projector,
        )
        torch.testing.assert_close(observed_weights[0], old)
        torch.testing.assert_close(
            model.weight,
            old - 0.1 * (expected_direction + extra),
        )

    def test_failed_probe_restores_clean_parameters(self):
        model = _VectorModel([1.0, -2.0])
        base = torch.optim.SGD([model.weight], lr=0.1)
        optimizer = FisherShape(
            [model.weight],
            base_optimizer=base,
            model=model,
        )
        old = model.weight.detach().clone()
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad()
            loss = model.weight.pow(2).sum()
            loss.backward()
            if calls == 2:
                raise RuntimeError("synthetic probing failure")
            return None, loss.detach()

        with self.assertRaisesRegex(RuntimeError, "synthetic probing failure"):
            optimizer.step(closure=closure)
        torch.testing.assert_close(model.weight, old)
        self.assertIsNone(model.weight.grad)
        self.assertNotIn("fs_g0", optimizer.state[model.weight])
        self.assertNotIn("fs_perturbation", optimizer.state[model.weight])

    def test_failed_projector_clears_transient_state_and_gradients(self):
        model = _VectorModel([1.0, -2.0])
        base = torch.optim.SGD([model.weight], lr=0.1)
        optimizer = FisherShape(
            [model.weight],
            base_optimizer=base,
            model=model,
        )
        old = model.weight.detach().clone()

        def closure():
            optimizer.zero_grad()
            loss = model.weight.pow(2).sum()
            loss.backward()
            return None, loss.detach()

        def failed_projector(_):
            raise RuntimeError("synthetic projector failure")

        optimizer.set_direction_projector(failed_projector)
        with self.assertRaisesRegex(RuntimeError, "synthetic projector failure"):
            optimizer.step(closure=closure)
        torch.testing.assert_close(model.weight, old)
        self.assertIsNone(model.weight.grad)
        self.assertNotIn("fs_g0", optimizer.state[model.weight])

    def test_rho_zero_fixed_mode_uses_clean_gradient(self):
        model = _VectorModel([1.0, -2.0])
        base = torch.optim.SGD([model.weight], lr=0.1)
        optimizer = FisherShape(
            [model.weight],
            base_optimizer=base,
            model=model,
            rho=0.0,
            alpha=0.2,
            alpha_mode="fixed",
        )
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad()
            # Mimic stochastic probes at identical weights by changing the
            # linear objective between closure calls.
            loss = float(calls) * model.weight.sum()
            loss.backward()
            return None, loss.detach()

        old = model.weight.detach().clone()
        optimizer.step(closure=closure)
        torch.testing.assert_close(model.weight, old - 0.1 * torch.ones_like(old))
        self.assertEqual(optimizer.last_stats["fs_coefficient"], 0.0)

    def test_probe_only_missing_parameter_keeps_clean_gradient(self):
        first = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        second = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        model = torch.nn.Module()
        model.register_parameter("first", first)
        model.register_parameter("second", second)
        base = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer = FisherShape(
            model.parameters(),
            base_optimizer=base,
            model=model,
            alpha=0.0,
        )
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad(set_to_none=True)
            loss = first.pow(2)
            if calls == 1:
                loss = loss + second.pow(2)
            loss.backward()
            return None, loss.detach()

        optimizer.step(closure=closure)
        self.assertAlmostEqual(first.item(), 0.8)
        self.assertAlmostEqual(second.item(), 1.6)

    def test_zero_delta_gradient_stays_finite(self):
        model = _VectorModel([1.0, -2.0])
        base = torch.optim.SGD([model.weight], lr=0.1)
        optimizer = FisherShape(
            [model.weight],
            base_optimizer=base,
            model=model,
            alpha=0.2,
            alpha_mode="ratio",
        )

        def closure():
            optimizer.zero_grad()
            # A linear objective has H g0 = 0, hence g1-g0 is exactly zero.
            loss = model.weight.sum()
            loss.backward()
            return None, loss.detach()

        old = model.weight.detach().clone()
        optimizer.step(closure=closure)
        self.assertTrue(torch.isfinite(model.weight).all())
        torch.testing.assert_close(model.weight, old - 0.1 * torch.ones_like(old))
        self.assertEqual(optimizer.last_stats["fs_coefficient"], 0.0)

    def test_non_finite_hyperparameters_are_rejected(self):
        model = _VectorModel([1.0])
        base = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, "rho"):
            FisherShape(
                model.parameters(),
                base_optimizer=base,
                model=model,
                rho=float("nan"),
            )

    def test_clean_closure_without_parameter_gradients_is_rejected(self):
        model = _VectorModel([1.0])
        base = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer = FisherShape(
            model.parameters(),
            base_optimizer=base,
            model=model,
        )

        def closure():
            optimizer.zero_grad()
            detached_loss = torch.ones((), requires_grad=True)
            detached_loss.backward()
            return None, detached_loss.detach()

        with self.assertRaisesRegex(RuntimeError, "no parameter gradients"):
            optimizer.step(closure=closure)
        self.assertIsNone(model.weight.grad)


class FisherTransformTest(unittest.TestCase):
    def test_trace_normalization_and_isotropic_damping(self):
        fisher = {
            "a": torch.tensor([1.0, 3.0]),
            "b": torch.tensor([2.0]),
        }
        normalized, raw_trace = normalize_fisher_by_trace(fisher)
        self.assertEqual(raw_trace, 6.0)
        self.assertAlmostEqual(fisher_trace(normalized), 1.0, places=7)

        damped = damp_trace_normalized_fisher(normalized, beta=0.2)
        self.assertAlmostEqual(fisher_trace(damped), 1.0, places=7)
        expected_a = 0.8 * normalized["a"] + 0.2 / 3.0
        torch.testing.assert_close(damped["a"], expected_a)

    def test_invalid_damping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "beta"):
            damp_trace_normalized_fisher({"a": torch.ones(1)}, beta=1.1)


if __name__ == "__main__":
    unittest.main()
