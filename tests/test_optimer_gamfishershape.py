import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optimer_gamfishershape import GAMFisherShapeSAM  # noqa: E402
from optimer_sam import SAM  # noqa: E402
from uie_trainer_lora import UIETrainer  # noqa: E402


class _VectorModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))


class _ToyLoRAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(
            torch.tensor([[1.0, -0.5, 0.25], [-0.2, 0.4, 0.8]], dtype=torch.float64),
            requires_grad=False,
        )
        self.loranew_A = torch.nn.Parameter(
            torch.tensor([[0.3, 0.1, -0.4], [0.6, -0.2, 0.5]], dtype=torch.float64)
        )
        self.loranew_B = torch.nn.Parameter(
            torch.tensor([[0.1, -0.2], [0.2, 0.3], [-0.4, 0.5]], dtype=torch.float64)
        )


class GAMFisherShapeSAMTest(unittest.TestCase):
    def test_quadratic_direction_is_probe_gradient(self):
        model = _VectorModel([1.0, -2.0])
        curvature = torch.tensor([2.0, 4.0], dtype=torch.float64)
        rho = 0.05
        learning_rate = 0.1
        base = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = GAMFisherShapeSAM(
            model.parameters(),
            base_optimizer=base,
            model=model,
            rho=rho,
            mode="sam",
        )

        old = model.weight.detach().clone()
        g0 = curvature * old
        probe = old + rho * g0 / g0.norm()
        g1 = curvature * probe
        (0.5 * (curvature * model.weight.pow(2)).sum()).backward()
        calls = 0

        def probe_closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * (curvature * model.weight.pow(2)).sum()
            loss.backward()
            return None, loss.detach()

        optimizer.step(closure=probe_closure, first_grad_ready=True)

        torch.testing.assert_close(model.weight, old - learning_rate * g1)
        self.assertEqual(calls, 1)
        self.assertEqual(optimizer.last_stats["fs_coefficient"], 1.0)
        self.assertEqual(optimizer.last_stats["gamfs_sam_equivalent"], 1.0)
        self.assertEqual(optimizer.last_stats["gamfs_gam_probe_count"], 0.0)

    def test_three_steps_match_legacy_sam_direction_and_adamw(self):
        torch.manual_seed(19)
        sam_model = torch.nn.Linear(3, 2, bias=False).double()
        gamfs_model = torch.nn.Linear(3, 2, bias=False).double()
        gamfs_model.load_state_dict(sam_model.state_dict())
        features = torch.tensor(
            [[0.2, -0.4, 0.7], [1.1, 0.3, -0.2]], dtype=torch.float64
        )
        targets = torch.tensor(
            [[0.1, -0.3], [0.8, 0.2]], dtype=torch.float64
        )
        adamw_kwargs = dict(
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.0,
        )
        sam_optimizer = SAM(
            sam_model.parameters(),
            base_optimizer=torch.optim.AdamW,
            rho=0.05,
            **adamw_kwargs,
        )
        gamfs_base = torch.optim.AdamW(gamfs_model.parameters(), **adamw_kwargs)
        gamfs_optimizer = GAMFisherShapeSAM(
            gamfs_model.parameters(),
            base_optimizer=gamfs_base,
            model=gamfs_model,
            rho=0.05,
            mode="sam_equivalent",
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

            gamfs_optimizer.zero_grad(set_to_none=True)
            loss_for(gamfs_model).backward()
            closure_calls = 0

            def probe_closure():
                nonlocal closure_calls
                closure_calls += 1
                gamfs_optimizer.zero_grad(set_to_none=True)
                probe_loss = loss_for(gamfs_model)
                probe_loss.backward()
                return None, probe_loss.detach()

            gamfs_optimizer.step(
                closure=probe_closure,
                first_grad_ready=True,
            )
            self.assertEqual(closure_calls, 1)
            self.assertEqual(gamfs_optimizer.last_stats["fs_coefficient"], 1.0)
            torch.testing.assert_close(
                gamfs_model.weight,
                sam_model.weight,
                rtol=1e-10,
                atol=1e-11,
            )

    def test_only_sam_mode_is_accepted(self):
        model = _VectorModel([1.0])
        base = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, "only supports mode='sam'"):
            GAMFisherShapeSAM(
                model.parameters(),
                base_optimizer=base,
                model=model,
                mode="gam",
            )

    def test_optimizer_family_routing_is_mutually_exclusive(self):
        expected = {
            "adamw_hf": "standard",
            "sam-adamw_hf": "sam",
            "gam-adamw_hf": "gam",
            "fishershape-adamw_hf": "fishershape",
            "gamfishershape-adamw_hf": "gamfishershape",
        }
        for optimizer_type, family in expected.items():
            with self.subTest(optimizer_type=optimizer_type):
                self.assertEqual(
                    UIETrainer._optimizer_family(optimizer_type),
                    family,
                )

    def test_strategy_uses_ewclora_sidecar_without_old_gam_projection(self):
        strategy = "ewclora_gamfishershape"
        self.assertIn(strategy, UIETrainer.EWCLORA_STRATEGIES)
        self.assertIn(strategy, UIETrainer.FISHERSHAPE_EWCLORA_STRATEGIES)
        self.assertNotIn(strategy, UIETrainer.BASE_EWCLORA_STRATEGIES)

    def test_clean_point_olora_gradient_is_added_to_sam_direction(self):
        """OLoRA is differentiated once at clean weights and added to g1."""

        model = _ToyLoRAModel()
        trainer = object.__new__(UIETrainer)
        trainer.args = SimpleNamespace(
            lora_strategy="ewclora_gamfishershape",
            ewclora_lambda=0.0,
            gamfs_include_olora_regularization=True,
            gamfs_xi=1e-12,
            olora_lamda_1=0.5,
            olora_lamda_2=0.0,
        )
        trainer.ewclora_omega = {}
        params = [model.loranew_A, model.loranew_B]

        regularization_cache, metrics = (
            trainer._compute_fishershape_regularization_grad_cache(model, params)
        )
        expected_loss = 0.5 * torch.abs(
            model.lora_A @ model.loranew_A.T
        ).sum()
        (expected_a_grad,) = torch.autograd.grad(expected_loss, [model.loranew_A])

        self.assertAlmostEqual(metrics["ewc_loss"], 0.0)
        self.assertAlmostEqual(metrics["olora_loss"], expected_loss.item())
        self.assertIn(id(model.loranew_A), regularization_cache)
        self.assertNotIn(id(model.loranew_B), regularization_cache)
        torch.testing.assert_close(
            regularization_cache[id(model.loranew_A)],
            expected_a_grad,
        )

        sam_direction = torch.full_like(model.loranew_A, 0.25)
        model.loranew_A.grad = sam_direction.clone()
        projector = trainer._build_fishershape_additive_projector(
            regularization_cache,
            metrics,
        )
        projector([{"params": params}])
        torch.testing.assert_close(
            model.loranew_A.grad,
            sam_direction + expected_a_grad,
        )
        self.assertGreater(metrics["regularization_grad_norm"], 0.0)

    def test_clean_point_olora_and_ewc_gradients_are_added_together(self):
        """The joint cache is grad(L_OLoRA + L_EWC), with no projection."""

        model = _ToyLoRAModel()
        trainer = object.__new__(UIETrainer)
        ewc_lambda = 0.7
        trainer.args = SimpleNamespace(
            lora_strategy="ewclora_gamfishershape",
            ewclora_lambda=ewc_lambda,
            gamfs_include_olora_regularization=True,
            gamfs_xi=1e-12,
            olora_lamda_1=0.5,
            olora_lamda_2=0.0,
        )
        delta_w = model.loranew_B @ model.loranew_A
        trainer.ewclora_omega = {"": torch.ones_like(delta_w)}
        params = [model.loranew_A, model.loranew_B]

        regularization_cache, metrics = (
            trainer._compute_fishershape_regularization_grad_cache(model, params)
        )
        olora_loss = 0.5 * torch.abs(
            model.lora_A @ model.loranew_A.T
        ).sum()
        ewc_loss = 0.5 * ewc_lambda * delta_w.pow(2).sum()
        expected_grads = torch.autograd.grad(olora_loss + ewc_loss, params)

        self.assertAlmostEqual(metrics["olora_loss"], olora_loss.item())
        self.assertAlmostEqual(metrics["ewc_loss"], ewc_loss.item())
        for param, expected_grad in zip(params, expected_grads):
            torch.testing.assert_close(
                regularization_cache[id(param)],
                expected_grad,
            )

        sam_directions = [torch.full_like(param, 0.25) for param in params]
        for param, sam_direction in zip(params, sam_directions):
            param.grad = sam_direction.clone()
        projector = trainer._build_fishershape_additive_projector(
            regularization_cache,
            metrics,
        )
        projector([{"params": params}])
        for param, sam_direction, expected_grad in zip(
            params,
            sam_directions,
            expected_grads,
        ):
            torch.testing.assert_close(
                param.grad,
                sam_direction + expected_grad,
            )


if __name__ == "__main__":
    unittest.main()
