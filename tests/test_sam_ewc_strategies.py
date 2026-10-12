import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from peft.tuners.lora import Linear, LoraConfig  # noqa: E402
from peft.utils.save_and_load import get_peft_model_state_dict  # noqa: E402
from uie_trainer_lora import UIETrainer  # noqa: E402


class _ToyLoRAModel(torch.nn.Module):
    def __init__(self, *, save_lora_only=False):
        super().__init__()
        self.layer = Linear(
            "default",
            in_features=3,
            out_features=2,
            r=2,
            lora_alpha=2,
            lora_dropout=0.0,
            r_sum=2,
        )
        self.peft_config = {
            "default": LoraConfig(
                r=2,
                lora_alpha=2,
                lora_dropout=0.0,
                r_sum=2,
                save_lora_only=save_lora_only,
            )
        }
        self.modules_to_save = None

        with torch.no_grad():
            self.layer.lora_A["default"].weight.copy_(
                torch.tensor(
                    [[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]],
                    dtype=torch.float32,
                )
            )
            self.layer.lora_B["default"].weight.copy_(
                torch.tensor(
                    [[0.2, 0.3], [-0.4, 0.1]],
                    dtype=torch.float32,
                )
            )
            self.layer.loranew_A["default"].weight.copy_(
                torch.tensor(
                    [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    dtype=torch.float32,
                )
            )
            self.layer.loranew_B["default"].weight.copy_(
                torch.tensor(
                    [[10.0, 11.0], [12.0, 13.0]],
                    dtype=torch.float32,
                )
            )


def _bare_trainer(strategy="seqlora_sam_ewc", **arg_overrides):
    args = dict(
        lora_strategy=strategy,
        ewclora_lambda=0.7,
        ewclora_gamma=0.8,
        ewclora_normalize_fisher=False,
        ewclora_normalize_accumulated_fisher=False,
        ewclora_fisher_damping_beta=0.0,
        ewclora_fisher_eps=1e-12,
        ewclora_allow_state_mismatch=False,
        do_train=True,
        device=torch.device("cpu"),
    )
    args.update(arg_overrides)
    trainer = object.__new__(UIETrainer)
    trainer.args = SimpleNamespace(**args)
    trainer.ewclora_omega = {}
    trainer.ewclora_reference = {}
    trainer.ewclora_count_updates = 0
    return trainer


class SAMEWCStrategyTest(unittest.TestCase):
    def test_new_strategies_are_registered_in_the_fishershape_family(self):
        for strategy in ("inclora_sam_ewc", "inclora-sam-ewc"):
            with self.subTest(strategy=strategy):
                self.assertIn(strategy, UIETrainer.INCLORA_SAM_EWC_STRATEGIES)
                self.assertIn(
                    strategy,
                    UIETrainer.BASE_FISHERSHAPE_EWCLORA_STRATEGIES,
                )
                self.assertIn(strategy, UIETrainer.FISHERSHAPE_EWCLORA_STRATEGIES)
                self.assertIn(strategy, UIETrainer.EWCLORA_STRATEGIES)
                self.assertNotIn(strategy, UIETrainer.BASE_EWCLORA_STRATEGIES)
                self.assertNotIn(
                    strategy,
                    UIETrainer.GAMFISHERSHAPE_EWCLORA_STRATEGIES,
                )

        for strategy in ("seqlora_sam_ewc", "seqlora-sam-ewc"):
            with self.subTest(strategy=strategy):
                self.assertIn(strategy, UIETrainer.SEQLORA_SAM_EWC_STRATEGIES)
                self.assertIn(
                    strategy,
                    UIETrainer.BASE_FISHERSHAPE_EWCLORA_STRATEGIES,
                )
                self.assertIn(strategy, UIETrainer.FISHERSHAPE_EWCLORA_STRATEGIES)
                self.assertIn(strategy, UIETrainer.EWCLORA_STRATEGIES)
                self.assertNotIn(strategy, UIETrainer.BASE_EWCLORA_STRATEGIES)
                self.assertNotIn(
                    strategy,
                    UIETrainer.GAMFISHERSHAPE_EWCLORA_STRATEGIES,
                )

        inclora = _bare_trainer("inclora_sam_ewc")
        self.assertTrue(inclora._is_inclora_sam_ewc_strategy())
        self.assertFalse(inclora._is_seqlora_sam_ewc_strategy())
        self.assertTrue(inclora._is_fishershape_ewclora_strategy())

        seqlora = _bare_trainer("seqlora_sam_ewc")
        self.assertFalse(seqlora._is_inclora_sam_ewc_strategy())
        self.assertTrue(seqlora._is_seqlora_sam_ewc_strategy())
        self.assertTrue(seqlora._is_fishershape_ewclora_strategy())

    def test_seqlora_anchor_has_zero_loss_then_penalizes_delta_w_change(self):
        model = _ToyLoRAModel()
        trainer = _bare_trainer()
        trainer.ewclora_reference = trainer._snapshot_seqlora_ewc_reference(model)
        trainer.ewclora_omega = {"layer.": torch.ones(2, 3)}

        anchored_loss = trainer._compute_ewclora_loss(model)
        self.assertEqual(float(anchored_loss.detach().item()), 0.0)
        anchored_grads = torch.autograd.grad(
            anchored_loss,
            [
                model.layer.lora_A["default"].weight,
                model.layer.lora_B["default"].weight,
            ],
        )
        for grad in anchored_grads:
            torch.testing.assert_close(grad, torch.zeros_like(grad))

        reference = trainer.ewclora_reference["layer."]
        with torch.no_grad():
            model.layer.lora_B["default"].weight.add_(0.1)

        actual_loss = trainer._compute_ewclora_loss(model)
        delta_w = (
            model.layer.lora_B["default"].weight
            @ model.layer.lora_A["default"].weight
        )
        delta_ref = reference["b"] @ reference["a"]
        expected_loss = 0.5 * trainer.args.ewclora_lambda * (
            delta_w - delta_ref
        ).pow(2).sum()

        self.assertGreater(float(actual_loss.detach().item()), 0.0)
        torch.testing.assert_close(actual_loss, expected_loss)

    def test_seqlora_shared_delta_w_hook_captures_full_matrix_gradient(self):
        model = _ToyLoRAModel()
        layer = model.layer
        layer.seqlora_ewc_register_hook = True
        inputs = torch.tensor(
            [[1.0, 2.0, -1.0], [0.5, -1.0, 3.0]],
            dtype=torch.float32,
        )

        layer(inputs).sum().backward()

        expected = (
            torch.ones(2, 1) @ inputs.sum(dim=0, keepdim=True)
        ) * layer.scaling["default"]
        self.assertIsNotNone(layer.seqlora_ewc_delta_grad)
        torch.testing.assert_close(layer.seqlora_ewc_delta_grad, expected)
        self.assertIsNone(layer.ewclora_delta_grad)

    def test_save_lora_only_omits_fresh_branch_without_concatenation(self):
        model = _ToyLoRAModel(save_lora_only=True)
        shared_a = model.layer.lora_A["default"].weight.detach().clone()
        shared_b = model.layer.lora_B["default"].weight.detach().clone()

        saved = get_peft_model_state_dict(model)

        self.assertEqual(
            set(saved),
            {"layer.lora_A.weight", "layer.lora_B.weight"},
        )
        self.assertFalse(any("loranew_" in name for name in saved))
        torch.testing.assert_close(saved["layer.lora_A.weight"], shared_a)
        torch.testing.assert_close(saved["layer.lora_B.weight"], shared_b)
        self.assertEqual(tuple(saved["layer.lora_A.weight"].shape), (2, 3))
        self.assertEqual(tuple(saved["layer.lora_B.weight"].shape), (2, 2))
        self.assertEqual(model.peft_config["default"].r_sum, 2)

    def test_seqlora_v3_sidecar_metadata_and_round_trip(self):
        model = _ToyLoRAModel()
        trainer = _bare_trainer()
        trainer.ewclora_omega = {
            "layer.": torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                dtype=torch.float32,
            )
        }
        trainer.ewclora_reference = trainer._snapshot_seqlora_ewc_reference(model)
        trainer.ewclora_count_updates = 3
        trainer.is_world_process_zero = lambda: True

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.save_ewclora_state(temp_dir)
            state_path = Path(temp_dir) / "ewclora_fisher.pt"
            payload = torch.load(state_path, map_location="cpu")

            self.assertEqual(payload["format"], "full_delta_w_fisher_v3")
            self.assertEqual(payload["fisher_space"], "full_delta_w")
            self.assertEqual(payload["estimator"], "batch_empirical")
            self.assertEqual(payload["strategy"], "seqlora_sam_ewc")
            self.assertEqual(payload["adapter_update_space"], "sequential_shared")
            self.assertEqual(
                payload["anchor_mode"],
                "task_start_effective_delta_w",
            )
            self.assertEqual(payload["reference_encoding"], "lora_factors_v1")
            self.assertEqual(payload["count_updates"], 3)
            self.assertAlmostEqual(payload["lambda"], 0.7)
            self.assertAlmostEqual(payload["gamma"], 0.8)
            self.assertEqual(set(payload["reference"]), {"layer."})

            loaded = _bare_trainer(ewclora_fisher_path=temp_dir)
            loaded._load_ewclora_state()

            self.assertEqual(loaded.ewclora_count_updates, 3)
            self.assertEqual(set(loaded.ewclora_omega), {"layer."})
            self.assertEqual(set(loaded.ewclora_reference), {"layer."})
            torch.testing.assert_close(
                loaded.ewclora_omega["layer."],
                trainer.ewclora_omega["layer."],
            )
            for factor_name in ("a", "b"):
                torch.testing.assert_close(
                    loaded.ewclora_reference["layer."][factor_name],
                    trainer.ewclora_reference["layer."][factor_name],
                )

    def test_public_strategies_reject_partial_or_misshaped_fisher(self):
        model = _ToyLoRAModel()

        seqlora = _bare_trainer("seqlora_sam_ewc")
        seqlora.model = model
        seqlora.ewclora_reference = seqlora._snapshot_seqlora_ewc_reference(
            model
        )
        seqlora.ewclora_omega = {"unrelated.": torch.ones(2, 3)}
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            seqlora._validate_sam_ewc_state_against_model()
        seqlora.ewclora_omega = {"layer.": torch.ones(1, 3)}
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            seqlora._validate_sam_ewc_state_against_model()
        seqlora.ewclora_omega = {"layer.": torch.ones(2, 3)}
        seqlora._validate_sam_ewc_state_against_model()

        inclora = _bare_trainer("inclora_sam_ewc")
        inclora.model = model
        inclora.ewclora_omega = {}
        inclora._validate_sam_ewc_state_against_model()
        inclora.ewclora_omega = {"layer.": torch.ones(2, 2)}
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            inclora._validate_sam_ewc_state_against_model()
        inclora.ewclora_omega = {"layer.": torch.ones(2, 3)}
        inclora._validate_sam_ewc_state_against_model()

    def test_explicit_fisher_path_must_exist_and_be_nonempty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.pt"
            missing = _bare_trainer(
                "inclora_sam_ewc",
                ewclora_fisher_path=str(missing_path),
            )
            with self.assertRaisesRegex(FileNotFoundError, "explicitly requested"):
                missing._load_ewclora_state()

            empty_path = Path(temp_dir) / "empty.pt"
            torch.save(
                {
                    "format": "full_delta_w_fisher_v2",
                    "fisher_space": "full_delta_w",
                    "estimator": "batch_empirical",
                    "omega": {},
                },
                empty_path,
            )
            empty = _bare_trainer(
                "inclora_sam_ewc",
                ewclora_fisher_path=str(empty_path),
            )
            with self.assertRaisesRegex(ValueError, "empty omega"):
                empty._load_ewclora_state()


if __name__ == "__main__":
    unittest.main()
