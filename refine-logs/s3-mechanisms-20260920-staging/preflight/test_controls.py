"""CPU preflight for opt-in S3 boundary instrumentation (no model downloads)."""

import copy
import importlib.util
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "s3_mechanism_controls", ROOT / "source/utils/s3_mechanism_controls.py"
)
CONTROLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLS)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor_a = nn.Linear(2, 2, bias=False)
        self.factor_b = nn.Linear(2, 2, bias=False)
        self.fc = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.factor_a.weight.copy_(torch.tensor([[1., 2.], [3., 4.]]))
            self.factor_b.weight.copy_(torch.tensor([[2., 0.], [1., 3.]]))
            self.fc.weight.fill_(7.)
        self.train()
        self.fc.eval()  # A heterogeneous mode state must be restored exactly.


class FakeLearner:
    def __init__(self, args=None):
        self.args = dict(args or {})
        self.model = TinyModel()
        self._cur_task = 0
        self._known_classes = 0
        self._total_classes = 2
        self.calls = {"apply": 0, "fisher": 0, "weights": 0}
        self._in_task_snapshot_metrics = [{"snapshot_train_ce": 2.}, {"snapshot_train_ce": 1.}]
        factors = self._snapshot_lora_factor_weights(self.model)
        self._in_task_weight_snapshots = [
            (20, {key: val - 2 for key, val in factors.items()}),
            (40, {key: val - 1 for key, val in factors.items()}),
        ]
        self._in_task_path_weight_last_metrics = {}
        self._in_task_weight_merge_last_metrics = {}
        self._in_task_fisher_last_metrics = {}
        self._weight_merge_then_fisher_last_metrics = {}
        self._last_mechanism_metrics = {}
        self.raise_fisher = False

    @staticmethod
    def _capture_global_rng_state():
        return {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": None,
        }

    @staticmethod
    def _iter_lora_factor_weights(model):
        return [("factor_a.weight", model.factor_a.weight), ("factor_b.weight", model.factor_b.weight)]

    def _snapshot_lora_factor_weights(self, model):
        return {key: value.detach().cpu().clone() for key, value in self._iter_lora_factor_weights(model)}

    def _path_checkpoint_weights(self, selected):
        self.calls["weights"] += 1
        self._in_task_path_weight_last_metrics = {"in_task_path_weight_mode": "fake_weighted"}
        return [.25, .75]

    def _apply_in_task_weight_merge(self, model):
        self.calls["apply"] += 1
        selected = [(i, epoch, factors, self._in_task_snapshot_metrics[i])
                    for i, (epoch, factors) in enumerate(self._in_task_weight_snapshots)]
        weights = self._path_checkpoint_weights(selected)
        with torch.no_grad():
            for key, parameter in self._iter_lora_factor_weights(model):
                mean = sum(w * entry[2][key] for w, entry in zip(weights, selected))
                parameter.copy_(.7 * parameter + .3 * mean)
        self._in_task_weight_merge_last_metrics = {"in_task_weight_merge_applied": True}
        return dict(self._in_task_weight_merge_last_metrics)

    def _compute_delta_fisher(self, loader, model, class_offset):
        self.calls["fisher"] += 1
        # The existing call consumes RNG. Instrumentation must preserve its
        # exact consumption, rather than restoring or adding random draws.
        random.random()
        np.random.random()
        random_scale = torch.rand(())
        model.eval()
        if self.raise_fisher:
            raise RuntimeError("intentional Fisher failure")
        result = {"delta": (model.factor_b.weight @ model.factor_a.weight).detach().square() + random_scale}
        self._in_task_fisher_last_metrics = {
            "current_fisher_source": "post_weight_merge",
            "weight_merge_then_fisher_current_fisher_after_weight_merge": True,
        }
        self._weight_merge_then_fisher_last_metrics.update(self._in_task_fisher_last_metrics)
        return result

    def after_task(self):
        self._last_mechanism_metrics.update(self._weight_merge_then_fisher_last_metrics)
        self._last_mechanism_metrics["posterior_center_smoothing_before_fisher"] = True


class PatchedLearner(FakeLearner):
    pass


CONTROLS.install(PatchedLearner)


def seed_all():
    random.seed(12)
    np.random.seed(13)
    torch.manual_seed(14)


def clone_parameters(model):
    return {key: value.detach().clone() for key, value in model.named_parameters()}


class ControlsTest(unittest.TestCase):
    def setUp(self):
        seed_all()

    def assert_parameters_equal(self, model, before):
        self.assertEqual(set(dict(model.named_parameters())), set(before))
        for key, value in model.named_parameters():
            self.assertTrue(torch.equal(value, before[key]), key)

    def run_fisher(self, cls, args):
        seed_all()
        learner = cls(args)
        learner._apply_in_task_weight_merge(learner.model)
        fisher = learner._compute_delta_fisher(None, learner.model, 0)
        return learner, fisher, CONTROLS._rng_digest(learner)

    def test_default_numerical_noop_and_rng(self):
        native, expected, rng = self.run_fisher(FakeLearner, {})
        patched, actual, actual_rng = self.run_fisher(PatchedLearner, {})
        self.assert_parameters_equal(patched.model, clone_parameters(native.model))
        self.assertTrue(torch.equal(actual["delta"], expected["delta"]))
        self.assertEqual(actual_rng, rng)
        self.assertEqual(patched.calls, native.calls)
        self.assertFalse(hasattr(patched, "_s3_endpoint_factors"))
        self.assertEqual(patched._in_task_fisher_last_metrics, native._in_task_fisher_last_metrics)

    def test_capture_is_numeric_and_rng_noop_with_actual_metadata(self):
        native, expected, rng = self.run_fisher(FakeLearner, {})
        with tempfile.TemporaryDirectory() as directory:
            patched, actual, actual_rng = self.run_fisher(
                PatchedLearner, {"s3_boundary_capture": True, "outputs_root": directory}
            )
            self.assertTrue(torch.equal(actual["delta"], expected["delta"]))
            self.assert_parameters_equal(patched.model, clone_parameters(native.model))
            self.assertEqual(actual_rng, rng)
            self.assertEqual(patched.calls, native.calls)
            payload = torch.load(Path(directory) / "mechanism_states/task_00.pt", weights_only=False)
            self.assertEqual(payload["factor_space"], "lora_a_b")
            self.assertEqual([item["epoch"] for item in payload["snapshots"]], [20, 40])
            self.assertEqual([item["weight"] for item in payload["selected_path_weights"]], [.25, .75])
            self.assertEqual(payload["snapshots"][1]["metrics"]["snapshot_train_ce"], 1.)
            self.assertEqual(payload["rng_before_boundary_sha256"], payload["rng_after_boundary_sha256"])
            self.assertEqual(payload["capture_extra_fisher_count"], 0)
            self.assertNotIn("fc.weight", payload["endpoint_factors"])
            self.assertFalse(torch.equal(payload["endpoint_factors"]["factor_a.weight"], payload["retained_factors"]["factor_a.weight"]))

    def test_endpoint_uses_exactly_one_pass_and_restores_parameters_and_modes(self):
        learner = PatchedLearner({"s3_fisher_location": "endpoint"})
        endpoint = learner._snapshot_lora_factor_weights(learner.model)
        learner._apply_in_task_weight_merge(learner.model)
        retained = clone_parameters(learner.model)
        modes = [module.training for module in learner.model.modules()]
        before_rng = learner._capture_global_rng_state()
        expected_noise = torch.rand(())
        torch.set_rng_state(before_rng["torch_cpu"])
        actual = learner._compute_delta_fisher(None, learner.model, 0)
        expected = (endpoint["factor_b.weight"] @ endpoint["factor_a.weight"]).square() + expected_noise
        self.assertTrue(torch.equal(actual["delta"], expected))
        self.assert_parameters_equal(learner.model, retained)
        self.assertEqual([module.training for module in learner.model.modules()], modes)
        self.assertEqual(learner.calls["fisher"], 1)
        learner.after_task()
        self.assertEqual(learner._last_mechanism_metrics["current_fisher_source"], "pre_weight_merge_endpoint")
        self.assertFalse(learner._last_mechanism_metrics["posterior_center_smoothing_before_fisher"])
        self.assertFalse(learner._in_task_fisher_last_metrics["weight_merge_then_fisher_current_fisher_after_weight_merge"])

    def test_endpoint_rng_consumption_matches_original_one_pass(self):
        _, _, expected_rng = self.run_fisher(FakeLearner, {})
        _, _, actual_rng = self.run_fisher(PatchedLearner, {"s3_fisher_location": "endpoint"})
        self.assertEqual(actual_rng, expected_rng)

    def test_endpoint_restores_on_exception(self):
        learner = PatchedLearner({"s3_fisher_location": "endpoint"})
        learner._apply_in_task_weight_merge(learner.model)
        retained = clone_parameters(learner.model)
        modes = [module.training for module in learner.model.modules()]
        learner.raise_fisher = True
        with self.assertRaisesRegex(RuntimeError, "intentional Fisher failure"):
            learner._compute_delta_fisher(None, learner.model, 0)
        self.assert_parameters_equal(learner.model, retained)
        self.assertEqual([module.training for module in learner.model.modules()], modes)
        self.assertEqual(learner.calls["fisher"], 1)

    def test_missing_or_stale_endpoint_fails_before_estimator(self):
        learner = PatchedLearner({"s3_fisher_location": "endpoint"})
        with self.assertRaisesRegex(RuntimeError, "Missing current-task"):
            learner._compute_delta_fisher(None, learner.model, 0)
        learner._apply_in_task_weight_merge(learner.model)
        learner._cur_task = 1
        with self.assertRaisesRegex(RuntimeError, "Missing current-task"):
            learner._compute_delta_fisher(None, learner.model, 0)
        self.assertEqual(learner.calls["fisher"], 0)

    def test_bad_factor_keys_fail_without_mutation(self):
        learner = PatchedLearner({"s3_fisher_location": "endpoint"})
        learner._apply_in_task_weight_merge(learner.model)
        retained = clone_parameters(learner.model)
        learner._s3_endpoint_factors.pop("factor_a.weight")
        with self.assertRaisesRegex(RuntimeError, "factor set"):
            learner._compute_delta_fisher(None, learner.model, 0)
        self.assert_parameters_equal(learner.model, retained)
        self.assertEqual(learner.calls["fisher"], 0)

    def test_exclusive_capture_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = PatchedLearner({"s3_boundary_capture": True, "outputs_root": directory})
            learner._apply_in_task_weight_merge(learner.model)
            path = Path(directory) / "mechanism_states/task_00.pt"
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                learner._apply_in_task_weight_merge(learner.model)
            self.assertEqual(path.read_bytes(), original)

    def test_reject_ambiguous_diagnostics_or_second_merge(self):
        for flag in ("posterior_state_save", "mechanism_anchor_eval", "mechanism_future_surrogate_eval", "task_endpoint_weight_merge", "ewc_skip_task_fisher"):
            with self.subTest(flag=flag):
                learner = PatchedLearner({"s3_fisher_location": "endpoint", flag: True})
                with self.assertRaisesRegex(ValueError, "do not support"):
                    learner._apply_in_task_weight_merge(learner.model)
                self.assertEqual(learner.calls["apply"], 0)

    def test_invalid_location_and_install_idempotence(self):
        learner = PatchedLearner({"s3_fisher_location": "typo"})
        with self.assertRaisesRegex(ValueError, "Invalid s3_fisher_location"):
            learner._apply_in_task_weight_merge(learner.model)
        current = PatchedLearner._compute_delta_fisher
        self.assertIs(CONTROLS.install(PatchedLearner)._compute_delta_fisher, current)


if __name__ == "__main__":
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight must not initialize CUDA")
    unittest.main(verbosity=2)
