"""Numerical parity with the pinned upstream source, plus checkpoint integrity."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from treelora import TreeLoRARegularizer, UPSTREAM_COMMIT  # noqa: E402


def _load_oracle():
    path = ROOT / "references/treelora_upstream/utils/kd_lora_tree.py"
    provenance = json.loads((path.parents[1] / "UPSTREAM.json").read_text())
    assert provenance["commit"] == UPSTREAM_COMMIT
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["files"]["utils/kd_lora_tree.py"]
    spec = importlib.util.spec_from_file_location("_treelora_upstream_oracle", path)
    module = importlib.util.module_from_spec(spec)
    # Only remove logging infrastructure; run the original algorithm unchanged.
    stub = types.ModuleType("utils.utils")
    stub.print_rank_0 = lambda *args, **kwargs: None
    with patch.dict(sys.modules, {"utils.utils": stub}):
        spec.loader.exec_module(module)
    return module


def _assert_tree_equal(test, actual, expected):
    test.assertEqual(actual.task_indices, expected.task_indices)
    test.assertEqual(actual.depth, expected.depth)
    test.assertEqual(actual.is_leaf, expected.is_leaf)
    test.assertEqual(actual.median_similarity, expected.median_similarity)
    if actual.mean_vector is not None:
        torch.testing.assert_close(actual.mean_vector, expected.mean_vector, rtol=0, atol=0)
    if actual.left is not None:
        _assert_tree_equal(test, actual.left, expected.left)
        _assert_tree_equal(test, actual.right, expected.right)


def _assert_cpu_tensors(test, value):
    if torch.is_tensor(value):
        test.assertEqual(value.device.type, "cpu")
    elif isinstance(value, dict):
        for item in value.values():
            _assert_cpu_tensors(test, item)
    elif isinstance(value, list):
        for item in value:
            _assert_cpu_tensors(test, item)


class TreeLoRAParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle_module = _load_oracle()

    def test_four_tasks_match_upstream_loss_gradients_sampling_and_trajectory(self):
        # Two epochs exercise the upstream's last-epoch-only task summaries;
        # four tasks reach root branches that have their own split medians.
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                tree = TreeLoRARegularizer(reg=0.5, num_tasks=4)
                upstream = self.oracle_module.KD_LoRA_Tree(
                    types.SimpleNamespace(reg=0.5, num_tasks=4, global_rank=0)
                )
                data_rng = torch.Generator().manual_seed(1976)
                torch.manual_seed(619)
                for task_id in range(4):
                    for epoch in range(2):
                        tree.begin_epoch(7)
                        upstream.new_epoch_init(7)
                        for step in range(7):
                            values = torch.randn((3, 11), generator=data_rng, dtype=dtype)
                            values += 0.3 * task_id + epoch * 0.1
                            actual_a = values.clone().requires_grad_()
                            oracle_a = values.clone().requires_grad_()
                            actual_ce = 2 + actual_a.square().mean()
                            oracle_ce = 2 + oracle_a.square().mean()
                            before_rng = torch.random.get_rng_state()
                            upstream.step()
                            upstream.insert_grad(oracle_a)
                            expected = oracle_ce
                            selected = None
                            if task_id:
                                selected = upstream.tree_search(task_id, "cpu")
                                expected = oracle_ce - upstream.get_loss(
                                    oracle_a, oracle_ce, task_id, selected
                                )
                            expected_rng = torch.random.get_rng_state()
                            torch.random.set_rng_state(before_rng)
                            actual = tree.loss(actual_a, actual_ce)
                            torch.testing.assert_close(
                                torch.random.get_rng_state(), expected_rng, rtol=0, atol=0
                            )
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            torch.testing.assert_close(
                                torch.autograd.grad(actual, actual_a)[0],
                                torch.autograd.grad(expected, oracle_a)[0],
                                rtol=0, atol=0,
                            )
                            torch.testing.assert_close(
                                tree.current_grad, upstream.current_grad, rtol=0, atol=0
                            )
                            if selected is not None:
                                torch.testing.assert_close(tree.last_selection, selected, rtol=0, atol=0)
                                torch.testing.assert_close(tree.sim, upstream.sim, rtol=0, atol=0)
                                torch.testing.assert_close(
                                    tree.num_of_selected, upstream.num_of_selected, rtol=0, atol=0
                                )
                            # Mid-epoch serialization must preserve accumulated
                            # distances, counts, and ramp, not merely history.
                            if task_id == 2 and epoch == 1 and step == 3:
                                with tempfile.TemporaryDirectory() as folder:
                                    path = Path(folder) / "state.pt"
                                    state = tree.state_dict()
                                    _assert_cpu_tensors(self, state)
                                    torch.save(state, path)
                                    tree = TreeLoRARegularizer(reg=0.5, num_tasks=4)
                                    tree.load_state_dict(torch.load(path, weights_only=True))
                    with contextlib.redirect_stdout(io.StringIO()):
                        upstream.end_task(task_id)
                    tree.end_task()
                    self.assertEqual(tree.task_id, task_id + 1)
                    for index, history in enumerate(tree.history):
                        torch.testing.assert_close(
                            history, upstream.all_accumulate_grads[index], rtol=0, atol=0
                        )
                    _assert_tree_equal(self, tree.kd_tree_root, upstream.kd_tree_root)
                    # The actual experiment reloads between tasks.
                    restored = TreeLoRARegularizer(reg=0.5, num_tasks=4)
                    restored.load_state_dict(tree.state_dict())
                    tree = restored
                self.assertEqual(tree.diagnostics()["history_count"], 4)

    def test_preserves_layer_multiplier_and_rejects_duplicate_task_completion(self):
        tree = TreeLoRARegularizer()
        tree.begin_epoch(2)
        values = torch.ones((3, 5), requires_grad=True)
        tree.loss(values, values.sum())
        torch.testing.assert_close(tree.current_grad, torch.full((3, 5), 1.5))
        tree.loss(values, values.sum())
        tree.end_task()
        torch.testing.assert_close(tree.history[0], torch.full((3, 5), 3.0))
        with self.assertRaises(RuntimeError):
            tree.end_task()

    def test_checkpoint_is_independent_and_mismatches_fail(self):
        tree = TreeLoRARegularizer()
        tree.begin_epoch(1)
        values = torch.ones((3, 5), requires_grad=True)
        tree.loss(values, values.sum())
        tree.end_task()
        state = tree.state_dict()
        state["history"][0].zero_()
        self.assertGreater(float(tree.history[0].abs().sum()), 0)
        with self.assertRaises(ValueError):
            TreeLoRARegularizer(reg=0.2).load_state_dict(state)
        state["upstream_commit"] = "wrong-implementation"
        with self.assertRaises(ValueError):
            TreeLoRARegularizer().load_state_dict(state)

    def test_zero_regularization_is_identity(self):
        tree = TreeLoRARegularizer(reg=0)
        tree.begin_epoch(1)
        values = torch.ones((3, 5), requires_grad=True)
        ce = values.sum()
        self.assertIs(tree.loss(values, ce), ce)
        tree.end_task()
        self.assertEqual(tree.history, [])


class TreeLoRACheckpointTest(unittest.TestCase):
    def test_fixed_rank_peft_roundtrip_retains_A_B_and_logits(self):
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(21)
        config = LlamaConfig(
            vocab_size=43, hidden_size=24, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=32,
        )
        base = LlamaForCausalLM(config)
        original_base = copy.deepcopy(base.state_dict())
        model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"],
            r=2, r_sum=0, lora_alpha=4, lora_dropout=0.0, save_loranew=True,
        ))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "loranew_" in name:
                    parameter.copy_(torch.randn_like(parameter) * 0.2)
        model.eval()
        inputs = torch.tensor([[1, 4, 8, 2], [1, 3, 9, 6]])
        with torch.no_grad():
            expected_logits = model(input_ids=inputs).logits
        expected_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters() if "loranew_" in name
        }
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            first_bytes = (Path(folder) / "adapter_config.json").read_bytes()
            model.save_pretrained(folder)
            self.assertEqual(first_bytes, (Path(folder) / "adapter_config.json").read_bytes())
            saved_config = json.loads(first_bytes)
            self.assertEqual(saved_config["r_sum"], 0)
            self.assertTrue(saved_config["save_loranew"])
            reloaded_base = LlamaForCausalLM(config)
            reloaded_base.load_state_dict(original_base)
            reloaded = PeftModel.from_pretrained(reloaded_base, folder, is_trainable=True)
            reloaded.eval()
            with torch.no_grad():
                actual_logits = reloaded(input_ids=inputs).logits
            torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
            actual_parameters = dict(reloaded.named_parameters())
            self.assertTrue(expected_parameters)
            for name, expected in expected_parameters.items():
                torch.testing.assert_close(actual_parameters[name], expected, rtol=0, atol=0)
            for module in reloaded.modules():
                if hasattr(module, "lora_A") and "default" in module.lora_A:
                    self.assertEqual(module.lora_A["default"].weight.shape[0], 0)
                    self.assertEqual(module.loranew_A["default"].weight.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
