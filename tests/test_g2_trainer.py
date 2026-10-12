"""Effective-window, dropout, BF16 Llama and DDP contracts for Current-G2."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import nullcontext

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optimer_g2 import CurrentG2
from uie_trainer_lora import UIETrainer


class Quadratic(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.7, -0.4]))
        self.dropout = torch.nn.Dropout(dropout)
        self.observations = []

    def forward(self, features, targets):
        x = self.dropout(features)
        self.observations.append(x.detach().clone())
        return {"loss": ((x * self.weight - targets).square()).mean()}


def trainer_stub(model, output_dir, ga=4, strategy="inclora", inner="sgd_step", outer="sgd_step"):
    trainer = object.__new__(UIETrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(
        optimizer_type="g2-sgd", lora_strategy=strategy,
        gradient_accumulation_steps=ga, n_gpu=1, local_rank=-1,
        device=torch.device("cpu"), max_grad_norm=0.0, output_dir=output_dir,
        world_size=1, process_index=0,
        do_train=True, train_method="lora", rwp_type="", past_index=-1,
    )
    trainer.state = SimpleNamespace(global_step=0)
    trainer.current_flos = 0.0
    trainer.deepspeed = None
    trainer.do_grad_scaling = False
    trainer.use_apex = False
    trainer.ewclora_omega = {}
    trainer._prepare_inputs = lambda inputs: inputs
    trainer.compute_loss_context_manager = nullcontext
    trainer.compute_loss = lambda current_model, inputs: current_model(**inputs)["loss"]
    trainer.floating_point_ops = lambda inputs: 0
    base = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.03)
    trainer.optimizer = CurrentG2(
        [p for p in model.parameters() if p.requires_grad], base, model,
        inner_radius_mode=inner, outer_radius_mode=outer,
        inner_radius=0.05 if inner == "fixed_l2" else None,
        outer_radius=0.05 if outer == "fixed_l2" else None,
        closure_gradients_reduced=True,
    )
    return trainer


def clean_pass(trainer, model, batches):
    states = []
    for index, inputs in enumerate(batches):
        states.append(trainer._capture_g2_rng_state())
        context = model.no_sync() if hasattr(model, "no_sync") and index + 1 < len(batches) else nullcontext()
        with context:
            loss = trainer.compute_loss(model, inputs) / trainer.args.gradient_accumulation_steps
            loss.backward()
    return states


def closed_form(weight, batches, eta):
    x2 = torch.stack([x["features"].square() for x in batches]).mean(0)
    xy = torch.stack([x["features"] * x["targets"] for x in batches]).mean(0)
    # Each microbatch mean is over both coordinates.
    g0 = x2 * weight - xy
    g1 = x2 * (weight - eta * g0) - xy
    a = g0 - g1
    point = weight + eta * g0.norm() * a / a.norm()
    return x2 * point - xy


def ddp_worker(rank, rendezvous, output_root):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2,
    )
    try:
        model = torch.nn.parallel.DistributedDataParallel(Quadratic())
        trainer = trainer_stub(model, str(Path(output_root) / str(rank)), ga=2)
        trainer.args.world_size = 2
        trainer.args.process_index = rank
        local = [
            {"features": torch.tensor([1.0 + rank, 0.5 + i]),
             "targets": torch.tensor([0.2 + rank * 0.1, -0.3])}
            for i in range(2)
        ]
        old = model.module.weight.detach().clone()
        states = clean_pass(trainer, model, local)
        trainer._run_g2_optimizer_window(model, local, states)
        all_batches = [
            {"features": torch.tensor([1.0 + r, 0.5 + i]),
             "targets": torch.tensor([0.2 + r * 0.1, -0.3])}
            for r in range(2) for i in range(2)
        ]
        expected = old - 0.03 * closed_form(old, all_batches, 0.03)
        torch.testing.assert_close(model.module.weight, expected, rtol=1e-6, atol=1e-7)
        assert trainer.optimizer.last_stats["g2_ce_gradient_phases"] == 3
        assert trainer.optimizer.last_stats["g2_core_explicit_mean_reductions"] == 0
        gathered = [torch.zeros_like(old) for _ in range(2)]
        torch.distributed.all_gather(gathered, model.module.weight.detach())
        assert torch.equal(gathered[0], gathered[1])
    finally:
        torch.distributed.destroy_process_group()


class CurrentG2TrainerTest(unittest.TestCase):
    def test_family_and_strategy_are_independent_of_fisher_shape(self):
        for name in UIETrainer.G2_OPTIMIZERS:
            self.assertEqual(UIETrainer._optimizer_family(name), "g2")
        self.assertNotIn("inclora", UIETrainer.EWCLORA_STRATEGIES)
        self.assertEqual(UIETrainer._optimizer_family("gam-adamw_hf"), "gam")

    def test_full_and_short_windows_submit_the_same_correct_mean(self):
        for count in (4, 2):
            with self.subTest(micro_batches=count), tempfile.TemporaryDirectory() as output:
                model = Quadratic()
                trainer = trainer_stub(model, output)
                batches = [{"features": torch.tensor([1.0 + i * 0.1, 0.6 + i * 0.2]),
                            "targets": torch.tensor([0.2, -0.3])} for i in range(count)]
                old = model.weight.detach().clone()
                expected = old - 0.03 * closed_form(old, batches, 0.03)
                states = clean_pass(trainer, model, batches)
                trainer._run_g2_optimizer_window(model, batches, states)
                torch.testing.assert_close(model.weight, expected, rtol=1e-6, atol=1e-7)
                record = json.loads((Path(output) / "g2_step_trace.jsonl").read_text())
                self.assertEqual(record["actual_micro_batches"], count)
                self.assertEqual(record["tail_gradient_rescale"], 4 / count)
                self.assertEqual(record["g2_ce_gradient_phases"], 3)
                self.assertEqual(record["g2_base_optimizer_steps"], 1)
                self.assertFalse(record["g2_regularization_hook_used"])
                self.assertIsNotNone(record["g2_cos_g0_g2"])
                self.assertGreater(record["g2_g2_minus_g0_relative_norm"], 0.0)
                self.assertAlmostEqual(record["g2_gradient_clip_scale"], 1.0)
                self.assertFalse(record["g2_quantization_proxy"]["sampled"])
                self.assertEqual(len(model.observations), 3 * count)
                self.assertFalse(batches)
                self.assertFalse(states)

    def test_dropout_masks_replay_without_advancing_post_clean_rng(self):
        torch.manual_seed(37)
        with tempfile.TemporaryDirectory() as output:
            model = Quadratic(dropout=0.25)
            trainer = trainer_stub(model, output, inner="fixed_l2", outer="fixed_l2")
            batches = [{"features": torch.tensor([1.0 + i, 2.0]),
                        "targets": torch.tensor([0.2, -0.3])} for i in range(4)]
            states = clean_pass(trainer, model, batches)
            post_clean = torch.get_rng_state().clone()
            trainer._run_g2_optimizer_window(model, batches, states)
            self.assertTrue(torch.equal(torch.get_rng_state(), post_clean))
            self.assertEqual(len(model.observations), 12)
            for index in range(4):
                self.assertTrue(torch.equal(model.observations[index], model.observations[4 + index]))
                self.assertTrue(torch.equal(model.observations[index], model.observations[8 + index]))

    def test_invalid_pure_configuration_fails_closed(self):
        with tempfile.TemporaryDirectory() as output:
            trainer = trainer_stub(Quadratic(), output)
            trainer.args.lora_strategy = "ewclora"
            with self.assertRaisesRegex(ValueError, "Pure G2"):
                trainer._validate_g2_training_configuration()
            trainer.args.lora_strategy = "inclora"
            trainer.args.g2_replay_rng_state = False
            with self.assertRaisesRegex(ValueError, "RNG"):
                trainer._validate_g2_training_configuration()

    def test_two_adamw_backends_preserve_explicit_hyperparameters(self):
        for backend, module in (("g2-adamw_hf", "transformers.optimization"),
                                ("g2-adamw_torch", "torch.optim.adamw")):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as output:
                trainer = trainer_stub(Quadratic(), output)
                trainer.optimizer = None
                trainer.lr_scheduler = None
                trainer.args.optimizer_type = backend
                trainer.args.learning_rate = 0.003
                trainer.args.adam_beta1 = 0.7
                trainer.args.adam_beta2 = 0.95
                trainer.args.adam_epsilon = 1e-7
                trainer.args.weight_decay = 0.013
                trainer.args.g2_inner_radius_mode = "sgd_step"
                trainer.args.g2_outer_radius_mode = "sgd_step"
                trainer.args.g2_inner_radius = None
                trainer.args.g2_outer_radius = None
                trainer.args.g2_quantization_diagnostics_interval = 3
                trainer.create_optimizer_and_scheduler(num_training_steps=2)
                base = trainer.optimizer.base_optimizer
                self.assertEqual(type(base).__module__, module)
                self.assertEqual(base.param_groups[0]["betas"], (0.7, 0.95))
                self.assertEqual(base.param_groups[0]["eps"], 1e-7)
                self.assertEqual(base.param_groups[0]["weight_decay"], 0.013)
                self.assertEqual(base.param_groups[0]["lr"], 0.003)
                self.assertEqual(trainer.optimizer.quantization_diagnostics_interval, 3)

    def test_zero_gradient_fallback_logs_no_probe_loss(self):
        with tempfile.TemporaryDirectory() as output:
            model = Quadratic()
            trainer = trainer_stub(model, output)
            old = model.weight.detach().clone()
            batches = [{"features": torch.ones(2), "targets": old.clone()} for _ in range(4)]
            states = clean_pass(trainer, model, batches)
            trainer._run_g2_optimizer_window(model, batches, states)
            record = json.loads((Path(output) / "g2_step_trace.jsonl").read_text())
            self.assertIsNone(record["probe_loss"])
            self.assertEqual(record["g2_ce_gradient_phases"], 1)
            self.assertIsNotNone(record["g2_fallback_reason"])
            self.assertEqual(record["g2_base_optimizer_steps"], 1)
            self.assertTrue(torch.equal(model.weight, old))
            self.assertIsNone(record["g2_cos_g0_g2"])
            self.assertIsNone(record["g2_g2_minus_g0_relative_norm"])
            self.assertIsNone(record["g2_gradient_clip_scale"])

    def test_tiny_random_llama_bf16_forward_fp32_lora_full_and_tail(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        torch.set_num_threads(1)
        torch.manual_seed(11)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
            use_cache=False,
        )).to(torch.bfloat16)
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        ))
        for name, param in model.named_parameters():
            param.requires_grad_("loranew_" in name)
            if param.requires_grad:
                param.data = param.data.float()
        frozen = {name: param.detach().clone() for name, param in model.named_parameters()
                  if not param.requires_grad}
        with tempfile.TemporaryDirectory() as output:
            trainer = trainer_stub(model, output, inner="fixed_l2", outer="fixed_l2")
            trainer.optimizer.quantization_diagnostics_interval = 1
            trainer.compute_loss_context_manager = lambda: torch.autocast("cpu", dtype=torch.bfloat16)
            for count in (4, 2):
                batches = []
                for index in range(count):
                    tokens = torch.tensor([[1, 2 + index, 4, 6, 3]])
                    batches.append({"input_ids": tokens, "labels": tokens.clone(),
                                    "attention_mask": torch.ones_like(tokens)})
                states = clean_pass(trainer, model, batches)
                post_clean = torch.get_rng_state().clone()
                trainer._run_g2_optimizer_window(model, batches, states)
                self.assertEqual(trainer.optimizer.last_stats["g2_ce_gradient_phases"], 3)
                self.assertEqual(trainer.optimizer.last_stats["g2_base_optimizer_steps"], 1)
                self.assertTrue(trainer.optimizer.last_stats["g2_restore_base_exact"])
                self.assertTrue(torch.equal(post_clean, torch.get_rng_state()))
                self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad))
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                stats = trainer.optimizer.last_stats
                self.assertIsNotNone(stats["g2_cos_g0_g2"])
                self.assertTrue(math.isfinite(stats["g2_g2_minus_g0_relative_norm"]))
                self.assertTrue(stats["g2_quantization_proxy"]["sampled"])
                self.assertFalse(stats["g2_quantization_proxy"]["proves_forward_visibility"])
                for phase in ("inner", "outer"):
                    proxy = stats["g2_quantization_proxy"][phase]
                    self.assertEqual(proxy["source_parameter_dtypes"], ["torch.float32"])
                    self.assertEqual(proxy["status"], "finite_cast")
                    self.assertGreater(proxy["changed_coordinates"], 0)
                    self.assertGreater(proxy["represented_displacement_norm"], 0.0)
                json.dumps(stats, allow_nan=False)
                trainer.state.global_step += 1
            for name, param in model.named_parameters():
                if name in frozen:
                    self.assertTrue(torch.equal(param, frozen[name]))

    def test_two_rank_global_mean_matches_closed_form(self):
        with tempfile.TemporaryDirectory() as output:
            rendezvous = str(Path(output) / "gloo-store")
            torch.multiprocessing.spawn(ddp_worker, args=(rendezvous, output), nprocs=2, join=True)

    def test_native_training_loop_flushes_bf16_tiny_llama_tail(self):
        from transformers import LlamaConfig, LlamaForCausalLM, default_data_collator
        from peft import LoraConfig, get_peft_model, TaskType
        from run_uie_lora import UIETrainingArguments
        torch.set_num_threads(1)
        torch.manual_seed(12)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
            use_cache=False,
        )).to(torch.bfloat16)
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        ))
        for name, param in model.named_parameters():
            param.requires_grad_("loranew_" in name)
            if param.requires_grad:
                param.data = param.data.float()
        examples = []
        for index in range(6):
            tokens = torch.tensor([1, 2 + index, 4, 6, 3])
            examples.append({"input_ids": tokens, "labels": tokens.clone(),
                             "attention_mask": torch.ones_like(tokens)})
        with tempfile.TemporaryDirectory() as output:
            args = UIETrainingArguments(
                output_dir=output, do_train=True, train_method="lora", lora_strategy="inclora",
                optimizer_type="g2-sgd", learning_rate=0.03, weight_decay=0.0,
                per_device_train_batch_size=1, gradient_accumulation_steps=4,
                num_train_epochs=1, no_cuda=True, bf16=True,
                max_grad_norm=0.0, report_to=[], save_strategy="no",
                logging_strategy="no", disable_tqdm=True, remove_unused_columns=False,
                g2_inner_radius_mode="fixed_l2", g2_outer_radius_mode="fixed_l2",
                g2_inner_radius=0.05, g2_outer_radius=0.05,
                olora_lamda_1=0.0, olora_lamda_2=0.0,
            )
            trainer = UIETrainer(model=model, args=args, train_dataset=examples,
                                 data_collator=default_data_collator)
            result = trainer.train()
            self.assertEqual(result.global_step, 2)
            rows = [json.loads(line) for line in
                    (Path(output) / "g2_step_trace.jsonl").read_text().splitlines()]
            self.assertEqual([row["actual_micro_batches"] for row in rows], [4, 2])
            self.assertTrue(all(row["g2_ce_gradient_phases"] == 3 for row in rows))
            self.assertTrue(all(row["g2_restore_base_exact"] for row in rows))
            self.assertFalse(any(row["g2_regularization_hook_used"] for row in rows))
            audit = trainer.finish_g2_training_audit()
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["normal_three_read_steps"], 2)
            self.assertEqual(audit["tail_windows"], 1)
            self.assertEqual(audit["fisher_state_entries"], 0)
            self.assertTrue((Path(output) / "g2_runtime_audit.json").is_file())

    def test_native_bf16_llama_adamw_dynamic_radius_and_clipping(self):
        """Exercise the production geometry and both production AdamW commits."""
        from transformers import LlamaConfig, LlamaForCausalLM, default_data_collator
        from peft import LoraConfig, get_peft_model, TaskType
        from run_uie_lora import UIETrainingArguments
        torch.set_num_threads(1)
        variants = (
            ("g2-adamw_hf", "transformers.optimization", 1e-6, 0.0),
            ("g2-adamw_torch", "torch.optim.adamw", 1e-8, 0.01),
        )
        for backend, module, epsilon, weight_decay in variants:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as output:
                torch.manual_seed(0)
                model = LlamaForCausalLM(LlamaConfig(
                    vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2,
                    max_position_embeddings=32, use_cache=False,
                )).to(torch.bfloat16)
                model = get_peft_model(model, LoraConfig(
                    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32,
                    lora_dropout=0.1, target_modules=["q_proj", "v_proj"],
                ))
                for name, param in model.named_parameters():
                    param.requires_grad_("loranew_" in name)
                    if param.requires_grad:
                        param.data = param.data.float()
                frozen = {name: param.detach().clone() for name, param in model.named_parameters()
                          if not param.requires_grad}
                examples = []
                for index in range(6):
                    tokens = torch.tensor([1, 2 + index, 4, 6, 3])
                    examples.append({"input_ids": tokens, "labels": tokens.clone(),
                                     "attention_mask": torch.ones_like(tokens)})
                args = UIETrainingArguments(
                    output_dir=output, do_train=True, train_method="lora",
                    lora_strategy="inclora", optimizer_type=backend,
                    learning_rate=0.001, adam_beta1=0.9, adam_beta2=0.999,
                    adam_epsilon=epsilon, weight_decay=weight_decay,
                    per_device_train_batch_size=1, gradient_accumulation_steps=4,
                    num_train_epochs=1, no_cuda=True, bf16=True,
                    max_grad_norm=1.0, report_to=[], save_strategy="no",
                    logging_strategy="no", disable_tqdm=True,
                    remove_unused_columns=False, seed=0, data_seed=0,
                    g2_inner_radius_mode="sgd_step", g2_outer_radius_mode="sgd_step",
                    g2_inner_radius=None, g2_outer_radius=None,
                    olora_lamda_1=0.0, olora_lamda_2=0.0,
                )
                trainer = UIETrainer(model=model, args=args, train_dataset=examples,
                                     data_collator=default_data_collator)
                result = trainer.train()
                audit = trainer.finish_g2_training_audit()
                self.assertEqual(result.global_step, 2)
                self.assertEqual(type(trainer.optimizer.base_optimizer).__module__, module)
                group = trainer.optimizer.base_optimizer.param_groups[0]
                self.assertEqual(group["betas"], (0.9, 0.999))
                self.assertEqual(group["eps"], epsilon)
                self.assertEqual(group["weight_decay"], weight_decay)
                self.assertEqual(group["lr"], 0.001)
                rows = [json.loads(line) for line in
                        (Path(output) / "g2_step_trace.jsonl").read_text().splitlines()]
                self.assertEqual([row["actual_micro_batches"] for row in rows], [4, 2])
                self.assertEqual(audit["global_step"], 2)
                self.assertEqual(audit["tail_windows"], 1)
                self.assertEqual(audit["fisher_state_entries"], 0)
                self.assertFalse(audit["regularization_enabled"])
                self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad))
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                fallback_count = 0
                for row in rows:
                    self.assertTrue(row["g2_restore_base_exact"])
                    self.assertEqual(row["g2_base_optimizer_steps"], 1)
                    self.assertFalse(row["g2_regularization_hook_used"])
                    self.assertFalse(row["clean_regularization_enabled"])
                    self.assertEqual(row["g2_fisher_probe_calls"], 0)
                    self.assertEqual(row["g2_projection_calls"], 0)
                    self.assertEqual(row["g2_hvp_calls"], 0)
                    self.assertTrue(row["g2_final_clip_enabled"])
                    self.assertLessEqual(row["g2_submitted_postclip_norm"], 1.0 + 1e-6)
                    self.assertGreater(row["g2_actual_update_norm"], 0.0)
                    for key, value in row.items():
                        if isinstance(value, float):
                            self.assertTrue(math.isfinite(value), msg=key)
                    if row["g2_fallback_reason"] is None:
                        self.assertEqual(row["g2_ce_gradient_phases"], 3)
                        self.assertEqual(row["g2_final_direction"], "g2")
                    else:
                        fallback_count += 1
                        self.assertIn(row["g2_ce_gradient_phases"], (1, 2))
                        self.assertEqual(row["g2_final_direction"], "g0_fallback")
                self.assertEqual(audit["fallback_steps"], fallback_count)
                self.assertEqual(audit["normal_three_read_steps"], 2 - fallback_count)
                for name, param in model.named_parameters():
                    if name in frozen:
                        self.assertTrue(torch.equal(param, frozen[name]))
                print("G2 production-shape CPU receipt: " + json.dumps({
                    "optimizer": backend, "actual_module": module,
                    "global_step": audit["global_step"],
                    "actual_micro_batch_counts_in_order": [row["actual_micro_batches"] for row in rows],
                    "normal_three_read_steps": audit["normal_three_read_steps"],
                    "fallback_steps": fallback_count, "fallback_rate": fallback_count / 2,
                    "fallback_counts": audit["fallback_counts"],
                    "fp32_trainable": True, "bf16_forward": True,
                    "max_grad_norm": 1.0, "fisher_state_entries": 0,
                    "regularization_enabled": False,
                }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
