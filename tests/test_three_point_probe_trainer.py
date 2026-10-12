"""Native three-read NLP controls: forward, reflected, central and BASE AdamW."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import AdamW

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from optimer_three_point_probe import ThreePointProbe
from run_uie_lora import UIETrainingArguments
from uie_trainer_lora import UIETrainer
from test_g2_trainer import Quadratic, clean_pass, trainer_stub


def probe_stub(model, output, mode="central", ga=4):
    trainer = trainer_stub(model, output, ga=ga)
    trainer.args.optimizer_type = "threepoint-adamw_hf"
    trainer.args.probe_mode = mode
    trainer.args.probe_rho = .05
    trainer.args.probe_eps = 1e-12
    trainer.args.probe_quantization_diagnostics_interval = 1
    trainer.args.adam_epsilon = 1e-6
    trainer.args.weight_decay = 0.0
    trainer.args.bf16 = False
    trainer.data_collator = SimpleNamespace(decoder_pipeline_version="decoder_token_ids_v2")
    trainer.is_world_process_zero = lambda: trainer.args.process_index == 0
    params = [p for p in model.parameters() if p.requires_grad]
    base = AdamW(params, lr=.001, eps=1e-6, weight_decay=0, no_deprecation_warning=True)
    trainer.optimizer = ThreePointProbe(params, base, model, mode=mode, radius=.05,
        closure_gradients_reduced=True, quantization_diagnostics_interval=1)
    return trainer


def batch_rows(count):
    return [{"features": torch.tensor([1.0 + .1 * index, .6 + .2 * index]),
             "targets": torch.tensor([.2, -.3])} for index in range(count)]


def expected_direction(weight, batches, mode, rho=.05):
    x2 = torch.stack([batch["features"].square() for batch in batches]).mean(0)
    xy = torch.stack([batch["features"] * batch["targets"] for batch in batches]).mean(0)
    g0 = x2 * weight - xy
    direction = rho * g0 / (g0.norm() + 1e-12)
    gplus = x2 * (weight + direction) - xy
    gminus = x2 * (weight - direction) - xy
    return {"g1_plus": gplus, "backward_reflected": 2 * g0 - gminus,
            "central": g0 + .5 * (gplus - gminus), "g0": g0}[mode]


def tiny_llama_native_setup(output, mode, max_steps=-1):
    from transformers import AutoTokenizer, LlamaConfig
    from peft import LoraConfig, TaskType, get_peft_model
    from model.llama import LlamaForCausalLM_with_lossmask
    from uie_collator import DataCollatorForUIE
    torch.set_num_threads(1)
    torch.manual_seed(223)
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/140-1/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = LlamaConfig(vocab_size=128256, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, use_cache=False)
    config._name_or_path = "tiny/llama"
    model = get_peft_model(LlamaForCausalLM_with_lossmask(config), LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=.1,
        target_modules=["q_proj", "v_proj"]))
    for name, param in model.named_parameters():
        param.requires_grad_("loranew_" in name)
    collator = DataCollatorForUIE(tokenizer, model, max_source_length=32, max_target_length=8)
    labels = ["Athlete", "Educational Institution", "Natural Place", "Building", "Athlete", "Building"]
    rows = [{"Task": "TC", "Dataset": "dbpedia", "Samples": [], "subset": "train",
             "Instance": {"label": label, "sentence": "example", "instruction": "{0}\nAnswer:"}}
            for label in labels]
    args = UIETrainingArguments(
        output_dir=output, do_train=True, train_method="lora", lora_strategy="inclora",
        optimizer_type="threepoint-adamw_hf", probe_mode=mode, probe_rho=.05,
        probe_quantization_diagnostics_interval=1,
        learning_rate=.001, adam_epsilon=1e-6, weight_decay=0,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=1, max_steps=max_steps, no_cuda=True, bf16=True,
        max_grad_norm=1, report_to=[], save_strategy="no", logging_strategy="no",
        disable_tqdm=True, remove_unused_columns=False, seed=0, data_seed=0,
        olora_lamda_1=0, olora_lamda_2=0)
    return model, UIETrainer(model=model, args=args, train_dataset=rows, data_collator=collator)


def ddp_probe_worker(rank, rendezvous, output_root):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method="file://" + rendezvous,
                                         rank=rank, world_size=2)
    try:
        for mode in ("g1_plus", "backward_reflected", "central", "g0"):
            model = torch.nn.parallel.DistributedDataParallel(Quadratic())
            trainer = probe_stub(model, str(Path(output_root) / f"{mode}_{rank}"), mode, ga=2)
            trainer.args.world_size = 2
            trainer.args.process_index = rank
            local = [{"features": torch.tensor([1.0 + rank, .5 + i]),
                      "targets": torch.tensor([.2 + rank * .1, -.3])} for i in range(2)]
            global_batches = [{"features": torch.tensor([1.0 + r, .5 + i]),
                               "targets": torch.tensor([.2 + r * .1, -.3])}
                              for r in range(2) for i in range(2)]
            reference = torch.nn.Parameter(model.module.weight.detach().clone())
            expected = AdamW([reference], lr=.001, eps=1e-6, weight_decay=0,
                             no_deprecation_warning=True)
            reference.grad = expected_direction(reference.detach(), global_batches, mode)
            expected.step()
            states = clean_pass(trainer, model, local)
            trainer._run_three_point_probe_optimizer_window(model, local, states)
            torch.testing.assert_close(model.module.weight, reference, rtol=0, atol=0)
            assert trainer.optimizer.last_stats["probe_core_explicit_mean_reductions"] == 0
            assert trainer.optimizer.last_stats["probe_ce_gradient_phases"] == 3
            gathered = [torch.zeros_like(reference) for _ in range(2)]
            torch.distributed.all_gather(gathered, model.module.weight.detach())
            assert torch.equal(gathered[0], gathered[1])
    finally:
        torch.distributed.destroy_process_group()


class ThreePointProbeTrainerTest(unittest.TestCase):
    def test_new_family_keeps_legacy_sam_gam_and_g2_separate(self):
        for name in UIETrainer.THREE_POINT_PROBE_OPTIMIZERS:
            self.assertEqual(UIETrainer._optimizer_family(name), "three_point_probe")
        for name, family in [("sam-adamw_hf", "sam"), ("gam-adamw_hf", "gam"),
                             ("g2-adamw_hf", "g2"), ("probe-adamw_hf", "two_point_probe"),
                             ("adamw_hf", "standard")]:
            self.assertEqual(UIETrainer._optimizer_family(name), family)

    def test_full_and_tail_means_match_literal_direction_and_one_hf_adamw(self):
        for mode in ("g1_plus", "backward_reflected", "central", "g0"):
            for count in (4, 2):
                with self.subTest(mode=mode, count=count), tempfile.TemporaryDirectory() as output:
                    model = Quadratic()
                    trainer = probe_stub(model, output, mode)
                    batches = batch_rows(count)
                    reference = torch.nn.Parameter(model.weight.detach().clone())
                    expected = AdamW([reference], lr=.001, eps=1e-6, weight_decay=0,
                                     no_deprecation_warning=True)
                    reference.grad = expected_direction(reference.detach(), batches, mode)
                    expected.step()
                    states = clean_pass(trainer, model, batches)
                    trainer._run_three_point_probe_optimizer_window(model, batches, states)
                    torch.testing.assert_close(model.weight, reference, rtol=0, atol=0)
                    row = json.loads((Path(output) / "three_point_probe_step_trace.jsonl").read_text())
                    self.assertEqual(row["probe_ce_gradient_phases"], 3)
                    self.assertEqual(row["probe_replay_closure_calls"], 2)
                    self.assertEqual(row["actual_micro_batches"], count)
                    self.assertEqual(row["tail_gradient_rescale"], 4 / count)
                    self.assertEqual(row["probe_base_optimizer_steps"], 1)
                    self.assertTrue(row["probe_restore_base_exact"])
                    self.assertEqual(row["probe_g2_calls"], 0)
                    self.assertEqual(row["probe_g3_calls"], 0)
                    self.assertEqual(len(model.observations), count * 3)
                    self.assertFalse(batches)
                    self.assertFalse(states)

    def test_dropout_replay_and_post_clean_rng_restoration(self):
        for mode in ("g1_plus", "backward_reflected", "central", "g0"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as output:
                torch.manual_seed(37)
                model = Quadratic(dropout=.25)
                trainer = probe_stub(model, output, mode)
                batches = batch_rows(4)
                states = clean_pass(trainer, model, batches)
                after_clean = torch.get_rng_state().clone()
                trainer._run_three_point_probe_optimizer_window(model, batches, states)
                self.assertTrue(torch.equal(after_clean, torch.get_rng_state()))
                for offset in (4, 8):
                    for index in range(4):
                        self.assertTrue(torch.equal(model.observations[index], model.observations[offset + index]))

    def test_fixed_backend_and_pipeline_guards(self):
        with tempfile.TemporaryDirectory() as output:
            trainer = probe_stub(Quadratic(), output)
            trainer._validate_three_point_probe_training_configuration()
            for key, value in [("lora_strategy", "ewclora"), ("train_method", "use_probe"),
                               ("rwp_type", "rwp"), ("adam_epsilon", 1e-8),
                               ("weight_decay", .01), ("probe_mode", "gam")]:
                old = getattr(trainer.args, key)
                setattr(trainer.args, key, value)
                with self.assertRaises(ValueError):
                    trainer._validate_three_point_probe_training_configuration()
                setattr(trainer.args, key, old)
            trainer.data_collator.decoder_pipeline_version = "legacy"
            with self.assertRaisesRegex(ValueError, "decoder_token_ids_v2"):
                trainer._validate_three_point_probe_training_configuration()

    def test_first_window_fingerprints_match_all_four_controls(self):
        receipts = []
        for mode in ("g1_plus", "backward_reflected", "central", "g0"):
            with tempfile.TemporaryDirectory() as output:
                model = Quadratic()
                trainer = probe_stub(model, output, mode)
                batches = batch_rows(4)
                states = clean_pass(trainer, model, batches)
                trainer._run_three_point_probe_optimizer_window(model, batches, states)
                receipts.append(json.loads((Path(output) / "three_point_probe_step_trace.jsonl").read_text()))
        for field in ("initial_trainable_sha256", "clean_gradient_sha256", "first_window_batch_sha256"):
            self.assertEqual(len({row[field] for row in receipts}), 1)
            self.assertEqual(len(receipts[0][field]), 64)
        self.assertEqual(len(UIETrainer._two_point_probe_tensor_hash([("scalar", torch.tensor(1.0))])), 64)

    def test_native_corrected_llama_bf16_forward_all_modes_flush_tail(self):
        receipts = []
        for mode in ("g1_plus", "backward_reflected", "central", "g0"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as output:
                model, trainer = tiny_llama_native_setup(output, mode)
                frozen = {name: value.detach().clone() for name, value in model.named_parameters()
                          if not value.requires_grad}
                result = trainer.train()
                audit = trainer.finish_three_point_probe_training_audit()
                self.assertEqual(result.global_step, 2)
                self.assertEqual(audit["status"], "passed")
                self.assertEqual(audit["decoder_pipeline_version"], "decoder_token_ids_v2")
                self.assertEqual(audit["probe_quantization_diagnostics_interval"], 1)
                self.assertEqual(audit["mode"], mode)
                self.assertEqual(audit["total_ce_gradient_phases"], 6)
                self.assertEqual(audit["total_base_optimizer_steps"], 2)
                self.assertEqual(audit["tail_windows"], 1)
                self.assertEqual(audit["fisher_state_entries"], 0)
                self.assertFalse(audit["g2_enabled"])
                self.assertFalse(audit["g3_enabled"])
                rows = [json.loads(line) for line in (Path(output) / "three_point_probe_step_trace.jsonl").read_text().splitlines()]
                self.assertEqual([row["actual_micro_batches"] for row in rows], [4, 2])
                self.assertEqual(type(trainer.optimizer.base_optimizer).__module__, "transformers.optimization")
                self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad))
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                self.assertTrue(all(torch.equal(value, frozen[name]) for name, value in model.named_parameters()
                                    if name in frozen))
                for row in rows:
                    self.assertLessEqual(row["probe_submitted_postclip_norm"], 1 + 1e-6)
                    self.assertEqual(row["probe_base_optimizer_steps"], 1)
                    self.assertEqual(row["probe_quantization_proxy"]["sampled"], True)
                    self.assertFalse(row["probe_quantization_proxy"]["proves_forward_visibility"])
                    self.assertEqual(row["probe_fisher_probe_calls"], 0)
                    self.assertEqual(row["probe_projection_calls"], 0)
                    self.assertEqual(row["probe_hvp_calls"], 0)
                json.dumps(audit, allow_nan=False)
                receipts.append(audit)
        for field in ("initial_trainable_sha256", "clean_gradient_sha256", "first_window_batch_sha256"):
            self.assertEqual(len({receipt[field] for receipt in receipts}), 1)

    def test_native_max_steps_resets_epoch_windows_and_steps_scheduler_once(self):
        with tempfile.TemporaryDirectory() as output:
            _, trainer = tiny_llama_native_setup(output, "central", max_steps=3)
            result = trainer.train()
            audit = trainer.finish_three_point_probe_training_audit()
            rows = [json.loads(line) for line in (Path(output) / "three_point_probe_step_trace.jsonl").read_text().splitlines()]
            self.assertEqual(result.global_step, 3)
            self.assertEqual([row["actual_micro_batches"] for row in rows], [4, 2, 4])
            self.assertEqual(trainer.lr_scheduler.last_epoch, 3)
            self.assertEqual(audit["total_base_optimizer_steps"], 3)
            self.assertEqual(audit["fallback_steps"], 0)

    def test_native_gradient_checkpointing_keeps_bf16_reverse_transaction(self):
        with tempfile.TemporaryDirectory() as output:
            model, trainer = tiny_llama_native_setup(output, "central", max_steps=1)
            model.enable_input_require_grads()
            trainer.args.gradient_checkpointing = True
            result = trainer.train()
            audit = trainer.finish_three_point_probe_training_audit()
            self.assertEqual(result.global_step, 1)
            self.assertEqual(audit["total_ce_gradient_phases"], 3)
            self.assertEqual(audit["total_replay_closure_calls"], 2)
            self.assertEqual(audit["total_base_optimizer_steps"], 1)
            self.assertEqual(audit["fallback_steps"], 0)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_extra_side_reads_preserve_legacy_clean_sam_and_reflected_updates(self):
        from test_reverse_probe_trainer import tiny_llama_native_setup as legacy_setup
        # This compares native trajectories, rather than reusing the new core's
        # formula. Extra matched probes must not alter the selected method or
        # the sampler/dropout stream when their gradients are unused.
        for mode, legacy in (("g0", "sgd"), ("g1_plus", "sam"),
                             ("backward_reflected", "reverse")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as old_output, tempfile.TemporaryDirectory() as new_output:
                old_model, old_trainer = legacy_setup(old_output, legacy, max_steps=3)
                old_trainer.train()
                old_audit = old_trainer.finish_two_point_probe_training_audit()
                new_model, new_trainer = tiny_llama_native_setup(new_output, mode, max_steps=3)
                new_trainer.train()
                new_audit = new_trainer.finish_three_point_probe_training_audit()
                for (old_name, old), (new_name, new) in zip(old_model.named_parameters(), new_model.named_parameters()):
                    self.assertEqual(old_name, new_name)
                    self.assertTrue(torch.equal(old, new), old_name)
                for field in ("initial_trainable_sha256", "clean_gradient_sha256", "first_window_batch_sha256"):
                    self.assertEqual(old_audit[field], new_audit[field])
                self.assertEqual(new_audit["total_ce_gradient_phases"], 9)
                self.assertEqual(new_audit["total_replay_closure_calls"], 6)
                self.assertEqual(new_trainer.lr_scheduler.last_epoch, 3)

    @unittest.skipUnless(os.environ.get("THREEPOINT_TEST_GLOO") == "1", "explicit CPU loopback test")
    def test_two_rank_global_mean_all_modes_matches_one_hf_adamw(self):
        with tempfile.TemporaryDirectory() as output:
            rendezvous = str(Path(output) / "gloo-store")
            torch.multiprocessing.spawn(ddp_probe_worker, args=(rendezvous, output), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
