"""CPU-only checks of configuration identity, budgets and four-stage chaining."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("g2_llama_runner_tested", HERE / "run_g2_llama.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
CONFIGS = sorted((HERE.parent / "configs/g2_llama").glob("*_order1.json"))
HF_CONFIGS = sorted((HERE.parent / "configs/g2_llama").glob("inclora_g2_llama*_order*.json"))


class G2LauncherTests(unittest.TestCase):
    def test_four_pure_presets_have_explicit_matched_backends(self):
        self.assertEqual(len(CONFIGS), 4)
        for path in CONFIGS:
            cfg = runner.load_config(path)
            with self.subTest(path=path.name):
                self.assertEqual(cfg["training"]["ewclora_lambda"], 0)
                self.assertNotIn("g2_include_olora_regularization", cfg["training"])
                expected = "transformers.AdamW" if cfg["method"] == "inclora_g2" else "torch.optim.AdamW"
                self.assertEqual(cfg["optimizer_class"], expected)

    def test_six_hf_chains_match_historical_orders_budgets_and_evaluation(self):
        self.assertEqual(len(HF_CONFIGS), 6)
        expected_orders = {
            1: (["dbpedia", "amazon", "yahoo", "agnews"], [218, 78, 156, 62]),
            2: (["dbpedia", "amazon", "agnews", "yahoo"], [218, 78, 62, 156]),
            3: (["yahoo", "amazon", "agnews", "dbpedia"], [156, 78, 62, 218]),
        }
        observed = set()
        for path in HF_CONFIGS:
            cfg = runner.load_config(path)
            observed.add((cfg["model_size"], cfg["order"]))
            tasks, updates = expected_orders[cfg["order"]]
            with self.subTest(config=path.name):
                plan = runner.build_plan(cfg, [0], Path("/tmp/g2_hf_chain_dry"))
                formal = [stage for stage in plan["stages"] if not stage["smoke"]]
                self.assertEqual(plan["order"], cfg["order"])
                self.assertEqual(plan["formal_updates_per_complete_order"], 514)
                self.assertEqual([stage["task"] for stage in formal], tasks)
                self.assertEqual([stage["expected_updates"] for stage in formal], updates)
                self.assertEqual([stage["expected_test_samples"] for stage in formal], [7600, 15200, 22800, 30400])
                self.assertEqual([stage["expected_saved_rank"] for stage in formal], [8, 16, 24, 32])
                self.assertEqual(len([stage for stage in plan["stages"] if stage["smoke"]]), 2)
                for index, stage in enumerate(formal):
                    self.assertEqual(stage["seen_tasks"], tasks[:index + 1])
                    self.assertEqual(stage["order"], cfg["order"])
                    self.assertEqual(stage["options"]["optimizer_type"], "g2-adamw_hf")
                    self.assertEqual(stage["options"]["adam_epsilon"], 1e-6)
                    self.assertEqual(stage["options"]["weight_decay"], 0)
                    self.assertEqual(stage["options"]["task_config_dir"], str(HERE.parent / "configs" / f"order{cfg['order']}_configs" / tasks[index]))
                    if index:
                        self.assertEqual(stage["previous_adapter"], str(Path(formal[index - 1]["output_dir"]) / "adapter"))
        self.assertEqual(observed, {(size, order) for size in ("1B", "3B") for order in (1, 2, 3)})

    def test_each_requested_folder_has_an_executable_hf_wrapper(self):
        for path in HF_CONFIGS:
            cfg = runner.load_config(path)
            size = cfg["model_size"].lower()
            wrapper = HERE.parent / f"scripts_order{cfg['order']}_llama{cfg['model_size']}" / f"IncLoRA-G2_LLaMA3-{size}.sh"
            with self.subTest(wrapper=str(wrapper)):
                self.assertTrue(wrapper.is_file())
                self.assertTrue(os.access(wrapper, os.X_OK))
                env = dict(os.environ, G2_LLAMA_PYTHON=sys.executable)
                completed = subprocess.run(["bash", str(wrapper), "--dry-run", "--seed", "0"], env=env,
                                           capture_output=True, text=True, check=True)
                plan = json.loads(completed.stdout)
                self.assertEqual(plan["order"], cfg["order"])
                self.assertEqual(plan["model_size"], cfg["model_size"])
                self.assertEqual(plan["formal_stages"], 4)

    def test_changed_task_config_is_rejected_before_launch(self):
        cfg = runner.load_config(next(path for path in HF_CONFIGS if path.name == "inclora_g2_llama1b_order3.json"))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            config = source / "configs/order3_configs/yahoo"
            config.mkdir(parents=True)
            (config / "train_tasks.json").write_text(json.dumps({"TC": [{"dataset name": "dbpedia"}]}))
            with self.assertRaisesRegex(ValueError, "stage1 train task configuration differs"):
                runner.build_plan(cfg, [0], source / "not_created", source=source)
            self.assertFalse((source / "not_created").exists())

    def test_missing_metrics_tokenizer_fails_before_gpu_query_or_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            incomplete_repo = Path(tmp)
            with patch.object(runner, "REPO", incomplete_repo), \
                 patch.object(runner, "gpu_snapshot", side_effect=AssertionError("queried GPUs before asset validation")):
                with self.assertRaisesRegex(FileNotFoundError, "local metrics tokenizer"):
                    runner.claim_gpus("0,1,2,3")
                with self.assertRaisesRegex(FileNotFoundError, "local metrics tokenizer"):
                    runner.freeze_source(incomplete_repo / "run_not_created")
            self.assertFalse((incomplete_repo / "run_not_created").exists())

    def test_frozen_entrypoint_imports_real_tokenizer_offline_without_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "frozen_cpu_import"
            source, hashes = runner.freeze_source(run_root)
            for name in runner.METRICS_TOKENIZER_FILES:
                relative = "data/gpt2tokenizer/" + name
                self.assertIn(relative, hashes)
                self.assertEqual(hashes[relative], runner.sha256(HERE.parent / relative))
            runner.verify_source(source, hashes)
            entrypoint = source / "src/run_uie_lora.py"
            script = (
                "import pathlib, runpy, socket\n"
                "def no_network(*args, **kwargs):\n"
                "    raise AssertionError('offline frozen import attempted network access')\n"
                "socket.socket.connect = no_network\n"
                "socket.create_connection = no_network\n"
                f"runpy.run_path({str(entrypoint)!r}, run_name='frozen_g2_import_regression')\n"
                "import compute_metrics, torch\n"
                f"assert pathlib.Path(compute_metrics.GPT2TOKENIZER).resolve() == pathlib.Path({str(source / 'data/gpt2tokenizer')!r})\n"
                "assert compute_metrics.GPTTokenizer.gpt_tokenizer.encode('Current G2 offline import regression.')\n"
                "assert not torch.cuda.is_initialized()\n"
                "print('FROZEN_ENTRYPOINT_IMPORT_OFFLINE_OK')\n"
            )
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(source / "src"),
                       TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       NLTK_DATA=str(HERE.parent / ".cache/nltk_data"), TOKENIZERS_PARALLELISM="false")
            completed = subprocess.run([sys.executable, "-c", script], cwd=tmp, env=env,
                                       capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("FROZEN_ENTRYPOINT_IMPORT_OFFLINE_OK", completed.stdout)

    def test_full_plan_has_paired_budget_and_separate_adapter_chains(self):
        cfg = runner.load_config(CONFIGS[0])
        plan = runner.build_plan(cfg, [0, 42, 1993], Path("/tmp/g2_cpu_plan_only"))
        self.assertEqual(plan["formal_orders"], 3)
        self.assertEqual(plan["formal_stages"], 12)
        self.assertEqual(len(plan["stages"]), 18)
        for seed in runner.SEEDS:
            formal = [s for s in plan["stages"] if s["seed"] == seed and not s["smoke"]]
            self.assertEqual([s["expected_updates"] for s in formal], [218, 78, 156, 62])
            self.assertEqual(sum(s["expected_updates"] for s in formal), 514)
            self.assertEqual([s["expected_test_samples"] for s in formal], [7600, 15200, 22800, 30400])
            self.assertEqual([s["expected_saved_rank"] for s in formal], [8, 16, 24, 32])
            self.assertIsNone(formal[0]["previous_adapter"])
            for stage in range(1, 4):
                self.assertEqual(formal[stage]["previous_adapter"], str(Path(formal[stage - 1]["output_dir"]) / "adapter"))
            for stage in formal:
                self.assertEqual(stage["options"]["seed"], stage["options"]["data_seed"])
                self.assertEqual(stage["options"]["learning_rate"], .001 if stage["stage"] == 1 else .0001)
                self.assertNotIn("smoke", stage["output_dir"])

    def test_fisher_and_ewc_are_disabled_in_every_preset(self):
        for path in CONFIGS:
            cfg = runner.load_config(path)
            stage = runner.make_stage(cfg, 0, 2, Path("/tmp/g2_cpu_only"), smoke=False)
            with self.subTest(path=path.name):
                self.assertNotIn("ewclora_fisher_path", stage["options"])
                self.assertEqual(stage["options"]["ewclora_lambda"], 0)
                self.assertEqual(stage["options"]["olora_lamda_1"], 0)
                self.assertEqual(stage["options"]["olora_lamda_2"], 0)

    def test_dry_run_creates_nothing_and_never_queries_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "must_not_exist"
            stdout = io.StringIO()
            with patch.object(runner, "gpu_snapshot", side_effect=AssertionError("dry run queried GPUs")), \
                 patch.object(runner, "freeze_source", side_effect=AssertionError("dry run copied files")), \
                 contextlib.redirect_stdout(stdout):
                result = runner.main(["--config", str(CONFIGS[0]), "--dry-run", "--run-root", str(run_root), "--seeds", "0", "42", "1993"])
            self.assertEqual(result, 0)
            self.assertFalse(run_root.exists())
            self.assertEqual(json.loads(stdout.getvalue())["formal_stages"], 12)

    def test_fixed_probes_require_explicit_finite_positive_radii(self):
        cfg = runner.load_config(CONFIGS[0])
        training = cfg["training"]
        training["g2_inner_radius_mode"] = "fixed_l2"
        with self.assertRaises(ValueError):
            runner.validate_radii(training)
        for radius in (0, -1, float("nan"), float("inf"), True):
            training["g2_inner_radius"] = radius
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                runner.validate_radii(training)
        training["g2_inner_radius"] = .05
        runner.validate_radii(training)
        training["g2_outer_radius_mode"] = "fixed_l2"
        training["g2_outer_radius"] = .05
        runner.validate_radii(training)

    def test_dry_cli_exposes_independent_radii(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            runner.main(["--config", str(CONFIGS[0]), "--dry-run", "--inner-radius-mode", "fixed_l2", "--inner-radius", "0.05",
                         "--outer-radius-mode", "fixed_l2", "--outer-radius", "0.1"])
        stage = json.loads(out.getvalue())["stages"][-1]
        self.assertEqual(stage["options"]["g2_inner_radius"], .05)
        self.assertEqual(stage["options"]["g2_outer_radius"], .1)
        self.assertEqual(stage["options"]["max_steps"], 62)

    def test_duplicate_seeds_and_geometry_drift_fail_closed(self):
        cfg = runner.load_config(CONFIGS[0])
        with self.assertRaises(ValueError):
            runner.build_plan(cfg, [0, 0], Path("/tmp/g2_not_created"))
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            cfg["execution"]["gradient_accumulation_steps"] = 8
            bad.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                runner.load_config(bad)

    def test_ewc_or_olora_presets_are_rejected_by_the_pure_launcher(self):
        cfg = runner.load_config(CONFIGS[0])
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            for method in ("ewclora_g2", "olora_g2"):
                cfg["method"] = method
                bad.write_text(json.dumps(cfg))
                with self.subTest(method=method), self.assertRaises(ValueError):
                    runner.load_config(bad)

    def test_runtime_audit_checks_actual_backend_and_optimizer_values(self):
        cfg = runner.load_config(next(p for p in CONFIGS if p.name == "inclora_g2_llama1b_order1.json"))
        with tempfile.TemporaryDirectory() as tmp:
            spec = runner.make_stage(cfg, 0, 1, Path(tmp), smoke=True)
            output = Path(spec["output_dir"])
            output.mkdir(parents=True)
            audit = {
                "schema": "CurrentG2TrainerAudit_v1", "status": "passed", "global_step": 2, "num_records": 2,
                "normal_three_read_steps": 2, "fallback_steps": 0, "tail_windows": 0,
                "trainable_precision": "torch.float32", "bf16_forward": True, "strategy": "inclora",
                "fisher_state_entries": 0, "regularization_enabled": False, "history_enabled": False,
                "projection_enabled": False, "trace_sha256": "observed_trace_hash", "actual_micro_batch_counts": [16],
                "world_size": 4, "trainable_elements": 100,
                "base_optimizer": "transformers.optimization.AdamW",
                "base_optimizer_param_groups": [{"betas": [.9, .999], "eps": 1e-6, "weight_decay": 0., "lr": .001}],
            }
            file = output / "g2_runtime_audit.json"
            file.write_text(json.dumps(audit))
            self.assertEqual(runner.verify_runtime_audit(spec, "observed_trace_hash", 0)["actual_base_optimizer"], "transformers.optimization.AdamW")
            audit["base_optimizer"] = "torch.optim.adamw.AdamW"
            file.write_text(json.dumps(audit))
            with self.assertRaisesRegex(RuntimeError, "optimizer class"):
                runner.verify_runtime_audit(spec, "observed_trace_hash", 0)
            audit["base_optimizer"] = "transformers.optimization.AdamW"
            audit["base_optimizer_param_groups"][0]["eps"] = 1e-8
            file.write_text(json.dumps(audit))
            with self.assertRaisesRegex(RuntimeError, "hyperparameters"):
                runner.verify_runtime_audit(spec, "observed_trace_hash", 0)

    def test_real_run_requires_gpu_list_before_execution(self):
        with patch.object(runner, "execute", side_effect=AssertionError("unrequested launch")), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                runner.main(["--config", str(CONFIGS[0])])
        self.assertEqual(exc.exception.code, 2)

    def test_saved_adapter_checks_reject_changed_history_and_nonfinite_weights(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root / "previous"
            previous.mkdir()
            old = {}
            for module in ("q_proj", "v_proj"):
                old[f"layers.0.{module}.lora_A.weight"] = torch.ones((8, 3), dtype=torch.float32)
                old[f"layers.0.{module}.lora_B.weight"] = torch.ones((3, 8), dtype=torch.float32)
            torch.save(old, previous / "adapter_model.bin")
            new = {name: torch.cat((value, torch.zeros_like(value)), dim=0 if ".lora_A." in name else 1)
                   for name, value in old.items()}
            checkpoint = root / "adapter_model.bin"
            torch.save(new, checkpoint)
            spec = {"expected_saved_rank": 16, "expected_target_modules": 2, "previous_adapter": str(previous)}
            good = runner.verify_adapter_tensors(spec, checkpoint)
            self.assertTrue(good["adapter_finite_fp32"])
            self.assertTrue(good["historical_prefix_exact"])
            first_name = next(iter(new))
            new[first_name][0, 0] += 1
            torch.save(new, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "frozen historical"):
                runner.verify_adapter_tensors(spec, checkpoint)
            new[first_name][0, 0] = float("nan")
            torch.save(new, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                runner.verify_adapter_tensors(spec, checkpoint)

    def test_stage_accepts_a_partial_clean_fallback_without_fabricating_probe_loss(self):
        cfg = runner.load_config(next(p for p in CONFIGS if p.name == "inclora_g2_llama1b_order1.json"))
        with tempfile.TemporaryDirectory() as tmp:
            spec = runner.make_stage(cfg, 0, 1, Path(tmp), smoke=True)
            output = Path(spec["output_dir"])
            adapter = output / "adapter"
            adapter.mkdir(parents=True)
            (output / "trainer_state.json").write_text(json.dumps({"global_step": 2}))
            (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "r_sum": 8, "lora_alpha": 32,
                                                                   "lora_dropout": .1, "target_modules": ["q_proj", "v_proj"]}))
            (adapter / "adapter_model.bin").write_bytes(b"tensor loading is tested separately")
            (output / "predict_results.json").write_text(json.dumps({"predict_exact_match": 50.}))
            predictions = [{"Dataset": "dbpedia", "Instance": {"id": i}, "Prediction": "label"} for i in range(16)]
            (output / "predict_eval_predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
            base = {
                "schema": "CurrentG2TrainerStep_v1", "world_size": 4, "actual_micro_batches": 16,
                "configured_gradient_accumulation_steps": 16, "effective_window_mean_denominator": 16,
                "tail_gradient_rescale": 1., "replay_rng_state": True, "replay_ddp_no_sync_all_micro_batches": True,
                "replay_reduction": "explicit_global_mean_once_per_probe", "clean_gradient_reduction": "native_ddp_mean",
                "history_enabled": False, "g2_history_used": False, "g2_base_optimizer_steps": 1,
                "g2_restore_base_exact": True, "g2_fisher_probe_calls": 0, "g2_projection_calls": 0,
                "g2_hvp_calls": 0, "g2_core_explicit_mean_reductions": 0, "g2_gradient_reduction_owner": "trainer",
                "strategy": "inclora", "g2_regularization_hook_used": False, "clean_regularization_enabled": False,
                "configured_clip_grad_norm": 1., "g2_final_clip_enabled": True, "g2_lr": .001,
                "base_optimizer": "transformers.optimization.AdamW", "g2_inner_radius_mode": "sgd_step",
                "g2_outer_radius_mode": "sgd_step", "g2_inner_radius": None, "g2_outer_radius": None,
                "g2_inner_radius_theoretical": .001, "g2_outer_radius_theoretical": .001, "g2_h_ref": .001,
                "g2_actual_update_norm": .001, "g2_g0_norm": 1.,
            }
            fallback = {**base, "step": 1, "g2_final_direction": "g0_fallback", "g2_fallback_reason": "base_gradient_at_or_below_eps",
                        "g2_ce_gradient_phases": 1, "g2_replay_closure_calls": 0, "probe_loss": None,
                        "g2_h_ref": 0., "g2_g0_norm": 0., "g2_actual_update_norm": 0.,
                        "g2_inner_radius_theoretical": 0., "g2_outer_radius_theoretical": 0.}
            normal = {**base, "step": 2, "g2_final_direction": "g2", "g2_fallback_reason": None,
                      "g2_ce_gradient_phases": 3, "g2_replay_closure_calls": 2, "g2_relocation_origin": "base", "probe_loss": 1.}
            trace_file = output / "g2_step_trace.jsonl"
            def save_trace():
                trace_file.write_text(json.dumps(fallback) + "\n" + json.dumps(normal) + "\n")
            save_trace()
            with patch.object(runner, "verify_adapter_tensors", return_value={}), patch.object(runner, "verify_runtime_audit", return_value={}):
                receipt = runner.verify_stage(spec)
                self.assertEqual(receipt["g2_fallback_steps"], 1)
                normal["probe_loss"] = None
                save_trace()
                with self.assertRaisesRegex(RuntimeError, "probe_loss"):
                    runner.verify_stage(spec)


if __name__ == "__main__":
    unittest.main()
