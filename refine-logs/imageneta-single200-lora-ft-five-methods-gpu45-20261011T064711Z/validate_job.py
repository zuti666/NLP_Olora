"""Accept exact single-task budgets and real update/state/CUDA evidence."""
from collections import Counter
import math
from pathlib import Path
from common import HERE, finite_tree, read_json, sha256

READS = {"sgd": 1, "sam": 2, "mirror": 2, "g2": 3, "gam": 4}
ZERO_COUNTERS = ("fisher_estimation_passes", "weight_merge_updates", "hvp_calls", "create_graph_calls", "gradient_clipping_calls", "final_norm_rescale_calls", "forbidden_work_attempts", "training_hvp_passes", "training_create_graph_passes")
ZERO_STEP = ("fisher_estimation_passes", "weight_merge_updates", "hvp_calls", "qr_calls", "create_graph_calls", "gradient_clipping_calls", "final_norm_rescale_calls", "ewc_calls", "replay_calls", "sgv_calls", "extra_gradient_projection_calls")

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def validate_job(job):
    out = Path(job["output_dir"])
    path = out / job.get("report_file", "final_summary.json")
    a = read_json(path)
    finite_tree(a)
    if job["kind"] == "pairing":
        require(a.get("status") in {"passed", "completed"} and bool(a.get("checks")) and all(a["checks"].values()), "Pairing gate failed")
        require(a.get("smoke_count", 10) == 10, "Pairing did not cover all ten smokes")
        for p, expected in a.get("evidence_hashes", {}).items():
            require(sha256(p) == expected, "Pairing evidence changed: " + p)
        return dict(summary_sha256=sha256(path), gate="passed", full_formal_training=False)
    raw = read_json(job["config"])
    cfg = read_json(out / "effective_config.json")
    method, scope = job["method"], job["train_scope"]
    epochs, steps = (1, 2) if job["smoke"] else (50, 2350)
    lr0 = .04 if scope == "lora" else .001
    require(a["status"] == "completed" and a["completed_epochs"] == epochs and a["completed_steps"] == steps and a["completed_tasks"] == 1, "Incomplete requested single-task budget")
    require((a["method"], a["train_scope"], a["seed"], a["smoke"]) == (method, scope, 2, job["smoke"]), "Run identity differs from registered job")
    require(a["input_config_sha256"] == job["config_sha256"] == sha256(job["config"]), "Config hash changed")
    require(a["trainer_sha256"] == sha256(HERE / "trainer/train_single200.py") and a["algorithm_sha256"] == sha256(HERE / "trainer/five_method_step.py"), "Execution code hash changed")
    require(a["source_integrity"]["passed"] and a["source_integrity"]["file_count"] == 190, "Source integrity missing")
    require(a["source_integrity"]["manifest_sha256"] == sha256(HERE / "source_manifest.json"), "Wrong source manifest")
    require(bool(a.get("checks")) and all(a["checks"].values()), "Run-level checks failed")
    fixed = dict(train_scope=scope, method=method, seed=2, init_cls=200, increment=200, nb_tasks=1, epochs=50, init_epoch=50, batch_size=128, lora_rank=10, rho=.05, r=1., momentum=0., weight_decay=0., lr=lr0, min_lr_ratio=.1, ewc_lambda=0., ewc_skip_task_fisher=True, strict_deterministic=True, attention_backend="native", gamma=.1 if method == "gam" else 0.)
    for key, value in fixed.items():
        require(raw.get(key) == value and cfg.get(key) == value, "Changed scientific setting: " + key)
    for key in ("approximate_fisher_in_task", "in_task_weight_merge", "task_endpoint_weight_merge", "use_sgv", "gam_use_sgv", "mechanism_eval", "as1_delta_w_penalty", "final_update_norm_matching"):
        require(cfg.get(key, False) is False, "Forbidden active module: " + key)
    require(cfg["continual_learning"] is False and cfg["replay"] is False and cfg["final_update_normalized"] is False, "CL, replay or final normalization was enabled")
    d = a["dataset_partition_receipt"]
    require(d["full_train_samples"] == 5981 and d["full_test_samples"] == 1519 and d["nb_classes"] == 200 and d["class_order"] == list(range(200)) and d["task_increments"] == [200] and d["batches_per_epoch"] == 47, "Wrong complete ImageNet-A split")
    scope_receipt = a["model_scope_receipt"]
    require(scope_receipt["train_scope"] == scope and scope_receipt["trainable_parameter_count"] > 0, "Wrong parameter scope")
    if scope == "full_ft":
        require(scope_receipt["frozen_parameter_count"] == 0 and scope_receipt["lora_factor_tensor_count"] == 0 and scope_receipt["no_lora_modules"] and scope_receipt["all_backbone_and_head_trainable"], "FT still contains LoRA or frozen backbone")
        displacement = a["actual_first_update_scope_displacement"]
        require(displacement["backbone_and_fc_both_updated"] and displacement["backbone_parameter_displacement_norm"] > 0 and displacement["fc_parameter_displacement_norm"] > 0 and displacement["backbone_nonzero_updated_tensor_count"] > 0 and displacement["fc_nonzero_updated_tensor_count"] > 0, "FT did not actually update both backbone and classifier")
    else:
        require(scope_receipt["lora_factor_tensor_count"] > 0 and scope_receipt["frozen_parameter_count"] > 0 and not scope_receipt["all_backbone_and_head_trainable"], "LoRA scope is not frozen-backbone tuning")
    require(a["initial_identity"]["reference_initialization_exact"], "Historical initialization pairing failed")
    rows = [__import__("json").loads(line) for line in (out / "step_trace.jsonl").read_text().splitlines() if line.strip()]
    require(len(rows) == steps, "Incomplete step trace")
    for i, r in enumerate(rows):
        finite_tree(r)
        epoch, batch = (1, i) if job["smoke"] else (i // 47 + 1, i % 47)
        require((r["step_index"], r["task"], r["epoch"], r["batch"], r["seed"], r["method"], r["train_scope"]) == (i, 0, epoch, batch, 2, method, scope), "Trace coordinates or condition changed")
        require(r["ordinary_gradient_calls"] == READS[method] and r["sgd_updates"] == 1 and r["real_sgd_update"] and r["exact_torch_sgd_update"], "Not one exact SGD step with native read budget")
        require(r["training_forward_count"] == READS[method] and r["actual_model_forward_count"] == READS[method] + 1 and r["actual_model_forward_count_measured"] and r["diagnostic_loss_forward_count"] == 1, "Forward accounting differs")
        require(r["strict_deterministic"] and not r["deterministic_warn_only"] and r["only_base_R1_rng_advance"] and all(r["state_restore_checks"].values()) and all(r["diagnostic_state_restoration"].values()), "State/RNG/determinism checks failed")
        require(all(r[k] == 0 for k in ZERO_STEP) and r["history_used"] is False and r["history_available"] is False, "Forbidden method work executed")
        require(r["projection_calls"] == int(method == "gam") and r["gam_intrinsic_projection_calls"] == int(method == "gam") and r["final_update_normalized"] is False, "Wrong projection or final normalization")
        require(r["final_update_origin"] == "base" and r["class_slice"] == [0, 200] and r["seen_classes"] == 200 and r["fc_shape"] == [200, 768], "Wrong commit origin or classifier")
        eta = lr0 * (.1 + .9 * (1 + math.cos(math.pi * (epoch - 1) / 50)) / 2)
        require(math.isclose(r["nominal_lr"], eta, rel_tol=1e-12, abs_tol=1e-15), "Scope learning rate schedule changed")
        require(r["identity"]["data_seed"] == 200000 + epoch, "Sampler/augmentation seed changed")
        require(r["batch_samples"] == (93 if not job["smoke"] and batch == 46 else 128), "Epoch sample budget changed")
        require(r["final_gradient_sha256"] == r["U_sha256"], "Submitted vector differs from final update")
        if scope == "full_ft" and i == 0:
            require(r["actual_first_update_scope_displacement"] == a["actual_first_update_scope_displacement"], "FT first-step displacement receipt differs from summary")
        if method in {"sgd", "sam", "g2"}:
            key = {"sgd": "g0_sha256", "sam": "g1_sha256", "g2": "g2_sha256"}[method]
            require(r["U_sha256"] == r[key], "Wrong selected gradient")
        if method == "mirror":
            require(r["g1_minus_sha256"] and r["g1_sha256"] is None and r["first_probe_sign"] == -1, "Mirror did not use the negative first probe")
        if method == "sgd":
            require(all(r[k] is None for k in ("g1_sha256", "g1_minus_sha256", "g2_sha256", "g3_sha256")), "SGD executed an extra probe")
        if method in {"g2", "gam"}:
            require(r["outer_origin"] == "base" and r["q_sha256"], "Outer relocation did not start from base")
        if method == "gam":
            require(r["g3_sha256"] and r["gamma_GAM"] == .1 and r["projection"], "Original GAM intrinsic projection missing")
    c = a["counters"]
    require(c["training_sgd_updates"] == steps and c["training_batches"] == steps and c["training_ce_gradient_passes"] == READS[method] * steps and c["training_forward_count"] == READS[method] * steps and c["diagnostic_loss_forward_count"] == steps, "Total read/update accounting differs")
    require(c["training_samples"] == (256 if job["smoke"] else 5981 * 50), "Total train sample budget differs")
    require(all(c[k] == 0 for k in ZERO_COUNTERS) and c["projection_calls"] == int(method == "gam") * steps, "Forbidden aggregate work")
    for file, key in (("step_trace.jsonl", "step_trace_sha256"), ("effective_config.json", "effective_config_sha256"), ("epoch_test_curve.json", "epoch_curve_sha256"), ("checkpoints/final.pt", "checkpoint_sha256")):
        require(sha256(out / file) == a[key], "Saved artifact hash changed: " + file)
    curve = read_json(out / "epoch_test_curve.json")["epochs"]
    require(len(curve) == epochs and [x["epoch"] for x in curve] == list(range(1, epochs + 1)), "Incomplete epoch test curve")
    endpoint = a["endpoint_test"]
    require(endpoint["test_samples"] == 1519 and endpoint["classification_classes"] == 200 and not endpoint["test_used_for_selection"] and endpoint["passive_state_and_rng_restored"], "Test selection or state protocol differs")
    require(endpoint["test_accuracy"] == curve[-1]["test_accuracy"] and endpoint["test_ce"] == curve[-1]["test_ce"] and sum(endpoint["test_group_counts"]) == 1519, "Endpoint is not the fixed final epoch")
    require(0 <= endpoint["test_accuracy"] <= 100 and endpoint["test_ce"] >= 0, "Invalid endpoint metrics")
    witness = read_json(out / "cuda_witness.json")
    require(witness["status"] == "passed" and witness["assigned_device_match"] and witness["visible_device_count"] == 1 and witness["cuda_logical_device"] == 0 and not witness["tf32_matmul"] and not witness["tf32_cudnn"], "Invalid actual trainer CUDA witness")
    runtime = HERE / "runtime/jobs" / job["id"]
    assigned = read_json(runtime / "job.json")
    runner = read_json(runtime / "runner_identity.json")["process"]
    require(int(witness["expected_physical_gpu"]) == assigned["gpu"] in {4, 5} and witness["expected_uuid"] == assigned["gpu_uuid"] == witness["cuda_visible_devices"], "CUDA witness differs from physical assignment")
    require(witness["pid"] == runner["pid"] and str(witness["process_starttime"]) == str(runner["proc_starttime"]), "CUDA witness is not from the actual trainer process")
    return dict(summary_sha256=sha256(path), full_formal_training=not job["smoke"], completed_steps=steps, completed_epochs=epochs, ordinary_gradient_calls=READS[method], train_scope=scope, method=method, test_accuracy=endpoint["test_accuracy"] if not job["smoke"] else None)
