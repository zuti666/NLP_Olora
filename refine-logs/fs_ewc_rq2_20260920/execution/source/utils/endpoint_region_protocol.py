"""Single source of truth for the frozen ImageNet-R RQ3 v2 protocol."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping


PROTOCOL_VERSION = "imagenetr-endpoint-region-v2"
SOURCE_CLOSURE_DIRECTORIES = (
    "src",
    "utils",
    "models_LoRAbasedCL",
    "models_CL",
    "backbone",
    "optimer_PerturabtionType",
    "eval_flat",
    "evaluation",
    "evaluation_attention",
    "evaluation_feature",
    "evaluation_ood",
    "evaluation_performance",
    "evaluation_weight_sharpness",
    "sharpness_evaluation_core",
)
FROZEN_FORMAL = {
    "mode": "heldout",
    "tag": "imagenetr_endpoint_region_heldout_s345_v2",
    "seeds": [3, 4, 5],
    "methods": ["sgd", "sam", "ratio"],
    "physical_gpus": [5],
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 0.02,
    "scheduler": "cosine",
    "momentum": 0.0,
    "weight_decay": 0.0,
    "ratio_c": 0.20,
    "rho": 0.05,
    "fisher_damping": 0.20,
    "lambda_ewc": 1_200_000.0,
    "split_seed": 19073,
    "split_ratio": 0.20,
    "lora_rank": 10,
    "random_directions": 32,
    "task2_gradient_directions": 32,
    "gradient_batch_size": 8,
    "diagnostic_per_class": 8,
    "radii": [0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8],
    "tolerances": [0.005, 0.01, 0.02],
    "taus": [0.005, 0.01, 0.02],
    "accuracy_tolerances_pp": [1.0, 2.0, 5.0],
    "primary_statistic": "common_task2_direction_q05_radius",
    "matched_new_tolerance_pp": 0.2,
    "matched_progress_max_gap_pp": 5.0,
    "matched_progress_max_overshoot_pp": 5.0,
    "matched_task1_tolerance_pp": 1.0,
    "trajectory_new_accuracy_targets": [50.0, 60.0, 70.0, 80.0],
    "python": "/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python3.11",
}


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


FROZEN_PROTOCOL_SHA256 = canonical_sha256(FROZEN_FORMAL)


def formal_manifest_mismatches(value: Mapping[str, object]) -> dict[str, object]:
    """Return exact frozen-protocol mismatches; an empty mapping means valid."""
    observed = value.get("frozen_formal")
    mismatches: dict[str, object] = {}
    if value.get("protocol_version") != PROTOCOL_VERSION:
        mismatches["protocol_version"] = value.get("protocol_version")
    if observed != FROZEN_FORMAL:
        mismatches["frozen_formal"] = observed
    if value.get("frozen_protocol_sha256") != FROZEN_PROTOCOL_SHA256:
        mismatches["frozen_protocol_sha256"] = value.get(
            "frozen_protocol_sha256"
        )
    if value.get("mode") != "heldout" or bool(value.get("smoke", True)):
        mismatches["formal_role"] = {
            "mode": value.get("mode"),
            "smoke": value.get("smoke"),
        }
    if value.get("tag") != FROZEN_FORMAL["tag"]:
        mismatches["tag"] = value.get("tag")
    if value.get("seeds") != FROZEN_FORMAL["seeds"]:
        mismatches["seeds"] = value.get("seeds")
    if value.get("task1_methods") != FROZEN_FORMAL["methods"]:
        mismatches["task1_methods"] = value.get("task1_methods")
    direct_fields = {
        "physical_gpus": "physical_gpus",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "learning_rate": "learning_rate",
        "scheduler": "scheduler",
        "momentum": "momentum",
        "weight_decay": "weight_decay",
        "ratio_c": "ratio_c",
        "rho": "rho",
        "fisher_damping": "fisher_damping",
        "lambda_ewc": "lambda_ewc",
        "lora_rank": "lora_rank",
        "python": "python",
    }
    for manifest_field, frozen_field in direct_fields.items():
        if value.get(manifest_field) != FROZEN_FORMAL[frozen_field]:
            mismatches[manifest_field] = value.get(manifest_field)
    split = value.get("split", {})
    expected_split = {
        "source": "train",
        "ratio": FROZEN_FORMAL["split_ratio"],
        "seed": FROZEN_FORMAL["split_seed"],
    }
    if split != expected_split:
        mismatches["split"] = split
    diagnostic = value.get("diagnostic", {})
    diagnostic_expected = {
        "random_directions": FROZEN_FORMAL["random_directions"],
        "task2_gradient_directions_per_family": FROZEN_FORMAL[
            "task2_gradient_directions"
        ],
        "gradient_batch_size": FROZEN_FORMAL["gradient_batch_size"],
        "old_samples_per_class": FROZEN_FORMAL["diagnostic_per_class"],
        "primary_statistic": FROZEN_FORMAL["primary_statistic"],
        "radii": FROZEN_FORMAL["radii"],
        "tolerances": FROZEN_FORMAL["tolerances"],
        "taus": FROZEN_FORMAL["taus"],
        "accuracy_tolerances_pp": FROZEN_FORMAL["accuracy_tolerances_pp"],
    }
    for field, expected in diagnostic_expected.items():
        if diagnostic.get(field) != expected:
            mismatches[f"diagnostic.{field}"] = diagnostic.get(field)
    trajectory = value.get("task2_trajectory", {})
    trajectory_expected = {
        "new_accuracy_targets": FROZEN_FORMAL["trajectory_new_accuracy_targets"],
        "terminal_match_tolerance_pp": FROZEN_FORMAL[
            "matched_new_tolerance_pp"
        ],
        "matched_progress_max_gap_pp": FROZEN_FORMAL[
            "matched_progress_max_gap_pp"
        ],
        "matched_progress_max_overshoot_pp": FROZEN_FORMAL[
            "matched_progress_max_overshoot_pp"
        ],
        "matched_task1_tolerance_pp": FROZEN_FORMAL[
            "matched_task1_tolerance_pp"
        ],
    }
    for field, expected in trajectory_expected.items():
        if trajectory.get(field) != expected:
            mismatches[f"task2_trajectory.{field}"] = trajectory.get(field)
    if not bool(value.get("cohort_evidence_eligible", False)):
        mismatches["cohort_evidence_eligible"] = value.get(
            "cohort_evidence_eligible"
        )
    for field in (
        "producer_source_bundle_sha256",
        "producer_env_spec_hash",
        "producer_env_receipt_sha256",
    ):
        observed_value = str(value.get(field, ""))
        if len(observed_value) != 64:
            mismatches[field] = observed_value
    return mismatches
