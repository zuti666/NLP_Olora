"""CPU-only source audit; no fitting, checkpoint loading, or metric modification."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "execution/report_seed0"


def rows(name):
    with (SOURCE / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    neighborhood = rows("B_neighborhood_deterministic.csv")
    tolerance = rows("B_tolerance_deterministic.csv")
    bridge = rows("B_actual_direction_bridge.csv")
    trajectory = rows("C_current_FP32_task_trajectory.csv")
    paths = rows("C_path_summary.csv")
    config_path = HERE.parent / "execution/diagnostic_config_recovery1.json"
    config = json.loads(config_path.read_text())
    by_radius = {(r["method"], r["split"], float(r["radius"])): r for r in neighborhood}
    radii = sorted({float(r["radius"]) for r in neighborhood})
    comparisons = []
    for radius in radii:
        s, g = (by_radius[m, "test", radius] for m in ("sgd", "gam"))
        comparisons.append({
            "radius": radius,
            "raw_trace_gam_over_sgd_train256": float(g["raw_trace_mean_train256"]) / float(s["raw_trace_mean_train256"]),
            "test_G_sgd": float(s["G"]), "test_G_gam": float(g["G"]),
            "test_Q95_sgd": float(s["Q95"]), "test_Q95_gam": float(g["Q95"]),
        })
    endpoints = {}
    for method in ("sgd", "gam"):
        sequence = sorted((r for r in trajectory if r["method"] == method), key=lambda r: int(r["task"]))
        end = sequence[-1]
        path = next(r for r in paths if r["method"] == method)
        damage = [float(r["test_local_loss_delta_recomputed"]) for r in sequence]
        drift = [float(r["delta_norm_cpu_recomputed"]) for r in sequence]
        endpoints[method] = {
            "tasks_code": [int(r["task"]) for r in sequence],
            "T_total_learned_tasks": [int(r["task"]) + 1 for r in sequence],
            "signed_test_damage": damage, "actual_displacement": drift,
            "stage_T_with_damage_decrease": [i + 1 for i in range(1, len(damage)) if damage[i] < damage[i - 1]],
            "T10_test_accuracy": float(end["test_local_accuracy"]),
            "T10_S_path_train256": float(path["S_path"]),
            "T10_I_train256": float(path["I"]),
            "S_path_recomputed": float(path["I"]) / drift[-1] ** 2,
            "r2_bridge_over_final_actual_displacement": 2.0 / drift[-1],
        }
    sources = [SOURCE / n for n in (
        "B_neighborhood_deterministic.csv", "B_tolerance_deterministic.csv",
        "B_actual_direction_bridge.csv", "C_current_FP32_task_trajectory.csv", "C_path_summary.csv",
        "Figure_B_new_deterministic.png", "Figure_B_bridge.png", "Figure_C_sequence_and_path.png",
        "MANIFEST.json")]
    sources.extend([config_path, HERE.parent / "execution/postprocess.py"])
    output = {
        "scope": "Existing ImageNet-A seed0 B/C figures, NOT historical ImageNet-R table",
        "dataset_root": config["data_root"],
        "shared_config": str(config_path),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "neighborhood_comparisons": comparisons,
        "tolerance_G": [r for r in tolerance if r["metric"] == "G"],
        "bridge_at_r1": [r for r in bridge if float(r["radius"]) == 1.0],
        "sequence": endpoints,
    }
    (HERE / "SOURCE_AND_NUMBERS.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"dataset_root": output["dataset_root"], "source_files": len(sources), "sequence_rows": len(trajectory), "output": str(HERE / "SOURCE_AND_NUMBERS.json")}))


if __name__ == "__main__":
    main()
