#!/usr/bin/env python3
"""CPU-only validation, collection, and Figure C for historical seed0 chords.

Use --allow-partial to inspect progress. Missing/failed paths are never filled,
and a figure is emitted only when all 16 new paths plus 2 legacy paths validate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from paper_plot_style import (
    FIG_DIR, EXECUTION, RAW, METHODS, COLORS, MARKERS, LINES, LABELS,
    load_verified_data, save_figure, sha256, plt, np, pd,
)

REPORT = FIG_DIR / "report"
COHORT_KEYS = (
    "data_manifest_sha256", "coordinate_sha256", "coordinate_order",
    "coordinate_numel", "head_sha256", "anchor_effective_sha256",
    "source_files", "common_code_sha256",
)


def read_json(path):
    return json.loads(path.read_text())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, *, atol=1e-12, rtol=1e-10):
    check(np.isfinite(actual) and np.isfinite(expected), f"{name}: non-finite value")
    check(bool(np.isclose(actual, expected, atol=atol, rtol=rtol)),
          f"{name}: {actual!r} != {expected!r}")


def provenance(path):
    return {"path": str(path), "sha256": sha256(path)}


def validate_data_manifest(path, expected_hash):
    data = read_json(path)
    recorded = data.pop("manifest_sha256")
    check(recorded == canonical_hash(data) == expected_hash, f"Data manifest hash: {path}")
    check(Path(data["data_root"]).name == "imagenet-a", "Wrong dataset")
    check(data["classes"] == list(range(20)), "Wrong Task0 classes")
    check(len(data["subset_indices"]) == len(set(data["subset_indices"])) == 256,
          "Fisher subset must contain 256 unique train examples")
    check(data["subset_seed"] == 20260920, "Wrong subset seed")
    check(data["subset_ids"] == [data["records"]["train"][i]["id"]
                                  for i in data["subset_indices"]], "Subset membership mismatch")
    return data


def validate_path(method, target, directory, trajectory, legacy_meta, legacy_data):
    legacy = target == 9
    summary_path, manifest_path, states_path = [directory / name for name in
                                               ("summary.json", "manifest.json", "states.jsonl")]
    summary, meta = read_json(summary_path), read_json(manifest_path)
    check(summary["status"] == "complete", "Summary is not complete")
    check(summary["method"] == meta["method"] == method, "Method mismatch")
    check(meta["seed"] == 0, "Wrong seed")
    check(summary["protocol_sha256"] == meta["protocol_sha256"], "Summary protocol mismatch")
    protocol_meta = {k: v for k, v in meta.items() if k != "protocol_sha256"}
    check(canonical_hash(protocol_meta) == meta["protocol_sha256"], "Manifest protocol hash mismatch")
    for key in COHORT_KEYS:
        check(meta[key] == legacy_meta[key], f"Cohort mismatch: {key}")
    config = meta["config"]
    check(config["fisher_samples"] == 256 and config["subset_seed"] == 20260920,
          "Fisher sample scope mismatch")
    check(config["training_seed"] == 0, "Config training seed mismatch")
    if not legacy:
        check(summary["target_task"] == target, "Target mismatch")
        check(summary["samples"] == 256 and summary["fisher_scope"] == "subset256",
              "Summary Fisher scope mismatch")
        check(summary["numerical_gate_passed"] is True, "Numerical gate failed")
        data_path = directory.parent / "data_manifest.json"
        data = validate_data_manifest(data_path, meta["data_manifest_sha256"])
        check(data == legacy_data, "Data/subset membership differs from historical cohort")
        check(meta["path_fisher_scope"] == "subset256" and meta["fisher_samples"] == 256,
              "Manifest Fisher scope mismatch")
        check(meta["path_task"] == target and summary["display_T"] == target + 1, "Stage mapping mismatch")
        check(meta["fisher_sample_ids"] == data["subset_ids"]
              and meta["fisher_sample_ids_sha256"] == canonical_hash(data["subset_ids"])
              and meta["fisher_scope_tensor_sha256"] == data["subset_image_tensor_sha256"],
              "Manifest Fisher sample identity mismatch")
        check(meta["legacy_compatibility"]["passed"] is True, "Runner legacy compatibility failed")
        check(meta["production_precision"] == "FP32 model/gradients, FP64 scalar reductions; TF32 off",
              "Production precision differs")
        check(meta["fisher_definition"] == "mean per-example true-label CE gradient outer product in effective Q/V; fixed old20 head; no normalization/floor/regularizer",
              "Fisher definition differs")
    else:
        data_path = RAW / method / "data_manifest.json"
        check(meta["production_precision"] == legacy_meta["production_precision"], "Legacy precision mismatch")
    direction_name = method if legacy else f"{method}_t{target:02d}"

    stage = trajectory.query("method == @method").set_index("task")
    for task in (0, target):
        check(meta["checkpoints"][method][str(task)] == stage.loc[task, "checkpoint"],
              f"Checkpoint path differs: task{task}")
        check(meta["checkpoint_sha256"][method][str(task)] == stage.loc[task, "checkpoint_sha256"],
              f"Checkpoint hash differs: task{task}")
    drift = float(summary["drift"])
    check(drift > 0, "A non-anchor chord must have positive drift")
    close(drift, stage.loc[target, "r_T"], "Historical displacement")
    close(drift, meta["actual_drift"][direction_name], "Manifest displacement")
    integral, sensitivity = float(summary["integral"]), float(summary["S_path"])
    check(integral >= 0 and sensitivity >= 0, "Negative Fisher quadratic form")
    close(sensitivity, integral / drift**2, "S_path = I/r^2")
    count = int(summary["selected_nodes"])
    check(count in (9, 17, 33), "Unsupported quadrature node count")
    path = summary["path"]
    loss_key = "subset_loss" if legacy else "fisher_scope_loss_for_bound"
    check(len(path) == count, "Selected path node count mismatch")
    check([p["s"] for p in path] == np.linspace(0, 1, count).tolist(), "Invalid Simpson node grid")
    states = [json.loads(line) for line in states_path.read_text().splitlines() if line.strip()]
    check(len({s["state_id"] for s in states}) == len(states), "Duplicate state IDs")
    by_amount = {s["amount"]: s for s in states}
    check(len(by_amount) == len(states), "Duplicate path positions")
    check(summary["states_completed"] == len(states), "Committed state count mismatch")
    flat_nodes = []
    for node in path:
        state = by_amount[node["s"]]
        check(state["protocol_sha256"] == meta["protocol_sha256"], "State protocol mismatch")
        for key in ("coordinate_sha256", "data_manifest_sha256"):
            check(state[key] == meta[key], f"State {key} mismatch")
        check(state["method"] == method, "State method mismatch")
        check(state["checkpoint_sha256"] == meta["checkpoint_sha256"][method], "State checkpoints mismatch")
        fisher = state["fisher"] if legacy else state["fisher_by_scope"]["subset"]
        check(fisher["samples"] == 256 and fisher["normalized"] is False
              and fisher["floor"] is None and fisher["regularizer"] is None,
              "Fisher estimator scope differs")
        close(node["q"], fisher["q"][direction_name], "Node q vs journal")
        close(node["raw_trace"], fisher["raw_trace"], "Node trace vs journal")
        close(node[loss_key], fisher["subset_loss"], "Node subset loss vs journal")
        if not legacy:
            check(state["anchor_restored_exactly"] is True, "State restoration failed")
            close(state["effective_weight_drift"], node["s"] * drift,
                  "Realized chord displacement", atol=1e-5, rtol=2e-6)
        if node["s"] > 0:
            check(state["point"]["direction_sha256"] == meta["actual_direction_sha256"][direction_name],
                  "State chord direction differs")
        check(node["metrics"] == state["metrics"], "Node metrics vs journal mismatch")
        check(node["q"] >= 0 and node["raw_trace"] >= 0, "Negative raw Fisher measurement")
        for split, n in (("train", 1259), ("test", 307)):
            check(node["metrics"][split]["samples"] == n, f"Wrong {split} metric scope")
            metrics = node["metrics"][split]
            check(np.isfinite(metrics["loss"]) and metrics["loss"] >= 0
                  and np.isfinite(metrics["accuracy"]) and 0 <= metrics["accuracy"] <= 100,
                  "Invalid measured endpoint metrics")
        flat_nodes.append({"method": method, "seed": 0, "target_task": target, "T": target + 1,
                           "s": node["s"], "q": node["q"], "raw_trace": node["raw_trace"],
                           "subset_loss": node[loss_key],
                           "train_loss": node["metrics"]["train"]["loss"],
                           "test_loss": node["metrics"]["test"]["loss"],
                           "test_accuracy": node["metrics"]["test"]["accuracy"],
                           "samples": 256, "fisher_scope": "subset256",
                           "point_effective_sha256": state.get("point_effective_sha256"),
                           "source_states": str(states_path)})
    if legacy:
        check(by_amount[0.0]["point_effective_sha256"] == meta["anchor_effective_sha256"],
              "Origin is not the method's Task0 anchor")
    else:
        check(summary["end_restoration"]["effective_sha256"] == meta["anchor_effective_sha256"],
              "Final anchor restoration differs")
        close(by_amount[0.0]["effective_weight_drift"], 0.0, "Zero origin displacement")
    q = np.asarray([p["q"] for p in path])
    simpson = lambda values: float((values[0] + values[-1] + 4 * values[1:-1:2].sum()
                                   + 2 * values[2:-1:2].sum()) / (3 * (len(values) - 1)))
    close(integral, simpson(q), "Simpson integral from measured nodes")
    for n, recorded in summary["integrals_simpson"].items():
        n = int(n)
        check(n in (5, 9, 17, 33) and n <= count, "Invalid Simpson refinement")
        close(recorded if legacy else recorded["q_integral"],
              simpson(q[::(count - 1) // (n - 1)]), f"Simpson {n} nodes")
    fine, coarse = simpson(q), simpson(q[::2])
    difference = abs(fine - coarse)
    damage = path[-1][loss_key] - path[0][loss_key]
    if legacy:
        relative = difference / max(abs(integral), 1e-12)
        close(relative, summary["quadrature_relative_difference"], "Quadrature relative difference")
        stable = relative <= .05
        bound_tolerance = max(1e-8, .02 * abs(integral))
        bound_key = "bound_passed_with_2pct_numeric_tolerance"
    else:
        relative = difference / max(abs(fine), abs(coarse), 1e-30)
        check(float(config.get("quadrature_atol", 1e-8)) == 1e-8
              and float(config.get("quadrature_rtol", .05)) == .05
              and float(config.get("bound_rtol", .02)) == .02,
              "Numerical tolerances differ from the declared protocol")
        tolerance = float(config.get("quadrature_atol", 1e-8)) + float(config.get("quadrature_rtol", .05)) * max(abs(fine), abs(coarse))
        bound_tolerance = max(1e-8, float(config.get("bound_rtol", .02)) * abs(fine))
        stable = difference <= tolerance
        gate = summary["quadrature_gates"][-1]
        check(gate["nodes"] == count and gate["passed"] is True and gate["stable"] is True,
              "Final quadrature gate mismatch")
        close(relative, gate["relative_difference"], "Quadrature relative difference")
        close(tolerance, gate["stability_tolerance"], "Quadrature tolerance")
        close(bound_tolerance, gate["bound_tolerance"], "Bound tolerance")
        trace_integral = simpson(np.asarray([node["raw_trace"] for node in path]))
        close(summary["trace_integral"], trace_integral, "Trace integral")
        close(summary["trace_upper_bound"], drift**2 * trace_integral, "Trace upper bound")
        check(integral <= drift**2 * trace_integral + max(1e-8, 1e-5 * abs(drift**2 * trace_integral)),
              "Directional integral exceeds trace upper bound")
        bound_key = "bound_passed_with_numeric_tolerance"
    check(summary["quadrature_converged"] is True and stable, "Quadrature not converged")
    close(summary["D_subset" if legacy else "D_fisher_scope"], damage, "Scope-matched damage")
    close(summary["D_subset_squared" if legacy else "D_fisher_scope_squared"], damage**2,
          "Scope-matched squared damage")
    check(summary[bound_key] is True and damage**2 <= integral + bound_tolerance, "Scope-matched bound failed")
    close(summary["bound_gap_I_minus_D2"], integral - damage**2, "Bound gap")

    recovery = {}
    for endpoint, task in ((0, 0), (-1, target)):
        for split in ("train", "test"):
            for metric in ("loss", "accuracy"):
                value = path[endpoint]["metrics"][split][metric]
                reference = float(stage.loc[task, f"{split}_local_{metric}"])
                close(value, reference, f"Endpoint {task} {split} {metric}", atol=1e-5, rtol=0)
                recovery[f"task{task}_{split}_{metric}_abs_error"] = abs(value - reference)
    if not legacy:
        check(summary["endpoint_recovery"]["passed"] is True, "Runner endpoint recovery failed")
        check(summary["origin_metrics"] == path[0]["metrics"]
              and summary["endpoint_metrics"] == path[-1]["metrics"], "Endpoint metric receipt mismatch")
        receipt = summary["endpoint_recovery"]
        check(receipt["task"] == target and receipt["checkpoint"] == stage.loc[target, "checkpoint"]
              and receipt["checkpoint_sha256"] == stage.loc[target, "checkpoint_sha256"],
              "Endpoint recovery source mismatch")
        check(receipt["effective_delta_sha256"] == meta["actual_direction_sha256"][direction_name],
              "Endpoint recovery direction mismatch")
        check(np.isfinite(receipt["max_logit_error"]) and 0 <= receipt["max_logit_error"] < 5e-5,
              "Endpoint logit recovery failed")
        for key in ("local_loss_errors", "path_endpoint_loss_errors"):
            check(all(np.isfinite(v) and abs(v) < 3e-6 for v in receipt[key].values()),
                  "Endpoint loss recovery failed")
    damage_split = {split: path[-1]["metrics"][split]["loss"] - path[0]["metrics"][split]["loss"]
                    for split in ("train", "test")}
    if not legacy:
        for split in damage_split:
            close(summary[f"D_{split}"], damage_split[split], f"D_{split}")
    row = {"method": method, "display_name": LABELS[method], "seed": 0, "dataset": "ImageNet-A",
           "target_task": target, "T": target + 1, "fisher_scope": "subset256", "samples": 256,
           "r_T": drift, "integral": integral, "S_path": sensitivity,
           "D_fisher_scope": damage, "D_fisher_scope_squared": damage**2,
           "D_train": damage_split["train"], "D_test": damage_split["test"],
           "selected_nodes": count, "quadrature_relative_difference": relative,
           "quadrature_converged": True, "numerical_gate_passed": True,
           "bound_passed_with_numeric_tolerance": True, "bound_tolerance": bound_tolerance,
           "endpoint_recovery_passed": True,
           "endpoint_max_loss_abs_error": max(v for k, v in recovery.items() if "loss" in k),
           "checkpoint": stage.loc[target, "checkpoint"],
           "checkpoint_sha256": stage.loc[target, "checkpoint_sha256"],
           "anchor_checkpoint_sha256": stage.loc[0, "checkpoint_sha256"],
           "data_manifest_sha256": meta["data_manifest_sha256"],
           "subset_image_tensor_sha256": legacy_data["subset_image_tensor_sha256"],
           "coordinate_sha256": meta["coordinate_sha256"], "head_sha256": meta["head_sha256"],
           "source_kind": "reused_legacy_T10" if legacy else "stage_extension",
           "source_summary": str(summary_path), "source_summary_sha256": sha256(summary_path),
           "protocol_sha256": meta["protocol_sha256"]}
    return row, flat_nodes, [provenance(p) for p in (summary_path, manifest_path, states_path, data_path)]


def draw_figure(trajectory, paths, checks):
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.5),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    fig.subplots_adjust(left=.065, right=.993, bottom=.205, top=.89, wspace=.5)
    specs = [("r_T", r"Displacement $r_T$", (-.065, 1.85)),
             ("test_local_loss_delta_recomputed", "Signed Task0 CE change", (-.035, .34)),
             ("test_local_accuracy", "Task0 accuracy (%)", (78.5, 94)),
             ("S_path", r"Chord sensitivity $\bar S_T$", None)]
    for index, (ax, (column, ylabel, limits)) in enumerate(zip(axes, specs)):
        ax.text(0, 1.045, f"({'abcd'[index]})", transform=ax.transAxes,
                fontsize=9, va="bottom", ha="left")
        data = trajectory if index < 3 else paths
        for method in METHODS:
            rows = data.query("method == @method").sort_values("T")
            expected = list(range(1, 11)) if index < 3 else list(range(2, 11))
            check(rows["T"].tolist() == expected, "Incomplete plotted curve")
            line, = ax.plot(rows["T"], rows[column], color=COLORS[method], marker=MARKERS[method],
                            linestyle=LINES[method], linewidth=1.3, markersize=3.1,
                            label=LABELS[method], zorder=3)
            np.testing.assert_array_equal(line.get_xdata(), rows["T"].to_numpy())
            np.testing.assert_array_equal(line.get_ydata(), rows[column].to_numpy())
            if index < 2:
                ax.plot([1], [rows.iloc[0][column]], linestyle="none", marker=MARKERS[method],
                        markersize=5.3 if method == "sgd" else 3.1, markerfacecolor="white",
                        markeredgecolor=COLORS[method], markeredgewidth=1.2, zorder=4)
        ax.set_xlim(.65, 10.35)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel(r"Learning stage $T$")
        ax.set_ylabel(ylabel, labelpad=4)
        ax.tick_params(axis="both", length=3, pad=3)
        if limits:
            check(data[column].min() > limits[0] and data[column].max() < limits[1], "Y-axis clipping")
            ax.set_ylim(*limits)
        else:
            upper = max(float(paths.S_path.max()) * 1.38, .05)
            ax.set_ylim(0, upper)
            # Matplotlib's automatic locator also creates ticks beyond ylim;
            # keep only visible ticks for export and text-boundary validation.
            ax.set_yticks([tick for tick in ax.get_yticks() if 0 <= tick <= upper])
        ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=1,
                  handlelength=1.3, handletextpad=.35, borderaxespad=.35, labelspacing=.35)
    axes[0].set_yticks([0, .5, 1, 1.5])
    axes[1].set_yticks([0, .1, .2, .3])
    axes[1].axhline(0, color=".7", linewidth=.7, zorder=0)
    axes[2].set_yticks([80, 85, 90])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        box, bounds = ax.get_legend().get_window_extent(renderer), ax.get_window_extent(renderer)
        check(bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1), "Legend outside axes")
        for line in ax.lines:
            check(not line.get_path().transformed(line.get_transform()).intersects_bbox(box, filled=False),
                  "Legend overlaps a measured curve")
        ticks = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
        check(all(not a.overlaps(b) for a, b in zip(ticks, ticks[1:])), "Overlapping stage tick labels")
    checks.update({"figure_size_inches": [7.6, 2.5], "minimum_native_font_size_points": 8,
                   "layout": "one row, four columns", "display_names": LABELS,
                   "panels": [s[0] for s in specs], "T1_S_path": "undefined; no point drawn",
                   "all_four_legends_inside_axes_without_curve_overlap": True,
                   "fourth_panel_method_legend_two_rows": True,
                   "footer_or_dataset_annotation_in_image": False,
                   "smoothing_clipping_monotonic_enforcement": False,
                   "script": provenance(Path(__file__)),
                   "selected_nodes_observed": sorted(paths.selected_nodes.unique().tolist())})
    save_figure(fig, "Figure_C_sequence_stage_paths", checks)


def write_report(paths, completeness):
    lines = ["# ImageNet-A seed0 stage chord results", "",
             f"Status: **{completeness['status']}**; validated {len(paths)}/18 paths "
             f"({completeness['new_validated']}/16 new, {completeness['legacy_validated']}/2 legacy).", "",
             "Each target uses its own straight effective-weight chord from the method's Task0 anchor. "
             "Fisher uses the same fixed 256 Task0 training examples and fixed 20-class head. "
             "S_path = I/r_T²; T1 has zero displacement and undefined S_path, so it is omitted. "
             "The first three panels retain the original 20 measured checkpoint rows. "
             "No missing stage is interpolated or imputed.", "",
             "Numerical validation recomputes Simpson quadrature, S_path, the same-subset loss bound, "
             "and endpoint recovery against the original checkpoint trajectory. Full train/test damage "
             "is reported separately and is not substituted into the subset bound. "
             "Legacy T10 gates are derived from their original quadrature/bound receipts; endpoint "
             "recovery is recomputed from their measured nodes. Checkpoint identity is checked against "
             "the existing hashes; this CPU collector does not rehash the large weight files.", ""]
    if len(paths):
        lines += ["| Method | T | r_T | I | S_path | Nodes |", "|---|---:|---:|---:|---:|---:|"]
        for row in paths.itertuples(index=False):
            lines.append(f"| {row.display_name} | {row.T} | {row.r_T:.8g} | {row.integral:.8g} | "
                         f"{row.S_path:.8g} | {row.selected_nodes} |")
    if completeness["issues"]:
        lines += ["", "Pending or invalid paths:", ""]
        lines += [f"- {item['method']} / task{item['target_task']}: {item['reason']}"
                  for item in completeness["issues"]]
    lines += ["", "One seed per method; these curves do not estimate between-seed uncertainty.", ""]
    (REPORT / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true", help="Write progress CSVs without drawing incomplete curves")
    parser.add_argument("--no-figure", action="store_true", help="Validate and collect only")
    args = parser.parse_args()
    REPORT.mkdir(exist_ok=True)
    trajectory, _, checks = load_verified_data()
    all_rows, all_nodes, sources, issues = [], [], [], []
    for method in METHODS:
        legacy_meta = read_json(RAW / method / "path" / "manifest.json")
        legacy_data = validate_data_manifest(RAW / method / "data_manifest.json", legacy_meta["data_manifest_sha256"])
        for target in range(1, 10):
            directory = RAW / method / "path" if target == 9 else FIG_DIR / "results" / method / f"path_t{target:02d}"
            try:
                row, nodes, receipts = validate_path(method, target, directory, trajectory, legacy_meta, legacy_data)
                all_rows.append(row)
                all_nodes.extend(nodes)
                sources.extend(receipts)
            except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
                issues.append({"method": method, "target_task": target, "T": target + 1,
                               "reason": f"{type(error).__name__}: {error}", "directory": str(directory)})
    paths = pd.DataFrame(all_rows)
    complete = len(all_rows) == 18 and not issues
    completeness = {"status": "complete" if complete else "partial", "expected_paths": 18,
                    "validated_paths": len(all_rows), "new_validated": sum(r["target_task"] < 9 for r in all_rows),
                    "legacy_validated": sum(r["target_task"] == 9 for r in all_rows),
                    "T1_S_path": None, "T1_S_path_reason": "undefined: r_T = 0",
                    "issues": issues, "checkpoint_weight_files_rehashed": False,
                    "source_hashes": sources, "figure_generated_this_run": False}
    paths.to_csv(REPORT / "C_stage_path.csv", index=False)
    pd.DataFrame(all_nodes).to_csv(REPORT / "path_nodes.csv", index=False)
    write_report(paths, completeness)
    checks.update({"stage_path_completeness": {k: v for k, v in completeness.items()
                                               if k != "figure_generated_this_run"},
                   "source_hashes": checks["source_hashes"] + sources,
                   "Sbar": "I/r_T^2 on separate Task0-to-target chords; fixed train256; adaptive Simpson 9/17/33 nodes"})
    if complete and not args.no_figure:
        draw_figure(trajectory, paths, checks)
        completeness["figure_generated_this_run"] = True
        used = ", ".join(map(str, sorted(paths.selected_nodes.unique())))
        caption = ("Figure C. ImageNet-A, seed0. (a) Actual joint effective-Q/V displacement "
                   r"$r_T=\|\Delta_T\|_2$, where $\Delta_T=w_T-w_1$, from each method's own Task0 anchor; "
                   "(b) signed Task0 test cross-entropy change relative to that same anchor; "
                   "(c) Task0 test accuracy. Panels (b,c) use each method's fixed Task0 20-class head "
                   "on all 307 Task0 test examples. (d) Chord-normalized directional empirical-Fisher "
                   r"sensitivity, $\bar S_T=I_T/r_T^2$, where "
                   r"$I_T=\int_0^1\Delta_T^\top F_0^{\mathrm{emp}}(w_1+s\Delta_T)\Delta_T\,ds$, "
                   r"and $F_0^{\mathrm{emp}}$ is the unnormalized, true-label, per-example empirical "
                   "Fisher on the same fixed 256 Task0 training examples for every stage. "
                   "T is the number of learned tasks, with T=1 denoting Task0. Each point in (d) integrates "
                   "a separate straight chord from Task0 to that checkpoint, not the optimizer-step trajectory. "
                   f"Adaptive Simpson quadrature uses 9, 17, or 33 measured nodes (observed counts: {used}). "
                   r"T1 sensitivity is undefined because $r_1=0$ and is not plotted. One seed per method; "
                   "no error bars across seeds.\n")
        (FIG_DIR / "Figure_C_sequence_stage_paths_caption.md").write_text(caption)
        (FIG_DIR / "Figure_C_sequence_stage_paths_include.tex").write_text(
            "\\begin{figure*}[t]\n\\centering\n"
            "\\includegraphics[width=\\textwidth]{Figure_C_sequence_stage_paths.pdf}\n"
            "\\caption{" + caption.replace("Task0", "Task~0").strip() + "}\n"
            "\\label{fig:sequence-stage-paths}\n\\end{figure*}\n")
    (REPORT / "completeness.json").write_text(json.dumps(completeness, indent=2) + "\n")
    print(json.dumps({k: v for k, v in completeness.items() if k not in ("source_hashes", "issues")}, indent=2))
    if not complete and not args.allow_partial:
        raise SystemExit("Incomplete or invalid paths; see report/completeness.json. Figure was not generated.")


if __name__ == "__main__":
    main()
