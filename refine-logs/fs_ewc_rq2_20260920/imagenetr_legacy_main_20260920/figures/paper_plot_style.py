"""CPU-only plotting and provenance for existing ImageNet-A seed0 checkpoints."""
from pathlib import Path
import hashlib
import json
import os
import shutil

FIG_DIR = Path(__file__).resolve().parent
EXECUTION = FIG_DIR.parent.parent / "execution"
REPORT = EXECUTION / "report_seed0"
RAW = EXECUTION / "results_seed0_recovery1"
SOURCE_DIR = FIG_DIR / "source_data"
SOURCE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(FIG_DIR / ".mplcache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd

METHODS = ("sgd", "gam")
COLORS = {"sgd": "#0072B2", "gam": "#D55E00"}
MARKERS = {"sgd": "o", "gam": "D"}
LINES = {"sgd": "-", "gam": (0, (4, 2))}
LABELS = {"sgd": "SGD → SGD + EWC", "gam": "GAM → GAM + EWC"}
PROVENANCE = "ImageNet-A · existing seed0 checkpoints"
STAGES = r"$T=1$: Task0 anchor (open markers)    |    $T=2$–$10$: subsequent learning"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 10,
    "figure.dpi": 140, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .8, "lines.linewidth": 1.6,
    "text.usetex": False, "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_data():
    """Verify all plotted values against immutable reports and raw JSON summaries."""
    trajectory_path = REPORT / "C_current_FP32_task_trajectory.csv"
    path_summary_path = REPORT / "C_path_summary.csv"
    config_path = EXECUTION / "diagnostic_config_recovery1.json"
    config = json.loads(config_path.read_text())
    assert Path(config["data_root"]).name == "imagenet-a"
    assert config["fisher_samples"] == 256 and config["path_task"] == 9
    trajectory = pd.read_csv(trajectory_path, float_precision="round_trip")
    paths = pd.read_csv(path_summary_path, float_precision="round_trip")
    assert len(trajectory) == 20 and len(paths) == 2
    assert set(trajectory.method) == set(paths.method) == set(METHODS)
    assert set(trajectory.seed) == set(paths.seed) == {0}
    assert set(trajectory.precision) == {"current full FP32"}
    assert set(trajectory.test_local_classes) == {20}
    assert set(trajectory.test_local_samples) == {307}
    assert not trajectory.duplicated(["method", "seed", "task"]).any()
    assert not paths.duplicated(["method", "seed"]).any()
    assert paths.numerical_gate_passed.all() and paths.quadrature_converged.all()
    assert paths.bound_passed_with_2pct_numeric_tolerance.all()
    assert set(paths.nodes) == {9}
    sources = [trajectory_path, path_summary_path, config_path]
    methods = {}
    for method in METHODS:
        d = trajectory.query("method == @method").sort_values("task")
        assert list(d.task) == list(range(10))
        assert d.iloc[0].delta_norm_cpu_recomputed == 0
        assert np.isfinite(d[["test_local_accuracy", "test_local_loss_delta_recomputed",
                                "delta_norm_cpu_recomputed"]].to_numpy()).all()
        np.testing.assert_allclose(d.test_local_loss_delta_recomputed,
                                   d.test_local_loss - d.iloc[0].test_local_loss,
                                   rtol=0, atol=1e-14)
        raw_trajectory_path = RAW / method / "trajectory" / "summary.json"
        raw_path_path = RAW / method / "path" / "summary.json"
        sources.extend([raw_trajectory_path, raw_path_path])
        raw_trajectory = json.loads(raw_trajectory_path.read_text())
        raw_path = json.loads(raw_path_path.read_text())
        assert raw_trajectory["status"] == raw_path["status"] == "complete"
        assert raw_trajectory["task_count"] == len(raw_trajectory["tasks"]) == 10
        raw_tasks = {row["task"]: row for row in raw_trajectory["tasks"]}
        for row in d.itertuples(index=False):
            raw = raw_tasks[row.task]
            assert row.checkpoint == raw["checkpoint"] == config["checkpoints"][method][str(row.task)]
            assert row.checkpoint_sha256 == raw["checkpoint_sha256"]
            np.testing.assert_array_equal(
                [row.delta_norm_cpu_recomputed, row.test_local_loss_delta_recomputed,
                 row.test_local_accuracy, row.test_local_loss],
                [raw["effective_weight_drift"], raw["D0"]["test"],
                 raw["metrics"]["test"]["local"]["accuracy"],
                 raw["metrics"]["test"]["local"]["loss"]])
        p = paths.query("method == @method").iloc[0]
        np.testing.assert_allclose(p.I / p.drift ** 2, p.S_path, rtol=1e-12)
        np.testing.assert_array_equal(
            [d.iloc[-1].delta_norm_cpu_recomputed, p.drift, p.I, p.S_path],
            [raw_path["drift"], raw_path["drift"], raw_path["integral"], raw_path["S_path"]])
        np.testing.assert_array_equal([node["s"] for node in raw_path["path"]], np.linspace(0, 1, 9))
        q = np.array([node["q"] for node in raw_path["path"]])
        integral = (q[0] + q[-1] + 4 * q[1:-1:2].sum() + 2 * q[2:-1:2].sum()) / 24
        np.testing.assert_allclose(integral, p.I, rtol=1e-12)
        damage = d.test_local_loss_delta_recomputed.to_numpy()
        methods[method] = {
            "actual_row_count": len(d), "loss_peak_T": int(np.argmax(damage) + 1),
            "loss_peak": float(damage.max()), "loss_endpoint": float(damage[-1]),
            "loss_decrease_arrival_T": (np.flatnonzero(np.diff(damage) < 0) + 2).tolist(),
            "negative_damage_T": (np.flatnonzero(damage < 0) + 1).tolist(),
            "endpoint_r_T": float(d.iloc[-1].delta_norm_cpu_recomputed),
            "endpoint_Sbar": float(p.S_path),
            "quadrature_relative_difference": float(p.quadrature_relative_difference),
        }
    assert methods["gam"]["loss_peak_T"] == 7
    assert methods["gam"]["loss_decrease_arrival_T"][-3:] == [8, 9, 10]
    assert methods["sgd"]["loss_decrease_arrival_T"] == [2, 5]
    # Copies retain the input bytes; enriched source CSVs retain round-trip precision.
    for path in [trajectory_path, path_summary_path, config_path]:
        shutil.copyfile(path, SOURCE_DIR / path.name)
        assert sha256(path) == sha256(SOURCE_DIR / path.name)
    trajectory = trajectory.assign(
        dataset="ImageNet-A", T=trajectory.task + 1,
        learning_stage=np.where(trajectory.task == 0, "Task0 anchor", "subsequent learning"),
        r_T=trajectory.delta_norm_cpu_recomputed,
        r_T_definition="joint effective-weight distance from own Task0 anchor")
    paths = paths.assign(dataset="ImageNet-A", endpoint_T=10, endpoint_task_code=9,
                         Sbar=paths.S_path, fisher_split="train", fisher_samples=256,
                         path_kind="Task0-to-Task9 effective-weight straight chord")
    trajectory.to_csv(SOURCE_DIR / "C_stage_trajectory.csv", index=False)
    paths.to_csv(SOURCE_DIR / "C_T10_chord_sensitivity.csv", index=False)
    checks = {
        "status": "passed", "dataset_verified_from_config": "ImageNet-A",
        "training_seeds": [0], "training_runs_per_method": 1,
        "all_20_rows_match_raw_JSON_exactly": True,
        "checkpoint_path_and_hash_match_raw_JSON": True,
        "checkpoint_weight_files_rehashed": False,
        "fixed_Task0_head_classes": 20, "Task0_test_samples": 307,
        "T_definition": "total learned tasks = task_code + 1; T1 is Task0 anchor",
        "source_bytes_preserved": True, "signed_damage_preserved": True,
        "smoothing_clipping_monotonic_enforcement": False,
        "r_T": "actual cumulative distance from own Task0 effective weights, not sum of step lengths or a damage-tolerance radius",
        "Sbar": "T10 only; I/r_T^2 on Task0-to-Task9 straight chord; training n=256; 9 Simpson nodes",
        "Sbar_and_integral_match_raw_JSON": True,
        "Simpson_integral_recomputed_from_raw_nodes": True,
        "both_reported_path_numerical_gates_pass": True,
        "optimizer_sequences": {"sgd": "SGD to SGD + EWC", "gam": "GAM to GAM + EWC"},
        "methods": methods,
        "source_hashes": [{"path": str(p), "sha256": sha256(p)} for p in sources],
        "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    return trajectory, paths, checks


def stage_axis(ax):
    ax.set_xlim(.65, 10.35)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel(r"Total learned tasks, $T$")
    ax.axvline(1.5, color=".65", linestyle=(0, (2, 3)), linewidth=.7, zorder=0)
    ax.get_xticklabels()[0].set_fontweight("bold")


def draw_trajectory(ax, trajectory, column):
    lines = []
    for method in METHODS:
        d = trajectory.query("method == @method").sort_values("T")
        line, = ax.plot(d["T"], d[column], color=COLORS[method],
                        linestyle=LINES[method], marker=MARKERS[method], markersize=4,
                        markevery=list(range(1, 10)), zorder=2)
        # Co-located zero anchors: a larger circle surrounds a smaller diamond.
        ax.plot([1], [d.iloc[0][column]], linestyle="none", marker=MARKERS[method],
                markersize=7 if method == "sgd" else 4, markerfacecolor="white",
                markeredgecolor=COLORS[method], markeredgewidth=1.3, zorder=4)
        np.testing.assert_array_equal(line.get_xdata(), d["T"].to_numpy())
        np.testing.assert_array_equal(line.get_ydata(), d[column].to_numpy())
        lines.append(line)
    stage_axis(ax)
    return lines


def legend_handles():
    return [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle=LINES[m],
                   label=LABELS[m], markersize=4) for m in METHODS]


def save_figure(fig, name, checks):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for obj in fig.findobj(match=Text):
        if not obj.get_visible() or not obj.get_text().strip():
            continue
        box = obj.get_window_extent(renderer=renderer)
        if box.width == 0 or box.height == 0:
            continue
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            outside.append(obj.get_text())
    assert not outside, f"Text outside figure canvas: {outside}"
    assert all(not ax.get_title() for ax in fig.axes)
    for suffix in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"{name}.{suffix}", dpi=300)
    checks.update({
        "figure": name, "all_plotted_line_coordinates_match_source_rows": True,
        "text_extents_inside_canvas": True, "axes_titles_empty": True,
        "rendered_visual_review": "pending root inspection and independent figure review",
        "outputs": {suffix: {"path": str(FIG_DIR / f"{name}.{suffix}"),
                             "sha256": sha256(FIG_DIR / f"{name}.{suffix}")}
                    for suffix in ["pdf", "png"]},
        "output_format": "vector PDF with embedded text; PNG at 300 dpi",
    })
    (FIG_DIR / f"{name}_validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    plt.close(fig)
