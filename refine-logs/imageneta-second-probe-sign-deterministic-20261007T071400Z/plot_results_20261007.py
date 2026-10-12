"""Plot only the completely validated r=1 slice; no partial endpoints."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
record = json.loads((ROOT / "results.json").read_text())
rows = {r["variant"]: r for r in record["rows"] if r["accepted"] and r["radius"] == 1.0}
assert set(rows) == {"B0", "B1", "N1", "N2", "N3", "N4", "N5", "N6"}
assert all(r["tasks_completed"] == 10 and not r["reused"] for r in rows.values())
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})

groups = [(r"$g_0$ anchor", [["B0", "N1"], ["N3", "N5"]]),
          (r"$g_1$ anchor", [["B1", "N2"], ["N4", "N6"]])]
fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.5), constrained_layout=True)
for i, (anchor, grid) in enumerate(groups):
    for j, metric in enumerate(("AAA", "FAA")):
        ax = axes[i, j]
        values = np.array([[rows[v][metric] for v in line] for line in grid])
        im = ax.imshow(values, vmin=0, vmax=70 if metric == "AAA" else 55,
                       cmap="viridis", aspect="equal")
        ax.set_xticks([0, 1], ["+1", "−1"])
        ax.set_yticks([0, 1], ["+1", "−1"])
        ax.set_xlabel("Residual coefficient")
        ax.set_ylabel("Third probe sign")
        ax.set_title(f"{anchor}: {metric}")
        for y in range(2):
            for x in range(2):
                color = "white" if values[y, x] < 30 else "black"
                ax.text(x, y, f"{grid[y][x]}\n{values[y, x]:.3f}%", color=color,
                        ha="center", va="center", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label=f"{metric} (%)")
fig.suptitle("ImageNet-A: second local probe and residual sign\nr=1, rho=0.05, rank10, seed2; 8/8 validated endpoints", fontsize=12)
fig.savefig(ROOT / "SECOND_PROBE_R1_RESULTS_20261007.pdf", bbox_inches="tight")
fig.savefig(ROOT / "SECOND_PROBE_R1_RESULTS_20261007.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)
colors = ["#2864A5", "#CD4E46", "#E6A33D", "#2A8E69"]
for ax, (anchor, grid) in zip(axes, groups):
    for variant, color in zip([grid[0][0], grid[0][1], grid[1][0], grid[1][1]], colors):
        row = rows[variant]
        epochs = json.loads((Path(row["source_summary"]).parent / "metrics.json").read_text())["epochs"]
        task0 = [e for e in epochs if e["task"] == 0]
        assert len(task0) == 50
        assert all(np.isfinite(e["train_ce"]) for e in task0)
        ax.plot([e["epoch"] for e in task0], [e["train_ce"] for e in task0],
                color=color, label=variant, linewidth=1.7)
    ax.set_yscale("log")
    ax.set_xlabel("Task 0 epoch")
    ax.set_ylabel("Mean training CE (log scale)")
    ax.set_title(anchor)
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
fig.suptitle("Learning collapse starts within the first task\nr=1, seed2; observed losses remain finite", fontsize=12)
fig.savefig(ROOT / "SECOND_PROBE_TASK0_STABILITY_20261007.pdf", bbox_inches="tight")
fig.savefig(ROOT / "SECOND_PROBE_TASK0_STABILITY_20261007.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Generated r=1 endpoint heatmaps and Task0 CE curves from 8 validated runs.")
