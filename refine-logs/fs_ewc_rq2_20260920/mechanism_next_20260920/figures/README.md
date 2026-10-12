## 当前 B 图：训练均值三图版（2026-09-21）

当前 B 图为 Times New Roman 的一行三列：平均正损伤、训练准确率、平均 raw Fisher trace。已按用户要求删除第四幅容忍半径图。所有轴为线性，横轴为 Weight drift，仅显示训练均值。前三图来自同批历史 ImageNet-A seed0 扰动状态；loss/accuracy 使用 1259 个训练样本，Fisher 使用其中固定的 256 个样本。两处 Figure_B_neighborhood / Figure_B_neighborhood_merged 输出一致。最新图注见 B_caption.md（figures_redesign_20260920 目录），最新来源核查见 B_THREE_PANEL_SOURCES.md；下方早期记录与历史审查对应各归档版本。


Reproduce the figures and tables from any working directory:

```bash
python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/mechanism_next_20260920/figures/regenerate_all.py
```

The three independent generators use the shared `paper_plot_style.py` and require Python, NumPy, pandas, and Matplotlib. They run on CPU, read immutable existing reports and raw neighborhood JSONL, and write only within this directory. Every numerical plotted value is loaded or recomputed from those sources. Output PDFs contain vector graphics and embedded TrueType fonts; PNGs are exported at 300 dpi.

- `Figure_B_neighborhood.pdf` / `.png`: three training-mean panels: positive loss damage, accuracy and raw Fisher trace. All axes are linear; x is Weight drift. All metrics share the same perturbed states per method, while Fisher uses a fixed 256-image subset of the 1259-image training evaluation set.
- `Figure_C_sequence_and_endpoint.pdf` / `.png`: three full task-checkpoint curves and one endpoint-only sensitivity comparison. The path is an effective-weight endpoint straight segment, not a training-step trajectory.
- `TABLE_A_controlled_continuation.csv` / `.md` / `.tex`: both anchors, all three class-incremental accuracies, signed diagnostic loss changes, raw anchor trace, and effective displacement. The CSV preserves all original columns, including provenance and additional diagnostic values.
- `TABLE_B_Q95_tolerance.*`: appendix percentile tolerance table with explicit right censoring.
- `TABLE_B_existing_bridge.*`: appendix table of the pre-existing historical-sequence bridge. Its direction source is separate from the new controlled continuation in A.
- `source_data/`: full-precision derived and copied numeric sources, including every neighborhood direction state and raw-trace summaries.
- `CAPTIONS.md`, `captions.json`, and `latex_includes.tex`: self-contained captions and editable LaTeX integration. The LaTeX assumes `figures/` is next to the manuscript and requires `graphicx` and `booktabs`.
- `*_provenance.json`: original-source SHA256 hashes, library versions, and successful numerical checks.

Interpretation constraints: one training seed; 16 perturbation directions are direction statistics, not training replicates. The main Gmean tolerance radii tie on both splits at every tested threshold. GAM's test Gmean is higher at every positive measured radius despite lower train Gmean and raw trace. A retains the lower new-task accuracy of its GAM-anchor continuation and the inherited-Fisher limitation. C retains GAM's larger effective endpoint displacement. No unmeasured sensitivity trajectory or new GPU experiment is added.
