# Reference figure style audit (2026-09-20)

The requested reference is the **restyled main AA figure**, not the earlier default output of `plot_lora_aa.py`. `restyle_aa_main.py` generated the final main PDF and its sidecar `output/lora_taskwise_aa_display.json` (there is no generic `display.json`). The O-LoRA and CopyFlatCV copies of the PDF, PNG, both plotting scripts and metadata are byte-identical. Neither mirror is a newer version.

Reference PDF SHA256: `5e152c9c66be17d444cd7f0b68d30ac44d35f6dd9e9c9dc2bb96966a5428539f`.

Reference locations:

- [Final PDF](/data/140-1/users/liying/O-LoRA/figures/lora_taskwise_20260918/aa_times/output/lora_taskwise_aa.pdf)
- [Base plotting implementation](/data/140-1/users/liying/O-LoRA/figures/lora_taskwise_20260918/aa_times/plot_lora_aa.py)
- [Final restyle overrides](/data/140-1/users/liying/O-LoRA/figures/lora_taskwise_20260918/aa_times/restyle_aa_main.py)
- [Final display metadata](/data/140-1/users/liying/O-LoRA/figures/lora_taskwise_20260918/aa_times/output/lora_taskwise_aa_display.json)
- CopyFlatCV mirror root: `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/Figures/lora_taskwise_20260918/aa_times`.

## Exact final style

```python
REFERENCE_STYLE = {
    "font_family": "Times New Roman",
    "font_dir": "/data/140-1/users/liying/O-LoRA/figures/lora_taskwise_20260918/aa_times/fonts",
    "rcparams": {
        "font.family": "Times New Roman", "font.size": 11,
        # These two base defaults are overridden for the final main figure below.
        "axes.labelsize": 11, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    },
    "xlabel": {"text": "Number of learned tasks", "fontsize": 14},
    "ylabel": {"text": "Average accuracy (%)", "fontsize": 14},
    "title": {"fontsize": 13, "fontweight": "bold", "pad": 7},
    "tick_params": {"labelsize": 10, "pad": 3.5},
    "line": {"linestyle": "-", "linewidth": 1.5, "markersize": 4, "zorder": 3},
    "band": {"alpha": 0.14, "linewidth": 0, "zorder": 1},
    "grid": {"color": "#d8d8d8", "linewidth": 0.55, "alpha": 0.7},
    "legend": {
        "loc": "best", "frameon": True, "facecolor": "white",
        "edgecolor": "#dddddd", "framealpha": 0.94, "fontsize": 8.6,
        "handlelength": 1.65, "handletextpad": 0.5,
        "borderpad": 0.3, "labelspacing": 0.28,
    },
    "x_padding": 0.2, # xlim = (0.8, number_of_tasks + 0.2)
    "y_margin": 0.09,
    "y_major_locator": "MaxNLocator(nbins=5)",
    "y_limits": "auto after margins, clipped to [0, 100]",
    "axisbelow": True,
    "save": {"bbox_inches": "tight", "pad_inches": 0.055},
    "formats": ["pdf", "png", "svg"],
    "main_figsize_inches": (13.4, 3.65), # original 1x4 composition
    "main_subplots_adjust": {
        "left": 0.05, "right": 0.996, "top": 0.90,
        "bottom": 0.24, "wspace": 0.36,
    },
    "main_note": {"x": 0.515, "y": 0.025, "ha": "center", "fontsize": 9},
}

REFERENCE_METHOD_STYLES = {
    "Vanilla LoRA": {"color": "#777777", "marker": "o"},
    "InfLoRA":     {"color": "#0072B2", "marker": "s"},
    "SD-LoRA":     {"color": "#D55E00", "marker": "^"},
    "EWCLoRA":     {"color": "#009E73", "marker": "P"},
    "FS-EWC":      {"color": "#CC3377", "marker": "*"},
}
REFERENCE_LEGEND_ORDER = ["Vanilla LoRA", "InfLoRA", "SD-LoRA", "EWCLoRA", "FS-EWC"]
```

Every reference line is solid and has the same width. Marker `P` is a filled plus and `*` is a star. No customized marker face/edge colors are set, so they follow the line color. The final main legend removes the dagger symbols used by the earlier base plotting function; the explanatory text below the axes carries the single-seed qualification instead. There are no error bars.

The base script's independent-dataset figures use `figsize=(5.0,3.75)` and `subplots_adjust(left=.145,right=.98,top=.90,bottom=.23)`; note position `(x=.53,y=.028)` at 8.3pt. Those existing independent outputs were not restyled to the main figure's 14pt axis labels/bold titles. For the new requested single-dataset figures, use the **single-dataset canvas** with the **final main style overrides** above, then visually check the seven-entry legend against curves.

All integer stages are shown as x ticks: `x = np.arange(1, T + 1)`; the y-axis is not forced to begin at zero. The grids are thin, light gray and below curves. The legend is inside each axes, automatically located. Reference main figure titles are bold, centered dataset names, without subplot letters.

## Shading and metric meanings

The reference uses current-stage task-macro accuracy:

`AA_t = mean(a[t, 1:t])`.

It is neither cumulative AAA nor the sample-weighted global top1 curve. Future-task NaNs are excluded. For multiple seeds, the plotted center is the per-stage mean of per-run AA and the band is **mean ± sample SD (`ddof=1`)**, with each run's AA computed before aggregating seeds. For one seed, the reference substitutes a fixed **±0.2 percentage-point illustrative band**, explicitly not an SD, variance or confidence interval. `fill_between` uses the same color at alpha .14 and has no boundary line. It does not numerically clip band bounds; axes clip the visible area at 0–100.

Reference single-seed methods are Vanilla LoRA, InfLoRA and SD-LoRA. Reference EWCLoRA and FS-EWC have multiple seeds and real SD bands. The new table's runs do not inherit those replicate counts or SDs. If the new plots copy the single-seed ±0.2 visual band, label it as illustrative; never reuse the reference's multi-seed footnote or imply uncertainty estimated from one run. Omitting such artificial bands is also compatible with an honest single-seed plot, but would be a documented departure from the reference presentation.

## Mapping to the requested seven methods

Preserve the four unchanged method color/marker identities. The reference's **FS-EWC label is an explicit historical display alias for GAM-EWC+Merge**, not proof that the new table's GAM-EWCLoRA is that implementation. For the new table retain the actual **GAM-EWCLoRA** label, while reusing magenta `#CC3377` and star `*` as a visual identity. Do not propagate the old FS-EWC alias to a different method.

OLoRA and CL-LoRA do not exist in the reference, so their style cannot be “extracted.” A distinct extension, if needed, is OLoRA light blue `#56B4E9` with diamond `D`, and CL-LoRA purple `#7B3294` with down-triangle `v`. This is a new styling choice, not a property of the reference. Suggested table-aligned legend order:

`Vanilla LoRA, InfLoRA, OLoRA, SD-LoRA, CL-LoRA, EWCLoRA, GAM-EWCLoRA`.

## PDF verification and visual inspection

The final PDF is one vector page, approximately **13.3408 × 3.5597 inches** after tight cropping (nominal canvas 13.4×3.65). PyMuPDF reads embedded Type0 TrueType subsets `TimesNewRomanPSMT` and `TimesNewRomanPS-BoldMT`. Actual text spans confirm axes labels at14pt, title bold13pt, tick labels10pt, legend8.6pt and note9pt. The matching PNG was visually inspected: it has four wide side-by-side axes, internal legends, all solid lines and unobtrusive shading. No original figure or script was modified.
