"""Generate table fragments from audited records without mixing cohorts."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)


def read_csv(name):
    with (ROOT / name).open() as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cell(mean, decimals, sd=None):
    s = f"{mean:.{decimals}f}"
    if sd is not None:
        s += r"\pm" + f"{sd:.{decimals}f}"
    return "$" + s + "$"


v = json.loads((ROOT / "lora_log_validation.json").read_text())
assert sha(v["source_table"]) == v["source_sha256"]
for record in v["runs"]:
    assert sha(record["path"]) == record["sha256"]

legacy = {r["setting"]: r for r in read_csv("lora_summary_recomputed.csv")}
legacy_specs = [
    ("baseline", "Vanilla LoRA", "Adam", "Adam"),
    ("gam_only", "Shaping-only LoRA", "GAM", "GAM"),
    ("control", "EWCLoRA", "Adam", "Adam+EWC"),
    ("full", "Shaping + EWCLoRA", "GAM", "P-GAM+EWC"),
    ("setting1", "EWCLoRA + Task-1 shaping", "GAM", "Adam+EWC"),
    ("setting2", "EWCLoRA + Task-2 shaping", "Adam", "P-GAM+EWC"),
]
original_display = [
    ["90.73", "19.885", "84.85", "89.59", "-5.87", "22.469", "1.560"],
    ["93.34", "10.946", "84.80", "93.81", "-8.55", "10.471", "3.131"],
    ["90.73", "19.885", "87.95", "83.45", "-2.78", "24.599", "0.0106"],
    ["93.34", "10.946", "89.10", "88.33", "-4.24", "17.931", "0.0204"],
    ["93.34", "10.946", "91.82", "82.57", "-1.52", "18.734", "0.392"],
    ["90.73", "19.885", "83.07", "89.21", "-7.66", "41.316", "0.129"],
]
legacy_fields = [
    ("a11", 2), ("T1_fisher", 3), ("a21", 2), ("a22", 2),
    ("BWT", 2), ("T2_current_fisher", 3), ("r_effective", 3)
]
legacy_template = r"""% Requires \usepackage{booktabs,caption,amsmath}.
% Alternative versions with the same label should not both be included.
\begin{table*}[t]
\centering
\captionsetup{font=footnotesize,labelfont=bf}
\caption{Two-task diagnostics for the legacy fresh-adapter LoRA implementation
on ImageNet-R (rank 10). @@AGGREGATION@@.
Accuracies are percentages and BWT is in percentage points.
The update rules are listed explicitly; all GAM variants use Adam as the base
optimizer. P-GAM+EWC projects the GAM gradient against the EWC gradient separately
for each parameter tensor before adding the EWC gradient.
$\mathcal B_j=\operatorname{tr}(\widehat F^{\mathrm{batch}}_{j,\mathrm{cur},KV})$
is the current-task batch-gradient Fisher proxy trace at that task's endpoint.
It is neither a per-example empirical Fisher trace nor a cumulative buffer,
and $\mathcal B_2$ is not old-task sensitivity re-estimated after Task~2.
$r_{KV}$ is the effective K/V weight displacement between the two evaluated
endpoints, excluding classifier heads.
Both endpoints are evaluated before task-end factor accumulation.
No trajectory averaging is used.}
\label{tab:two_task_lora_diagnostics}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}
\toprule
& \multicolumn{2}{c}{Update rule}
& \multicolumn{2}{c}{Task 1 endpoint}
& \multicolumn{4}{c}{Task 2 endpoint}
& Drift \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}
Variant & Task 1 & Task 2
& $a_{11}\uparrow$ & $\mathcal B_1$
& $a_{21}\uparrow$ & $a_{22}\uparrow$ & BWT$\uparrow$ & $\mathcal B_2$
& $r_{KV}$ \\
\midrule
@@BODY@@
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{\textwidth}\scriptsize
For this fresh-adapter two-task implementation,
$r_{KV}=[\sum_{\ell}\sum_{c\in\{K,V\}}\|B^{(2)}_{\ell c,\mathrm{new}}
A^{(2)}_{\ell c,\mathrm{new}}\|_F^2]^{1/2}$.
The proxy uses
$\widehat F^{\mathrm{batch}}_{\mathrm{diag}}
=N^{-1}\sum_b |b|\,(\nabla_{W^{\mathrm{eff}}_{KV}}\overline{\ell}_b)^{\odot2}$,
with sample averaging and no trace rescaling or positive floor.
The Fisher batch size is 128; EWC accumulation uses $\gamma=1$.
BWT is $a_{21}-a_{11}$ within each seed before rounding;
each seed's norm is computed before aggregation.
Adam learning rate is $5\times10^{-4}$, weight decay is $0.005$,
and EWC uses $\lambda=10^7$ where enabled.
\end{minipage}
\end{table*}
"""
legacy_export = []
for with_sd in (False, True):
    body = []
    for idx, (key, label, stage1, stage2) in enumerate(legacy_specs):
        r = legacy[key]
        cells, means = [], []
        for field, decimals in legacy_fields:
            if field == "r_effective":
                decimals = 4 if key in ("control", "full") else 3
            mean, sd = float(r[field + "_mean"]), float(r[field + "_sd"])
            means.append(f"{mean:.{decimals}f}")
            cells.append(cell(mean, decimals, sd if with_sd else None))
        assert means == original_display[idx], (key, means)
        if idx == 4:
            body.append(r"\midrule")
        body.append(" & ".join([label, stage1, stage2] + cells) + r" \\")
        if not with_sd:
            legacy_export.append({**r, "variant": label, "stage1_update": stage1,
                                  "stage2_update": stage2})
    aggregation = ("Means and sample standard deviations over seeds 0, 1, and 2"
                   if with_sd else "Means over seeds 0, 1, and 2")
    name = "legacy_lora_mean_sd.tex" if with_sd else "legacy_lora_corrected.tex"
    (OUT / name).write_text(legacy_template.replace("@@BODY@@", "\n".join(body))
                           .replace("@@AGGREGATION@@", aggregation))
write_csv("legacy_lora_corrected_values.csv", legacy_export)

new_sources = json.loads((ROOT / "new_lora_sources.json").read_text())["imagenetr_ratio"]
new_runs = []
for src in new_sources:
    assert sha(src["metrics"]["path"]) == src["metrics"]["sha256"]
    d = json.loads(Path(src["metrics"]["path"]).read_text())
    m = d["cnn"]["matrices"]["final"]
    sq = d["mechanism"]["steps"]["1"]["delta_norm"]
    new_runs.append(dict(setting=src["setting"], seed=src["seed"],
                         a11=m[0][0], a21=m[1][0], a22=m[1][1],
                         BWT=m[1][0]-m[0][0], r=math.sqrt(sq)))
new_specs = [
    ("baseline", "Vanilla LoRA", "SGD", "SGD"),
    ("fs_only", "Ratio-FS only", "Ratio-FS", "Ratio-FS"),
    ("control", "EWCLoRA", "SGD", "SGD+EWC"),
    ("fs_ewc", "Ratio-FS + EWC", "Ratio-FS", "Ratio-FS+EWC"),
    ("factored_s1", "EWC + Task-1 Ratio-FS", "Ratio-FS", "SGD+EWC"),
    ("factored_s2", "EWC + Task-2 Ratio-FS", "SGD", "Ratio-FS+EWC"),
]
new_summary, new_body = [], []
for idx, (key, label, s1, s2) in enumerate(new_specs):
    rr = [r for r in new_runs if r["setting"] == key]
    assert sorted(r["seed"] for r in rr) == [3, 4, 5]
    row = dict(setting=key, variant=label, stage1_update=s1, stage2_update=s2,
               seeds="3,4,5", n=3)
    cells = []
    for field in ("a11", "a21", "a22", "BWT", "r"):
        row[field + "_mean"] = statistics.mean(r[field] for r in rr)
        row[field + "_sd"] = statistics.stdev(r[field] for r in rr)
        cells.append(cell(row[field+"_mean"], 4 if field == "r" else 2, row[field+"_sd"]))
    if idx == 4:
        new_body.append(r"\midrule")
    new_body.append(" & ".join([label, s1, s2] + cells) + r" \\")
    new_summary.append(row)
new_template = r"""% Requires \usepackage{booktabs,caption,amsmath}.
% Independent experiment; do not splice these values into the legacy Fisher table.
\begin{table*}[t]
\centering
\captionsetup{font=footnotesize,labelfont=bf}
\caption{Two-task results for the continuous-adapter Ratio-FS LoRA protocol
on ImageNet-R. Values are means and sample standard deviations over seeds 3, 4,
and 5. All variants use rank 10, SGD with learning rate $0.02$, and 50 epochs
per task; EWC uses $\lambda=1.2\times10^6$ where enabled.
Accuracies are percentages and BWT is in percentage points.
Ratio-FS uses $c=0.20$ and $\rho=0.05$.
The effective displacement is
$r_{QV}=[\sum_{\ell}\sum_{c\in\{Q,V\}}\|B^{(2)}_{\ell c}A^{(2)}_{\ell c}
-B^{(1)}_{\ell c}A^{(1)}_{\ell c}\|_F^2]^{1/2}$,
excluding classifier heads. Norms and BWT are computed per seed before
aggregation. Uniform raw per-example Fisher endpoint diagnostics are not
available for this cohort and are not populated using protection-buffer traces.}
\label{tab:two_task_lora_ratiofs}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}
\toprule
Variant & Task 1 update & Task 2 update
& $a_{11}\uparrow$ & $a_{21}\uparrow$ & $a_{22}\uparrow$
& BWT$\uparrow$ & $r_{QV}$ \\
\midrule
@@BODY@@
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{\textwidth}\scriptsize
This protocol uses normalized, damped batch-gradient protection buffers
and a continuously trained Q/V adapter.
It differs from the legacy Adam/fresh-KV protocol in the optimizer,
protection rule, shaping method, and seeds; it is a separate experiment.
Accuracy and drift statistics for all 18 runs are recovered from original
metrics. Independent tensor reconstruction additionally verifies the six
EWC and Task-1 Ratio-FS endpoints.
\end{minipage}
\end{table*}
"""
(OUT / "new_imagenetr_ratiofs.tex").write_text(new_template.replace("@@BODY@@", "\n".join(new_body)))
write_csv("new_imagenetr_ratiofs_mean_sd.csv", new_summary)

today = read_csv("today_imageneta_pair_recomputed.csv")
today_body = []
for r in today:
    label = "SGD" if r["anchor_method"] == "sgd" else "GAM"
    cells = [cell(float(r[k]), d) for k, d in [
        ("a11_fp32", 2), ("a21_fp32", 2), ("a22_trainer_rounded", 2),
        ("BWT_fp32", 2), ("old_test_local_loss_delta", 6), ("radius_fp64", 6)]]
    today_body.append(" & ".join([label, "SGD+EWC"]+cells) + r" \\")
today_template = r"""% Requires \usepackage{booktabs,caption,amsmath}.
\begin{table*}[t]
\centering
\captionsetup{font=footnotesize,labelfont=bf}
\caption{Preliminary two-task ImageNet-A results with fixed future SGD+EWC
(seed 0; LoRA rank 10; $\lambda=2000$; 50 continuation epochs; learning rate
$0.02$). Each branch restores its own Task-1 model and protection buffer.
Task-1 accuracies are re-evaluated in the current FP32 protocol rather than
copied from historical mixed-precision records.
$D_{0,\mathrm{local}}$ is the old-task test cross-entropy increment using
the fixed old 20-class head; accuracies use the classes seen at each endpoint.
$r_{QV}$ is the effective shared-backbone displacement, excluding classifiers.
Dedicated endpoint raw Fisher measurements are not yet available in this
snapshot. BWT uses unrounded within-run accuracies.}
\label{tab:two_task_imageneta_fixed_future_preliminary}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}
\toprule
Task 1 update & Task 2 update
& $a_{11}\uparrow$ & $a_{21}\uparrow$ & $a_{22}\uparrow$
& BWT$\uparrow$ & $D_{0,\mathrm{local}}$ & $r_{QV}$ \\
\midrule
@@BODY@@
\bottomrule
\end{tabular*}
\end{table*}
"""
(OUT / "new_imageneta_fixed_future_seed0.tex").write_text(
    today_template.replace("@@BODY@@", "\n".join(today_body)))

for path in OUT.glob("*.tex"):
    s = path.read_text()
    assert "@@" not in s
    assert s.count("{") == s.count("}"), path.name
    for env in ("table*", "tabular*", "minipage"):
        assert s.count("\\begin{"+env+"}") == s.count("\\end{"+env+"}"), path.name
    assert "\\bottomrule" in s and "\\end{table*}" in s
validation = dict(legacy_displayed_values_verified=42, legacy_log_hashes_verified=len(v["runs"]),
                  new_ratio_raw_jsons_verified=len(new_runs),
                  old_and_new_cohorts_mixed=False,
                  compilation="LaTeX engine not found in PATH; structural checks only",
                  generated_files=[str(p.name) for p in sorted(OUT.iterdir())])
(OUT / "TABLE_VALIDATION.json").write_text(json.dumps(validation, indent=2)+"\n")
print(json.dumps(validation, indent=2))
