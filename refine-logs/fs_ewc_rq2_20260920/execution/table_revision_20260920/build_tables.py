"""Export tables from audited, completed results; never launches experiments."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

OUT = Path(__file__).resolve().parent
EX = OUT.parent
LEGACY = EX.parents[1] / 'lora_full_two_task_drift_audit_20260920'


def load(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


manifest = load(EX / 'report_seed0/MANIFEST.json')
assert manifest['status'] == 'complete'
assert manifest['A_paired_rng']['passed']
sources = {}
current = []
for method in ('sgd', 'gam'):
    source = manifest['sources'][f'{method}/table_a']
    p = Path(source['path'])
    assert sha(p) == source['sha256']
    sources[str(p)] = sha(p)
    d = load(p)
    lp = EX / f'training/A_{method}_to_sgd_seed0/task0_local_metrics.jsonl'
    local = [json.loads(x) for x in lp.read_text().splitlines()]
    sources[str(lp)] = sha(lp)
    a11 = local[0]['test']['seen_accuracy']
    a21 = local[-1]['test']['seen_accuracy']
    f1, f2 = d['anchor']['fisher'], d['endpoint']['fisher']
    assert f1['samples'] == f2['samples'] == 256
    assert not f1['normalized'] and not f2['normalized']
    current.append(dict(
        variant='EWCLoRA' if method == 'sgd' else 'EWC + Task-1 shaping',
        first_update='SGD' if method == 'sgd' else 'GAM',
        second_update='SGD+EWC', dataset='ImageNet-A', seed=0,
        a11=a11, a21=a21, a22=d['accuracy_CIL']['a22'], BWT=a21-a11,
        old_task_trace_at_first_endpoint=f1['raw_trace'],
        old_task_trace_at_second_endpoint=f2['raw_trace'],
        r_QV=d['effective_weight_drift'],
        old_test_local_CE_delta=d['D0']['test'],
        summary_source=str(p),
    ))
write_csv(OUT / 'imageneta_current_complete.csv', current)

template = r'''% Requires booktabs, caption, amsmath. Task 1/2 correspond to code task 0/1.
\begin{table*}[t]
\centering
\captionsetup{font=footnotesize,labelfont=bf}
\caption{Completed two-task LoRA diagnostics on ImageNet-A (seed 0).
Both continuations use SGD+EWC for 50 epochs with $\lambda=2000$,
learning rate $0.02$, and rank 10. Each branch restores its own Task-1
model and Fisher protection buffer. Accuracies use all classes seen
at the corresponding endpoint; BWT is $a_{21}-a_{11}$ in percentage points.
$\mathcal T_1(\theta_j)$ is the unnormalized per-example empirical-Fisher
trace evaluated at endpoint $j$ on the same 256 old-task training images,
using a fixed old 20-class head and deterministic evaluation transforms.
It is not a new-task Fisher statistic.
The coordinate space is the 24 effective Q/V LoRA matrices.
All measurements use FP32 with TF32 disabled and no trajectory averaging.}
\label{tab:two_task_imageneta_fixed_future_complete}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}
\toprule
& \multicolumn{2}{c}{Update rule}
& \multicolumn{2}{c}{Task 1 endpoint}
& \multicolumn{4}{c}{Task 2 endpoint} & Drift \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}
Variant & Task 1 & Task 2 & $a_{11}\uparrow$ & $\mathcal T_1(\theta_1)$
& $a_{21}\uparrow$ & $a_{22}\uparrow$ & BWT$\uparrow$
& $\mathcal T_1(\theta_2)$ & $r_{QV}$ \\
\midrule
@@BODY@@
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{\textwidth}\scriptsize
$r_{QV}=[\sum_{\ell,c\in\{Q,V\}}
\|B^{(2)}_{\ell c}A^{(2)}_{\ell c}-B^{(1)}_{\ell c}A^{(1)}_{\ell c}\|_F^2]^{1/2}$
excludes classifier heads. This implementation continues the same adapter;
the drift is not the norm of a fresh Task-2 adapter.
Fisher traces are diagnostic measurements; training uses each branch's
own trace-normalized batch-gradient protection matrix.
BWT is computed from current-protocol accuracies before display rounding.
Only seed 0 is available; across-seed standard deviations are not estimable.
\end{minipage}
\end{table*}
'''
lines=[]
for r in current:
    numbers=[f'{r[k]:.{n}f}' for k,n in [('a11',2),('old_task_trace_at_first_endpoint',3),
             ('a21',2),('a22',2),('BWT',2),('old_task_trace_at_second_endpoint',3),('r_QV',6)]]
    lines.append(' & '.join([r['variant'],r['first_update'],r['second_update']]+numbers)+r' \\')
(OUT / 'imageneta_current_complete.tex').write_text(template.replace('@@BODY@@','\n'.join(lines)))

# Preserve the verified legacy numbers while using corrected estimator/update-rule labels.
for source_name, dest in [('legacy_lora_corrected.tex','imagenetr_legacy_verified.tex'),
                          ('legacy_lora_mean_sd.tex','imagenetr_legacy_mean_sd.tex')]:
    p=LEGACY/'tables'/source_name
    sources[str(p)]=sha(p)
    (OUT/dest).write_text(p.read_text())
rp=LEGACY/'lora_per_seed_recomputed.csv'
sources[str(rp)]=sha(rp)
with rp.open() as f:
    legacy=list(csv.DictReader(f))
assert len(legacy)==18
lookup={(r['setting'],int(r['seed'])):r for r in legacy}
assert len(lookup)==18
contrast={}
for field in ('a11','a21','a22','BWT','T1_fisher','T2_current_fisher','r_effective'):
    diffs=[float(lookup['setting1',s][field])-float(lookup['control',s][field]) for s in (0,1,2)]
    contrast[field]={'seed_differences':dict(zip((0,1,2),diffs)),
                     'mean':statistics.mean(diffs),'sample_sd':statistics.stdev(diffs)}
diffs=[(float(lookup['setting1',s]['a21'])+float(lookup['setting1',s]['a22'])
       -float(lookup['control',s]['a21'])-float(lookup['control',s]['a22']))/2 for s in (0,1,2)]
contrast['mean_accuracy']={'seed_differences':dict(zip((0,1,2),diffs)),
                          'mean':statistics.mean(diffs),'sample_sd':statistics.stdev(diffs)}
(OUT/'legacy_task1_shaping_contrast.json').write_text(json.dumps({
    'contrast':'setting1 minus control, matched by seed label; historical RNG pairing not established',
    'seeds':[0,1,2], 'n':3,'metrics':contrast},indent=2)+'\n')
(OUT/'TABLE_SOURCES.json').write_text(json.dumps({
    'sources_sha256':sources,
    'current_validated_expected':'2/2 continuation arms, completed seed0',
    'legacy_expected':'6 settings x seeds 0/1/2 = 18 logs; validation in USER_TABLE_PROVENANCE',
    'do_not_pool':'Different datasets, optimizers, parameter channels, Fisher estimators and adapter lifecycles.'
},indent=2)+'\n')
print(json.dumps({'current_rows':len(current),'legacy_rows':len(legacy),'output':str(OUT)},indent=2))
