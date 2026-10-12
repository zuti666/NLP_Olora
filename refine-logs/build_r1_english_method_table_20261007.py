"""Produce an English, protocol-separated summary from validated results."""
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/140-1/users/liying/O-LoRA')
BASE = ROOT / 'refine-logs/imageneta-gam-cumulative-results-20261007T065249Z'
OUT = ROOT / 'refine-logs/imageneta-r1-english-method-summary-20261007'
OUT.mkdir(exist_ok=True)

def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))

master = read_csv(BASE / 'ALL_UNIQUE_10TASK_RESULTS.csv')
aggs = {a['condition']: a for a in read_csv(BASE / 'THREE_SEED_COMPLETE_32_CONDITIONS.csv')}
specs = [
    ('A', 'Accelerated GAM (original)', r'h_+-0.1\Pi^\perp_{h_+}(h_-)', 'N01'),
    ('A', 'Unprojected branch difference (gamma=1)', r'h_+-h_-', None),
    ('A', 'GAM without orthogonal correction (gamma=0)', r'g_1+g_3', 'N02'),
    ('A', 'Mean of g1 and g3', r'(g_1+g_3)/2', 'N10'),
    ('A', 'SGD-direction control', r'g_0', 'N05'),
    ('A', 'SAM-direction control', r'g_1', 'N03'),
    ('A', 'SGD + second local difference', r'g_0+(g_3-g_2)', 'N07'),
    ('A', 'Outer SAM-probe gradient', r'g_3', 'N04'),
    ('A', 'Outer-location gradient', r'g_2', 'N06'),
    ('B', 'SAM + second local difference', r'g_1+(g_3-g_2)', 'G12_g1_plus_d32'),
    ('B', 'Sum of g1 and g2', r'g_1+g_2', 'G12_g1_plus_g2'),
    ('B', 'Mean of g1 and g2', r'(g_1+g_2)/2', 'G12_half_g1_plus_g2'),
    ('B', 'SAM + outer displacement residual', r'g_1+(g_2-g_0)', 'formal_sum12minus0_r1_k10_s2'),
]
rows, sources = [], []
common = {
    'dataset': 'imageneta', 'class_shuffle': False, 'init_cls': 20,
    'increment': 20, 'nb_tasks': 10, 'lora_rank': 10,
    'gam_grad_rho': .05, 'gam_grad_norm_rho': 1., 'init_epoch': 50,
    'epochs': 50, 'batch_size': 128, 'optimizer': 'sgd',
    'init_lr': .04, 'lrate': .02, 'init_momentum': 0., 'momentum': 0.,
    'init_weight_decay': 0., 'weight_decay': 0., 'scheduler': 'cosine',
    'ewc_lambda': 0., 'ewc_skip_task_fisher': True,
    'in_task_weight_merge': False, 'task_endpoint_weight_merge': False,
    'approximate_fisher_in_task': False,
}
for panel, method, formula, condition in specs:
    row = {'panel': panel, 'method': method, 'gradient': formula, 'condition': condition or '',
           'n_seeds': 0, 'seeds': '', 'status': 'no_matching_completed_run'}
    selected = []
    if condition in aggs:
        a = aggs[condition]
        for seed in (0, 1, 2):
            matches = [r for r in master if int(r['seed']) == seed and float(r['radius']) == 1.
                       and math.isclose(float(r['AAA']), float(a[f'AAA_seed{seed}']), abs_tol=1e-10)
                       and math.isclose(float(r['FAA']), float(a[f'FAA_seed{seed}']), abs_tol=1e-10)]
            assert len(matches) == 1, (condition, seed, len(matches))
            selected.append(matches[0])
    elif condition:
        selected = [r for r in master if r['job'] == condition and float(r['radius']) == 1. and int(r['seed']) == 2]
        assert len(selected) == 1
    for r in selected:
        cfg = json.loads(Path(r['source_config']).read_text())
        for k, value in common.items():
            assert k in cfg and cfg[k] == value, (r['job'], k, cfg.get(k), value)
        assert cfg.get('replay', False) is False
        assert cfg.get('memory_size', 0) == 0
        raw = json.loads(Path(r['source_matrix']).read_text())
        if 'matrix' in raw:
            matrix = raw['matrix']
        else:
            matrix = [m[:i+1] for i, m in enumerate(raw['cnn']['matrices']['t09'])]
        measured = {'AAA': statistics.mean(statistics.mean(m) for m in matrix),
                    'FAA': statistics.mean(matrix[-1]),
                    'BWT': statistics.mean(matrix[-1][i] - matrix[i][i] for i in range(9))}
        for k, value in measured.items():
            assert math.isclose(value, float(r[k]), abs_tol=1e-8)
        sources.append({'panel': panel, 'method': method, 'condition': condition, **r})
    if selected:
        row['n_seeds'] = len(selected)
        row['seeds'] = ';'.join(r['seed'] for r in selected)
        row['status'] = 'validated_complete'
        for metric in ('FAA', 'AAA', 'BWT'):
            values = [float(r[metric]) for r in selected]
            row[metric + '_mean'] = statistics.mean(values)
            row[metric + '_sample_sd'] = statistics.stdev(values) if len(values) > 1 else None
    rows.append(row)
assert len(sources) == 34
assert len({r['source_receipt'] for r in sources}) == 34

def format_metric(row, metric):
    if not row['n_seeds']:
        return '—'
    v = row[metric + '_mean']
    sd = row[metric + '_sample_sd']
    return f'{v:.3f}±{sd:.3f}' if sd is not None else f'{v:.3f}'

def table(panel):
    lines = ['| Method | Gradient direction | FAA ↑ | AAA ↑ | BWT ↑ |',
             '|---|---|---:|---:|---:|']
    for r in rows:
        if r['panel'] != panel:
            continue
        method = r['method'] + ('†' if r['n_seeds'] == 1 else '')
        lines.append('| ' + ' | '.join([method, '$' + r['gradient'] + '$'] +
                     [format_metric(r, m) for m in ('FAA', 'AAA', 'BWT')]) + ' |')
    return '\n'.join(lines)

report = f'''# ImageNet-A at r=1: method and gradient summary

After correcting the original GAM coefficient to gamma=0.1, 10 of the 11 rows in the supplied image have completed counterparts. The literal unprojected difference, h+−h−, has no matching completed result or identified queue entry. Its dash is a missing result, not zero accuracy.

Common training configuration: ImageNet-A, 200 classes in a fixed order, 10×20-class tasks; ViT-B/16 with augreg2_in21k_ft_in1k pretraining, LoRA rank10; outer radius r=1 and inner radius rho=.05; 50 epochs per task, batch128, SGD with zero momentum and weight decay; learning rate .04 for Task0 and .02 thereafter, cosine decay to 10% of each initial learning rate. EWC/Fisher, weight merging, and replay are disabled. Natural-step results use the original learning rates without additional final update normalization.

Values are task-macro FAA/AAA (%) and BWT (percentage points). Completed three-seed conditions use seeds0/1/2 and mean±sample standard deviation (ddof=1). † denotes seed2 only; its standard deviation is not defined. Higher is better for all three metrics, including negative BWT.

Define h+=g1+g3 and h−=g0+g2. Pi-perp-u(v) is the component of v orthogonal to u, computed using a global inner product over trainable LoRA parameters and the seen classifier. The positive probes are g0=g(theta), g1=g(theta+rho N(g0)), g2=g(theta+r N(g1−g0)), and g3=g(theta+r N(g1−g0)+rho N(g2)).

## A. Original execution protocol

Eight completed methods, 24/24 validated seed cells. Four gradient probes plus one clean CE forward; global RNG progression and native perturbation subtraction/restoration. The missing unprojected direction is listed for completeness.

{table('A')}

The gamma=0 row is also the branch sum g1+g3; these are one experiment, not two independent conditions. Sum and mean are collinear at the same parameter state. The original GAM orthogonal correction can change the direction, so they should not be labelled “the same direction as original GAM.” The g3 row is an outer SAM-probe gradient; original GAM minus SAM also contains the orthogonal correction. “Mean” is used instead of attributing the setting to an unverified paper hyperparameter.

## B. Exact restoration and RNG replay protocol

Three three-seed methods plus one seed2 method, 10/10 validated seed cells. Four positive probes with exact full-state restoration and per-probe RNG replay; unused probes leave no final state change. These rows share the common training configuration above but belong to a different execution protocol from Panel A. Compare within each panel rather than treat both as one controlled ranking.

{table('B')}

The last row is algebraically g1+g2−g0, not g1+g2. Its seed2 result is listed without a standard deviation. The two g1+g2 rows extend the supplied image with the newly completed three-seed experiments.

## Formula corrections and remaining gap

Original GAM uses h+−0.1 Pi-perp-h+(h−). Setting gamma=0 removes the complete orthogonal correction and gives h+. Removing only the projection while retaining gamma=.1 would instead give h+−.1h−; no matching completed run or queue entry was found for that direction either. The image's h+−h− additionally sets the correction coefficient to1.

For a single strictly uniform execution-protocol table, full GAM, branch sum, branch mean, g3 and g0+(g3−g2) need exact-restoration/RNG-replay counterparts; the no-projection directions also remain unrun. No new training or queue changes were made for this summary.

## Sources and verification

All34 included receipts are distinct and complete. All included configurations were checked against the common settings, including the literal r=1 value, and their raw task matrices independently reproduce FAA/AAA/BWT. Three-seed means and sample standard deviations were recomputed. The native SAM baseline uses the predefined audited S-control for seed2, not the separate earlier physical repeat.

[Table CSV]({OUT}/METHOD_GRADIENT_FAA_AAA_BWT.csv) · [Per-seed source ledger]({OUT}/SOURCE_LEDGER_34.csv) · [Full cumulative report]({BASE}/REPORT.md).
'''
(OUT/'REPORT_EN.md').write_text(report)
table_fields = ['panel', 'method', 'gradient', 'condition', 'n_seeds', 'seeds', 'status',
                'FAA_mean', 'FAA_sample_sd', 'AAA_mean', 'AAA_sample_sd', 'BWT_mean', 'BWT_sample_sd']
with (OUT/'METHOD_GRADIENT_FAA_AAA_BWT.csv').open('w', newline='') as f:
    w=csv.DictWriter(f,table_fields);w.writeheader();w.writerows(rows)
with (OUT/'SOURCE_LEDGER_34.csv').open('w',newline='') as f:
    fields=list(sources[0]);w=csv.DictWriter(f,fields);w.writeheader();w.writerows(sources)
audit = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'training_and_queues_unchanged': True,
         'common_configuration': common, 'validated_cells': 34, 'expected_registered_cells': 34,
         'failed_included_cells': 0, 'missing_unregistered_formulas': ['h_plus-h_minus', 'h_plus-0.1*h_minus'],
         'single_seed_sample_sd': None, 'source_files': {}, 'derived_files': {}}
for r in sources:
    for key in ('source_config','source_matrix','source_summary','source_receipt'):
        p=Path(r[key]);audit['source_files'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in OUT.iterdir():
    if p.is_file():audit['derived_files'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(report)
