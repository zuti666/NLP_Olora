"""English protocol-separated r=1 table with LR/step controls and |U|/|g1|.

Extends build_r1_english_method_table_20261007.py with Panel B baselines
(P01/P02/P04), Panel A LR/step controls (S x2, Mean x2, Gmatch) and the
per-step update-norm ratio |U|/|g1| read from the audited step traces.
Only literal r=1 cells are aggregated; cells at any other radius are excluded.
"""
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/140-1/users/liying/O-LoRA')
BASE = ROOT / 'refine-logs/imageneta-gam-cumulative-results-20261007T065249Z'
OUT = ROOT / 'refine-logs/imageneta-r1-english-method-summary-20261012'
OUT.mkdir(exist_ok=True)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


master = read_csv(BASE / 'ALL_UNIQUE_10TASK_RESULTS.csv')
aggs = {a['condition']: a for a in read_csv(BASE / 'THREE_SEED_COMPLETE_32_CONDITIONS.csv')}
# (panel, method, formula, condition, lr_scale_label)
specs = [
    ('A', 'Accelerated GAM (original)', r'h_+-0.1\Pi^\perp_{h_+}(h_-)', 'N01', '1'),
    ('A', 'Unprojected branch difference (gamma=1)', r'h_+-h_-', None, '1'),
    ('A', 'GAM without orthogonal correction (gamma=0)', r'g_1+g_3', 'N02', '1'),
    ('A', 'Mean of g1 and g3', r'(g_1+g_3)/2', 'N10', '1'),
    ('A', 'Mean of g1 and g3, 2x LR', r'(g_1+g_3)/2', 'N13', '2'),
    ('A', 'SGD-direction control', r'g_0', 'N05', '1'),
    ('A', 'SAM-direction control', r'g_1', 'N03', '1'),
    ('A', 'SAM-direction control, 2x LR', r'g_1', 'N12', '2'),
    ('A', 'SGD + second local difference', r'g_0+(g_3-g_2)', 'N07', '1'),
    ('A', 'Outer SAM-probe gradient', r'g_3', 'N04', '1'),
    ('A', 'Outer SAM-probe gradient, SAM-matched step (Gmatch)', r'g_3', 'N14', 'matched'),
    ('A', 'Outer-location gradient', r'g_2', 'N06', '1'),
    ('B', 'SGD-direction control', r'g_0', 'P01', '1'),
    ('B', 'SAM-direction control', r'g_1', 'P02', '1'),
    ('B', 'Outer-location gradient', r'g_2', 'P04', '1'),
    ('B', 'SAM + second local difference', r'g_1+(g_3-g_2)', 'G12_g1_plus_d32', '1'),
    ('B', 'Sum of g1 and g2', r'g_1+g_2', 'G12_g1_plus_g2', '1'),
    ('B', 'Mean of g1 and g2', r'(g_1+g_2)/2', 'G12_half_g1_plus_g2', '1'),
    ('B', 'SAM + outer displacement residual', r'g_1+(g_2-g_0)', 'formal_sum12minus0_r1_k10_s2', '1'),
]
common = {
    'dataset': 'imageneta', 'class_shuffle': False, 'init_cls': 20,
    'increment': 20, 'nb_tasks': 10, 'lora_rank': 10,
    'gam_grad_rho': .05, 'gam_grad_norm_rho': 1., 'init_epoch': 50,
    'epochs': 50, 'batch_size': 128, 'optimizer': 'sgd',
    'init_momentum': 0., 'momentum': 0.,
    'init_weight_decay': 0., 'weight_decay': 0., 'scheduler': 'cosine',
    'ewc_lambda': 0., 'ewc_skip_task_fisher': True,
    'in_task_weight_merge': False, 'task_endpoint_weight_merge': False,
    'approximate_fisher_in_task': False,
}
BASE_LR = {'init_lr': .04, 'lrate': .02}
EXPECTED_STEPS = 2550


def raw_matrix(path):
    raw = json.loads(Path(path).read_text())
    if 'matrix' in raw:
        return raw['matrix']
    return [m[:i+1] for i, m in enumerate(raw['cnn']['matrices']['t09'])]


def measure(matrix):
    return {'AAA': statistics.mean(statistics.mean(m) for m in matrix),
            'FAA': statistics.mean(matrix[-1]),
            'BWT': statistics.mean(matrix[-1][i] - matrix[i][i] for i in range(9))}


def trace_stats(r, panel):
    """Mean per-step |U|/|g1| and actual-step/base-LR scale from the audited trace."""
    if panel == 'A':
        path = Path(r['source_receipt']).parent / 'direction_audit.jsonl'
    else:
        path = Path(r['source_matrix']).parent / 'step_trace.jsonl'
    if not path.exists():
        return None
    ratios, scales = [], []
    with path.open() as f:
        for line in f:
            s = json.loads(line)
            if panel == 'A':
                u, g1 = s['selected_actual']['l2_norm'], s['gradients']['g1']['l2_norm']
                nominal = s['lr'][0]
                actual = s.get('step_scale') or nominal
            else:
                u, g1 = s['norms']['U'], s['norms']['g1_plus']
                nominal, actual = s['nominal_lr'], s['step_scale']
            ratios.append(u / g1)
            # Relative to the common base schedule, so 2x-LR runs report 2.
            scales.append(actual / nominal * float(r['lr_multiplier']))
    assert len(ratios) == EXPECTED_STEPS, (r['job'], len(ratios))
    return {'trace': str(path), 'ratio_mean': statistics.mean(ratios),
            'scale_mean': statistics.mean(scales)}


rows, sources, excluded = [], [], []
for panel, method, formula, condition, lr_label in specs:
    row = {'panel': panel, 'method': method, 'gradient': formula, 'condition': condition or '',
           'lr_scale': lr_label, 'n_seeds': 0, 'seeds': '', 'status': 'no_matching_completed_run'}
    selected = []
    if condition in aggs:
        a = aggs[condition]
        for seed in (0, 1, 2):
            same = [r for r in master if int(r['seed']) == seed
                    and math.isclose(float(r['AAA']), float(a[f'AAA_seed{seed}']), abs_tol=1e-10)
                    and math.isclose(float(r['FAA']), float(a[f'FAA_seed{seed}']), abs_tol=1e-10)]
            r1 = [r for r in same if float(r['radius']) == 1.]
            if len(r1) == 1:
                selected.append(r1[0])
            else:
                assert not r1 and len(same) == 1, (condition, seed, len(same))
                excluded.append({'panel': panel, 'method': method, 'condition': condition,
                                 'reason': f"radius {same[0]['radius']} != 1", **same[0]})
    elif condition:
        selected = [r for r in master if r['job'] == condition and float(r['radius']) == 1. and int(r['seed']) == 2]
        assert len(selected) == 1
    ratio_values, scale_values, trace_seeds = [], [], []
    for r in selected:
        cfg = json.loads(Path(r['source_config']).read_text())
        for k, value in common.items():
            assert k in cfg and cfg[k] == value, (r['job'], k, cfg.get(k), value)
        mult = float(r['lr_multiplier'])
        assert mult == (2. if lr_label == '2' else 1.), (r['job'], mult)
        for k, value in BASE_LR.items():
            assert math.isclose(cfg[k], value * mult), (r['job'], k, cfg[k])
        assert cfg.get('replay', False) is False and cfg.get('memory_size', 0) == 0
        measured = measure(raw_matrix(r['source_matrix']))
        for k, value in measured.items():
            assert math.isclose(value, float(r[k]), abs_tol=1e-8)
        ts = trace_stats(r, panel)
        if ts:
            ratio_values.append(ts['ratio_mean'])
            scale_values.append(ts['scale_mean'])
            trace_seeds.append(r['seed'])
        sources.append({'panel': panel, 'method': method, 'condition': condition,
                        'U_over_g1_mean': ts['ratio_mean'] if ts else None,
                        'lr_scale_mean': ts['scale_mean'] if ts else None,
                        'trace': ts['trace'] if ts else None, **r})
    if selected:
        row['n_seeds'] = len(selected)
        row['seeds'] = ';'.join(r['seed'] for r in selected)
        row['status'] = 'validated_complete'
        for metric in ('FAA', 'AAA', 'BWT'):
            values = [float(r[metric]) for r in selected]
            row[metric + '_mean'] = statistics.mean(values)
            row[metric + '_sample_sd'] = statistics.stdev(values) if len(values) > 1 else None
        row['U_over_g1'] = statistics.mean(ratio_values) if ratio_values else None
        row['lr_scale_measured'] = statistics.mean(scale_values) if scale_values else None
        row['trace_seeds'] = ';'.join(trace_seeds)
    rows.append(row)

assert len({r['source_receipt'] for r in sources}) == len(sources)
assert {(e['condition'], e['seed']) for e in excluded} == {('P01', '2'), ('P02', '2')}, excluded

# Reference only: three-seed values if the radius-0.25 seed2 cells were admitted.
alt = {}
for e in excluded:
    seed01 = [s for s in sources if s['condition'] == e['condition']]
    vals = {m: [float(s[m]) for s in seed01] + [float(e[m])] for m in ('FAA', 'AAA', 'BWT')}
    alt[e['condition']] = {m: (statistics.mean(v), statistics.stdev(v)) for m, v in vals.items()}


def format_metric(row, metric):
    if not row['n_seeds']:
        return '—'
    v, sd = row[metric + '_mean'], row[metric + '_sample_sd']
    return f'{v:.3f}±{sd:.3f}' if sd is not None else f'{v:.3f}'


def format_lr(row):
    if row['lr_scale'] == 'matched':
        return f"S-matched (×{row['lr_scale_measured']:.2f})"
    return '×' + row['lr_scale']


def mark(row):
    if row['n_seeds'] == 1:
        return '†'
    if row['condition'] in alt:
        return '‡'
    return ''


def table(panel):
    lines = ['| Method | Gradient direction | LR scale | \\|U\\|/\\|g₁\\| | FAA ↑ | AAA ↑ | BWT ↑ |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['panel'] != panel:
            continue
        ratio = '—' if not r['n_seeds'] else f"{r['U_over_g1']:.2f}" + ('*' if r['trace_seeds'] != r['seeds'] else '')
        lines.append('| ' + ' | '.join([r['method'] + mark(r), '$' + r['gradient'] + '$',
                     format_lr(r) if r['n_seeds'] else '—', ratio] +
                     [format_metric(r, m) for m in ('FAA', 'AAA', 'BWT')]) + ' |')
    return '\n'.join(lines)


n_cells = len(sources)
alt_text = '; '.join(f"{c}: FAA {v['FAA'][0]:.3f}±{v['FAA'][1]:.3f}, AAA {v['AAA'][0]:.3f}±{v['AAA'][1]:.3f}, "
                     f"BWT {v['BWT'][0]:.3f}±{v['BWT'][1]:.3f}" for c, v in sorted(alt.items()))
report = f'''# ImageNet-A at r=1: method, gradient and step-size summary

This version adds the Panel B baselines (SGD, SAM and g2 directions), the Panel A learning-rate and step-matched controls (SAM ×2 LR, Mean ×2 LR, Gmatch), and the update-norm ratio |U|/|g1|. Every included cell uses the literal outer radius r=1.

Common training configuration: ImageNet-A, 200 classes in a fixed order, 10×20-class tasks; ViT-B/16 (augreg2_in21k_ft_in1k), LoRA rank10; outer radius r=1, inner radius rho=.05; 50 epochs per task, batch128; SGD with zero momentum and weight decay; base learning rate .04 for Task0 and .02 thereafter, cosine decay to 10% of each initial rate. EWC/Fisher, weight merging and replay are disabled. The only deliberate deviation is the LR scale column: ×2 rows double both the peak and floor learning rates; the Gmatch row keeps the base schedule but rescales each step so that its effective LoRA-BA plus classifier displacement equals the independent same-seed SAM run at the same task and step.

Values are task-macro FAA/AAA (%) and BWT (percentage points); mean±sample SD (ddof=1) over the listed seeds. † seed2 only. ‡ seeds0/1 only (see note below). Higher is better for all three metrics.

|U|/|g1| is the per-step ratio of the final update-direction norm to the SAM-probe gradient norm at the same parameter state and batch, over all trainable LoRA and seen-classifier parameters, averaged over the 2550 steps of each run and then over seeds. It excludes the LR scale, so the step relative to a SAM step at the same state is approximately LR scale × |U|/|g1|. * marks ratios computed from seeds0/1 only, because the historical seed2 runs of those conditions have no per-step trace.

Define h+=g1+g3 and h−=g0+g2. The positive probes are g0=g(theta), g1=g(theta+rho N(g0)), g2=g(theta+r N(g1−g0)) and g3=g(theta+r N(g1−g0)+rho N(g2)).

## A. Original execution protocol

Four gradient probes plus one clean CE forward; global RNG progression and native perturbation subtraction/restoration.

{table('A')}

The gamma=0 row is the branch sum g1+g3. Mean ×2 LR is the same nominal update as the branch sum, so the Mean and Mean ×2 rows differ only in step size. Gmatch uses the g3 direction with the SAM run's step length.

## B. Exact restoration and RNG replay protocol

Four probes with exact full-state restoration, per-probe RNG replay, and a per-epoch data seed of seed×100000+task×1000+epoch. The SGD, SAM and g2 baselines evaluate the probe set {{g0, g1, g1 at theta−rho N(g0), g2}} instead of {{g0, g1, g2, g3}}; the g2 location is identical, and exact restoration means the unused probes leave no state change. Compare within each panel.

{table('B')}

‡ The SGD and SAM baselines have no r=1 seed2 run. Their seed2 runs used r=0.25 and are excluded here. Their final update (g0 or g1) does not depend on r, and the r-dependent probe is discarded under exact restoration, so including them would be mathematically equivalent but not literally identical. For reference, the three-seed values with that cell would be: {alt_text}.

## Remaining gaps

The unprojected difference h+−h− and the gamma=.1 unprojected variant h+−.1h− have no completed runs. No step-matched control exists for the branch sum or original GAM (|U|/|g1|≈3), and SAM was tested up to 2× LR only.

## Sources and verification

{n_cells} validated cells. Each configuration was checked against the common settings, including r=1 and the stated LR scale. Each raw task matrix reproduces its FAA/AAA/BWT, and each trace covers exactly 2550 steps. Per-seed values and trace paths are in the source ledger.

[Table CSV]({OUT}/METHOD_GRADIENT_STEP_FAA_AAA_BWT.csv) · [Per-seed source ledger]({OUT}/SOURCE_LEDGER.csv) · [Excluded cells]({OUT}/EXCLUDED_CELLS.csv) · [Previous version](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-english-method-summary-20261007/REPORT_EN.md)
'''
(OUT / 'REPORT_EN.md').write_text(report)
table_fields = ['panel', 'method', 'gradient', 'condition', 'lr_scale', 'lr_scale_measured', 'U_over_g1',
                'trace_seeds', 'n_seeds', 'seeds', 'status', 'FAA_mean', 'FAA_sample_sd',
                'AAA_mean', 'AAA_sample_sd', 'BWT_mean', 'BWT_sample_sd']
with (OUT / 'METHOD_GRADIENT_STEP_FAA_AAA_BWT.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, table_fields, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
with (OUT / 'SOURCE_LEDGER.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, list(sources[0]))
    w.writeheader()
    w.writerows(sources)
with (OUT / 'EXCLUDED_CELLS.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, list(excluded[0]))
    w.writeheader()
    w.writerows(excluded)
audit = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'training_and_queues_unchanged': True,
         'common_configuration': common, 'base_lr': BASE_LR, 'validated_cells': n_cells,
         'excluded_cells': [(e['condition'], e['seed'], e['reason']) for e in excluded],
         'missing_unregistered_formulas': ['h_plus-h_minus', 'h_plus-0.1*h_minus'],
         'source_files': {}, 'derived_files': {}}
for r in sources:
    for key in ('source_config', 'source_matrix', 'source_summary', 'source_receipt', 'trace'):
        if r.get(key):
            p = Path(r[key])
            audit['source_files'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
for p in OUT.iterdir():
    if p.is_file() and p.name != 'AUDIT.json':
        audit['derived_files'][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
(OUT / 'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
print(report)
