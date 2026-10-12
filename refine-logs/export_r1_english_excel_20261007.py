"""Export existing ImageNet-A results; does not modify training or queues."""
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter

BASE = Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-english-method-summary-20261007')
DEST = BASE / 'ImageNetA_r1_Gradient_Results.xlsx'

def read_csv(name):
    with (BASE / name).open() as f:
        return list(csv.DictReader(f))

stats = read_csv('METHOD_GRADIENT_FAA_AAA_BWT.csv')
sources = read_csv('SOURCE_LEDGER_34.csv')
by_condition = {r['condition']: r for r in stats}
specs = [
    ('N01', 'Accelerated GAM with orthogonal correction',
     '(g₁+g₃) − 0.1(g₀+g₂)⊥(g₁+g₃)', r'(g_1+g_3)-0.1\Pi^\perp_{g_1+g_3}(g_0+g_2)'),
    ('', 'Accelerated GAM without projection (untested)',
     '(g₁+g₃) − (g₀+g₂)', r'(g_1+g_3)-(g_0+g_2)'),
    ('N02', 'Summed direction (correction removed)', 'g₁+g₃', r'g_1+g_3'),
    ('N10', 'Half-scaled summed direction', '½g₁ + ½g₃', r'\tfrac12g_1+\tfrac12g_3'),
    ('N05', 'SGD direction', 'g₀', r'g_0'),
    ('N03', 'SAM direction', 'g₁ = g₀ + (g₁−g₀)', r'g_1=g_0+(g_1-g_0)'),
    ('N07', 'SGD + GAM residual', 'g₀ + (g₃−g₂)', r'g_0+(g_3-g_2)'),
    ('G12_g1_plus_d32', 'SAM + GAM residual', 'g₁ + (g₃−g₂)', r'g_1+(g_3-g_2)'),
    ('N04', 'Summed direction minus SAM', 'g₃ = g₀ + (g₃−g₂) + (g₂−g₀)', r'g_3=g_0+(g_3-g_2)+(g_2-g_0)'),
    ('N06', 'Relocated-gradient update', 'g₂ = g₀ + (g₂−g₀)', r'g_2=g_0+(g_2-g_0)'),
    ('formal_sum12minus0_r1_k10_s2', 'SAM + relocation residual', 'g₁ + (g₂−g₀)', r'g_1+(g_2-g_0)'),
    ('G12_g1_plus_g2', 'Sum of SAM and relocated gradients', 'g₁+g₂', r'g_1+g_2'),
    ('G12_half_g1_plus_g2', 'Mean of SAM and relocated gradients', '(g₁+g₂)/2', r'(g_1+g_2)/2'),
]
assert len(specs) == 13
protocols = {'A': 'Native restoration / global RNG', 'B': 'Exact restoration / RNG replay'}
wb = Workbook()
summary = wb.active
summary.title = 'Summary'
numeric = wb.create_sheet('Numeric results')
runs = wb.create_sheet('Per-seed runs')
settings = wb.create_sheet('Settings')
wb.properties.title = 'ImageNet-A r=1: gradient-direction experiments'
wb.properties.subject = 'English summary of validated ImageNet-A gradient-direction results'
wb.properties.creator = 'Research results export'
wb.properties.description = '13 methods in one table; multi-seed mean/sample SD, actual run IDs, protocol metadata and per-seed numeric values.'

summary.merge_cells('A1:H1')
summary['A1'] = 'ImageNet-A | r = 1.0 | Gradient-direction results'
summary.merge_cells('A2:H2')
summary['A2'] = 'FAA / AAA: task-macro percentages. BWT: percentage points. Mean ± sample SD for seeds 0, 1, 2; the single-seed result is seed 2. Blank metrics = untested.'
summary.merge_cells('A3:H3')
summary['A3'] = '10 × 20 classes; LoRA rank 10; inner radius 0.05; 50 epochs/task; batch 128; SGD; LR 0.04 / 0.02; no EWC/Fisher, weight merging or replay. Execution protocols are recorded per row.'
summary.append([])
for c, value in enumerate(['Method', 'Gradient', 'FAA', 'AAA', 'BWT', 'run', 'Seeds', 'Protocol'], 1):
    summary.cell(5, c, value)
numeric_headers = ['Method', 'Gradient', 'FAA', 'AAA', 'BWT', 'run', 'n_seeds', 'Seeds',
                   'FAA sample SD', 'AAA sample SD', 'BWT sample SD', 'Protocol', 'Status', 'Gradient (LaTeX)', 'Condition ID']
numeric.append(numeric_headers)
run_headers = ['Method', 'Gradient', 'Seed', 'FAA', 'AAA', 'BWT', 'run', 'Protocol',
               'Outer radius', 'Inner radius', 'LoRA rank', 'Tasks', 'Total epochs', 'Updates',
               'Status', 'Configuration path', 'Matrix path', 'Summary path', 'Receipt path']
runs.append(run_headers)
selected_by_condition = {}
for idx, (condition, method, gradient, latex) in enumerate(specs, 6):
    row = by_condition[condition]
    n = int(row['n_seeds'])
    selected = sorted([r for r in sources if r['condition'] == condition], key=lambda r: int(r['seed']))
    assert len(selected) == n
    selected_by_condition[condition] = selected
    run_text = '\n'.join(f"s{r['seed']}: {r['job']}" for r in selected) or None
    seeds = ', '.join(r['seed'] for r in selected)
    protocol = protocols[row['panel']] if n else 'Untested'
    means = [float(row[f'{m}_mean']) if n else None for m in ('FAA', 'AAA', 'BWT')]
    sds = [float(row[f'{m}_sample_sd']) if n > 1 else None for m in ('FAA', 'AAA', 'BWT')]
    displayed = [f'{mean:.3f} ± {sd:.3f}' if n > 1 else mean for mean, sd in zip(means, sds)]
    vals = [method, gradient, *displayed, run_text, f'{seeds} (n={n})' if n else 'Not run', protocol]
    for col, value in enumerate(vals, 1):
        summary.cell(idx, col, value)
    numeric.append([method, gradient, *means, run_text, n, seeds or None, *sds, protocol,
                    'Validated complete' if n else 'Untested', latex, condition or None])
    for r in selected:
        runs.append([method, gradient, int(r['seed']), float(r['FAA']), float(r['AAA']), float(r['BWT']),
                     r['job'], protocol, float(r['radius']), float(r['rho']), int(r['rank']),
                     int(r['tasks']), int(r['epochs']), int(r['updates']), 'Validated complete',
                     r['source_config'], r['source_matrix'], r['source_summary'], r['source_receipt']])
    if n:
        for metric, mean, sd in zip(('FAA', 'AAA', 'BWT'), means, sds):
            raw = [float(r[metric]) for r in selected]
            assert abs(statistics.mean(raw) - mean) < 1e-10
            if n > 1:
                assert abs(statistics.stdev(raw) - sd) < 1e-10

settings.append(['Setting', 'Value'])
notes = [
    ('Dataset', 'ImageNet-A; the ImageNet-R values in the supplied template were not used.'),
    ('Data', 'Local train: 5,981; test: 1,519; 200 classes in fixed order.'),
    ('Task split', '10 tasks, 20 classes per task'),
    ('Backbone', 'ViT-B/16; augreg2_in21k_ft_in1k pretraining'),
    ('LoRA', 'Rank = 10; scale = 1; no alpha/rank scaling'),
    ('Outer radius', 'gam_grad_norm_rho = 1.0 in every included source configuration'),
    ('Inner radius', 'gam_grad_rho = 0.05'),
    ('Training budget', '50 epochs/task; 500 total epochs; 2,550 updates per run'),
    ('Batch size', '128'),
    ('Optimizer', 'SGD; momentum = 0; weight decay = 0'),
    ('Learning rate', 'Task 0: 0.04; subsequent tasks: 0.02'),
    ('Schedule', 'Cosine; minimum learning rate = 0.1 × task initial learning rate'),
    ('Disabled modules', 'EWC/Fisher, actual weight merging, replay'),
    ('Update scale', 'Original learning rates; no final norm matching; half-scaled directions multiply the final direction by 0.5 once'),
    ('FAA', 'Final-stage equal-weight average over 10 tasks (%)'),
    ('AAA', 'Average of 10 stage-wise equal-weight seen-task accuracies (%)'),
    ('BWT', 'Mean final minus acquisition accuracy over the first 9 tasks (percentage points); higher is better'),
    ('Statistics', 'Seeds 0, 1, 2: mean ± sample standard deviation, ddof = 1; single-seed SD is blank'),
    ('Native protocol', 'Four gradient probes plus a clean CE forward; global RNG progression; native perturbation subtraction/restoration'),
    ('Exact-restoration protocol', 'Four positive probes; exact full-state restoration; per-probe RNG replay; only one g0/R1 RNG advance'),
    ('Protocol comparison', 'Rows share training hyperparameters. Keep the protocol labels when presenting a combined table; cross-protocol differences are not a strictly controlled comparison.'),
    ('Projection notation', '(v)⊥u denotes the component of v orthogonal to u, using a global inner product over trainable LoRA and the seen classifier.'),
    ('Original GAM', 'U = (g1+g3) - 0.1 * perpendicular_to_(g1+g3)(g0+g2)'),
    ('Correction removed', 'gamma = 0 gives g1+g3; this is the summed-direction row, not a separate condition'),
    ('Untested template row', '(g1+g3)-(g0+g2) has no matching completed run; leaving its metrics blank is intentional'),
    ('Fixed-gamma projection removal', 'Removing only the projection at gamma = 0.1 would yield (g1+g3)-0.1(g0+g2); this direction also has no matching completed result'),
    ('SAM baseline identity', 'The predefined audited S-control is used for seed 2; the earlier physical duplicate is not treated as an extra seed'),
    ('SGD/SAM names', 'These denote final update-direction controls; extra probes are still executed'),
    ('g3 label', 'g3 equals (g1+g3) minus g1; it does not equal full original GAM minus SAM when the orthogonal correction is present'),
    ('Summed/mean directions', 'At the same parameter state the summed and half-scaled directions are collinear; the corrected original GAM can change direction'),
    ('Validation', '34/34 complete runs; 34 distinct receipts; exact r=1 configurations; raw-matrix metric checks and recomputed sample SD'),
    ('Source report', str(BASE / 'REPORT_EN.md')),
    ('Created UTC', datetime.now(timezone.utc).isoformat()),
]
for pair in notes:
    settings.append(pair)

dark = '17365D'
blue = 'EAF1FA'
teal = 'E6F4EF'
grey = 'F2F2F2'
edge = Side(style='thin', color='D8E2EF')
for ws in wb:
    ws.sheet_view.showGridLines = False
    for row in ws:
        for c in row:
            c.font = Font(name='Calibri', size=11, color='1F2937')
            c.alignment = Alignment(vertical='center', wrap_text=True)
    header = 5 if ws is summary else 1
    for c in ws[header]:
        c.fill = PatternFill('solid', fgColor=dark)
        c.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
    ws.row_dimensions[header].height = 30
    ws.freeze_panes = 'C6' if ws is summary else 'C2'
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1 if ws is summary else 0
    ws.print_title_rows = f'1:{header}'
    ws.oddFooter.center.text = 'ImageNet-A | r = 1 | Page &P of &N'
    ws.sheet_properties.outlinePr.summaryRight = False

for idx, spec in enumerate(specs, 6):
    r = by_condition[spec[0]]
    shade = grey if not int(r['n_seeds']) else blue if r['panel'] == 'A' else teal
    for c in summary[idx]:
        c.fill = PatternFill('solid', fgColor=shade)
        c.border = Border(bottom=edge)
    for col in (3, 4, 5, 7):
        summary.cell(idx, col).alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    for col in (3, 4, 5):
        summary.cell(idx, col).number_format = '0.000'
    summary.cell(idx, 6).font = Font(name='Calibri', size=9, color='374151')
    summary.row_dimensions[idx].height = 62 if int(r['n_seeds']) > 1 else 48
for col, width in {'A': 44, 'B': 44, 'C': 21, 'D': 21, 'E': 21, 'F': 63, 'G': 17, 'H': 34}.items():
    summary.column_dimensions[col].width = width
summary.row_dimensions[1].height = 32
summary.row_dimensions[2].height = 32
summary.row_dimensions[3].height = 36
summary['A1'].font = Font(name='Calibri', size=17, bold=True, color=dark)
summary['A2'].font = Font(name='Calibri', size=11, italic=True, color='475569')
summary['A3'].font = Font(name='Calibri', size=11, color='475569')
summary.print_area = 'A1:H18'

for ws, table_name, top in [(summary, 'MethodSummary', 5), (numeric, 'NumericResults', 1),
                             (runs, 'PerSeedRuns', 1), (settings, 'ExperimentSettings', 1)]:
    t = Table(displayName=table_name, ref=f'A{top}:{get_column_letter(ws.max_column)}{ws.max_row}')
    t.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=ws is not summary)
    ws.add_table(t)
for ws in (numeric, runs):
    for c in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(c)].width = 20
    ws.column_dimensions['A'].width = 45
    ws.column_dimensions['B'].width = 44
    ws.column_dimensions['F' if ws is numeric else 'G'].width = 63
    for row in range(2, ws.max_row + 1):
        ws.row_dimensions[row].height = 60 if ws is numeric else 40
    metric_cols = (3, 4, 5, 9, 10, 11) if ws is numeric else (4, 5, 6)
    for row in range(2, ws.max_row + 1):
        for col in metric_cols:
            ws.cell(row, col).number_format = '0.000'
    if ws is runs:
        for col in ('P', 'Q', 'R', 'S'):
            ws.column_dimensions[col].width = 90
numeric.column_dimensions['L'].width = 34
numeric.column_dimensions['N'].width = 64
numeric.column_dimensions['O'].width = 40
settings.column_dimensions['A'].width = 34
settings.column_dimensions['B'].width = 125
for row in range(2, settings.max_row + 1):
    settings.row_dimensions[row].height = 36
wb.save(DEST)

# Reopen the final artifact to check export integrity, order and blank cells.
check = load_workbook(DEST, data_only=True)
assert check['Summary'].max_row == 18
assert [check['Summary'].cell(5, c).value for c in range(1, 7)] == ['Method', 'Gradient', 'FAA', 'AAA', 'BWT', 'run']
assert [check['Summary'].cell(i, 1).value for i in range(6, 19)] == [s[1] for s in specs]
assert all(check['Summary'].cell(7, c).value is None for c in (3, 4, 5, 6))
assert check['Per-seed runs'].max_row == 35
assert check['Numeric results'].max_row == 14
for i, (condition, *_rest) in enumerate(specs, 2):
    row = by_condition[condition]
    n = int(row['n_seeds'])
    for col, metric in zip((3, 4, 5), ('FAA', 'AAA', 'BWT')):
        v = check['Numeric results'].cell(i, col).value
        assert v is None if n == 0 else abs(v - float(row[f'{metric}_mean'])) < 1e-10
    if n <= 1:
        assert all(check['Numeric results'].cell(i, c).value is None for c in (9, 10, 11))
validation = {'file': str(DEST), 'sha256': hashlib.sha256(DEST.read_bytes()).hexdigest(),
              'method_rows': 13, 'completed_methods': 12, 'untested_methods': 1,
              'three_seed_methods': 11, 'single_seed_methods': 1,
              'unique_per_seed_runs': 34, 'round_trip_checks_passed': True,
              'training_and_queues_unchanged': True}
(BASE / 'EXCEL_EXPORT_VALIDATION.json').write_text(json.dumps(validation, indent=2) + '\n')
print(json.dumps(validation, indent=2))
