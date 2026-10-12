"""Build the requested table from audited seed cohorts and verify raw metrics."""
import csv
import hashlib
import json
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ['aircraft', 'cars196', 'cub200', 'flowers', 'het5']
LABELS = dict(zip(DATASETS, ['Aircraft', 'Cars', 'CUB-200', 'Flowers', '5Datasets']))
METHODS = ['Vanilla LoRA', 'InfLoRA', 'OLoRA', 'SDLoRA', 'CL-LoRA', 'EWCLoRA', 'GAM-EWCLoRA']


def read(name):
    return json.loads((ROOT / name).read_text())


def verify_run(run):
    path = Path(run['source'])
    raw = path.read_bytes()
    expected = run.get('source_sha256', run.get('metrics_sha256'))
    actual = hashlib.sha256(raw).hexdigest()
    assert actual == expected, path
    matrix = json.loads(raw)['cnn']['matrices']['final']
    stages = []
    for i, row in enumerate(matrix):
        seen = row[:i + 1]
        assert len(seen) == i + 1 and all(math.isfinite(x) for x in seen), path
        stages.append(st.mean(seen))
    faa, aaa = stages[-1], st.mean(stages)
    assert abs(faa - run['faa']) < 1e-9, path
    assert abs(aaa - run['aaa_stage']) < 1e-9, path
    return dict(seed=run['seed'], faa=faa, aaa_stage=aaa,
                source=str(path), source_sha256=actual)


groups = []
for r in read('baselines.json')['records']:
    groups.append(dict(method=r['method'], dataset=r['dataset'],
                       runs=[dict(seed=r['seeds'][0], faa=r['faa_mean'],
                                  aaa_stage=r['aaa_stage_mean'], source=r['source'],
                                  source_sha256=r['source_sha256'])]))
cl_gam = read('cl_gam.json')
for r in cl_gam['current_table_groups']:
    groups.append(dict(method=r['method'], dataset=r['dataset'],
                       runs=[dict(run, seed=r['seed']) for run in r['runs']]))
for r in read('ewc.json')['rows']:
    groups.append(dict(method=r['method'], dataset=r['dataset'], runs=r['runs']))

cohorts = {}
for g in groups:
    runs = [verify_run(run) for run in g['runs']]
    seeds = [run['seed'] for run in runs]
    assert len(set(seeds)) == len(seeds)
    entry = dict(method=g['method'], dataset=g['dataset'], n=len(runs), seeds=seeds, runs=runs)
    for metric in ['faa', 'aaa_stage']:
        values = [run[metric] for run in runs]
        entry[metric] = dict(mean=st.mean(values),
                             sample_sd=st.stdev(values) if len(values) > 1 else None,
                             sample_variance=st.variance(values) if len(values) > 1 else None)
    key = (g['method'], g['dataset'])
    assert key not in cohorts
    cohorts[key] = entry
assert len(cohorts) == 35

original = (ROOT / 'input_table.tex').read_text()
lines = []
for line in original.splitlines():
    method = line.split(' & ', 1)[0]
    if method not in METHODS:
        lines.append(line)
        continue
    parts = line.removesuffix('\\\\').split(' & ')
    assert len(parts) == 11
    cells = []
    for i, old in enumerate(parts[1:]):
        cohort = cohorts[method, DATASETS[i // 2]]
        metric = 'faa' if i % 2 == 0 else 'aaa_stage'
        result = cohort[metric]
        old_number = re.search(r'\d+\.\d+', old).group()
        assert abs(float(old_number) - max(run[metric] for run in cohort['runs'])) < 0.006
        number = f"{result['mean']:.2f}" if cohort['n'] > 1 else old_number
        center = old.replace(old_number, number)
        sd = f"{result['sample_sd']:.2f}" if cohort['n'] > 1 else r'\mathrm{NA}'
        cells.append(r'\ExpRes{' + center + '}{' + sd + '}')
        result['display_center'] = number
        result['display_sd'] = sd
        result['original_single_run_display'] = old_number
    lines.append(method + ' & ' + ' & '.join(cells) + r'\\')

note = r'''\par\vspace{3pt}
\begin{minipage}{\linewidth}\scriptsize
Subscripts report sample standard deviations across matched seeds.
EWCLoRA on Cars and Flowers reports means over seeds 1 and 2 ($n=2$),
replacing the previous best-run scores; all other entries are single-run
results ($n=1$), with undefined standard deviations marked NA.
Boldface and underlining retain the original formatting.
Configurations differ across methods and datasets.
\end{minipage}'''
table = '\n'.join(lines).replace(r'\end{table*}', note + '\n' + r'\end{table*}') + '\n'
assert table.count(r'\ExpRes{') == 70
assert table.count(r'\mathrm{NA}') == 66
assert 'Pets' not in table
assert table.count(r'\textbf{') == original.count(r'\textbf{')
assert table.count(r'\underline{') == original.count(r'\underline{')
(ROOT / 'finegrained_std.tex').write_text(table)

statistics = dict(date='2026-09-22', sd_definition='sample standard deviation across seeds, ddof=1',
                  metric_definition='FAA: last matrix row mean; AAA_stage: mean of seen-task row means',
                  cohorts=[cohorts[m, d] for m in METHODS for d in DATASETS],
                  matched_multiseed_cells=2, single_run_cells=33,
                  original_emphasis_preserved=True)
(ROOT / 'statistics.json').write_text(json.dumps(statistics, indent=2) + '\n')
with (ROOT / 'statistics.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['method', 'dataset', 'n', 'seeds', 'faa_mean', 'faa_sample_sd', 'faa_sample_variance',
                     'aaa_stage_mean', 'aaa_stage_sample_sd', 'aaa_stage_sample_variance'])
    for r in statistics['cohorts']:
        writer.writerow([r['method'], r['dataset'], r['n'], ','.join(map(str, r['seeds']))] +
                        [r[m][s] for m in ['faa', 'aaa_stage'] for s in ['mean', 'sample_sd', 'sample_variance']])

separate_lines = []
for g in cl_gam['separate_protocol_groups']:
    runs = [verify_run(run) for run in g['runs']]
    cells = []
    for metric in ['faa', 'aaa_stage']:
        values = [run[metric] for run in runs]
        mean, sd = st.mean(values), st.stdev(values)
        assert abs(mean - g[metric]) < 1e-9
        assert abs(sd - g[metric + '_sd']) < 1e-9
        cells.append(f'\\ExpRes{{{mean:.2f}}}{{{sd:.2f}}}')
    separate_lines.append('GAM-EWCLoRA & ' + LABELS[g['dataset']] + ' & 3 & ' + ' & '.join(cells) + r'\\')
(ROOT / 'separate_protocol_multiseed.tex').write_text(r'''% Separate RNG-controlled protocol; do not merge with historical scores.
\begin{tabular}{llrrr}
\toprule
Method & Dataset & $n$ & FAA & AAA\\
\midrule
''' + '\n'.join(separate_lines) + '\n' + r'''\bottomrule
\end{tabular}
''')

preamble = r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=15mm]{geometry}
\usepackage{booktabs,array}
\newcommand{\ExpRes}[2]{\ensuremath{{#1}_{\scriptscriptstyle (#2)}}}
\makeatletter\setlength{\@fptop}{0pt}\makeatother
\pagestyle{empty}
\begin{document}
\input{finegrained_std.tex}
\end{document}
'''
(ROOT / 'finegrained_std_preview.tex').write_text(preamble)
print('Verified 37 current-table source matrices and 6 separate-protocol matrices; generated 70 metric cells.')
for d in ['cars196', 'flowers']:
    r = cohorts['EWCLoRA', d]
    print(d, r['n'], r['faa'], r['aaa_stage'])
