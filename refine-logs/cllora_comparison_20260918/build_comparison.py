"""Read-only audit of experiment sources; write a separate comparison package."""
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = Path('/data/140-1/users/liying/CopyFlatCV')
RESULTS = ROOT / 'GAM-EWC/results'
RUN = ROOT / 'refine-logs/gam-ewc-fine-gapfill-20260918-attempt2'
DATASETS = ['cub200', 'cars196', 'aircraft', 'flowers', 'pets', 'het5']
NAMES = ['CUB-200', 'Cars196', 'Aircraft', 'Flowers102', 'Pets', '5Datasets']
METHODS = ['Vanilla LoRA', 'InfLoRA', 'OLoRA', 'SDLoRA', 'CL-LoRA', 'EWCLoRA', 'GAM-EWCLoRA']
EXPANDED = METHODS[:4] + ['Vanilla LoRA + GAM'] + METHODS[4:]
MAP = dict(seqlora='Vanilla LoRA', inflora='InfLoRA', olora='OLoRA', sdlora='SDLoRA',
           native_ewc='EWCLoRA', GAM='Vanilla LoRA + GAM', **{'GAM+EWC': 'GAM-EWCLoRA'})

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def csv_write(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()})

selected = read(RESULTS / 'fine_nlp_tables_20260918/vision_selected.json')
alignment = {r['metrics']: r for r in read(RESULTS / 'protocol_alignment_20260918/legacy_source_config_audit.json')}
records = []
for r in selected:
    if r['table_method'] in MAP and r['status'] == 'verified':
        records.append(dict(r, method=MAP[r['table_method']], campaign='archived'))
for r in read(RESULTS / 'existing_seeds_20260918/het5_existing.json'):
    method = MAP.get(r['model'])
    if r['model'] == 'as1_normfisher_fishershape' and float(r['params'].get('ewc_lambda', 0)) > 0:
        method = 'EWCLoRA'
    if method:
        records.append(dict(r, method=method, campaign='archived'))
for r in read(RESULTS / 'ewc_cars_flowers_wiki_20260918/per_run.json'):
    if r['family'] == 'alpha0_historical_sweep' and str(r['seed']) == '0' and r['status'] == 'verified' and float(r['lambda']) == 2000:
        records.append(dict(r, method='EWCLoRA', campaign='archived_alpha0_wrapper'))

receipts = []
for p in sorted((RUN / 'receipts').glob('*.json')):
    r = read(p)
    if r.get('stage') != 'formal':
        continue
    receipts.append(r)
    if r['status'] != 'completed':
        continue
    config = read(r['config'])
    assert sha(r['config']) == r['config_sha256']
    assert sha(r['log']) == r['log_sha256']
    records.append(dict(source=r['metric_path'], sha256=r['metric_sha256'], log=r['log'],
                        params=config, model=config['model_name'], method='CL-LoRA' if r['method'] == 'CL-LoRA' else 'GAM-EWCLoRA',
                        dataset=r['dataset'], seed=r['seed'], tasks=r['expected_tasks'],
                        faa=r['metrics']['faa'], aaa_stage=r['metrics']['aaa_stage'],
                        status='verified', campaign='aligned_port_20260918',
                        config_path=r['config'], receipt=str(p)))

manifest, configs, checks = [], [], []
for r in records:
    p = r['params']
    a = read(r['source'])['cnn']['matrices']['final']
    n = len(a)
    assert n == int(r['tasks']), r['source']
    assert all(len(row) == n for row in a)
    assert all(math.isfinite(a[i][j]) for i in range(n) for j in range(i + 1)), r['source']
    assert sha(r['source']) == r['sha256'], r['source']
    faa = sum(a[-1]) / n
    aaa = sum(sum(a[i][:i+1]) / (i+1) for i in range(n)) / n
    assert abs(faa - r['faa']) < 1e-8
    assert abs(aaa - r['aaa_stage']) < 1e-8
    prior = alignment.get(r['source'])
    config = r.get('config_path') or (prior or {}).get('resolved_config') or p.get('config')
    if config:
        candidates = [Path(config), ROOT / config,
                      Path(str(config).replace('/data/140-0/', '/data/140-1/'))]
        config = next((str(c) for c in candidates if c.is_file()), None)
    if prior:
        assert sha(r['log']) == prior['log_sha256']
        assert sha(config) == prior['config_sha256']
    for role, path in [('metrics', r['source']), ('log', r['log']), ('config', config), ('receipt', r.get('receipt'))]:
        if path:
            manifest.append(dict(method=r['method'], dataset=r['dataset'], role=role, path=path, sha256=sha(path)))
    r['config_path'] = config
    r['recomputed_faa'] = faa
    r['recomputed_aaa_stage'] = aaa
    r['aaa_triangle'] = sum(a[i][j] for i in range(n) for j in range(i+1)) / (n*(n+1)/2)
    r['final_task_accuracy'] = a[-1]
    r['bwt_final'] = sum(a[-1][i]-a[i][i] for i in range(n-1))/(n-1)
    r['forget_avg'] = sum(max(a[t][i] for t in range(i,n-1))-a[-1][i] for i in range(n-1))/(n-1)
    checks.append(dict(method=r['method'], dataset=r['dataset'], tasks=n,
                       source_hash_matches=True, complete_finite_observed_matrix=True, metrics_match=True))
    keys = ['model_name', 'optimizer', 'optimizer_type', 'scheduler', 'init_epoch', 'epochs',
            'init_lr', 'lrate', 'lora_rank', 'batch_size', 'init_momentum', 'momentum',
            'init_weight_decay', 'weight_decay', 'memory_size', 'class_shuffle', 'init_cls',
            'increment', 'nb_tasks', 'ewc_lambda', 'ewc_gamma', 'fisher_mode', 'ewc_max_batches',
            'ewc_normalize_fisher', 'ewc_normalize_accumulated_fisher', 'ewc_fisher_damping_beta',
            'fs_direction_mode', 'fs_alpha', 'grad_rho', 'grad_norm_rho', 'grad_gamma',
            'msa', 'shared_pos', 'specific_pos', 'kd_ratio', 'temperature', 'orth_ratio']
    row = dict(method=r['method'], dataset=r['dataset'], seed=r['seed'], campaign=r['campaign'],
               tasks=n, faa=r['faa'], aaa_stage=r['aaa_stage'])
    row.update({k: p.get(k, 'not_recorded') for k in keys})
    row.update(metrics_path=r['source'], log_path=r['log'], config_path=config)
    configs.append(row)

lookup = {(r['method'], r['dataset']): r for r in records}
assert len(lookup) == len(records), 'Duplicate method/dataset selection'
for m in METHODS:
    for d in DATASETS:
        assert (m,d) in lookup or (m,d) == ('GAM-EWCLoRA','het5')

def pair(method, dataset):
    r = lookup.get((method, dataset))
    return '--' if r is None else f"{r['faa']:.2f} / {r['aaa_stage']:.2f}"

def markdown(methods):
    return '\n'.join(['| Method | ' + ' | '.join(NAMES) + ' |',
                     '|---|' + '---:|' * 6] +
                    ['| ' + m + ' | ' + ' | '.join(pair(m,d) for d in DATASETS) + ' |' for m in methods])

NOTE = r'''ViT-B/16; checkpoint identifier \textnormal{augreg2\_in21k\_ft\_in1k}; rank 16 per adapter and batch 128; SGD base optimizer (momentum and weight decay zero), cosine schedule, no replay, fixed class order. FAA is the final row's task-macro mean; AAA is the mean of stage-wise task-macro means, not the legacy lower-triangle mean. All records are single runs. Five individual datasets use 40 epochs/task; 5Datasets uses 30 for Vanilla LoRA, CL-LoRA and EWC, versus 40 for InfLoRA, OLoRA and SDLoRA (the pending GAM-EWC run uses 30). Seed 0 except EWC/Pets and EWC/5Datasets (seed 2). SDLoRA/Aircraft uses LR .02; other Aircraft runs use .05. $^{w}$Historical two-probe $\alpha=0$ wrapper, not native one-pass EWC. EWC $\lambda$: CUB 200000, Cars/Flowers 2000, Aircraft 200, Pets 200000, 5Datasets 50000; GAM-EWC uses 2000. GAM-specific settings are incompletely logged, so GAM versus GAM-EWC is not a lambda-only comparison. $^{p}$CL-LoRA is an aligned port with shared/specific adapters, KD and its native prototype classifier; equal rank does not imply equal parameter or compute budgets. Historical data manifests, checkpoint file identity, code and RNG are not fully certified; Available Flowers loaders differ in indexing, but agree on the current data copy; historical identity remains unverified. Results support a qualified descriptive comparison, not a strictly matched ablation or a statistical superiority claim. Dashes indicate no eligible completed run. No best-value bolding.'''

def latex(methods, label):
    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{Fine-grained continual learning under documented configurations. Each dataset reports FAA and stage-averaged AAA (\%). 5Datasets is the cross-dataset task stream. Configuration exceptions are stated below.}',
             r'\label{' + label + '}',
             r'\scriptsize\setlength{\tabcolsep}{2.2pt}\renewcommand{\arraystretch}{1.08}',
             r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l*{12}{r}@{}}',r'\toprule',
             'Method & ' + ' & '.join(r'\multicolumn{2}{c}{'+n+'}' for n in NAMES) + r'\\',
             ''.join(r'\cmidrule(lr){'+str(2*i+2)+'-'+str(2*i+3)+'}' for i in range(6)),
             ' & ' + ' & '.join(['FAA', 'AAA'] * 6) + r'\\\midrule']
    for m in methods:
        cells = [m + (r'$^{p}$' if m == 'CL-LoRA' else '')]
        for d in DATASETS:
            r = lookup.get((m,d))
            suffix = r'$^{w}$' if m == 'EWCLoRA' and d in ['cars196','flowers'] else ''
            cells.extend(['--','--'] if r is None else [f"{r['faa']:.2f}"+suffix, f"{r['aaa_stage']:.2f}"+suffix])
        lines.append(' & '.join(cells) + r'\\')
    lines += [r'\bottomrule\end{tabular*}', r'\par\vspace{3pt}\begin{minipage}{\linewidth}\scriptsize',NOTE,
              r'\end{minipage}', r'\end{table*}']
    return '\n'.join(lines) + '\n'

for name, methods in [('main',METHODS), ('expanded',EXPANDED)]:
    (OUT / f'table_{name}.md').write_text(markdown(methods)+'\n')
    (OUT / f'table_{name}.tex').write_text(latex(methods, 'tab:exp_fine' if name=='main' else 'tab:exp_fine_expanded'))
    assert all(line.count('&')==12 for line in (OUT/f'table_{name}.tex').read_text().splitlines()
               if '&' in line and '\\multicolumn' not in line)

deltas=[]
for r in records:
    base = lookup[('Vanilla LoRA',r['dataset'])]
    deltas.append(dict(method=r['method'],dataset=r['dataset'],
                       delta_faa_pp=r['faa']-base['faa'],delta_aaa_stage_pp=r['aaa_stage']-base['aaa_stage'],
                       faa_relative_change_pct=100*(r['faa']/base['faa']-1),
                       interpretation='descriptive only; see configuration and data caveats'))
summary=[dict(method=r['method'],dataset=r['dataset'],seed=r['seed'],faa=r['faa'],bwt_final=r['bwt_final'],forget_avg=r['forget_avg'],
              aaa_stage=r['aaa_stage'],aaa_triangle=r['aaa_triangle'],tasks=r['tasks'],
              campaign=r['campaign'],source=r['source'],sha256=r['sha256']) for r in records]
csv_write('results.csv',summary)
csv_write('configurations.csv',configs)
csv_write('deltas_vs_vanilla.csv',deltas)
(OUT/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
(OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
status=dict(checked_at=datetime.now(ZoneInfo('Europe/Madrid')).isoformat(),
            verified_completed_cells=len(records),main_completed_cells=sum(r['method'] in METHODS for r in records),
            checks=checks,formal_receipts=receipts,
            additional_gam_het5='Historical SeqLoRA+GAM has only 3/5 stages; excluded.')
(OUT/'verification.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
preview = r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=15mm]{geometry}
\usepackage{booktabs,array,amsmath}
\makeatletter\setlength{\@fptop}{0pt}\makeatother
\begin{document}
\input{table_expanded.tex}
\clearpage
\noindent\textbf{Configuration comparison (single-run records)}\par\medskip
\small
\noindent\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}
\toprule
Dataset & CUB & Cars & Aircraft & Flowers & Pets & 5Datasets\\
\midrule
Tasks & 10 & 10 & 10 & 10 & 9 & 5\\
Initial classes & 20 & 16 & 10 & 12 & 5 & 200\\
Increment & 20 & 20 & 10 & 10 & 4 & variable\\
Default LR & .01 & .05 & .05 & .0025 & .0025 & .02\\
Epochs/task & 40 & 40 & 40 & 40 & 40 & 30/40\\
EWC lambda & 200000 & 2000 & 200 & 2000 & 200000 & 50000\\
GAM-EWC lambda & 2000 & 2000 & 2000 & 2000 & 2000 & 2000\\
\bottomrule
\end{tabular*}
\par\medskip
5Datasets task sizes: 200, 100, 196, 102, 37; total 635 classes.
SDLoRA/Aircraft LR is .02. EWC/Pets and EWC/5Datasets use seed 2; others use seed 0.
Cars/Flowers EWC use an alpha-zero wrapper. No claim of equal compute or strict component isolation is made.
\par\medskip
For task accuracy $a_{t,i}$ after stage $t$,
\[\mathrm{FAA}=\frac1T\sum_{i=1}^{T}a_{T,i},\qquad
\mathrm{AAA}_{\rm stage}=\frac1T\sum_{t=1}^{T}\frac1t\sum_{i=1}^{t}a_{t,i}.\]
The archived JSON field \textnormal{cnn.final.AAA} averages lower-triangular cells, which weights later stages more. Every numeric table cell above is recomputed from the same completed task matrix using the stage definition.
\par\medskip
Historical checkpoint hub identifiers match the new port; exact historical file revisions and image/label manifests are not available for all runs. Available Flowers loaders changed indexing code, but reproduce identical image/label mappings on the current data copy; the historical copy is unavailable. The new port's frozen data manifests therefore do not certify historical equivalence. Its native prototype classifier and adapter accumulation also differ from linear-head baselines. These differences prevent a strictly controlled superiority claim.
\end{document}
'''
(OUT/'comparison_preview.tex').write_text(preview)
print(json.dumps({k:status[k] for k in ['checked_at','verified_completed_cells','main_completed_cells']},indent=2))
