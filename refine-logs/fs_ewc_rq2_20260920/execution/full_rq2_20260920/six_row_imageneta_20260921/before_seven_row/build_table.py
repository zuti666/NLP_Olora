"""Read completed formal results and build the requested six-row table on CPU.

Does not train, evaluate a model, alter frozen inputs, or impute missing arms.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
SOURCES = {}


def read_json(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def stats(values):
    assert len(values) == 3 and all(math.isfinite(v) for v in values)
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "n": 3}


def fmt(stat, digits=2, latex=False):
    if stat is None:
        return r"\textemdash" if latex else "待补"
    a, b = stat["mean"], stat["sample_sd"]
    return (rf"\ExpRes{{{a:.{digits}f}}}{{{b:.{digits}f}}}" if latex
            else f"{a:.{digits}f}±{b:.{digits}f}")


raw_csv = BASE / 'final_report/individual.csv'
SOURCES[str(raw_csv)] = hashlib.sha256(raw_csv.read_bytes()).hexdigest()
existing = list(csv.DictReader(raw_csv.open()))
by_cell = {(int(x['seed']), x['branch']): x for x in existing}
anchors = {}
checks = []
for method, branch in [('sgd', 'A'), ('gam', 'B')]:
    for seed in range(3):
        run = BASE / f'formal_training/anchor_{method}_seed{seed}'
        receipt = read_json(run / 'COMPLETE.json')
        manifest = read_json(run / 'run_manifest.json')
        metrics_path = Path(receipt['validation']['metrics'])
        metrics = read_json(metrics_path)
        assert SOURCES[str(metrics_path)] == receipt['validation']['metrics_sha256']
        assert receipt['status'] == 'complete' and manifest['seed'] == seed
        assert manifest['anchor_method'] == method
        m = metrics['mechanism']['steps']['0']
        assert m['fisher_estimator_family'] == 'batch_gradient_square_proxy'
        assert not m['fisher_is_per_sample_empirical']
        assert m['fisher_num_samples'] == 1259 and not m['in_task_fisher_used_path']
        assert m['in_task_weight_merge_applied'] == 0
        assert m['train_ewc_mean'] == 0  # Task 1 has no historical task to protect.
        row = by_cell[(seed, branch)]
        anchors[(method, seed)] = {
            'a11': float(row['a11']),
            'B1_batch_proxy': m['in_task_fisher_final_raw_trace'],
            'raw_trace_anchor_train256': float(row['raw_trace_anchor_train256']),
            'checkpoint_sha256': row['anchor_checkpoint_sha256'],
            'checkpoint': str(run / 'checkpoints/task_00.pt'),
            'metrics': str(metrics_path),
        }
        checks.append({'kind': 'anchor', 'method': method, 'seed': seed, 'passed': True})

arms = [
    ('vanilla', 'Vanilla LoRA', 'sgd', 'sgd', 0, 'C', 'C', 'A'),
    ('shaping_only', 'Shaping only', 'gam', 'gam', 0, 'C+F', 'C+F', None),
    ('ewclora', 'EWCLoRA', 'sgd', 'sgd', 2000, 'C', 'C+P', 'C'),
    ('fs_ewclora', 'FS-EWCLoRA', 'gam', 'gam', 2000, 'C+F', 'C+F+P', 'sequence_gam'),
    ('task1_shaping', 'Task-1 shaping', 'gam', 'sgd', 2000, 'C+F', 'C+P', 'D'),
    ('task2_shaping', 'Task-2 shaping', 'sgd', 'gam', 2000, 'C', 'C+F+P', None),
    ('task1_shaping_no_ewc', 'Task-1 shaping without EWC (extra)', 'gam', 'sgd', 0, 'C+F', 'C', 'B'),
]
numeric = ['a11', 'B1_batch_proxy', 'raw_trace_anchor_train256', 'a21', 'a22', 'BWT', 'average_accuracy', 'd_QV_cont']
records, grouped, ledger = [], [], []
for arm, label, parent, future, lam, obj1, obj2, source in arms:
    rows = []
    for seed in range(3):
        anchor = anchors[(parent, seed)]
        row = {'arm': arm, 'variant': label, 'seed': seed, 'parent_method': parent,
               'future_optimizer': future, 'ewc_lambda': lam, 'task1_objective': obj1,
               'task2_objective': obj2, 'status': 'complete' if source else 'not_run',
               'a11': anchor['a11'], 'B1_batch_proxy': anchor['B1_batch_proxy'],
               'raw_trace_anchor_train256': anchor['raw_trace_anchor_train256'],
               'parent_checkpoint': anchor['checkpoint'], 'anchor_metrics': anchor['metrics']}
        row.update({k: None for k in ['a21', 'a22', 'BWT', 'average_accuracy', 'd_QV_cont']})
        if source == 'sequence_gam':
            run = BASE / f'formal_training/sequence_gam_seed{seed}'
            receipt = read_json(run / 'COMPLETE.json')
            manifest = read_json(run / 'run_manifest.json')
            path = Path(receipt['validation']['metrics'])
            metrics = read_json(path)
            assert SOURCES[str(path)] == receipt['validation']['metrics_sha256']
            assert receipt['status'] == 'complete'
            config = manifest['config']
            assert config['optimizer_type'] == 'gam' and config['lambda_flat'] == 1
            assert config['ewc_lambda'] == 2000 and config['epochs'] == 50
            for flag in ['weight_merge', 'in_task_weight_merge', 'approximate_fisher_in_task']:
                assert config[flag] is False
            matrix = metrics['cnn']['matrices']['t01']
            assert abs(matrix[0][0] - anchor['a11']) < .0051
            trajectory_path = BASE / f'formal_results/seed{seed}/gam/trajectory_fisher/summary.json'
            trajectory = read_json(trajectory_path)
            assert trajectory['status'] == 'complete'
            t0 = next(x for x in trajectory['tasks'] if x['task'] == 0)
            t1 = next(x for x in trajectory['tasks'] if x['task'] == 1)
            assert t0['endpoint_recovery']['checkpoint_sha256'] == anchor['checkpoint_sha256']
            assert t1['endpoint_recovery']['passed']
            row.update(a21=matrix[1][0], a22=matrix[1][1],
                       d_QV_cont=t1['effective_weight_drift'],
                       result_source=str(path), drift_source=str(trajectory_path))
            assert math.isclose(row['d_QV_cont'] ** 2, t1['effective_weight_drift_squared'], rel_tol=1e-10)
        elif source:
            old = by_cell[(seed, source)]
            summary_path = Path(old['summary'])
            read_json(summary_path)
            assert SOURCES[str(summary_path)] == old['summary_sha256']
            assert old['anchor_method'] == parent and old['task1_optimizer'] == future
            assert float(old['ewc_lambda']) == lam
            row.update(a21=float(old['a21']), a22=float(old['a22']), d_QV_cont=float(old['drift_l2']),
                       result_source=str(summary_path), drift_source=str(summary_path))
        if source:
            row['BWT'] = row['a21'] - row['a11']
            row['average_accuracy'] = (row['a21'] + row['a22']) / 2
            assert all(math.isfinite(row[k]) for k in numeric)
        ledger.append({'arm': arm, 'seed': seed, 'requested_six_row': arm != 'task1_shaping_no_ewc',
                       'status': row['status'], 'anchor_available': True,
                       'observed_metrics_valid': bool(source), 'missing_metrics': [k for k in numeric if row[k] is None]})
        records.append(row); rows.append(row)
    grouped.append({'arm': arm, 'variant': label, 'task1_objective': obj1, 'task2_objective': obj2,
                    'status': 'complete' if source else 'not_run',
                    'statistics': {k: stats([x[k] for x in rows]) if all(x[k] is not None for x in rows) else None for k in numeric}})

paired = []
for left, right in [('fs_ewclora', 'ewclora'), ('fs_ewclora', 'task1_shaping'), ('task1_shaping', 'ewclora')]:
    aa = {x['seed']: x for x in records if x['arm'] == left}
    bb = {x['seed']: x for x in records if x['arm'] == right}
    for key in ['a21', 'a22', 'BWT', 'average_accuracy', 'd_QV_cont']:
        deltas = [aa[s][key] - bb[s][key] for s in range(3)]
        paired.append({'contrast': f'{left}-{right}', 'metric': key, **stats(deltas),
                       'seed0_delta': deltas[0], 'seed1_delta': deltas[1], 'seed2_delta': deltas[2],
                       'positive_count': sum(v > 0 for v in deltas)})

data = {'scope': 'Formal ImageNet-A; six requested arms plus one existing supplementary arm',
        'seeds': [0, 1, 2], 'expected_main_cells': 18, 'validated_main_cells': 12,
        'missing_main_cells': 6, 'new_training_launched': False,
        'definitions': {'B1_batch_proxy': 'Task-1 endpoint training batch-gradient delta-W Fisher proxy, before normalization; 1259 samples',
                        'd_QV_cont': 'Frobenius norm of total effective Q/V weight change from Task 1 endpoint to Task 2 endpoint; excludes heads',
                        'BWT': 'a21 minus common unrounded FP32 a11, computed per seed',
                        'missing': 'not_run; never zero or an estimated result'},
        'aggregates': grouped, 'individual': records, 'paired_contrasts': paired,
        'source_sha256': SOURCES}
(OUT / 'table_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
(OUT / 'VALIDITY_LEDGER.json').write_text(json.dumps({'expected_main_cells': 18, 'validated_main_cells': 12,
    'missing_main_cells': 6, 'invalid_observed_cells': 0, 'anchor_checks': checks, 'cells': ledger}, ensure_ascii=False, indent=2) + '\n')
for filename, rows in [('individual.csv', records), ('paired_contrasts.csv', paired)]:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / filename).open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=keys); w.writeheader(); w.writerows(rows)

md = ['# ImageNet-A：六行两任务诊断表（已有结果与缺口）', '',
      '状态：六行目标中4行完整、2行缺失；按seed计12/18已有正式结果、6/18未训练。另有GAM→SGD无EWC的3个已完成结果，单独列在附表。未启动新GPU实验。', '',
      '第一/第二任务对应代码task0/task1。数值为seed0/1/2均值±样本标准差；a21/a22为40类已见类别下的任务分组准确率，BWT逐seed用a21−a11计算。第一任务无历史保护损失；第二任务含P的行λ=2000，其余λ=0。', '',
      '| Variant | Task 1 | Task 2 | a11 | B1 batch proxy | a21 | a22 | BWT | d_QV_cont | 两任务平均 |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
def mdrow(g):
    s = g['statistics']
    return '| ' + ' | '.join([g['variant'], g['task1_objective'], g['task2_objective']] +
        [fmt(s[k], d) for k, d in [('a11',2),('B1_batch_proxy',3),('a21',2),('a22',2),('BWT',2),('d_QV_cont',4),('average_accuracy',2)]]) + ' |'
md += [mdrow(g) for g in grouped[:6]]
md += ['', '缺失行中的a11/B1是其已完成、待复用父起点的测量，不代表第二任务已执行。', '',
       '**六行之外的已完成消融**', '', md[6], md[7], mdrow(grouped[6]), '',
       '**指标对应**', '',
       '- B1取正式anchor训练日志的in_task_fisher_final_raw_trace：原生batch-gradient平方proxy，未归一化、无路径平均。SGD/GAM的原始逐seed值与路径见individual.csv。',
       '- 之前报告的1451.954/634.510是固定旧train256上的逐样本raw Fisher；它与本表B1不同，不能混填。两种指标均已记录。',
       '- 实际LoRA作用在Q/V，且第二任务延续同一LoRA分支，故将旧表fresh-adapter r_KV改为d_QV_cont（持续分支的端点间位移）；该量排除分类头，不是A/B因子的参数距离，也不可与旧fresh-KV范数直接横比。',
       '- F表示GAM实现的shaping更新，不声称代码直接优化一个显式Fisher-trace惩罚。',
       '- FS-EWCLoRA复用正式sequence_gam的t01矩阵及B2 t1有效位移；父anchor与其它同seed分支一致，无需重训这3条。',
       '- FS分支尚无B3同口径的endpoint-train256逐样本raw Fisher，但用户本表只需第一任务B1，因此不阻塞这行。',
       '- 当前表不包含full-model对应实验；caption不声称尚未完成的full-model附录结果存在。', '',
       '**已完成分支的配对差值**', '', '| 对照 | 指标 | 均值±SD | seed0/1/2差值 |', '|---|---|---:|---|']
for p in paired:
    if p['metric'] in ['a21','a22','average_accuracy']:
        md.append(f"| {p['contrast']} | {p['metric']} | {p['mean']:+.3f}±{p['sample_sd']:.3f} | {p['seed0_delta']:+.3f}/{p['seed1_delta']:+.3f}/{p['seed2_delta']:+.3f} |")
md += ['', '上述差值仅用于已有匹配分支的描述性比较；完整六行尚缺两行，不能估计缺失的保护或Task-2-only shaping对照。之前“平均收益小”的结论对应GAM起点→未来SGD；不能套用到未来也使用GAM的完整方法。', '',
       '**审查与复现**', '',
       '- [配置实现与最少补训](CONFIG_IMPLEMENTATION_AUDIT.md)',
       '- [连续GAM结果复用](CONTINUOUS_GAM_REUSE.md)',
       '- [指标定义审查](METRIC_MAPPING_AUDIT.md)',
       '- [缺失实验清单](MISSING_EXPERIMENTS.md)',
       '- [LaTeX表](table_imageneta.tex) / [逐seed CSV](individual.csv) / [完整数据与来源SHA](table_data.json) / [完整性台账](VALIDITY_LEDGER.json)',
       '- CPU复现：python build_table.py；此脚本不启动模型或训练。', '']
(OUT / 'TABLE_IMAGENETA.md').write_text('\n'.join(md))

def obj(x):
    return '$' + '+'.join(r'\mathsf '+v for v in x.split('+')) + '$'
tex = [r'% Requires booktabs; values are mean with sample-SD subscripts.',
       r'\providecommand{\ExpRes}[2]{\ensuremath{#1_{\!\pm #2}}}',
       r'\begin{table*}[t]', r'\centering',
       r'\caption{Two-task diagnostics of solution shaping and historical protection with rank-10 Q/V LoRA on ImageNet-A. $\mathsf C$, $\mathsf F$, and $\mathsf P$ denote current-task fitting, the implemented GAM shaping update, and historical EWC protection. Values are means over seeds 0, 1, and 2, with sample standard deviations as subscripts. $\mathcal B_1$ is the unnormalized Task-1 endpoint batch-gradient Fisher proxy in effective Q/V coordinates (fit loader, batch size 128); $d_{QV}^{\mathrm{cont}}$ is the endpoint-to-endpoint effective Q/V weight change of the same continued LoRA branch during Task 2, excluding classifier heads; this is not a fresh-adapter norm. References are endpoints without trajectory averaging. Dashes indicate unrun Task-2 branches; their Task-1 entries reuse the measured parent anchors.}',
       r'\label{tab:two_task_lora_diagnostics_imageneta}', r'\scriptsize',
       r'\setlength{\tabcolsep}{3.0pt}', r'\renewcommand{\arraystretch}{1.08}',
       r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}',
       r'\toprule',
       r'& \multicolumn{2}{c}{Training objective} & \multicolumn{2}{c}{Task 1 endpoint} & \multicolumn{3}{c}{Task 2 endpoint} & Task 2 update \\',
       r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}',
       r'Variant & Task 1 & Task 2 & $a_{11}\uparrow$ & $\mathcal B_1$ & $a_{21}\uparrow$ & $a_{22}\uparrow$ & BWT$\uparrow$ & $d_{QV}^{\mathrm{cont}}$ \\',
       r'\midrule']
for i, g in enumerate(grouped[:6]):
    if i == 4: tex.append(r'\midrule')
    s = g['statistics']
    tex.append(' & '.join([g['variant'], obj(g['task1_objective']), obj(g['task2_objective'])] +
        [fmt(s[k], d, True) for k,d in [('a11',2),('B1_batch_proxy',3),('a21',2),('a22',2),('BWT',2),('d_QV_cont',4)]]) + r' \\')
tex += [r'\bottomrule', r'\end{tabular*}', r'\end{table*}', '']
(OUT / 'table_imageneta.tex').write_text('\n'.join(tex))
print(json.dumps({'main_validated':12,'main_expected':18,'missing_training_runs':6,
                  'batch_proxy_sgd':grouped[0]['statistics']['B1_batch_proxy'],
                  'batch_proxy_gam':grouped[3]['statistics']['B1_batch_proxy'],
                  'fs_ewclora':grouped[3]['statistics']}, ensure_ascii=False))
