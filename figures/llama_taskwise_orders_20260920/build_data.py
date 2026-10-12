"""Extract measured task-wise scores. Never infer curves from final FAA."""
import csv
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ['SeqLoRA', 'IncLoRA', 'OLoRA', 'TreeLoRA', 'EWCLoRA', 'FS-EWCLoRA']
DATASETS = ['dbpedia', 'amazon', 'yahoo', 'agnews']
ORDERS = {1: DATASETS, 2: ['dbpedia', 'amazon', 'agnews', 'yahoo'],
          3: ['yahoo', 'amazon', 'agnews', 'dbpedia']}
ALIASES = {'IncLoRA+EWC': 'EWCLoRA', 'IncLoRA-GAM-EWC': 'FS-EWCLoRA'}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return str(Decimal(str(value)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def main():
    with (ROOT / 'refine-logs/nlp_table_audit_20260919/llama_table_evidence.csv').open() as f:
        old = list(csv.DictReader(f))
    mappings = {}
    for row in old:
        method = ALIASES.get(row['method'], row['method'])
        model, order = row['model'], int(row['order'])
        if model.endswith('1B') and method in ['OLoRA', 'FS-EWCLoRA']:
            continue  # Superseded rows must not feed this table or its figures.
        mappings[model, method, order] = dict(
            run_dir=str(Path(row['metric_file']).parent.parent),
            final_summary=None, logs=row['console_logs'], seed=int(row['seed']),
            world_size=int(row['world_size']), ga=int(row['gradient_accumulation_steps']))
    cohorts = [
        ('Llama-3.2-1B', 'EWCLoRA', 'inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z', None),
        ('Llama-3.2-1B', 'FS-EWCLoRA', 'gam_ewc_bf16_b64_s1993_ddp4_20260919T131729Z', None),
        ('Llama-3.2-1B', 'TreeLoRA', 'treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z', 'llama1b'),
        ('Llama-3.2-3B', 'TreeLoRA', 'treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z', 'llama3b')]
    for model, method, name, subdir in cohorts:
        cohort = ROOT / 'experiment_queue' / name
        sub = cohort / subdir if subdir else cohort
        summary = next(sub.glob('final_results*.json'))
        result = read_json(summary)
        protocol = result['protocol']
        for order in ORDERS:
            mappings[model, method, order] = dict(
                run_dir=str(sub / 'full' / f'order_{order}'),
                final_summary=str(summary), logs=str(cohort / 'logs'),
                seed=protocol['seed'], world_size=protocol['world_size'],
                ga=protocol['gradient_accumulation_steps'])

    points, stages, finals, sources = [], [], [], []
    for (model, method, order), spec in sorted(mappings.items()):
        final_result = None
        if spec['final_summary']:
            final_result = next(r for r in read_json(spec['final_summary'])['orders'] if r['order'] == order)
        for stage, learned in enumerate(ORDERS[order], start=1):
            folder = Path(spec['run_dir']) / f'{stage}-{learned}'
            metric = folder / 'predict_results.json'
            train = folder / 'train_results.json'
            assert metric.exists() and train.exists(), (metric, train)
            data = read_json(metric)
            measured = {name: data[f'predict_exact_match_for_{name}'] for name in DATASETS
                        if f'predict_exact_match_for_{name}' in data}
            seen = ORDERS[order][:stage]
            assert set(seen) <= set(measured), (metric, seen, measured)
            assert data['predict_samples'] == 7600 * len(measured), metric
            source = dict(model=model, method=method, order=order, stage=stage,
                          trained_dataset=learned, metric_path=str(metric), metric_sha256=sha(metric),
                          train_results_path=str(train), train_results_sha256=sha(train),
                          measured_datasets=list(measured), plotted_datasets=seen,
                          masked_unseen_datasets=sorted(set(measured)-set(seen)), **spec)
            sources.append(source)
            values = []
            for dataset in seen:
                recorded = float(measured[dataset])
                value, basis, count_source = recorded, 'predict_results.json (4 decimal places)', ''
                correct = ''
                if stage == 4 and final_result:
                    task = final_result['task_results'][dataset]
                    value = task['correct'] * 100.0 / task['samples']
                    assert abs(recorded-value) <= .000051, (metric, dataset, recorded, value)
                    correct = task['correct']
                    basis, count_source = 'integer counts in verified final_results.json', spec['final_summary']
                values.append(value)
                points.append(dict(model=model, method=method, order=order, stage=stage,
                                   trained_dataset=learned, evaluated_dataset=dataset,
                                   accuracy_percent=value, recorded_metric_percent=recorded,
                                   samples=7600, correct_if_recorded=correct, value_basis=basis,
                                   seed=spec['seed'], world_size=spec['world_size'], ga=spec['ga'],
                                   metric_path=str(metric), metric_sha256=source['metric_sha256'],
                                   count_source=count_source))
            aa = sum(values)/len(values)
            stages.append(dict(model=model, method=method, order=order, stage=stage,
                               trained_dataset=learned, seen_tasks=stage, seen_task_aa=aa,
                               logged_aggregate=data['predict_exact_match'],
                               logged_aggregate_includes_unseen=bool(set(measured)-set(seen)),
                               metric_path=str(metric)))
            if stage == 4:
                assert abs(aa-data['predict_exact_match']) < .00011, (metric, aa)
                finals.append(dict(model=model, method=method, order=order, faa=aa,
                                   displayed_faa=fmt(aa), seed=spec['seed'], world_size=spec['world_size'],
                                   ga=spec['ga'], metric_path=str(metric),
                                   exact_final_counts_source=spec['final_summary'] or ''))

    missing = [dict(model='Llama-3.2-1B', method='OLoRA', order=n,
                    user_reported_faa=v, status='No matching training/result source located; excluded from all curves and summaries')
               for n, v in enumerate([56.96, 55.74, 67.32], 1)]
    manifest = dict(schema_version=1, datasets=DATASETS, orders=ORDERS, methods=METHODS,
                    fs_ewclora_implementation='ewclora_gam / IncLoRA-GAM-EWC', sources=sources,
                    missing=missing, metric='Normalized exact match (%)',
                    replication='One seed per method/model; orders are not independent seeds',
                    exclusions='Unseen datasets are masked even when historical logs evaluated them; SC/TC aggregates are not dataset scores',
                    protocol_note='Nominal batch64; seed42/1993, DDP4/GA16 vs DDP8/GA8, optimizer defaults and adapter budgets differ. See protocol_configs.csv.')
    (HERE/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    write_csv(HERE/'taskwise_accuracy.csv', points)
    write_csv(HERE/'stage_seen_task_aa.csv', stages)
    write_csv(HERE/'final_faa_by_order.csv', finals)
    write_csv(HERE/'missing_sources.csv', missing)
    config = ROOT/'refine-logs/nlp_table_config_summary_20260920/method_configs.csv'
    (HERE/'protocol_configs.csv').write_bytes(config.read_bytes())
    write_reports(points, stages, finals, manifest)
    checks = dict(source_stages=len(sources), plotted_points=len(points), matched_final_cells=len(finals),
                  missing_final_cells=len(missing), all_final_means_match_recorded_aggregate=True,
                  all_measured_samples_per_dataset=7600,
                  unseen_masked_stages=[s['metric_path'] for s in sources if s['masked_unseen_datasets']],
                  raw_predictions_recomputed_this_turn=False,
                  basis='Original stage metrics and previously verified final integer counts; checks performed by build_data.py')
    (HERE/'data_verification.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps({k:v for k,v in checks.items() if k!='unseen_masked_stages'}, indent=2))


def write_reports(points, stages, finals, manifest):
    lines = ['# Llama 逐任务真实结果（2026-09-20）', '',
             '来源先定位、后绘图。数值来自各训练阶段 `predict_results.json` 的四个数据集 exact-match 字段；新运行最后阶段采用已经保存并核验的整数正确数。未从 FAA 倒推中间点，也未对缺失结果插值。', '',
             '表中 FS-EWCLoRA 对应实际 `ewclora_gam`。新 1B OLoRA 56.96/55.74/67.32 尚缺对应原始来源，以下标缺失；旧 OLoRA 1B 数值未混入。', '',
             '共同名义 batch64，但 seed、卡数/GA、底层优化器、适配器参数预算存在差异，详见 `protocol_configs.csv`。每个方法/模型只有一个 seed，不绘制置信区间。', '',
             '横轴表示依次训练完第1至第4个任务。每条数据集曲线从该任务首次出现开始；未训练任务的早期分数即使在旧日志存在，也不纳入本图。Accuracy 为 normalized exact match (%)。', '']
    for model in ['Llama-3.2-1B', 'Llama-3.2-3B']:
        lines += [f'## {model}：最终 FAA', '', '| Method | Order-1 | Order-2 | Order-3 | Avg. |', '|---|---:|---:|---:|---:|']
        for method in METHODS:
            rows = sorted([r for r in finals if r['model']==model and r['method']==method], key=lambda r:r['order'])
            vals = [r['displayed_faa'] for r in rows]
            avg = fmt(sum(r['faa'] for r in rows)/3) if len(rows)==3 else '—'
            lines.append('| '+method+' | '+' | '.join(vals if rows else ['—']*3)+' | '+avg+' |')
        for order in ORDERS:
            lines += ['', f'### Order-{order}: '+ ' → '.join(ORDERS[order]), '',
                      '下表为各阶段已见任务平均准确率 AA；第4列即最终 FAA。早期阶段评估任务集合不同，AA 的下降不能单独解释为遗忘。', '',
                      '| Method | After task 1 | After task 2 | After task 3 | After task 4 (FAA) |', '|---|---:|---:|---:|---:|']
            for method in METHODS:
                rows = sorted([r for r in stages if r['model']==model and r['method']==method and r['order']==order],key=lambda r:r['stage'])
                lines.append('| '+method+' | '+' | '.join([fmt(r['seen_task_aa']) for r in rows] if rows else ['—']*4)+' |')
            lines += ['', '每个数据集的具体曲线点（—为尚未学到该任务，不代表0）：', '',
                      '| Method | Dataset | Task 1 | Task 2 | Task 3 | Task 4 |', '|---|---|---:|---:|---:|---:|']
            for method in METHODS:
                if not any(r['model']==model and r['method']==method for r in points):
                    continue
                for dataset in ORDERS[order]:
                    subset = {r['stage']:r for r in points if r['model']==model and r['method']==method and r['order']==order and r['evaluated_dataset']==dataset}
                    lines.append('| '+method+' | '+dataset+' | '+' | '.join(fmt(subset[t]['accuracy_percent']) if t in subset else '—' for t in range(1,5))+' |')
    lines += ['', '## 复现与文件', '',
              '- `source_manifest.json`：132个阶段的运行路径、日志目录、指标与训练结果 SHA256。',
              '- `taskwise_accuracy.csv`：每个真实曲线点及其原始字段来源。',
              '- `stage_seen_task_aa.csv`：每阶段已见任务 AA，并记录日志总均值是否含未见任务。',
              '- `final_faa_by_order.csv`：最终 FAA；`missing_sources.csv`：未核实的表格行。',
              '- `data_verification.json`：来源/样本量/终点核对记录；本次未重新遍历原始预测 JSONL。',
              '- 复现：在本目录执行 `python build_data.py`，然后 `python gen_llama1b.py` 与 `python gen_llama3b.py`。', '']
    (HERE/'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
