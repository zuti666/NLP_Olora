"""Generate a concise source inventory and result tables from extracted scores."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from build_data import HERE, METHODS, fmt, write_csv


def main():
    manifest = json.loads((HERE/'source_manifest.json').read_text())
    with (HERE/'final_faa_by_order.csv').open(encoding='utf-8-sig') as f:
        finals = list(csv.DictReader(f))
    with (HERE/'taskwise_accuracy.csv').open(encoding='utf-8-sig') as f:
        points = list(csv.DictReader(f))
    runs = []
    for source in manifest['sources']:
        if source['stage']==4:
            runs.append({k:source[k] for k in ['model','method','order','run_dir','metric_path','logs','seed','world_size','ga']})
    write_csv(HERE/'run_sources.csv', runs)
    lines = ['# 原始结果路径', '',
             '以下为匹配当前表格的33个模型/方法/Order运行。每个运行目录下有 `1-任务名` 至 `4-任务名` 四个阶段；绘图读取其中的 `predict_results.json`，训练证据为 `train_results.json` 及日志。', '',
             '1B新OLoRA三个Order未定位，未列入已确认源。具体每个点的来源和SHA256见 `source_manifest.json` 与 `taskwise_accuracy.csv`。', '',
             '| Model | Method | Order | 原始运行目录 |', '|---|---|---|---|']
    for r in runs:
        lines.append(f"| {r['model']} | {r['method']} | {r['order']} | [{Path(r['run_dir']).name}]({r['run_dir']}) |")
    (HERE/'SOURCE_PATHS.md').write_text('\n'.join(lines)+'\n')
    summary = []
    for model in ['Llama-3.2-1B','Llama-3.2-3B']:
        for method in METHODS:
            rows = sorted([r for r in finals if r['model']==model and r['method']==method], key=lambda r:int(r['order']))
            values = [float(r['faa']) for r in rows]
            row = dict(model=model,method=method,order1=values[0] if values else '',
                       order2=values[1] if values else '',order3=values[2] if values else '',
                       average=sum(values)/3 if values else '',
                       order_range_pp=max(values)-min(values) if values else '',
                       status='verified_source' if values else 'missing_source')
            summary.append(row)
    write_csv(HERE/'final_summary.csv',summary)
    table = ['% Generated from final_summary.csv; unlocated 1B OLoRA omitted.',
             '\\begin{tabular}{lrrrrrrrr}', '\\toprule',
             'Method & \\multicolumn{4}{c}{Llama-3.2-1B} & \\multicolumn{4}{c}{Llama-3.2-3B} \\\\',
             '& Order-1 & Order-2 & Order-3 & Avg. & Order-1 & Order-2 & Order-3 & Avg. \\\\', '\\midrule']
    for method in METHODS:
        values=[]
        for model in ['Llama-3.2-1B','Llama-3.2-3B']:
            row=next(r for r in summary if r['model']==model and r['method']==method)
            values.extend(fmt(row[k]) if row[k]!='' else '--' for k in ['order1','order2','order3','average'])
        table.append(method+' & '+' & '.join(values)+' \\\\')
    table.extend(['\\bottomrule','\\end{tabular}'])
    (HERE/'verified_final_table.tex').write_text('\n'.join(table)+'\n')
    changes=[]
    for model in ['Llama-3.2-1B','Llama-3.2-3B']:
        for method in METHODS:
            for order in ['1','2','3']:
                for dataset in manifest['datasets']:
                    rows=sorted([r for r in points if r['model']==model and r['method']==method and r['order']==order and r['evaluated_dataset']==dataset],key=lambda r:int(r['stage']))
                    if rows:
                        first,last=float(rows[0]['accuracy_percent']),float(rows[-1]['accuracy_percent'])
                        changes.append(dict(model=model,method=method,order=order,dataset=dataset,
                                            first_seen_stage=rows[0]['stage'],accuracy_after_first_learning=first,
                                            final_accuracy=last,final_minus_first_pp=last-first,
                                            subsequent_training_stages=4-int(rows[0]['stage'])))
    write_csv(HERE/'dataset_changes.csv',changes)
    expected = {
        'Llama-3.2-1B': {'SeqLoRA':[66.90,55.71,68.06], 'IncLoRA':[64.52,56.19,65.44],
                         'OLoRA':[56.96,55.74,67.32], 'TreeLoRA':[66.93,61.53,66.16],
                         'EWCLoRA':[67.19,61.92,66.54], 'FS-EWCLoRA':[69.85,68.13,70.62]},
        'Llama-3.2-3B': {'SeqLoRA':[67.86,67.26,67.40], 'IncLoRA':[65.23,66.74,69.55],
                         'OLoRA':[63.37,62.97,63.92], 'TreeLoRA':[72.02,69.01,73.46],
                         'EWCLoRA':[68.36,69.18,73.04], 'FS-EWCLoRA':[72.71,73.51,72.66]}}
    comparisons=[]
    for model,methods in expected.items():
        for method,reported in methods.items():
            for order,value in enumerate(reported,1):
                matches=[r for r in finals if r['model']==model and r['method']==method and r['order']==str(order)]
                actual=matches[0]['displayed_faa'] if matches else ''
                comparisons.append(dict(model=model,method=method,order=order,user_reported_faa=fmt(value),
                                        source_verified_faa=actual,
                                        status=('match' if actual==fmt(value) else 'MISMATCH') if matches else 'missing_source'))
    assert not any(r['status']=='MISMATCH' for r in comparisons)
    write_csv(HERE/'user_table_comparison.csv',comparisons)
    plot_checks=[]
    for file in HERE.glob('*_plotted_data.json'):
        for curve in json.loads(file.read_text()):
            matching=sorted([r for r in points if r['model']==curve['model'] and r['method']==curve['method'] and int(r['order'])==curve['order'] and r['evaluated_dataset']==curve['dataset']],key=lambda r:int(r['stage']))
            assert [int(r['stage']) for r in matching]==curve['x']
            assert [float(r['accuracy_percent']) for r in matching]==curve['y']
        plot_checks.append(dict(file=file.name,curves=len(json.loads(file.read_text())),all_points_equal_csv=True))
    (HERE/'plot_data_verification.json').write_text(json.dumps(plot_checks,indent=2)+'\n')
    README = '''# Llama 各任务顺序的准确率曲线

主图按模型分两组，每组横排三个 Order。颜色区分数据集，线型/标记区分方法；每条曲线从该数据集第一次学习后开始。另提供仅表格第一行 SeqLoRA 的两组图。

| 内容 | 文件 |
|---|---|
| Llama-1B 主图 | [PNG](llama_1b_taskwise_orders.png) / [PDF](llama_1b_taskwise_orders.pdf) / [SVG](llama_1b_taskwise_orders.svg) |
| Llama-3B 主图 | [PNG](llama_3b_taskwise_orders.png) / [PDF](llama_3b_taskwise_orders.pdf) / [SVG](llama_3b_taskwise_orders.svg) |
| 仅 SeqLoRA 1B | [PNG](llama_1b_seqlora_taskwise_orders.png) / [PDF](llama_1b_seqlora_taskwise_orders.pdf) |
| 仅 SeqLoRA 3B | [PNG](llama_3b_seqlora_taskwise_orders.png) / [PDF](llama_3b_seqlora_taskwise_orders.pdf) |
| 每个 Order、每个阶段、每个数据集的数值 | [RESULTS.md](RESULTS.md) / [CSV](taskwise_accuracy.csv) |
| 最终结果与顺序变化幅度 | [final_summary.csv](final_summary.csv) |
| 数据集首次学到后至最终的变化 | [dataset_changes.csv](dataset_changes.csv) |
| 原始结果路径 | [SOURCE_PATHS.md](SOURCE_PATHS.md) / [CSV](run_sources.csv) |
| 论文插图及表格 | [latex_includes.tex](latex_includes.tex) / [verified_final_table.tex](verified_final_table.tex) |

已定位132个阶段，绘制330个真实数据点；33个可追溯的最终FAA均与用户表格两位小数一致。1B新OLoRA的3个数值未找到匹配来源，主图明确标缺失，没有用旧OLoRA或插值替代。

1B：已确认的五种方法均在Order-2取得各自最低FAA。SeqLoRA的Order-1/2/3分别为66.90/55.71/68.06，顺序极差12.35个百分点；FS-EWCLoRA为69.85/68.13/70.62，极差2.49个百分点。SeqLoRA在Order-2中，Amazon从首次学习后的52.82降至25.82，AGNews从86.66降至62.79。

3B：SeqLoRA三个Order的FAA为67.86/67.26/67.40，极差0.59个百分点；FS-EWCLoRA为72.71/73.51/72.66，极差0.85个百分点。FS-EWCLoRA在Order-1/2取得最高FAA，Order-3最高为TreeLoRA73.46。总体FAA相近不表示每个数据集都稳定，例如3B SeqLoRA在Order-2的Amazon从55.43降至38.21。

以上为当前单seed结果的描述，seed、DDP卡数/GA、优化器及适配器预算不完全一致，不能把差值解释为仅由某一个组件造成。FS-EWCLoRA曲线对应实际ewclora_gam分支。详见protocol_configs.csv与图注。

绘图数据直接读取原始阶段指标，新运行最终点使用已经保存且核验的整数正确数；本轮没有重新计算原始预测JSONL。四位小数的历史记录按其实际精度保留。SC/TC任务类型聚合字段不作为数据集指标。3B SeqLoRA Order-1前三阶段日志包含未学任务，已按真实任务顺序屏蔽；未使用包含未学任务的总均值替代已见任务表现。

复现：先在原服务器执行`python build_data.py`提取原始指标；已有CSV时可直接运行`python gen_llama1b.py`和`python gen_llama3b.py`。第一行单方法版本使用`gen_llama1b_seqlora.py`与`gen_llama3b_seqlora.py`。最后运行`python package_results.py`汇总并打包。依赖Python和Matplotlib；字体优先使用仓库已有Times New Roman，否则使用DejaVu Serif。
'''
    (HERE/'README.md').write_text(README)
    artifact_manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix not in ['.zip'] and p.name!='ARTIFACT_MANIFEST.json'}
    (HERE/'ARTIFACT_MANIFEST.json').write_text(json.dumps(artifact_manifest,indent=2)+'\n')
    with zipfile.ZipFile(HERE/'llama_taskwise_orders.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(HERE.iterdir()):
            if p.is_file() and p.suffix!='.zip': z.write(p,arcname=HERE.name+'/'+p.name)
    print('Table comparisons:', {s:sum(r['status']==s for r in comparisons) for s in ['match','missing_source']})
    print('Plotted data checks:',plot_checks)


if __name__ == '__main__':
    main()
