"""Add the user-selected historical OLoRA cohort to the v2 source data."""
import csv
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREVIOUS=HERE.parent/'methodcolors_v2'
DATASETS=['dbpedia','amazon','yahoo','agnews']
ORDERS={1:DATASETS,2:['dbpedia','amazon','agnews','yahoo'],3:['yahoo','amazon','agnews','dbpedia']}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(x):
    return str(Decimal(str(x)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))


def main():
    previous_points=read_csv(PREVIOUS/'taskwise_accuracy.csv')
    assert not any(r['model']=='Llama-3.2-1B' and r['method']=='OLoRA' for r in previous_points)
    evidence=read_csv(ROOT/'refine-logs/nlp_table_audit_20260919/llama_table_evidence.csv')
    selected=sorted([r for r in evidence if r['model']=='Llama-3.2-1B' and r['method']=='OLoRA'],key=lambda r:int(r['order']))
    assert len(selected)==3
    manifest=json.loads((PREVIOUS/'source_manifest.json').read_text())
    points=[];finals=[];stages=[];run_sources=[]
    for ref in selected:
        order=int(ref['order']);run=Path(ref['metric_file']).parent.parent
        assert Path(ref['console_logs']).is_file()
        assert sha(ref['metric_file'])==ref['metric_sha256']
        for stage,learned in enumerate(ORDERS[order],1):
            metric=run/f'{stage}-{learned}'/'predict_results.json'
            train=metric.parent/'train_results.json'
            assert train.is_file()
            data=json.loads(metric.read_text())
            measured={d:data['predict_exact_match_for_'+d] for d in DATASETS if 'predict_exact_match_for_'+d in data}
            seen=ORDERS[order][:stage]
            assert set(seen)<=set(measured) and data['predict_samples']==7600*len(measured)
            source=dict(model='Llama-3.2-1B',method='OLoRA',order=order,stage=stage,trained_dataset=learned,
                        metric_path=str(metric),metric_sha256=sha(metric),train_results_path=str(train),
                        train_results_sha256=sha(train),measured_datasets=list(measured),plotted_datasets=seen,
                        masked_unseen_datasets=sorted(set(measured)-set(seen)),run_dir=str(run),
                        final_summary=None,logs=ref['console_logs'],seed=42,world_size=8,ga=8,
                        cohort='historical_OLoRA_b64_s42_explicitly_selected_by_user')
            manifest['sources'].append(source)
            for dataset in seen:
                value=measured[dataset]
                points.append(dict(model='Llama-3.2-1B',method='OLoRA',order=order,stage=stage,
                                   trained_dataset=learned,evaluated_dataset=dataset,
                                   accuracy_percent=value,recorded_metric_percent=value,samples=7600,
                                   correct_if_recorded='',value_basis='predict_results.json (4 decimal places)',
                                   seed=42,world_size=8,ga=8,metric_path=str(metric),metric_sha256=sha(metric),count_source=''))
            aa=sum(measured[d] for d in seen)/stage
            stages.append(dict(model='Llama-3.2-1B',method='OLoRA',order=order,stage=stage,
                               trained_dataset=learned,seen_tasks=stage,seen_task_aa=aa,
                               logged_aggregate=data['predict_exact_match'],
                               logged_aggregate_includes_unseen=bool(set(measured)-set(seen)),metric_path=str(metric)))
            if stage==4:
                assert abs(aa-data['predict_exact_match'])<.00011
                assert fmt(aa)==ref['verified_faa']
                finals.append(dict(model='Llama-3.2-1B',method='OLoRA',order=order,faa=aa,
                                   displayed_faa=fmt(aa),seed=42,world_size=8,ga=8,
                                   metric_path=str(metric),exact_final_counts_source=''))
                run_sources.append({k:source[k] for k in ['model','method','order','run_dir','metric_path','logs','seed','world_size','ga']})
    assert len(points)==30 and len(stages)==12
    assert fmt(sum(r['faa'] for r in finals)/3)=='56.71'
    write_csv(HERE/'taskwise_accuracy.csv',previous_points+points)
    write_csv(HERE/'historical_olora_added_points.csv',points)
    write_csv(HERE/'stage_seen_task_aa.csv',read_csv(HERE.parent/'stage_seen_task_aa.csv')+stages)
    all_finals=read_csv(HERE.parent/'final_faa_by_order.csv')+finals
    write_csv(HERE/'final_faa_by_order.csv',all_finals)
    write_csv(HERE/'run_sources.csv',read_csv(HERE.parent/'run_sources.csv')+run_sources)
    manifest['missing']=[]
    manifest['version']='methodcolors_v3'
    manifest['selection_note']='User explicitly selected historical 1B OLoRA 58.61/53.12/58.42, mean56.71; newer unmatched table values are not used.'
    manifest['selection_references']=[dict(path=str(ROOT/'refine-logs/bf16_ewc_b64_s42_20260918'/name),sha256=sha(ROOT/'refine-logs/bf16_ewc_b64_s42_20260918'/name)) for name in ['STATUS_REPORT.md','RESULTS_INDEX.md']]
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    configs=read_csv(PREVIOUS/'protocol_configs.csv')
    old3b=next(r for r in configs if r['model']=='Llama-3.2-3B' and r['method_in_user_table']=='OLoRA')
    config1b=dict(old3b,model='Llama-3.2-1B',result_mapping='matched_user_selected_historical_58.61_53.12_58.42',evidence='refine-logs/nlp_table_audit_20260919/llama_table_evidence.csv')
    configs=[config1b if r['model']=='Llama-3.2-1B' and r['method_in_user_table']=='OLoRA' else r for r in configs]
    write_csv(HERE/'protocol_configs.csv',configs)
    summary=[]
    for model in ['Llama-3.2-1B','Llama-3.2-3B']:
        for method in manifest['methods']:
            rows=sorted([r for r in all_finals if r['model']==model and r['method']==method],key=lambda r:int(r['order']))
            assert len(rows)==3
            values=[float(r['faa']) for r in rows]
            summary.append(dict(model=model,method=method,order1=values[0],order2=values[1],order3=values[2],
                                average=sum(values)/3,order_range_pp=max(values)-min(values),status='verified_source'))
    write_csv(HERE/'final_summary.csv',summary)
    table=['# v3 最终 FAA（%）','','历史1B OLoRA已按用户选择纳入；其余结果沿用v2。所有列最后统一舍入。','']
    for model in ['Llama-3.2-1B','Llama-3.2-3B']:
        table += [f'## {model}','','| Method | Order-1 | Order-2 | Order-3 | Avg. |','|---|---:|---:|---:|---:|']
        for r in summary:
            if r['model']==model:table.append('| '+r['method']+' | '+' | '.join(fmt(r[k]) for k in ['order1','order2','order3','average'])+' |')
        table.append('')
    (HERE/'RESULTS.md').write_text('\n'.join(table)+'\n')
    checks=dict(previous_points_unchanged=read_csv(HERE/'taskwise_accuracy.csv')[:330]==previous_points,
                previous_points=330,historical_olora_added_points=30,total_points=360,
                total_source_stages=len(manifest['sources']),total_final_cells=len(all_finals),
                historical_olora_faa=[r['displayed_faa'] for r in finals],historical_olora_average='56.71',
                accuracy_min=min(float(r['accuracy_percent']) for r in previous_points+points),
                historical_values_are_four_decimal_metrics=True,raw_predictions_recomputed_this_turn=False)
    assert checks['previous_points_unchanged'] and len(manifest['sources'])==144 and len(all_finals)==36
    (HERE/'data_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
