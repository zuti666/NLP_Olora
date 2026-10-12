"""Replace both pure-OLoRA cohorts with logged OLoRA+GAM, preserving other data."""
import csv
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREVIOUS=HERE.parent/'methodcolors_v3'
DATASETS=['dbpedia','amazon','yahoo','agnews']
ORDERS={1:DATASETS,2:['dbpedia','amazon','agnews','yahoo'],3:['yahoo','amazon','agnews','dbpedia']}
EXPECTED={'1B':['65.81','58.28','66.92','63.67'],'3B':['71.62','70.94','72.05','71.54']}


def read_csv(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def write_csv(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fmt(x):return str(Decimal(str(x)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))


def main():
    previous=read_csv(PREVIOUS/'taskwise_accuracy.csv')
    kept=[r for r in previous if r['method']!='OLoRA']
    removed=[r for r in previous if r['method']=='OLoRA']
    assert len(kept)==300 and len(removed)==60
    manifest=json.loads((PREVIOUS/'source_manifest.json').read_text())
    excluded_sources=[r for r in manifest['sources'] if r['method']=='OLoRA']
    manifest['sources']=[r for r in manifest['sources'] if r['method']!='OLoRA']
    manifest['methods']=['OLoRA+GAM' if m=='OLoRA' else m for m in manifest['methods']]
    points=[];finals=[];stages=[];run_sources=[];protocol_sources=[]
    for size in ['1B','3B']:
        for order,sequence in ORDERS.items():
            run=ROOT/f'logs_and_outputs_llama_{size}'/f'order_{order}'/f'OLoRA-GAM-002_Llama-3.2-{size}_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4'
            log=ROOT/'Logs_Scripts_Short_Paper'/f'Llama-3.2-{size}_AllOrders_IncLoRA_OLoRA'/f'scripts_order{order}_llama{size}'/f'OLoRA-GAM_LLaMA2-{size.lower()}.log'
            assert log.exists()
            log_text=log.read_text(errors='replace')
            for witness in ['lora_strategy=olora,','optimizer_type=gam-adamw_hf,','seed=42,','data_seed=None,','gradient_accumulation_steps=8,','olora_lamda_1=0.5,']:
                assert witness in log_text,(log,witness)
            protocol_sources.append(dict(model='Llama-3.2-'+size,order=order,log_path=str(log),log_sha256=sha(log),
                                         strategy='olora',optimizer_type='gam-adamw_hf',seed=42,data_seed=None,
                                         world_size=8,per_device_batch=1,ga=8,global_batch=64,
                                         gam_rho=.02,gam_norm_rho=.2,gam_gamma=.03,gam_betas=[1,1,1],
                                         olora_lambda1=.5,olora_lambda2=0))
            for stage,learned in enumerate(sequence,1):
                metric=run/f'{stage}-{learned}'/'predict_results.json'
                train=metric.parent/'train_results.json'
                state=metric.parent/'trainer_state.json'
                assert metric.exists() and train.exists() and state.exists()
                data=json.loads(metric.read_text());tr=json.loads(train.read_text())
                assert str(tr['train_runtime']) in log_text,(log,train)
                measured={d:data['predict_exact_match_for_'+d] for d in DATASETS if 'predict_exact_match_for_'+d in data}
                seen=sequence[:stage]
                assert set(measured)==set(seen) and data['predict_samples']==7600*stage
                source=dict(model='Llama-3.2-'+size,method='OLoRA+GAM',order=order,stage=stage,trained_dataset=learned,
                            metric_path=str(metric),metric_sha256=sha(metric),train_results_path=str(train),
                            train_results_sha256=sha(train),measured_datasets=list(measured),plotted_datasets=seen,
                            masked_unseen_datasets=[],run_dir=str(run),final_summary=None,logs=str(log),
                            seed=42,world_size=8,ga=8,cohort='historical_OLoRA_GAM_b64_s42')
                manifest['sources'].append(source)
                for dataset in seen:
                    value=measured[dataset]
                    points.append(dict(model='Llama-3.2-'+size,method='OLoRA+GAM',order=order,stage=stage,
                                       trained_dataset=learned,evaluated_dataset=dataset,accuracy_percent=value,
                                       recorded_metric_percent=value,samples=7600,correct_if_recorded='',
                                       value_basis='predict_results.json (4 decimal places)',seed=42,world_size=8,ga=8,
                                       metric_path=str(metric),metric_sha256=sha(metric),count_source=''))
                aa=sum(measured[d] for d in seen)/stage
                stages.append(dict(model='Llama-3.2-'+size,method='OLoRA+GAM',order=order,stage=stage,
                                   trained_dataset=learned,seen_tasks=stage,seen_task_aa=aa,
                                   logged_aggregate=data['predict_exact_match'],logged_aggregate_includes_unseen=False,
                                   metric_path=str(metric)))
                if stage==4:
                    assert abs(aa-data['predict_exact_match'])<.00011
                    assert fmt(aa)==EXPECTED[size][order-1]
                    finals.append(dict(model='Llama-3.2-'+size,method='OLoRA+GAM',order=order,faa=aa,
                                       displayed_faa=fmt(aa),seed=42,world_size=8,ga=8,metric_path=str(metric),exact_final_counts_source=''))
                    run_sources.append({k:source[k] for k in ['model','method','order','run_dir','metric_path','logs','seed','world_size','ga']})
    assert len(points)==60 and len(stages)==24
    write_csv(HERE/'taskwise_accuracy.csv',kept+points)
    write_csv(HERE/'olora_gam_selected_points.csv',points)
    for name,new in [('stage_seen_task_aa.csv',stages),('final_faa_by_order.csv',finals),('run_sources.csv',run_sources)]:
        write_csv(HERE/name,[r for r in read_csv(PREVIOUS/name) if r['method']!='OLoRA']+new)
    manifest['missing']=[]
    manifest['version']='methodcolors_v4'
    manifest['selection_note']='User selected OLoRA+GAM in both models. Green curves explicitly labeled OLoRA+GAM; pure OLoRA cohorts are retained only in previous version v3.'
    manifest['selection_references']=protocol_sources
    manifest['previous_pure_olora_sources_excluded_from_plot']=excluded_sources
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    configs=read_csv(PREVIOUS/'protocol_configs.csv')
    for c in configs:
        if c['method_in_user_table']=='OLoRA':
            c.update(method_in_user_table='OLoRA+GAM',result_mapping='matched_user_selected_OLoRA_GAM',
                     strategy='olora',actual_optimizer='GAM(torch.optim.AdamW)',actual_adam_epsilon='1e-8',
                     actual_weight_decay='0.01',optimizer_evidence='historical_source_inference_no_runtime_object',
                     method_parameters='olora_lamda_1=0.5;olora_lamda_2=0;gam_grad_rho=0.02;gam_grad_norm_rho=0.2;gam_grad_gamma=0.03;gam_grad_beta_1/2/3=1;adaptive=false;grad_reduce=mean;perturb_eps=1e-12',
                     evidence='source_manifest.json: selection_references contains logs and hashes')
    write_csv(HERE/'protocol_configs.csv',configs)
    all_finals=read_csv(HERE/'final_faa_by_order.csv');summary=[]
    for size in ['1B','3B']:
        model='Llama-3.2-'+size
        for method in manifest['methods']:
            rows=sorted([r for r in all_finals if r['model']==model and r['method']==method],key=lambda r:int(r['order']))
            assert len(rows)==3
            values=[float(r['faa']) for r in rows]
            avg=sum(values)/3
            if method=='OLoRA+GAM':assert fmt(avg)==EXPECTED[size][3]
            summary.append(dict(model=model,method=method,order1=values[0],order2=values[1],order3=values[2],average=avg,
                                order_range_pp=max(values)-min(values),status='verified_source'))
    write_csv(HERE/'final_summary.csv',summary)
    report=['# v4 最终 FAA（%）','','两种模型的绿色曲线现为OLoRA+GAM。纯OLoRA结果保持在v3中，未混入当前OLoRA+GAM的均值。','']
    for size in ['1B','3B']:
        report += ['## Llama-3.2-'+size,'','| Method | Order-1 | Order-2 | Order-3 | Avg. |','|---|---:|---:|---:|---:|']
        for r in summary:
            if r['model']=='Llama-3.2-'+size:report.append('| '+r['method']+' | '+' | '.join(fmt(r[k]) for k in ['order1','order2','order3','average'])+' |')
        report.append('')
    (HERE/'RESULTS.md').write_text('\n'.join(report)+'\n')
    checks=dict(previous_unrelated_points_preserved=read_csv(HERE/'taskwise_accuracy.csv')[:300]==kept,
                pure_olora_points_replaced=60,olora_gam_points=60,total_points=360,total_stages=len(manifest['sources']),
                total_final_cells=len(all_finals),matched_training_log_stages=24,
                selected_results={size:EXPECTED[size] for size in EXPECTED},
                all_points_min=min(float(r['accuracy_percent']) for r in kept+points),
                raw_prediction_jsonl_recomputed_this_turn=False)
    assert checks['previous_unrelated_points_preserved'] and len(manifest['sources'])==144 and len(all_finals)==36
    (HERE/'data_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
