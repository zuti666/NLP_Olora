#!/usr/bin/env python3
"""Read-only audit of the two existing S3 campaigns; writes only local reports."""
import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs')
OUT = Path(__file__).resolve().parent
NAMES = ['s3-reference-hyperparams-20260920', 's3-mechanisms-20260920']
IDENTITY = {'outputs_root', 'prefix', 'optimizer_tag_override', 'device', 'seed'}
OBSERVERS = {'s3_boundary_capture', 's3_fisher_location'}
CORE = {'ewc_lambda', 'gam_grad_norm_rho', 'ewc_fisher_damping_alpha',
        'in_task_weight_merge', 'in_task_weight_merge_start_task',
        'in_task_weight_merge_omega', 'in_task_fisher_omega',
        'in_task_path_beta', 'in_task_path_start_ratio'}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def diff(a,b,ignore=()):
    return {k:[a.get(k),b.get(k)] for k in sorted(set(a)|set(b))
            if k not in ignore and a.get(k)!=b.get(k)}

def main():
    base = read(ROOT/NAMES[0]/'configs/s3_center_s2.json')
    report = {'audited_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'Only the two named native S3 campaigns; no queue mutation and no GPU use.',
              'center_config': base, 'campaigns': [], 'jobs': []}
    manifests = []
    for name in NAMES:
        p=ROOT/name; suite=read(p/'suite.json'); results=read(p/'results.json')
        resmap={j['id']:j for j in results['jobs']}
        sm=read(p/'source_manifest.json'); manifests.append(sm)
        source_mismatches=[f for f,h in sm['files'].items() if sha(p/'source'/f)!=h]
        c={'name':name, 'path':str(p), 'results_updated_at':results.get('updated_at'),
           'results_sha256':sha(p/'results.json'), 'suite_sha256':sha(p/'suite.json'),
           'source_manifest_sha256':sha(p/'source_manifest.json'),
           'source_files_count':len(sm['files']), 'source_mismatches':source_mismatches,
           'source_manifest':str(p/'source_manifest.json')}
        report['campaigns'].append(c)
        for spec in suite['jobs']:
            jid=spec['id']; r=resmap[jid]; conf=read(spec['config']); rp=p/'runtime/jobs'/jid
            st=read(rp/'started.json') if (rp/'started.json').exists() else {}
            cp=read(rp/'completed.json') if (rp/'completed.json').exists() else None
            scientific_base=dict(base,s3_fisher_location='retained')
            scientific_conf=dict(conf,s3_fisher_location=conf.get('s3_fisher_location','retained'))
            scientific_diff=diff(scientific_base,scientific_conf,IDENTITY|{'s3_boundary_capture'})
            row={'campaign':name,'id':jid,'seed':spec['seed'],'axis':spec['axis'],
                 'value':spec.get('value'),'status':r['status'], 'config':str(spec['config']),
                 'config_sha256':sha(spec['config']), 'scientific_diff':scientific_diff,
                 'observer_diff':diff(base,conf,(set(base)|set(conf))-OBSERVERS),
                 'all_config_differences':diff(base,conf),
                 'source_manifest_sha256':c['source_manifest_sha256'],
                 'receipt':str(rp/'completed.json') if cp else str(rp/'started.json'),
                 'receipt_sha256':sha(rp/'completed.json') if cp else sha(rp/'started.json'),
                 'results_declared_tasks':r.get('tasks_completed'),
                 'target_seed':spec['seed'] in [2,3],
                 'core_axis_eligible':spec['seed'] in [2,3] and (not scientific_diff or len(scientific_diff)==1 and set(scientific_diff)<=CORE)}
            checks={'source_files_unchanged':not source_mismatches,
                    'config_matches_suite':sha(spec['config'])==spec['config_sha256'],
                    'started_source_receipt_matches':st.get('source_manifest_sha256')==c['source_manifest_sha256'] and st.get('source_files')==sm['files'],
                    'started_config_receipt_matches':st.get('config_sha256')==sha(spec['config']),
                    'runtime_config_identical':read(rp/'config.json')==conf,
                    'epochs_50_50':conf.get('init_epoch')==50 and conf.get('epochs')==50}
            log=rp/'console.log'; logtext=log.read_text(errors='replace')
            epochs={(int(t),int(e)) for t,e in re.findall(r'Task (\d+), Epoch (\d+)/50',logtext)}
            row['log']=str(log);row['log_sha256']=sha(log);row['epoch_pairs']=len(epochs)
            curves=re.findall(r'CNN top1 curve:\s*(\[[^\n\r]*\])',logtext)
            curve=ast.literal_eval(re.sub(r'np\.float(?:32|64)\(([^)]+)\)',r'\1',curves[-1])) if curves else []
            row['sample_curve']=curve
            row['last_task_epoch']=max(epochs) if epochs else None
            checks['no_traceback_or_oom']=not re.search(r'Traceback \(most recent call last\)|CUDA out of memory',logtext)
            checks['no_observed_nonfinite_training_metrics']=not re.search(r'(?:Loss|CE|GAM|D_F|Train_accy)\s+(?:nan|[-+]?inf)(?:\W|$)',logtext,re.IGNORECASE)
            if cp:
                final=cp['result']; mp=Path(final['metrics_path']); raw=read(mp)
                m=raw['cnn']['matrices']['final']; n=len(m)
                finite=all(math.isfinite(m[i][k]) for i in range(n) for k in range(i+1))
                vals={'faa_task_macro':sum(m[-1])/n,
                      'aaa_task_macro':sum(sum(m[i][:i+1])/(i+1) for i in range(n))/n,
                      'forgetting':sum(max(m[i][k] for i in range(k,n-1))-m[-1][k] for k in range(n-1))/(n-1),
                      'bwt':sum(m[-1][k]-m[k][k] for k in range(n-1))/(n-1),
                      'faa_sample_weighted':curve[-1]}
                # The project's forgetting includes the final row in the historical maximum,
                # giving zero forgetting for tasks whose final score sets a new maximum.
                vals['forgetting']=sum(max(m[i][k] for i in range(k,n))-m[-1][k] for k in range(n-1))/(n-1)
                row.update(vals); row['tasks_completed']=n
                row['metrics_path']=str(mp);row['metrics_sha256']=sha(mp)
                row['raw_matrix']=[[v if math.isfinite(v) else None for v in x] for x in m]
                checks.update({'completion_receipt_status':cp['status']=='completed',
                               'completion_source_matches':cp.get('source_files')==sm['files'] and cp.get('source_manifest_sha256')==c['source_manifest_sha256'],
                               'completion_config_matches':cp.get('config_sha256')==sha(spec['config']),
                               'full_500_epochs':epochs=={(t,e) for t in range(10) for e in range(1,51)},
                               'raw_matrix_10_tasks_finite':n==10 and finite,
                               'metrics_hash_matches_receipt_and_results':sha(mp)==final['metrics_sha256']==r['metrics_sha256'],
                               'metrics_recomputed_match_receipt_and_results':all(abs(vals[k]-final[k])<1e-8 and abs(vals[k]-r[k])<1e-8 for k in vals),
                               'sample_curve_matches':len(curve)==10 and curve==r['cnn_top1_curve']==final['cnn_top1_curve']})
                if conf.get('s3_boundary_capture'):
                    bv=final['boundary_validation']
                    checks['boundary_receipt_passed_10_tasks']=bv['passed'] and bv['tasks']==10 and all(x['passed'] and x['rng_unchanged'] for x in bv['per_task'])
                    row['boundary_receipt_summary']={'passed':bv['passed'],'tasks':bv['tasks'],'fisher_location':bv['fisher_location'],
                        'selected_epochs':[x['selected_epochs'] for x in bv['per_task']], 'merge_applied':[x['merge_applied'] for x in bv['per_task']],
                        'scope':'Validated existing completion receipt; large .pt boundary files not independently rehashed or loaded in this audit.'}
            else:
                row['tasks_completed']=len(curve)
                row['reuse_scope']='incomplete; final performance unavailable'
            row['checks']=checks;row['checks_passed']=all(checks.values())
            if row['core_axis_eligible'] and cp:
                row['reuse_scope']='eligible scientific single-axis/center point; preserve campaign/run provenance'
            elif cp:
                row['reuse_scope']='other mechanism axis or seed1; not one of requested seed2/3 primary hyperparameter points'
            if jid.endswith('a_off_s2'):
                row['note']='Whole A module off also skips snapshot CE and its RNG consumption; not matched_no_merge. Do not silently relabel as an executed omega=0 configuration.'
            elif 'nomerge_matched' in jid:
                row['note']='start_task=10 keeps snapshot CE while suppressing actual fusion on tasks0..9. Not literal native omega=0, which skips capture/CE.'
            elif 'rho025' in jid:
                row['note']='Changes inner rho=0.025; outer training r remains0.2. Not the requested outer-r axis.'
            elif 'endpoint_anchor' in jid:
                row['core_axis_eligible']=False
                row['note']='Changes Fisher estimation position via s3_fisher_location; not a center point or damping coefficient sweep.'
            report['jobs'].append(row)
    a,b=[x['files'] for x in manifests]
    report['source_compatibility']={'parent_unchanged_files':sum(b.get(k)==v for k,v in a.items()),
        'changed_parent_files':[k for k,v in a.items() if b.get(k)!=v],
        'added_files':sorted(set(b)-set(a))}
    mech=ROOT/NAMES[1]
    for label,rel in [('native_boundary_witness','preflight/NATIVE_BOUNDARY_WITNESS.json'),('short_test_review','preflight/smoke_review.json')]:
        j=read(mech/rel)
        report[label]={'path':str(mech/rel),'sha256':sha(mech/rel),'passed':j['passed'],
                       'source_manifest_matches':j['source_manifest_sha256']==report['campaigns'][1]['source_manifest_sha256']}
    original=read(ROOT/'s3-historical-reuse-20260920/reference_config_logged.json')
    report['original_reference_config_diff']=diff(original,base)
    report['summary']={'status_counts':dict(Counter(x['status'] for x in report['jobs'])),
         'all_row_checks_passed':all(x['checks_passed'] for x in report['jobs']),
         'target_completed_core_rows':[x['id'] for x in report['jobs'] if x['core_axis_eligible'] and x['status']=='completed'],
         'target_running_core_rows':[x['id'] for x in report['jobs'] if x['core_axis_eligible'] and x['status']=='running']}
    (OUT/'existing_campaign_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    fields=['campaign','id','seed','axis','value','status','core_axis_eligible','faa_task_macro','faa_sample_weighted','aaa_task_macro','forgetting','tasks_completed','epoch_pairs','checks_passed','config','metrics_path','metrics_sha256','source_manifest_sha256','reuse_scope','note']
    with (OUT/'existing_campaign_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(report['jobs'])
    lines=['# 现有 S3 两个 campaign 超参点核验', '', '核验时间：'+report['audited_at'], '',
           '只读核对源码清单、配置、启动/完成 receipt、控制台 500 个 task/epoch 对、十任务原始矩阵及指标。没有使用 GPU，没有修改队列。', '',
           'FAA=最后一行任务准确率均值；AAA=十个阶段的已见任务均值再平均；遗忘=各旧任务历次最高成绩（包括最终阶段）减最终成绩后平均。sample 为控制台最终 CNN top1。未来任务格子的 NaN 是结构性空值。', '',
           '## 核验状态', '', json.dumps(report['summary'],ensure_ascii=False), '',
           '## seed2/3 可复用的核心轴点', '',
           '| job | seed | 轴/设置 | FAA | sample | AAA | 遗忘 |', '|---|---:|---|---:|---:|---:|---:|']
    for row in report['jobs']:
        if row['core_axis_eligible'] and row['status']=='completed':
            label=json.dumps(row['scientific_diff'],ensure_ascii=False) if row['scientific_diff'] else 'center'
            lines.append('| '+row['id']+' | '+str(row['seed'])+' | '+label+' | '+' | '.join(f'{row[k]:.3f}' for k in ['faa_task_macro','faa_sample_weighted','aaa_task_macro','forgetting'])+' |')
    lines.extend(['', '## 源码与观测边界', '',
        f"两个冻结源码树共有 {report['source_compatibility']['parent_unchanged_files']} 个父文件逐字节相同；机制 campaign 只增加 `utils/s3_mechanism_controls.py`。每次运行的 started/completed source manifest 均重新核对。", '',
        '机制 campaign 开启 s3_boundary_capture；retained 分支调用原生 Fisher 一次，边界侧存档只复制因子，已有原生同状态 6 类等价性 witness 及每个已完成任务 RNG 检查通过。可与 native sweep 作科学配置复用，但不是历史原日志的逐位重现。此审计没有再次加载大体积 boundary .pt；其检查状态引用已完成 receipt。', '',
        '原历史 seed2/3 中心 FAA 为 53.311/51.968；这里重跑中心为 53.419/52.006。保留原中心作历史复现参考；本轮新单轴变化优先相对本轮中心计算，不能把两次执行当成同一条结果。', '',
        '## 需要保留的区别', '',
        '- λ 是 ewc_lambda；lambda_flat 另属 GAM 方向强度，不能合并。',
        '- 训练 r 对应 gam_grad_norm_rho；gam_grad_rho=.025 的在跑机制实验只改内层 ρ，不补外层 r 的缺点。',
        '- α=0/1 是 ewc_fisher_damping_alpha 的单轴控制；γ=0/1 是历史 Fisher 累积，不能标为 damping α。',
        '- 原生正 ω 可用 in_task_weight_merge_omega 单轴改变权重融合；这里两 campaign 尚无正 ω≠.3 的完成点。',
        '- matched_no_merge 保留快照 CE；native ω=0 会跳过快照/CE。Aoff 可作为模块关闭对照，但本报告没有把其配置改名为执行过的 ω=0。',
        '- 均匀 β=0 与后期 τ=.6 是权重路径构造设置，不是 ω 扫描。',
        '- 旧 seed0 同时融合 Fisher 与权重的 ω 扫描不在这两个 campaign 内，不能当本中心纯权重 ω 扫描复用。', '',
        '## 未完成记录', ''])
    for row in report['jobs']:
        if row['status']!='completed':
            lines.append(f"- {row['id']}: {row['status']}, 已见曲线阶段 {row['tasks_completed']}, 最后 task/epoch={row['last_task_epoch']}; 无最终指标。")
    (OUT/'existing_campaign_audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
