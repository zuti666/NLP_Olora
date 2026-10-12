#!/usr/bin/env python3
"""Collect complete RQ2 receipts; missing work remains explicitly pending."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
from pathlib import Path

from rq2_common import ROOT, atomic_json, sha_file

METHODS=('sgd','gam')
COLORS={'sgd':'#3569A3','gam':'#D56D32'}
LABELS={'sgd':'SGD + EWC','gam':'GAM + EWC'}


def csv_read(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def csv_write(path,rows,fields=None):
    if fields is None:
        fields=list(rows[0]) if rows else ['status']
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def receipt(path):
    path=Path(path)
    if not path.exists():
        return None
    data=json.loads(path.read_text())
    return data if data.get('status')=='complete' else None


def status_for(path,data):
    if data is not None:
        if 'passed' in data and not data['passed']:
            return 'failed'
        return 'complete'
    progress=Path(path).parent/'progress.json'
    if progress.exists():
        p=json.loads(progress.read_text())
        return f"pending ({p.get('states_completed',0)} states saved)"
    return 'pending'


def number(v,digits=6):
    return f'{v:.{digits}g}' if isinstance(v,(int,float)) and math.isfinite(v) else 'pending'


def setup_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
                         'savefig.bbox':'tight'})
    return plt


def save_plot(fig,output,name):
    fig.tight_layout()
    fig.savefig(output/f'{name}.pdf')
    fig.savefig(output/f'{name}.png',dpi=220)


def paired_rng_check(receipts):
    """Compare Task1 head and actual sampler/augmentation receipts by loader."""
    artifacts={}
    events={}
    for method in METHODS:
        summary=receipts[method,'table_a']
        if summary is None:
            return {'status':'pending','passed':False,'reason':'Both completed A endpoint receipts are required'}
        path=Path(summary['training_manifest']).parent/'rng_receipts.jsonl'
        if not path.exists():
            return {'status':'pending','passed':False,'reason':f'Missing {path}'}
        artifacts[method]={'path':str(path),'sha256':sha_file(path)}
        events[method]=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    mismatches=[]
    compared={}
    schemas={
      'task_rng_reset': (('task',),('task_seed','rng_protocol','optimizer_type')),
      'head_initialized': (('task',),('old_classes','total_classes','task_seed','new_head_sha256')),
      'loader_created': (('task','loader_serial'),('shuffle','generator_seed','num_workers','sample_count','dataset_ids_sha256')),
      'loader_iteration': (('task','loader_serial','iteration'),('shuffle','generator_seed','num_workers','sample_count',
        'dataset_ids_sha256','batches','samples','indices_sha256','first_batch_pixels_targets_sha256')),
    }
    table={}
    for event,(key_fields,compare_fields) in schemas.items():
        table[event]={}
        for method in METHODS:
            selected=[r for r in events[method] if r.get('event')==event and r.get('task')==1 and
                      (event not in ('loader_created','loader_iteration') or r.get('shuffle') is True)]
            indexed={}
            for row in selected:
                key=tuple(row.get(k) for k in key_fields)
                if key in indexed:
                    mismatches.append({'event':event,'method':method,'key':list(key),'issue':'duplicate key; inspect resume history'})
                indexed[key]={k:row.get(k) for k in compare_fields}
            table[event][method]=indexed
        left,right=table[event]['sgd'],table[event]['gam']
        if not left or not right:
            mismatches.append({'event':event,'issue':'missing required Task1 receipt'})
        for key in sorted(set(left)|set(right)):
            if key not in left or key not in right:
                mismatches.append({'event':event,'key':list(key),'issue':'key present for only one method'})
            elif left[key]!=right[key]:
                mismatches.append({'event':event,'key':list(key),'differing_fields':
                    {k:{'sgd':left[key][k],'gam':right[key][k]} for k in compare_fields if left[key][k]!=right[key][k]}})
        compared[event]={'key_fields':list(key_fields),'value_fields':list(compare_fields),
                         'sgd_count':len(left),'gam_count':len(right)}
    groups={}
    for key,row in table['loader_iteration']['sgd'].items():
        groups.setdefault(key[1],set()).add(key[2])
    full_training_loaders=[serial for serial,iterations in groups.items() if set(range(50)).issubset(iterations)]
    if not full_training_loaders:
        mismatches.append({'event':'loader_iteration','issue':'No shuffled Task1 loader has all 50 epoch iterations 0..49'})
    return {'status':'complete' if not mismatches else 'mismatch','passed':not mismatches,
            'artifacts':artifacts,'compared':compared,'complete_50_epoch_loader_serials':full_training_loaders,
            'mismatches':mismatches,'scope':'Task1 new head, actual sample order and first-batch augmentation hashes; eval loaders excluded explicitly'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output-dir',default=str(ROOT/'report_seed0'))
    parser.add_argument('--a-root',help='Root containing method/table_a/summary.json; defaults to config output_dir')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    results=Path(config.get('output_dir',ROOT/'results'))
    aroot=Path(args.a_root) if args.a_root else results
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    receipts,manifest,statuses={},{},[]
    for method in METHODS:
        for job in ('sanity','neighborhood','bridge','trajectory','path','table_a'):
            path=(aroot if job=='table_a' else results)/method/job/'summary.json'
            data=receipt(path)
            receipts[method,job]=data
            statuses.append({'method':method,'job':job,'status':status_for(path,data)})
            if data is not None:
                manifest[f'{method}/{job}']={'path':str(path),'sha256':sha_file(path),
                    'protocol_sha256':data.get('protocol_sha256',data.get('signature_sha256'))}
    # Shared-ID requirement is established from each data manifest, not filenames.
    hashes={}
    for method in METHODS:
        path=results/method/'data_manifest.json'
        if path.exists():
            hashes[method]=json.loads(path.read_text())['manifest_sha256']
    shared_ids=(len(hashes)==2 and len(set(hashes.values()))==1)
    if len(hashes)==2 and not shared_ids:
        raise ValueError('SGD/GAM data manifests differ; do not create a paired comparison')
    radius_rows,tolerance_rows,bridge_rows,path_rows,path_summary,a_rows,current_trajectory=[],[],[],[],[],[],[]
    for method in METHODS:
        b=receipts[method,'neighborhood']
        if b:
            for r in b['radius_summary']:
                for split in ('train','test'):
                    radius_rows.append({'method':method,'seed':0,'transform':'deterministic_test',
                      'split':split,'radius':r['radius'],'directions':r['directions'],
                      'D_mean':r[split]['D_mean'],'G':r[split]['G'],'Q95':r[split]['Q95'],
                      'raw_trace_mean_train256':r['raw_trace_mean']})
            tolerance_rows.extend({'method':method,'seed':0,'transform':'deterministic_test',**r} for r in b['tolerance'])
        bridge=receipts[method,'bridge']
        if bridge:
            for r in bridge['bridge_summary']:
                bridge_rows.append({'anchor':method,'seed':0,'direction_method':r['direction_method'],
                    'radius':r['radius'],'raw_trace_train256':r['raw_trace'],
                    'D_train':r['damage']['train'],'D_test':r['damage']['test']})
        traj=receipts[method,'trajectory']
        if traj:
            for r in traj['tasks']:
                current={'method':method,'seed':0,'task':r['task'],
                         'delta_norm_cpu_recomputed':r['effective_weight_drift'],
                         'checkpoint':r['checkpoint'],'checkpoint_sha256':r['checkpoint_sha256'],
                         'precision':'current full FP32','train_transform':'deterministic_test'}
                for split in ('train','test'):
                    current[f'{split}_local_loss_delta_recomputed']=r['D0'][split]
                    for scope in ('local','seen'):
                        for metric,value in r['metrics'][split][scope].items():
                            current[f'{split}_{scope}_{metric}']=value
                current_trajectory.append(current)
        path=receipts[method,'path']
        if path:
            gate=path['quadrature_converged'] and path['bound_passed_with_2pct_numeric_tolerance']
            path_summary.append({'method':method,'seed':0,'drift':path['drift'],'I':path['integral'],
                'S_path':path['S_path'],'D_subset':path['D_subset'],'D_subset_squared':path['D_subset_squared'],
                'nodes':path['selected_nodes'],'quadrature_relative_difference':path['quadrature_relative_difference'],
                'quadrature_converged':path['quadrature_converged'],
                'bound_passed_with_2pct_numeric_tolerance':path['bound_passed_with_2pct_numeric_tolerance'],
                'numerical_gate_passed':gate,'bound_gap_I_minus_D2':path['bound_gap_I_minus_D2']})
            for r in path['path']:
                path_rows.append({'method':method,'seed':0,'s':r['s'],'q':r['q'],'raw_trace':r['raw_trace'],
                    'subset_loss':r['subset_loss'],'train_loss':r['metrics']['train']['loss'],
                    'test_loss':r['metrics']['test']['loss'],'test_local_accuracy':r['metrics']['test']['accuracy']})
        a=receipts[method,'table_a']
        if a:
            a_rows.append({'anchor_method':method,'future_optimizer':a['future_optimizer'],'seed':a['seed'],
              **{k:a['accuracy_CIL'][k] for k in ('a11','a21','a22')},
              'a11_historical_inherited':a['a11_historical_inherited'],
              'D0_train_local':a['D0']['train'],'D0_test_local':a['D0']['test'],
              'old_test_local_acc_t0':a['anchor']['metrics']['test']['accuracy'],
              'old_test_local_acc_t1':a['endpoint']['metrics']['test']['accuracy'],
              'effective_weight_drift':a['effective_weight_drift'],
              'raw_trace_t0_train256':a['anchor']['fisher']['raw_trace'],
              'raw_trace_t1_train256':a['endpoint']['fisher']['raw_trace'],
              'D0_subset':a['D0_subset'],'checkpoint_t1':a['checkpoints']['1']})
    paired_rng=paired_rng_check(receipts)
    atomic_json(output/'A_PAIRED_RNG_CHECK.json',paired_rng)
    csv_write(output/'status.csv',statuses)
    for name,rows in [('B_neighborhood_deterministic',radius_rows),('B_tolerance_deterministic',tolerance_rows),
                      ('B_actual_direction_bridge',bridge_rows),('C_path_nodes',path_rows),
                      ('C_path_summary',path_summary),('A_two_task',a_rows)]:
        csv_write(output/f'{name}.csv',rows)
    csv_write(output/'C_current_FP32_task_trajectory.csv',current_trajectory)
    history=csv_read(ROOT.parent/'radius_damage_seed0.csv')
    historical_rows=[dict(r,protocol='historical_unverified_pairing',
        transform='historical_cached_random' if r['split']=='train' else 'deterministic_test') for r in history]
    csv_write(output/'B_historical_damage_separate.csv',historical_rows)
    trajectory=csv_read(ROOT.parent/'task_trajectory_recomputed.csv')
    csv_write(output/'C_historical_task_trajectory.csv',trajectory)
    current_trajectory_complete=all(receipts[m,'trajectory'] for m in METHODS)
    plotted_trajectory=current_trajectory if current_trajectory_complete else trajectory
    trajectory_label='Current full FP32' if current_trajectory_complete else 'Historical TF32 reference'
    plt=setup_plot()
    figures=[]
    if all(receipts[m,'neighborhood'] for m in METHODS):
        assert shared_ids
        fig,axes=plt.subplots(1,3,figsize=(12.4,3.5))
        for method in METHODS:
            rr=receipts[method,'neighborhood']['radius_summary']
            rad=[r['radius'] for r in rr]
            for ax,split in zip(axes[:2],('train','test')):
                ax.plot(rad,[r[split]['G'] for r in rr],'-o',color=COLORS[method],markersize=3,label=f'{LABELS[method]} mean')
                ax.plot(rad,[r[split]['Q95'] for r in rr],'--',color=COLORS[method],alpha=.8,label=f'{LABELS[method]} Q95')
                ax.set(xlabel='Effective-weight radius r',ylabel='Positive CE damage',title=f'{split.capitalize()}: shared directions')
            axes[2].plot(rad,[r['raw_trace_mean'] for r in rr],'-o',color=COLORS[method],markersize=3,label=LABELS[method])
        axes[0].legend(fontsize=7)
        axes[2].set(xlabel='Effective-weight radius r',ylabel='Raw empirical Fisher trace',title='Fixed 256 train images')
        axes[2].legend(fontsize=7)
        save_plot(fig,output,'Figure_B_new_deterministic');plt.close(fig)
        figures.append('Figure_B_new_deterministic')
    if history:
        fig,axes=plt.subplots(1,2,figsize=(8.6,3.4))
        for split,ax in zip(('train','test'),axes):
            for method in METHODS:
                rr=sorted([r for r in history if r['split']==split and r['method']==method],key=lambda r:float(r['radius']))
                ax.plot([float(r['radius']) for r in rr],[float(r['G']) for r in rr],color=COLORS[method],label=LABELS[method])
            ax.set(xlabel='Effective-weight radius r',ylabel='Mean positive CE damage',
                   title=f'Historical {split} (pairing unverified)')
            ax.legend(fontsize=8)
        save_plot(fig,output,'Historical_radius_damage_separate');plt.close(fig)
        figures.append('Historical_radius_damage_separate')
    if plotted_trajectory:
        fig,axes=plt.subplots(1,3,figsize=(12.1,3.6))
        for method in METHODS:
            rr=sorted([r for r in plotted_trajectory if r['method']==method],key=lambda r:int(r['task']))
            damage=[float(r['test_local_loss_delta_recomputed']) for r in rr]
            axes[0].plot([int(r['task']) for r in rr],damage,'-o',color=COLORS[method],markersize=3,label=LABELS[method])
            axes[1].plot([float(r['delta_norm_cpu_recomputed']) for r in rr],damage,'-o',color=COLORS[method],markersize=3,label=LABELS[method])
        axes[0].set(xlabel='Last learned task',ylabel='Old-task test CE increase',title=f'{trajectory_label}: fixed local head')
        axes[1].set(xlabel='Effective-weight displacement',ylabel='Old-task test CE increase',title='Damage versus displacement')
        axes[0].legend(fontsize=8)
        valid_paths=[r for r in path_summary if r['numerical_gate_passed']]
        if len(valid_paths)==2:
            rr={r['method']:r for r in valid_paths}
            axes[2].bar([0,1],[rr[m]['S_path'] for m in METHODS],color=[COLORS[m] for m in METHODS],width=.55)
            axes[2].set(xticks=[0,1],xticklabels=['SGD','GAM'],ylabel='Integral q(s) ds / displacement squared',title='Task0 to Task9 straight path')
        else:
            axes[2].axis('off')
            axes[2].text(.5,.55,'Path sensitivity: pending\n(or numerical gate unresolved)',ha='center',va='center',transform=axes[2].transAxes)
        save_plot(fig,output,'Figure_C_sequence_and_path');plt.close(fig)
        figures.append('Figure_C_sequence_and_path')
    if len(bridge_rows)==24:
        fig,axes=plt.subplots(1,2,figsize=(8.6,3.4))
        for method,ax in zip(METHODS,axes):
            for direction in METHODS:
                rr=sorted([r for r in bridge_rows if r['anchor']==method and r['direction_method']==direction],key=lambda r:r['radius'])
                ax.plot([r['radius'] for r in rr],[r['D_test'] for r in rr],'-o',color=COLORS[direction],markersize=3,label=f'{direction.upper()} update direction')
            ax.set(xlabel='Effective-weight radius r',ylabel='Old-task test CE increase',title=f'{method.upper()} anchor')
            ax.legend(fontsize=8)
        save_plot(fig,output,'Figure_B_bridge');plt.close(fig)
        figures.append('Figure_B_bridge')
    lines=['# FS-EWC RQ2 seed0 测量结果', '',
      '更新时间：'+datetime.datetime.now(datetime.timezone.utc).isoformat(),'',
      '本报告仅将完成的作业计入正式数值；未完成项目记为 pending。所有比较目前均为 seed0 机制案例，随机方向不是训练 seed 重复。', '',
      '| 方法 | 检查/实验 | 状态 |','|---|---|---|']
    lines += [f"| {r['method']} | {r['job']} | {r['status']} |" for r in statuses]
    lines += ['',f'两方法样本ID与图像内容清单哈希一致：{shared_ids}。', '',
      '新诊断使用确定性 Resize256/bicubic→CenterCrop224→ToTensor，不做 Normalize；训练与测试都使用这一输入处理。记录 Pillow/JPEG 解码器版本与完整 train/test 解码后张量哈希。Fisher 固定同一分层256个旧任务训练样本、24个有效 Q/V 权重矩阵、旧任务20类 head，计算逐样本 CE 梯度，不归一化、不加 floor 或正则。', '',
      '当前正式测量统一使用 full FP32，并关闭 cuDNN/matmul TF32。四个历史端点可通过 TF32 patch-convolution 操作数舍入复现，这是历史精度差异的证据，不等同于取得旧运行时开关的直接记录。sanity 使用该临时 reference 核验历史端点，随即逐位恢复原始权重；reference 不用于 Fisher、邻域、路径或当前逐任务曲线。', '',
      '旧半径 train 曲线来自历史缓存随机增强，方向配对未经验证，独立保存为 B_historical_damage_separate.csv 和 Historical_radius_damage_separate 图；不与新 raw trace 拼成逐点关系。旧逐任务 train loss 也保留历史口径，正文序列图使用 test。', '',
      '## A：固定后续 SGD+EWC 的两任务表','',
      '| Task0 起点 | a11 | a21 | a22 | train D0 | test D0 | 位移 | raw trace T0 | raw trace T1 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        row=next((r for r in a_rows if r['anchor_method']==method),None)
        lines.append('| '+method+' | '+(' | '.join(number(row[k]) for k in ('a11','a21','a22','D0_train_local','D0_test_local','effective_weight_drift','raw_trace_t0_train256','raw_trace_t1_train256')) if row else 'pending | pending | pending | pending | pending | pending | pending | pending')+' |')
    lines += ['', 'a11/a21/a22 是所有已见类别上的 CNN class-IL accuracy（%）；a11 使用当前 FP32 起点评估并与续训日志核对，历史矩阵继承的 a11_historical_inherited 另外保留在 CSV/JSON。D0 与机制 accuracy 使用固定旧任务20类 head。两个起点继承的 EWC Fisher 矩阵可不同，不能据此声称完全隔离保护矩阵。', '',
              f"Task1 配对随机过程核验：{paired_rng['status']}，passed={paired_rng['passed']}。检查新 head、按任务/loader/iteration 分组的样本顺序及首 batch 图像/标签哈希；评估 loader 明确排除。完整细节及任何差异见 A_PAIRED_RNG_CHECK.json。未通过时不宣称严格配对，最终报告状态保持 partial。", '',
              '## B：等半径邻域与真实方向','']
    if all(receipts[m,'neighborhood'] for m in METHODS):
        lines += ['已完成两个起点的公共方向扫描；各半径 G、Q95 与训练子集 raw trace 见 B_neighborhood_deterministic.csv，容忍半径见 B_tolerance_deterministic.csv。', '',
                  '| 半径 | SGD raw trace | GAM raw trace | GAM/SGD |','|---|---:|---:|---:|']
        sgd={r['radius']:r for r in receipts['sgd','neighborhood']['radius_summary']}
        for r in receipts['gam','neighborhood']['radius_summary']:
            sr=sgd[r['radius']]['raw_trace_mean'];gr=r['raw_trace_mean']
            lines.append(f"| {r['radius']} | {number(sr)} | {number(gr)} | {number(gr/sr if sr else None)} |")
    else:
        lines += ['邻域正式对照 pending；已存中间状态仅用于恢复和进度，不替代完整半径方向预算。']
    lines += ['', '## C：真实序列与 t9 直线路径','',
      '| 方法 | 位移 | D_subset | I | S_path | 节点数 | 求积收敛 | D_subset²≤I（2%数值容差） |',
      '|---|---:|---:|---:|---:|---:|---|---|']
    for method in METHODS:
        row=next((r for r in path_summary if r['method']==method),None)
        if row:
            lines.append('| '+method+' | '+' | '.join(number(row[k]) for k in ('drift','D_subset','I','S_path','nodes'))+f" | {row['quadrature_converged']} | {row['bound_passed_with_2pct_numeric_tolerance']} |")
        else:
            lines.append(f'| {method} | pending | pending | pending | pending | pending | pending | pending |')
    lines += ['', '路径连接 Task0 与 Task9 的有效权重端点，固定 Task0 head，属于端点直线而非逐步训练轨迹。方向敏感度先跨全部矩阵累加每样本 g·Δ，再平方。D_subset 与积分使用同一256个样本；全 train/test 损伤不用于该数值界检查。', '',
              f'Figure C 前两幅曲线当前来源：{trajectory_label}。正式交付优先使用重新评估的全10任务 FP32 曲线；历史曲线独立保存在 C_historical_task_trajectory.csv。', '',
              '只有求积收敛与界检查均通过后，Figure C 才显示 S_path 对照。若 raw sensitivity 或 S_path 未下降，保留数值并收窄相应机制结论。', '',
              '## 图与导出','']
    lines += [f'- [{name}.pdf]({name}.pdf) / [{name}.png]({name}.png)' for name in figures]
    lines += ['', '数据来源与 SHA256：MANIFEST.json。所有 CSV 为显式分离的新诊断、历史曲线、历史序列或 A 表。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    complete=(all(r['status']=='complete' for r in statuses) and shared_ids and paired_rng['passed'] and
              len(path_summary)==2 and all(r['numerical_gate_passed'] for r in path_summary))
    report_status='complete' if complete else 'partial'
    atomic_json(output/'MANIFEST.json',{'status':report_status,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'code_sha256':sha_file(__file__),'config':str(Path(args.config).resolve()),'config_sha256':sha_file(args.config),
        'sources':manifest,'data_manifest_sha256':hashes,'shared_data_ids':shared_ids,
        'trajectory_plot_source':trajectory_label,'path_numerical_gates_passed':len(path_summary)==2 and all(r['numerical_gate_passed'] for r in path_summary),
        'A_paired_rng':paired_rng,
        'historical_sources':{str(p):sha_file(p) for p in (ROOT.parent/'radius_damage_seed0.csv',ROOT.parent/'task_trajectory_recomputed.csv') if p.exists()},
        'figures':figures,'status_rows':statuses,'scope':'seed0 only; missing jobs pending; no fabricated values'})
    print(json.dumps({'status':report_status,'report':str(output/'RESULTS.md'),'figures':figures,'jobs':statuses}),flush=True)


if __name__=='__main__':
    main()
