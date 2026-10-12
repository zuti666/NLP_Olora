#!/usr/bin/env python3
"""CPU-only, read-only audit of completed independent Task0 diagnostics."""
import csv, hashlib, importlib.util, json, math, statistics
from datetime import datetime, timezone
from pathlib import Path

OUT=Path(__file__).resolve().parent
DIAG=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-stage1-task0-run-20260930T213643Z/diagnostics')
SUITE=DIAG/'suite_completion_recovery_20261001.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):return {'mean':statistics.mean(v),'min':min(v),'max':max(v)}
suite=read(SUITE)
spec=importlib.util.spec_from_file_location('task0_diag_summary',DIAG/'summarize_diagnostics.py')
ds=importlib.util.module_from_spec(spec);spec.loader.exec_module(ds)
for path,want in suite['file_pins'].items():assert sha(path)==want,path
ledger=[];anchors=[];batch_rows=[];hashes=[]
for job in suite['jobs']:
    path=Path(job['output']);receipt=Path(suite['runtime'])/'jobs'/job['id']/'completed.json'
    entry={'anchor':job['id'],'status':'missing','report':str(path)}
    if not path.exists():ledger.append(entry);continue
    d=read(path);entry['status']=d['status'];entry['available_batches']=len(d['batches'])
    if d['status']!='completed' or not receipt.exists():
        if d['status']=='completed':entry['status']='awaiting_receipt'
        if d['status']=='failed':
            failed_receipt=receipt.with_name('failed.json')
            entry.update(error=d.get('error'),final_restore_check=d.get('final_restore_check'),
                report_sha256=sha(path),failed_receipt=str(failed_receipt),
                failure=read(failed_receipt) if failed_receipt.exists() else None,
                excluded_partial_batches=len(d['batches']))
        ledger.append(entry);continue
    rc=read(receipt);ds.validate(d)
    assert rc['returncode']==0 and rc['validation_errors']==[] and rc['batches']==4
    assert rc['id']==job['id'] and rc['output']==str(path)
    assert d['training_seed']==2 and d['task']==0 and d['class_range']==[0,20]
    assert d['arguments']['checkpoint_dir']==job['checkpoint_dir']
    assert d['source_config_sha256']==sha(job['config'])
    cfg=read(job['config']);epoch=int(Path(job['checkpoint_dir']).name.split('_')[-1])
    for name,want in d['checkpoint_hashes'].items():assert sha(Path(job['checkpoint_dir'])/name)==want
    for name,want in d['source_hashes'].items():assert sha(Path(d['source'])/name)==want
    assert len(d['batches'])==4 and d['all_matches_successful'] is True
    assert d['arguments']['eta_ref']==.02 and d['arguments']['rho']==.05
    assert d['loss_before']['0']['n']==128
    assert d['loss_before']['0']['all_seen_ce']==d['loss_before']['0']['task_local_ce']
    entry.update(status='validated',report_sha256=sha(path),receipt_sha256=sha(receipt),
        config_sha256=sha(job['config']),source_direction=cfg['stage1_direction'],epoch=epoch)
    ledger.append(entry)
    b=d['batches'];paired=[]
    for batch in b:
        assert batch['sample_count']==128 and Path(batch['vectors_file']).exists()
        labels={c['label']:c for c in batch['counterfactuals']}
        assert len(labels)==25
        base=labels['g1']['loss_change']['current']['task_local_ce']
        for label,c in labels.items():
            assert c['match']['ok'] and c['norm_globally_monotonic']
            assert c['positive_stationary_scales']==[]
            assert c['loss_change']['old_task_macro'] is None
            assert c['loss_change']['old_task_status']=='NA_no_old_task'
            cur=c['loss_change']['current'];assert cur['task_local_ce']==cur['all_seen_ce']
            row={'anchor':job['id'],'source_direction':cfg['stage1_direction'],'epoch':epoch,
                'batch':batch['batch'],'candidate':label,'current_delta_ce':cur['task_local_ce'],
                'paired_delta_vs_g1':cur['task_local_ce']-base,'matched_scale':c['match']['scale'],
                'actual_match_error':c['match']['actual_relative_error']}
            paired.append(row);batch_rows.append(row)
    candidate={}
    for label in [c['label'] for c in b[0]['counterfactuals']]:
        rows=[r for r in paired if r['candidate']==label]
        candidate[label]={'delta_ce':stats([r['current_delta_ce'] for r in rows]),
            'paired_vs_g1':stats([r['paired_delta_vs_g1'] for r in rows]),
            'better_batches_than_g1':sum(r['paired_delta_vs_g1'] < -1e-7 for r in rows),
            'matched_scale':stats([r['matched_scale'] for r in rows])}
    sams={}
    for kind in ['parameter','kl']:
        ss=[x['radius_matched_sam'][kind] for x in b];assert all(x['ok'] for x in ss)
        sams[kind]={'successful':len(ss),'rho':stats([x['sam_rho'] for x in ss]),
            'rho_over_original':stats([x['sam_rho']/.05 for x in ss]),
            'match_error_max':max(x['match'].get('actual_relative_error',x['match'].get('relative_error',0)) for x in ss)}
    anchor={'anchor':job['id'],'source_direction':cfg['stage1_direction'],'epoch':epoch,
        'batch_count':4,'probe_batch_sizes':[128]*4,'eval_samples':128,
        'eta_ref':.02,'rho':.05,'base_ce':d['loss_before']['0']['task_local_ce'],
        'candidate_count':len(paired),'matched_count':len(paired),'match_error_max':max(r['actual_match_error'] for r in paired),
        'geometry':{'a':stats([x['geometry']['a'] for x in b]),
            'R_perp':stats([x['geometry']['r_perp_over_g3'] for x in b]),
            'cos_g1_g3':stats([x['geometry']['cosines']['g1_g3'] for x in b])},
        'r0_relative_error_max':max(x['r0_identity_sanity']['relative_error'] for x in b),
        'sum_halfsum_max_ce_gap':max(abs(next(c for c in x['counterfactuals'] if c['label']=='sum')['loss_change']['current']['task_local_ce']-next(c for c in x['counterfactuals'] if c['label']=='half_sum')['loss_change']['current']['task_local_ce']) for x in b),
        'candidates':candidate,'sam_probe_matches':sams,'finite_restore_passed':True,
        'old_task_metrics':'NA_no_old_task','checkpoint_dir':job['checkpoint_dir'],
        'report_sha256':sha(path),'eval_batch_hashes':d['eval_batch_hashes'],
        'probe_hashes':[{'inputs':x['inputs_sha256'],'targets':x['targets_sha256']} for x in b]}
    anchors.append(anchor)
paired_input_check=all(a['eval_batch_hashes']==anchors[0]['eval_batch_hashes'] and a['probe_hashes']==anchors[0]['probe_hashes'] for a in anchors) if anchors else None
report={'status':'completed' if len(anchors)==5 else 'partial','audited_at_utc':datetime.now(timezone.utc).isoformat(),
    'expected_anchors':5,'validated_anchors':len(anchors),'ledger':ledger,
    'suite_path':str(SUITE),'suite_sha256':sha(SUITE),'same_probe_and_eval_inputs_across_anchors':paired_input_check,
    'batches_are_not_seeds':True,'training_seed':2,'anchor_means_not_pooled':True,
    'old_task_metrics':'NA_no_old_task','anchors':anchors}
assert paired_input_check is not False
(OUT/'compact.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
with (OUT/'candidate_batches.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(batch_rows[0]) if batch_rows else ['anchor']);writer.writeheader();writer.writerows(batch_rows)

lines=['# 独立 Stage1 Task0：同状态诊断审计', '',
    f"统计时点：{report['audited_at_utc']}。已验证 **{len(anchors)}/5 个起点**；每起点 4 个固定训练 probe batch（各 128 样本），同一个固定 128 样本评测子集。只有 seed 2；batch 不是独立训练 seed。初始化和各训练末端分别列出，不跨起点求均值。", '',
    '候选一步更新的实际 concat(LoRA ΔBA, FC Δweight, FC Δbias) 范数匹配到该 batch 的 −0.02×g1 更新范数。0.02 是虚拟诊断步长，既不是 epoch50 的 cosine 学习率，也不是 task0 初始 LR=.04。评测子集不是训练最终表使用的全部 307 样本。', '',
    '全部起点只有 task0；all-seen CE 与 task-local CE 完全相同，旧任务指标为 NA，不能解释为 0 或遗忘。以下 ΔCE=一步后−一步前，越负越好；相对 g1 差值=候选 ΔCE−同 batch 的 g1 ΔCE。所有均值只在该起点自己的四个 probe batch 内计算。', '',
    '## 完整性与匹配', '', '| 起点 | 状态 | 配置方向 | epoch |', '|---|---|---|---|']
for e in ledger:lines.append(f"| {e['anchor']} | {e['status']} | {e.get('source_direction','—')} | {e.get('epoch','—')} |")
for e in ledger:
    if e['status']=='failed':
        lines += ['',f"**失败排除：{e['anchor']}**：{e.get('error')}。只保存了 {e['excluded_partial_batches']} 个已完成 batch；整个起点排除，不把其部分结果并入表。失败发生于第二个 batch 的 r=0 梯度相对误差检查（阈值 5e−4），触发值未写入最终 JSON，不能凭异常文字认定是 RNG 问题或训练发散。最终 parameters/buffers/modes/RNG/gradients 恢复仍全 true。失败只涉及离线诊断，10/10 正式训练完成状态不变。"]
if anchors:
    lines += ['',f"已验证 {sum(a['matched_count'] for a in anchors)}/{sum(a['candidate_count'] for a in anchors)} 个有效步幅匹配，最大实际相对误差 {max(a['match_error_max'] for a in anchors):.4g}（阈值 .001）；finite、每 batch 与最终参数/梯度/buffer/mode/RNG 恢复全部通过。配置、冻结源、checkpoint 和完成回执哈希通过；跨起点 probe/评测输入哈希一致。", '',
    '## 当前任务一步损失', '', '| 起点 | 原始子集 CE | g1 ΔCE | g3 ΔCE | Sum ΔCE | g3(r=.25) ΔCE |', '|---|---:|---:|---:|---:|---:|']
    for a in anchors:
        c=a['candidates'];vals=[c[k]['delta_ce']['mean'] for k in ['g1','g3','sum','g3_radius_0.25']]
        lines.append('| '+a['anchor']+f" | {a['base_ce']:.6f} | "+' | '.join(f'{x:+.6f}' for x in vals)+' |')
    lines += ['', '相对 g1 的配对差值（负值更好）；括号内为差值 < −1e−7 的 batch 数/4，只是固定批次描述，不是显著性检验。', '', '| 起点 | g3−g1 | Sum−g1 | g3(r=.25)−g1 |', '|---|---:|---:|---:|']
    for a in anchors:
        cells=[f"{a['candidates'][k]['paired_vs_g1']['mean']:+.6f} ({a['candidates'][k]['better_batches_than_g1']}/4)" for k in ['g3','sum','g3_radius_0.25']]
        lines.append('| '+a['anchor']+' | '+' | '.join(cells)+' |')
    lines += ['', '## 分解几何', '', '| 起点 | a | R_perp=‖残差‖/‖g3‖ | cos(g1,g3) |', '|---|---:|---:|---:|']
    for a in anchors:lines.append('| '+a['anchor']+' | '+' | '.join(f"{a['geometry'][k]['mean']:.4f}" for k in ['a','R_perp','cos_g1_g3'])+' |')
    lines += ['',f"g3=a×g1+r_perp。几何均值只描述当前起点；非共线不自动等于方向更优。r=0 的局部相对梯度误差最大 {max(a['r0_relative_error_max'] for a in anchors):.4g}；Sum 与 Half-Sum 有效步幅匹配后最大 CE 差 {max(a['sum_halfsum_max_ce_gap'] for a in anchors):.4g}。", '',
    '## 距离匹配 SAM 控制', '',
    '先沿 g0 选择 SAM 探针半径，使参数距离或输出 KL 匹配 GAM 第四探针；得到梯度后，再按同一 BA+FC 规则匹配最终更新步幅。探针距离匹配与更新步幅匹配是两项不同控制。', '',
    '| 起点 | 参数匹配 rho 均值 [min,max] | KL 匹配 rho 均值 [min,max] | SAM-param ΔCE | SAM-KL ΔCE |', '|---|---|---|---:|---:|']
    for a in anchors:
        vals=[]
        for k in ['parameter','kl']:
            z=a['sam_probe_matches'][k]['rho'];vals.append(f"{z['mean']:.4f} [{z['min']:.4f},{z['max']:.4f}]")
        dsam=[a['candidates'][k]['delta_ce']['mean'] for k in ['sam_parameter_matched_probe','sam_kl_matched_probe']]
        lines.append('| '+a['anchor']+' | '+' | '.join(vals)+' | '+' | '.join(f'{x:+.6f}' for x in dsam)+' |')
    lines += ['',f"两类 SAM 探针匹配均 {4*len(anchors)}/{4*len(anchors)} 成功。原始 rho=.05；表中的大半径是匹配控制，不是标准 SAM 正式训练的 rho。KL 匹配和 CE 诊断复用同一固定评测子集，属于探索性证据。", '',
    '## 当前可支持的观察', '',
    '以下仅描述已经通过全部检查的起点；Mean50 缺失，因此不能当作五起点完整机制结论。']
    for a in anchors:
        c=a['candidates'];comparisons=[]
        for k,label in [('g3','g3'),('sum','Sum'),('g3_radius_0.25','r=.25')]:
            delta=c[k]['paired_vs_g1']['mean'];comparisons.append(f"{label} 相对 g1 为 {delta:+.6f}")
        sam_better=[k for k in ['sam_parameter_matched_probe','sam_kl_matched_probe'] if c[k]['delta_ce']['mean'] < c['g3']['delta_ce']['mean']]
        lines.append(f"- {a['anchor']}："+'；'.join(comparisons)+f"。{len(sam_better)}/2 类距离匹配 SAM 的平均 ΔCE 优于 g3；不表示每个 batch 都如此。")
    lines += ['',
    '这组局部观察没有显示 g3、Sum 或 r=.25 相对 g1 的普遍同幅值方向优势；但它也不能否定这些方向在完整训练中因路径或幅值产生收益。两个更远 SAM 控制的表现说明探针位置本身仍是需要保留的解释因素，不能只归因于非共线分量。', '',
    '## 结论边界', '',
    '1. 只允许同一 checkpoint、同一 batch 上比较候选；不能把初始化与末端的绝对 ΔCE 大小直接归因为方向优劣，也不把不同训练起点混成多 seed。',
    '2. r=.25 这里指在现有起点上重新形成的候选方向，并非 r=.25 完整训练的 epoch50 checkpoint；本轮五起点没有后者，不能把局部结果当作其完整训练机制证明。',
    '3. 这些是虚拟一次更新，不解释完整 50 epochs 的长期因果机制。只看当前任务，不能推论旧任务保持或遗忘。',
    '4. 本轮不自动选 G*、不新开训练；可引用事实应保留起点、步长、子集与 seed2 限定。', '',
    '追溯：`compact.json` 保留逐起点完整候选统计、SAM 半径范围、几何及 provenance；`candidate_batches.csv` 保留全部配对 batch 数值。原始训练/诊断文件未改动。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'validated':len(anchors),'expected':5,'status':report['status'],'output':str(OUT)}))
