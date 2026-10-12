import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data/140-1/users/liying/O-LoRA/refine-logs/results-audit-20261001')
OUT = ROOT / 'residual_direction_scan_20261001'
OUT.mkdir(exist_ok=True)
FAMILIES = {
    'g1_plus_residual_lambda_': ('g1_plus_residual', ['0', '0.5', '1', '2']),
    'sum_family_lambda_': ('sum_family', ['0', '0.5', '1', '2']),
    'unit_interpolation_alpha_': ('unit_interpolation', ['0', '0.25', '0.5', '0.75', '1']),
}
LABELS = {prefix + value: (name, float(value)) for prefix, (name, values) in FAMILIES.items() for value in values}
INPUTS = {}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_json(p):
    INPUTS[str(p)] = digest(p)
    return json.loads(Path(p).read_text())


def read_csv(p):
    INPUTS[str(p)] = digest(p)
    return list(csv.DictReader(Path(p).open()))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


ten_ledger = read_json(ROOT / 'ten_task/VALIDITY_LEDGER.json')
assert ten_ledger['status'] == 'PASS'
assert ten_ledger['diagnostics_validated'] == ten_ledger['diagnostics_expected'] == 9
assert ten_ledger['matched_counterfactuals'] == ten_ledger['counterfactuals_total'] == 825
ten_raw = read_csv(ROOT / 'ten_task/diagnostic_counterfactual_batches.csv')
ten_summary = read_csv(ROOT / 'ten_task/diagnostic_candidate_summary.csv')
task0 = read_json(ROOT / 'task0_diagnostics/final/compact.json')
task0_raw = read_csv(ROOT / 'task0_diagnostics/final/candidate_batches.csv')
assert task0['status'] == 'completed' and task0['validated_anchors'] == task0['expected_anchors'] == 5
assert sum(a['matched_count'] for a in task0['anchors']) == sum(a['candidate_count'] for a in task0['anchors']) == 500
for item in task0['ledger']:
    assert item['status'] == 'validated'
    assert digest(item['report']) == item['report_sha256']
    INPUTS[item['report']] = item['report_sha256']
for item in ten_ledger['diagnostics']:
    assert item['status'] == 'validated' and item['finite_pass'] and item['restore_pass']
    assert digest(item['receipt']) == item['receipt_sha256']
    INPUTS[item['receipt']] = item['receipt_sha256']

selected_ten = [r for r in ten_raw if r['label'] in LABELS]
selected_task0 = [r for r in task0_raw if r['candidate'] in LABELS]
assert len(selected_ten) == 429 and len(selected_task0) == 260
ten_groups = defaultdict(list)
for r in ten_raw:
    ten_groups[(r['source_arm'], int(r['task']), int(r['batch']))].append(r)
assert len(ten_groups) == 33
for group in ten_groups.values():
    assert sum(r['label'] in LABELS for r in group) == len(LABELS) == 13
    assert len(set(r['label'] for r in group)) == 25
for r in selected_ten:
    assert r['match.ok'] == 'True'
    assert float(r['match.actual_relative_error']) < 1e-3
    assert digest(r['report_path']) == r['report_sha256']
    INPUTS[r['report_path']] = r['report_sha256']
for r in selected_task0:
    assert math.isfinite(float(r['paired_delta_vs_g1'])) and float(r['actual_match_error']) < 1e-3

summary = []
def append(protocol, anchor, version, candidate, metric, values, weights=None):
    assert all(math.isfinite(x) for x in values)
    family, coefficient = LABELS[candidate]
    summary.append({'protocol': protocol, 'anchor': anchor, 'schema_version': version,
        'family': family, 'coefficient': coefficient, 'candidate': candidate, 'metric': metric,
        'probe_batches': len(values), 'mean_CE_difference_vs_g1': float(np.mean(values)),
        'probe_sample_weighted_mean_sensitivity': float(np.average(values, weights=weights)),
        'median_CE_difference_vs_g1': float(np.median(values)),
        'min_CE_difference_vs_g1': min(values), 'max_CE_difference_vs_g1': max(values),
        'strictly_negative_batches': sum(x < 0 for x in values)})

for source in ['g1_r1', 'g3_r1', 'sum_r1']:
    for task in [0, 4, 9]:
        anchor = source.split('_')[0] + '_task' + str(task)
        for candidate in LABELS:
            group = [r for r in selected_ten if r['source_arm'] == source and int(r['task']) == task and r['label'] == candidate]
            assert len(group) == (3 if task == 9 else 4)
            for metric, column, summary_column in [
                ('current_all_seen', 'current_delta.all_seen_ce', 'current_all_seen_ce_vs_g1_mean'),
                ('current_task_local', 'current_delta.task_local_ce', 'current_task_local_ce_vs_g1_mean'),
                ('old_all_seen', 'old_task_macro_delta.all_seen_ce', 'old_task_macro_all_seen_ce_vs_g1_mean'),
                ('old_task_local', 'old_task_macro_delta.task_local_ce', 'old_task_macro_task_local_ce_vs_g1_mean'),
            ]:
                if metric.startswith('old_') and task == 0:
                    assert all(r[column] == '' for r in group)
                    continue
                differences = []
                for r in group:
                    base = next(x for x in ten_groups[(source, task, int(r['batch']))] if x['label'] == 'g1')
                    differences.append(float(r[column]) - float(base[column]))
                append('ten_task', anchor, 1, candidate, metric, differences, [int(r['sample_count']) for r in group])
                audited = next(r for r in ten_summary if r['anchor'] == anchor and r['candidate'] == candidate)
                assert abs(summary[-1]['mean_CE_difference_vs_g1'] - float(audited[summary_column])) < 1e-12

for anchor in task0['anchors']:
    assert anchor['finite_restore_passed']
    for candidate in LABELS:
        group = [r for r in selected_task0 if r['anchor'] == anchor['anchor'] and r['candidate'] == candidate]
        assert len(group) == 4
        values = [float(r['paired_delta_vs_g1']) for r in group]
        append('independent_task0', anchor['anchor'], anchor['schema_version'], candidate, 'current_all_seen', values)
        assert abs(summary[-1]['mean_CE_difference_vs_g1'] - anchor['candidates'][candidate]['paired_vs_g1']['mean']) < 1e-12
write_csv('coefficient_curves.csv', summary)

ledger = {
    'status': 'PASS', 'ten_task': {'validated_anchors': 9, 'expected_anchors': 9, 'probe_batches': 33,
        'scan_counterfactuals_validated': 429, 'scan_counterfactuals_expected': 429, 'all_candidates_matched': '825/825'},
    'independent_task0': {'validated_anchors': 5, 'expected_anchors': 5, 'probe_batches': 20,
        'scan_counterfactuals_validated': 260, 'scan_counterfactuals_expected': 260, 'all_candidates_matched': '500/500',
        'schema1_anchors': 4, 'schema2_anchors': 1, 'old_mean_partial_result_excluded': True,
        'gpu_instrumentation_sensitivity_max': task0['gpu_instrumented_vs_native_g3_relative_difference_max']},
    'failed_or_missing_in_final_selected_scans': 0, 'training_seed': 2, 'independent_seed_count': 1,
    'sample_standard_deviation_across_seeds': None, 'coefficient_grid_is_offline_single_step': True,
    'complete_training_coefficient_runs_identified': 0,
    'formal_training_inventory_scope': 'two main campaigns, 56 nearby s3 suites / 468 job records, logs_try 267 files; no universal negative claim',
    'no_pooled_anchor_statistic': True, 'diagnostic_reference_eta': .02,
    'step_matching': 'joint norm of actual concatenated LoRA delta(BA), FC delta W, FC delta b',
    'not_matched': ['post-update prediction KL', 'full training trajectory'], 'input_sha256': INPUTS,
}
(OUT / 'VALIDITY_LEDGER.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2)+'\n')

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
titles = [r'$g_1+\lambda r_\perp$', r'$(1+a)g_1+\lambda r_\perp$', 'Normalized direction interpolation']
families = [name for name, values in FAMILIES.values()]
colors = ['#3264aa', '#5286bb', '#d4742b', '#c44732']
fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), constrained_layout=True)
for row, metric in enumerate(['current_all_seen', 'old_all_seen']):
    for col, family in enumerate(families):
        ax = axes[row, col]
        for anchor, color in zip(['g1_task4', 'g1_task9', 'g3_task4', 'g3_task9'], colors):
            data = sorted([r for r in summary if r['protocol'] == 'ten_task' and r['anchor'] == anchor and r['metric'] == metric and r['family'] == family], key=lambda r:r['coefficient'])
            ax.plot([r['coefficient'] for r in data], [1000*r['mean_CE_difference_vs_g1'] for r in data],
                label=anchor, color=color, marker='o', ms=3, linestyle='--' if anchor.endswith('9') else '-')
        ax.axhline(0, color='gray', lw=.8)
        ax.grid(axis='y', alpha=.15)
        ax.set_xlabel(r'$\alpha$' if col == 2 else r'$\lambda$')
        ax.set_ylabel(('Current task' if row == 0 else 'Old tasks, equal-task mean')+'\nall-seen CE difference vs g1 (x 1e-3)')
        if row == 0:
            ax.set_title(titles[col])
        if row == 0 and col == 0:
            ax.legend(frameon=False, fontsize=8)
for suffix in ['png','pdf']:
    fig.savefig(OUT / ('ten_task_residual_interpolation.'+suffix), dpi=300)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True)
for col, family in enumerate(families):
    ax = axes[col]
    for anchor, color in zip(['s_epoch50', 'g_epoch50', 'sum_epoch50', 'mean_epoch50_v2'], colors):
        data = sorted([r for r in summary if r['protocol']=='independent_task0' and r['anchor']==anchor and r['family']==family], key=lambda r:r['coefficient'])
        ax.plot([r['coefficient'] for r in data], [1000*r['mean_CE_difference_vs_g1'] for r in data], marker='o', ms=3, color=color,
            linestyle='--' if anchor.endswith('v2') else '-', label=anchor+' (schema2)' if anchor.endswith('v2') else anchor)
    ax.axhline(0,color='gray',lw=.8)
    ax.grid(axis='y',alpha=.15)
    ax.set(title=titles[col], xlabel=r'$\alpha$' if col==2 else r'$\lambda$', ylabel='Task0 CE difference vs g1 (x 1e-3)')
    if col==0:
        ax.legend(frameon=False,fontsize=7)
for suffix in ['png','pdf']:
    fig.savefig(OUT / ('task0_residual_interpolation_epoch50.'+suffix),dpi=300)
plt.close(fig)

report = '''**残差方向与方向插值：已有结果及可支持结论**

已经完成的是同 checkpoint、同 batch 下的离线等有效步幅单步扫描。在已检索的两主campaign、56份同级s3 suites/468条任务记录及logs_try的267文件中，未发现这三个系数家族的独立完整训练任务、候选训练checkpoint或AAA/FAA结果。不能将下述单步CE曲线作为完整训练超参曲线。本轮只分析既有结果，未启动训练或修改队列。

十任务诊断9/9 checkpoint、33个探测batch，全25候选825/825匹配；独立Task0诊断5/5起点、20个batch，全25候选500/500匹配。提取本次三族13个系数时，分别为429/429、260/260，有效缺失0。仅seed2，标准差不适用，不能把batch或checkpoint当独立seed。Task0四个schema1和一个Mean schema2分别显示、不跨起点汇总；旧Mean失败的部分结果排除。schema2 GPU插桩最大局部梯度差约0.413%，不声称逐位复现。

令 a=<g3,g1>/||g1||²，q=r_perp=g3−a*g1。分解在LoRA A/B+FC梯度参数坐标中定义，不意味着有效BA或函数空间中也正交。现有a均为正值。所有候选匹配同一实际concat(delta BA, delta FC weight/bias)范数：目标为本状态以eta=.02沿g1更新的有效步幅；每个方向单独求正标量、真实应用并检查后恢复。没有匹配更新后的预测KL。任务0评测固定128张图；十任务每任务至多128张，实际71–128，按旧任务等权求损失均值。参考步长.02不是末端训练学习率。

| 家族 | 定义 | 扫描点 | 端点说明 |
|---|---|---|---|
| 残差加入g1 | g1+lambda*q | 0,.5,1,2 | lambda=0为g1；lambda=1不是g3 |
| 残差加入Sum平行分量 | (1+a)*g1+lambda*q | 0,.5,1,2 | lambda=0为scale-only；lambda=1等于g1+g3 |
| 单位方向插值 | N((1−alpha)N(g1)+alpha*N(g3)) | 0,.25,.5,.75,1 | alpha=0/1为g1/g3方向；alpha=.5一般不等于原始Sum方向 |

所有数值为候选单步CE变化减去同batch g1单步CE变化；负值表示相对g1更好，正值更差，单位为CE，不是准确率百分点。当前任务all-seen和旧任务all-seen均在全部已见类别中计算；任务内local另外保留，绝不混合加总。

这里lambda是残差系数，不是ewc_lambda；alpha是单位梯度方向的插值比例，不是Fisher阻尼或权重融合系数。“变差”均指弱于同状态g1单步，不一定指损失比不更新时更高。

**1. 新增方向确实存在，但不是普遍有益。** 原g3=a*g1+q的scale-only对照为a*g1；在a>0且等有效步幅时，它应与g1重合。新增q会产生不同的损失响应，因此不能把全部局部响应解释为原始梯度幅值。零系数共线控制存在最高约3.6e−6的CE数值差，不能把这些微小正负当性能增益。

**2. 独立Task0不支持平均残差收益。** 每个起点的三族平均曲线都以零残差/零插值最好：50个非零系数单元没有一个平均优于g1。单位插值五个起点均随alpha增加而平均CE变差。这不是“所有batch都更差”：例如S50的g1+2q在3/4batch改善，但另一个batch恶化更大，平均仍为+.000217691。初始S0的方向插值alpha=.25/.5/.75/1相对g1平均为+.006241/+.018962/+.037457/+.059279。更强残差不能从这组数据推广为更好的单任务泛化。

**3. 十任务出现状态依赖的新旧任务取舍。** 下表固定g1+q（lambda=1），与零残差g1比较；全部使用all-seen CE：

| checkpoint来源 | 当前任务CE差 | 旧任务CE差 | 观察 |
|---|---:|---:|---|
| g1轨迹Task4 | +.002401 | −.001180 | 旧任务改善、当前任务变差 |
| g1轨迹Task9 | +.015420 | −.003687 | 同上，取舍更明显 |
| g3轨迹Task4 | +.011620 | +.004220 | 当前、旧任务均变差 |
| g3轨迹Task9 | +.011426 | +.006874 | 当前、旧任务均变差 |

g1_task9的lambda从.5→1→2时，旧任务平均差为−.003402→−.003687→−.003784，收益逐渐趋缓；当前任务为+.013276→+.015420→+.016545，代价持续增大。g3_task9则两种损失随残差增加而持续恶化。g1_task4的current-local对lambda=.5/1/2呈−.000776/−.000667/−.000408，但current-allseen为+.001137/+.002401/+.003563，说明所谓“更好方向”依赖损失口径，不能说所有目标均改善。

**4. 方向插值有局部内部最优提示，但很弱。** g1_task4在all-seen口径下，alpha=.25使当前/旧任务平均CE相对g1为−.000163/−.000228；alpha=1却为当前+.001158、旧−.000904。即当前任务的网格最优在内部，而旧任务更偏向g3端点。不过alpha=.25的当前任务改善仅1/4个探测batch出现，旧任务2/4；probe batch大小为128/128/128/42。三个128样本batch的当前all-seen差均为正，42样本batch为−.003219，驱动等批均值的负差；若以probe样本数加权作敏感性分析，当前差反向为+.000454（旧all-seen仍为−.000062）。这不是替换主估计量，而是说明内部最优对权重选择和批次构成敏感，不能确认稳定方向互补。g1_task9从alpha=.25到1则旧任务获益更大、当前代价更大；g3自身Task4/9是向g3插值越多，两类平均CE越差。没有统一最佳alpha。

**可以写的结论：** 在同状态、等实际BA+FC有效更新范数的局部干预中，g3相对于g1的残差分量能改变当前与旧任务的损失取舍，且效果明显随训练状态和评测口径变化。当前结果未建立普遍的残差方向优势或稳定的内部最优插值系数。

**不能写的结论：** 残差方向普遍减少梯度冲突；残差越大越好；已找到提高完整训练AAA/FAA的lambda/alpha；长期收益完全独立于幅值；g3新增方向无用。最后一句同样不成立，因为末端虚拟一步无法代表完整优化路径。候选方向之间的余弦不是旧任务梯度与更新的干扰指标。

**下一步建议，尚未启动。** 若要验证完整训练机制，固定当前rank10/r=1/rho=.05/seed2及所有数据和优化设置，实施训练全程等有效步幅的g1、残差候选（例如lambda=.5）及插值候选（例如alpha=.25）。这些仅为待验证点，不称最佳点。两任务实验需独立重训，并同时测a11/a21/a22、获取后保持变化，以及全评测集的local/all-seen accuracy与CE；旧任务样本只用于离线诊断，不参与训练。优先辨别前任务解形成与后任务更新的作用，再扩大系数网格。

**附件说明。** coefficient_curves.csv提供每起点、每家族、每系数、每损失口径的均值/中位数/范围/负差batch计数，未跨起点平均，也无统计显著性检验。ten_task_residual_interpolation.pdf的上、下行分别为当前任务及旧任务all-seen CE差，显示四个代表性起点；task0_residual_interpolation_epoch50.pdf分别显示四个终点，Mean版本单独标记。图的纵轴为CE差乘1000，零线是同起点g1，没有准确率单位和误差条。
'''
(OUT/'REPORT.md').write_text(report)
manifest = {'inputs': INPUTS, 'outputs': {str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p != OUT/'MANIFEST.json'}}
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','scan_counterfactuals':'689/689','full_counterfactuals':'1325/1325','curve_rows':len(summary),'output':str(OUT)},ensure_ascii=False))
