import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data/140-1/users/liying/O-LoRA/refine-logs/results-audit-20261001')
OUT = ROOT / 'g1_vs_g3_deeper'
CAMPAIGN = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-stage1-task0-run-20260930T213643Z')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return list(csv.DictReader(Path(path).open()))


def write_csv(name, data):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


pins = {}
def pin(path):
    pins[str(path)] = sha(path)


training_path = ROOT / 'ten_task/training_results.json'
pin(training_path)
training = json.loads(training_path.read_text())
validation_path = ROOT / 'ten_task/VALIDITY_LEDGER.json'
pin(validation_path)
validated = json.loads(validation_path.read_text())
assert validated['status'] == 'PASS'
pair = {x['arm']: x for x in training if x['arm'] in ['g1_r1', 'g3_r1']}
assert set(pair) == {'g1_r1', 'g3_r1'}
configs = {}
for arm, data in pair.items():
    assert data['status'] in ['completed', 'reused'] and data['seed'] == 2
    audit = next(x for x in validated['training'] if x['arm'] == arm)
    assert audit['status'] == 'validated' and audit['observed_cells_finite']
    assert sha(audit['receipt']) == audit['receipt_sha256']
    pin(audit['receipt'])
    assert json.loads(Path(audit['receipt']).read_text())['status'] == 'completed'
    assert data['tasks_completed'] == data['expected_tasks'] == 10
    for label in ['metrics', 'config']:
        p = Path(data[label + '_path'])
        assert sha(p) == data[label + '_sha256']
        pin(p)
    c = json.loads(Path(data['config_path']).read_text())
    configs[arm] = c
    expected = {'lora_rank': 10, 'gam_grad_rho': .05, 'gam_grad_norm_rho': 1.,
                'gam_grad_gamma': 0., 'lambda_flat': 1., 'ewc_lambda': 0.,
                'ewc_skip_task_fisher': True, 'in_task_weight_merge': False,
                'task_endpoint_weight_merge': False, 'approximate_fisher_in_task': False}
    for k, v in expected.items():
        assert c[k] == v, (arm, k, c[k])
    assert (c['gam_grad_beta_1'], c['gam_grad_beta_3']) == ((1., 0.) if arm == 'g1_r1' else (0., 1.))
    for t, r in enumerate(data['cnn_matrix']):
        assert all(v is not None and math.isfinite(v) for v in r[:t+1])
        assert all(v is None for v in r[t+1:])

completion_path = CAMPAIGN / 'results_recovery_20261001/TRAINING_COMPLETION_AUDIT.json'
pin(completion_path)
completion = json.loads(completion_path.read_text())
assert all(completion['checks'].values())
single = {}
for job in ['task0_s_k10_s2', 'task0_g_k10_s2']:
    entry = next(x for x in completion['receipts'] if x['job'] == job)
    p = Path(entry['receipt'])
    assert sha(p) == entry['receipt_sha256']
    pin(p)
    receipt = json.loads(p.read_text())
    summary = receipt['summary']
    assert receipt['returncode'] == 0 and not receipt['errors']
    assert summary['status'] == 'completed' and summary['completed_epochs'] == 50
    assert summary['seed'] == 2 and summary['trained_tasks'] == [0] and not summary['smoke']
    assert summary['finite_check']['passed']
    single[job] = summary['final']
assert single['task0_s_k10_s2']['eval_per_class_counts'] == single['task0_g_k10_s2']['eval_per_class_counts']

stage_path = OUT / 'agent_matrix/per_stage_seen_old_current.csv'
task_path = OUT / 'agent_matrix/per_task_acquisition_final_bwt.csv'
pin(stage_path)
pin(task_path)
stages = rows(stage_path)
tasks = rows(task_path)
assert len(stages) == len(tasks) == 10
g1, g3 = pair['g1_r1'], pair['g3_r1']
decomposition = {
    'faa_difference_pp': g3['faa_task_macro'] - g1['faa_task_macro'],
    'acquisition_difference_pp': g3['acquisition_task_macro'] - g1['acquisition_task_macro'],
    'bwt_difference_pp': g3['bwt'] - g1['bwt'],
    'retention_endpoint_contribution_pp': .9 * (g3['bwt'] - g1['bwt']),
    'aaa_difference_pp': g3['aaa_task_macro'] - g1['aaa_task_macro'],
}
assert abs(decomposition['faa_difference_pp'] - decomposition['acquisition_difference_pp'] - decomposition['retention_endpoint_contribution_pp']) < 1e-10
for arm in pair:
    d = pair[arm]
    assert abs(d['faa_task_macro'] - d['acquisition_task_macro'] - .9 * d['bwt']) < 1e-10

comparison = []
for label, arm, job in [('g1', 'g1_r1', 'task0_s_k10_s2'), ('g3(r=1)', 'g3_r1', 'task0_g_k10_s2')]:
    d, s = pair[arm], single[job]
    comparison.append({'arm': label, 'seed': 2, 'single_task_correct': sum(s['eval_per_class_correct']),
        'single_task_samples': s['eval_samples'], 'single_task_accuracy_pct': s['eval_accuracy'],
        'single_task_class_macro_pct': s['eval_macro_accuracy'], 'single_task_eval_ce': s['eval_ce'],
        'single_task_train_ce': s['train_ce'], 'ten_task_AAA_pct': d['aaa_task_macro'],
        'ten_task_FAA_pct': d['faa_task_macro'], 'ten_task_acquisition_pct': d['acquisition_task_macro'],
        'ten_task_BWT_pp': d['bwt'], 'ten_task_forgetting_pp': d['forgetting']})
write_csv('performance_comparison.csv', comparison)

geometry_path = ROOT / 'ten_task/diagnostic_geometry_batches.csv'
pin(geometry_path)
geom = [r for r in rows(geometry_path) if r['source_arm'] in pair]
geometry = []
for arm in pair:
    for task in [0, 4, 9]:
        batch = [r for r in geom if r['source_arm'] == arm and int(r['task']) == task]
        assert len(batch) == (3 if task == 9 else 4)
        for r in batch:
            assert sha(r['report_path']) == r['report_sha256']
            pin(r['report_path'])
        ratios = [float(r['gradient_norms.g3']) / float(r['gradient_norms.g1']) for r in batch]
        assert all(math.isfinite(v) for v in ratios)
        geometry.append({'anchor_arm': arm, 'task_index': task, 'probe_batches': len(batch),
             'mean_of_paired_gradient_norm_ratios_g3_over_g1': float(np.mean(ratios)),
             'min_paired_gradient_norm_ratio': min(ratios), 'max_paired_gradient_norm_ratio': max(ratios),
             'mean_cos_g1_g3': float(np.mean([float(r['cosines.g1_g3']) for r in batch])),
             'mean_perpendicular_fraction': float(np.mean([float(r['r_perp_over_g3']) for r in batch])),
             'mean_g3_probe_parameter_distance': float(np.mean([float(r['sam.parameter.match.target']) for r in batch]))})
write_csv('same_state_geometry.csv', geometry)

candidate_path = ROOT / 'ten_task/diagnostic_candidate_summary.csv'
pin(candidate_path)
diagnostics = [r for r in rows(candidate_path) if r['source_arm'] in pair and r['candidate'] == 'g3']
assert len(diagnostics) == 6
write_csv('matched_step_counterfactuals.csv', diagnostics)

epoch_path = ROOT / 'task0/epoch_curves.csv'
pin(epoch_path)
epochs = [r for r in rows(epoch_path) if r['job'] in single]
assert len(epochs) == 100
write_csv('single_task_epoch_curves.csv', epochs)

ledger = {'scope': 'g1 versus g3(r=1), seed 2; independent Task0 and original ten-task protocols kept separate',
    'ten_task_training': {'validated': 2, 'expected': 2, 'failed': 0, 'missing': 0, 'finite_observed_matrix_cells': 110},
    'independent_single_task_training': {'validated': 2, 'expected': 2, 'failed': 0, 'missing': 0},
    'ten_task_same_state_diagnostics': {'validated_anchors': 6, 'expected_anchors': 6, 'probe_batches': len(geom)},
    'paired_seed_count': 1, 'sample_standard_deviation': None,
    'tasks_and_probe_batches_are_not_independent_seeds': True,
    'decomposition_is_descriptive_not_causal': True,
    'single_task_not_the_first_diagonal_of_original_ten_task_run': True,
    'decomposition': decomposition, 'input_sha256': pins}
(OUT / 'VALIDITY_LEDGER.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + '\n')

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
c1, c3 = '#3264aa', '#d4742b'
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), constrained_layout=True)
x = np.arange(1, 11)
for key, label, color in [('g1_seen_task_mean', r'$g_1$', c1), ('g3_seen_task_mean', r'$g_3$, $r=1$', c3)]:
    axes[0].plot(x, [float(r[key]) for r in stages], marker='o', ms=4, label=label, color=color)
axes[0].set(xlabel='Number of learned tasks', ylabel='Mean accuracy of seen tasks (%)', xticks=x, title='(a) Ten-task performance trajectory')
axes[0].legend(frameon=False)
axes[0].grid(axis='y', alpha=.15)
a1, a3 = np.diag(np.array(g1['cnn_matrix'], dtype=float)), np.diag(np.array(g3['cnn_matrix'], dtype=float))
final_gap = np.array(g3['cnn_matrix'][9]) - np.array(g1['cnn_matrix'][9])
acq_gap = a3 - a1
ret_gap = final_gap - acq_gap
axes[1].bar(x-.17, acq_gap, width=.34, color=c3, label='Acquisition difference')
axes[1].bar(x+.17, ret_gap, width=.34, color='#43958c', label='Change since acquisition')
axes[1].plot(x, final_gap, color='#303030', marker='D', ms=4, lw=1, label='Final difference')
axes[1].axhline(0, color='gray', lw=.8)
axes[1].set(xlabel='Task (1-based display)', ylabel=r'$g_3-g_1$ accuracy difference (pp)', xticks=x, title='(b) Final difference by task')
axes[1].legend(frameon=False, fontsize=8)
axes[1].grid(axis='y', alpha=.15)
for suffix in ['png', 'pdf']:
    fig.savefig(OUT / ('ten_task_g1_g3_decomposition.'+suffix), dpi=300)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
for job, label, color in [('task0_s_k10_s2', r'$g_1$', c1), ('task0_g_k10_s2', r'$g_3$, $r=1$', c3)]:
    e = [r for r in epochs if r['job'] == job]
    for ax, metric in zip(axes, ['eval_accuracy', 'eval_ce']):
        ax.plot([int(r['epoch']) for r in e], [float(r[metric]) for r in e], label=label, color=color)
        ax.set_xlabel('Training epoch')
        ax.grid(axis='y', alpha=.15)
axes[0].set(ylabel='Held-out sample accuracy (%)', title='(a) Independently trained Task0, 307 images')
axes[1].set(ylabel='Held-out cross entropy', title='(b) Early gain narrows at later epochs')
axes[0].legend(frameon=False)
axes[1].legend(frameon=False)
for suffix in ['png', 'pdf']:
    fig.savefig(OUT / ('single_task_g1_g3_learning.'+suffix), dpi=300)
plt.close(fig)

report = '''**g1 与 g3(r=1) 的进一步分析，seed 2**

当前比较的独立 Task0 和十任务训练各有 2/2 完整有效结果；缺失 0、无效 0。十任务两矩阵共 110 个已观测值均有限。两套协议分别核验，不把独立 Task0 结果拼入原十任务矩阵。这里只分析既有结果，没有启动训练、改变队列或修改冻结源代码。所有差值均为 g3−g1，准确率差以百分点表示。单 seed，标准差不适用；任务、epoch、诊断 batch 均不作为独立重复。

| 指标 | g1 | g3(r=1) | 差值 |
|---|---:|---:|---:|
| 独立 Task0 样本准确率 | 85.016% (261/307) | 85.342% (262/307) | +0.326 pp |
| 独立 Task0 类平均准确率 | 86.951% | 85.959% | −0.993 pp |
| 独立 Task0 eval CE | 0.410457 | 0.445263 | +0.034806 |
| 独立 Task0 train CE | 0.409575 | 0.527122 | +0.117546 |
| 十任务 AAA，任务等权 | 62.166% | 65.993% | +3.828 pp |
| 十任务 FAA，任务等权 | 45.132% | 50.679% | +5.547 pp |
| 十任务 acquisition，对角均值 | 53.436% | 56.728% | +3.292 pp |
| 十任务 BWT，旧九任务 | −9.227 pp | −6.721 pp | +2.506 pp |
| 十任务 peak-forgetting，旧九任务 | 9.482 pp | 7.011 pp | −2.471 pp |

**1. 单任务的终点优势很弱，但早期优化过程有区别。** g3 在 epoch5 的评测准确率为 84.365%，g1 为 79.805%；epoch10 为 86.319% / 83.388%。到固定 epoch50 只剩净一张图的差异，且类平均准确率、CE 都是 g1 更好。不能仅根据 85.342>85.016 声称稳定的单任务泛化优势。已有数据与“g3 改变早期优化速度或路径”的解释相容，尚未确定成因。固定 epoch50 是预定比较终点，早期峰值不替代终点，也不用于事后重选 checkpoint。

**2. 十任务增益同时包含获取和保持，两者需要分别解释。** 记 A[t,j] 为训练完 t 后任务 j 的准确率。Acquisition=mean_j A[j,j]，BWT=mean_{j<9}(A[9,j]−A[j,j])。严格恒等式为 FAA=Acquisition+0.9×BWT。因此最终差值 5.547=3.292+2.255 pp，后项为 0.9×2.505556。它是两条训练轨迹的终点记账分解，不能视作因果机制贡献；peak-forgetting 基于历史最大值，不能替换这里的 BWT。

AAA 是十个阶段的已见任务均值的平均。不能仅因 FAA 增益大于 AAA 增益，就断言收益主要来自抗遗忘。原十任务的 Task0 对角值为 85.02%/85.67%，与独立 Task0 的 85.016%/85.342% 不是同一个实验点。

**3. 优势早期已出现，且存在明确任务反例。** 原十任务学完 Task1（第二个任务）时，g3 的新任务准确率比 g1 高 8.27 pp，旧 Task0 却低 2.28 pp，合成阶段差为 (8.27−2.28)/2=2.995 pp。该阶段 g3 的旧任务下降反而更大，不能描述成“一开始获取相同，只是后期少遗忘”。Task5（第六个任务）g3 的 acquisition 低 12.68 pp，最终准确率低 11.26 pp。Task6 则 acquisition 高 12.05 pp，后续下降比 g1 多 2.41 pp。最终十个任务为 8 个更高、1 个更低、1 个相同；Task4 的最终增益 13.73 pp 几乎全部来自获取增益 13.72 pp。

**4. 当前设置改变了探针位置、方向与最终更新幅值。** 两组都为 ImageNet-A、固定类别顺序、每任务20类、ViT-B/16、Q/V LoRA rank10+分类头、seed2、每任务50epochs、batch128。SGD momentum=0、weight_decay=0；首任务 LR=.04→.004，后续 .02→.002，cosine。rho=gam_grad_rho=.05，r=gam_grad_norm_rho=1。lambda_flat=1、EWC lambda=0、skip-task-Fisher=true、任务内及任务末权重融合关闭、近似Fisher关闭、gamma_GAM=0、SGV关闭。最终 beta1/beta3 分别为 g1:1/0，g3:0/1。两组仍执行相同原生四探针流程；gamma=0 去掉投影修正的最终贡献，并不删除 gradient_decompose 的算术调用。

用 N(g)=g/||g|| 表示归一化，忽略 epsilon 与浮点往返误差的数学形式是：

g0=∇L(theta)
g1=∇L(theta+rho N(g0))
v=N(g1−g0)
g2=∇L(theta+r v)
g3=∇L(theta+r v+rho N(g2))

理想数学更新分别为 theta−eta*g1 和 theta−eta*g3；原生 add/sub 恢复后的基点可能有浮点舍入残差，不宣称逐位等价。最终梯度没有自动做等范数匹配。r 是第二次移动的长度，g3 点相对于 theta 的总距离并不等于 r。它是对 LoRA A/B 和当前分类头共同参数空间的扰动，不能等同于有效 BA 权重距离。现有实际距离匹配诊断显示 g3 探针总参数距离约 1.01–1.04，而 g1 为 .05，约相差20倍。同状态同batch的真实 g3/g1 范数比见 same_state_geometry.csv，不能把不同训练轨迹的均值做向量代数。

十任务同状态逐batch求 norm(g3)/norm(g1) 后平均：g1训练轨迹 Task0/4/9 分别为 2.761/2.888/5.666，g3训练轨迹则为 2.142/1.791/1.907。它们是这几个 checkpoint 上的离线局部值，不是训练全过程的平均有效学习率；共同表明即使 beta 都设为1，两方向的幅值也没有受控。

局部光滑且 rho 足够小时，g1−g0≈rho H N(g0)，因此外层偏移是由局部梯度变化引导。它不是精确 Hessian 最大特征向量，也不是已经求得球内最大损失点；r=1 时不能默认小位移 Taylor 展开准确。更远的探测、不同的方向、较大的原始更新共同变化，当前原生训练比较没有拆开这三项。

**5. 已有等步幅诊断不支持“g3 在每个状态都更优”。** 在固定同一 checkpoint 和 batch 后，匹配实际 concat(LoRA ΔBA,分类头 ΔW/Δb) 的更新范数，再做单步评测。十任务 g1_task9 的 g3−g1 当前任务 local ΔCE 为 +.005463、旧任务 all-seen ΔCE 为 −.003203，存在取舍；g3_task4/g3_task9 则分别为当前 +.003606/+.004734、旧 +.003317/+.005824，两者均是 g1 更有利。独立 Task0 S50/G50 起点也没有平均方向优势。诊断 eta=.02、固定小评测子集、少量 batch 和末端状态不能代替完整训练路径，所以这也不能反推 g3 长期训练无效。

g1 与 g3 的余弦只是候选更新之间的差异，不能当成新旧任务梯度冲突。要支持“少干扰”，需另测旧任务损失梯度与实际更新的内积，而目前没有这一直接证据。

**6. 类别竞争是下一步分析不可遗漏的维度。** 当前训练 CE 只切当前任务20类，评测却在所有已见类别上取最大预测，再按真实标签分组形成矩阵。共享 LoRA 持续训练，分类头逐任务扩展并复制旧行。在当前 momentum=weight_decay=0 且 EWC/Fisher 关闭时，旧分类头行没有直接训练梯度；旧样本特征变化与新增分类头行参与预测仍可改变跨任务竞争。于是 acquisition 和 BWT 同时包含任务内判别能力及新旧类之间的评分竞争，BWT 并非纯粹的表征遗忘。两个候选方向又同时扰动 LoRA 与当前分类头，因此分类头校准/竞争是可行解释之一，但需要 local 与 all-seen 指标及模块控制验证。

**当前最合适的表述：** 在 seed2 的无 Fisher、无 EWC、无权重融合、gamma=0 设置下，g3(r=1) 的十任务轨迹获得更高的任务获取准确率和更好的最终保持；独立 Task0 的终点没有一致的指标优势。已有同状态诊断显示 g3 确有不同方向和更大幅值，但尚未证明新增方向普遍有益、减少梯度冲突或形成更平坦解。

**建议下一步，尚未启动。** 为解释当前 r=1 的现象，先固定 S=g1、G=g3(r=1)，独立从 pretrained 开始训练两任务四组 SS、GS、SG、GG，保持 seed2、类别顺序、50epochs/任务、学习率、当前20类 CE、分类头初始化和逐epoch采样/增强配对。除定义 S/G 所需的 beta 外，不再调整其他超参数。记录 a11、a21、a22；重点 R=a21−a11 的 GS−SS（第一任务形成作用）、SG−SS（第二任务继续学习作用），以及四格 interaction。用 a22 检查是否仅牺牲新任务获取换来旧任务保留。切换前、扩头后未更新、第二任务epoch1/10/20/30/40/50均评测完整held-out集合，分别保存 task-local 和 all-seen accuracy/CE；记录实际BA+FC步幅、旧任务梯度与候选更新内积，仅用于离线诊断。

若后续目标是排除规模解释，需要进一步加入训练层面等有效步幅的 g3 与 g1，以及与 g3 实际探针距离匹配的 SAM 对照。Mean/Sum 的二倍关系不能替代 g1/g3 的等幅控制。先完成两任务归因，再决定是否扩展三个任务，不先启动大网格。

**图与数据。** ten_task_g1_g3_decomposition.pdf 展示已见任务均值和逐任务终点分解；single_task_g1_g3_learning.pdf 展示单任务全过程 accuracy/CE。均为单seed描述图，无误差条，任务编号在图中显示1–10，对应原始Task0–9。所有数值来自带SHA清单的CSV/JSON，不修改旧实验记录。
'''
(OUT / 'ANALYSIS.md').write_text(report)

manifest = {'inputs': pins, 'outputs': {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'MANIFEST.json'}}
(OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'status': 'PASS', 'training_validated': '4/4, two protocols separate', 'decomposition': decomposition,
                  'output_dir': str(OUT)}, ensure_ascii=False))
