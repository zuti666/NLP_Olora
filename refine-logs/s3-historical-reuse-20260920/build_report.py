"""Build the historical reuse report from audited evidence; no training."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = Path('/data/140-1/users/liying/CopyFlatCV')
TARGET_DIRS = {
    'logs_s3_damp_vs_endpoint_merge',
    'logs_s3_component_ablation_seed2',
    'logs_s2d_delta_r005_ablation',
    'logs_s1h_endpoint_damp_ablation',
    'logs_step2d_weightmerge_omega005_020_030_040_050_gpu12345',
    'logs_step2d_weightedpath_damp_truefisher_smoke',
    'logs_step2d_weightedpath_damp_truefisher_omega030',
    'logs_step2d_weightedpath_damp_truefisher_fishersplit_omega030',
}


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    index = read('reverse_index.json')
    grid = read('historical_four_grid.json')
    mapping = read('current_job_reuse_map.json')
    assert index['summary']['rows'] == 205
    assert not index['summary']['hash_or_metric_mismatch_ids']
    assert grid['validated_cells'] == 12
    assert mapping['same_seed_science_equivalent_jobs'] == 4
    assert mapping['not_historical_matched_jobs'] == 14
    selected = []
    for source in ['s3_components.json', 'omega_delta.json', 'truefisher.json']:
        for run in read(source)['runs']:
            row = {k: run.get(k) for k in [
                'id', 'log_path', 'config_path', 'metrics_path', 'seed',
                'tasks_completed', 'completion_status', 'faa_macro', 'aaa_macro',
                'sample_final',
            ]}
            row['specified_subdirectory'] = Path(row['log_path']).parent.name in TARGET_DIRS
            row['evidence_file'] = source
            row['config'] = run.get('printed_config', run.get('config'))
            selected.append(row)
    target = [r for r in selected if r['specified_subdirectory']]
    completed = [r for r in target if r['tasks_completed'] == 10 and r['faa_macro'] is not None]
    assert len(target) == 21 and len(completed) == 18
    assert len({r['log_path'] for r in selected}) == len(selected)
    (HERE / 'selected_runs.json').write_text(json.dumps(selected, indent=2, ensure_ascii=False) + '\n')
    with (HERE / 'selected_runs.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0]))
        writer.writeheader()
        for row in selected:
            writer.writerow({**row, 'config': json.dumps(row['config'], ensure_ascii=False)})

    reference = next(r for r in index['runs'] if r['log_path'].endswith('s3_damp_c3_alpha02_int20_s2_gpu3.log'))
    (HERE / 'reference_config_logged.json').write_text(json.dumps(reference['config'], indent=2, ensure_ascii=False) + '\n')
    (HERE / 'reference_result.json').write_text(json.dumps(reference, indent=2, ensure_ascii=False) + '\n')
    grid_rows = '\n'.join(
        f"| {r['arm']} | {r['faa_mean']:.3f} ± {r['faa_sample_sd']:.3f} | {r['aaa_mean']:.3f} ± {r['aaa_sample_sd']:.3f} |"
        for r in grid['arms']
    )
    report = f'''# S3 历史结果反查与复用审计

核验日期：2026-09-20。按最终成绩追溯原日志、运行时参数、配置、逐任务矩阵和源码行为，未按文件名直接归组。

**指定的8个子目录共21次训练尝试，其中18次具有完整10任务×50epochs结果；3个truefisher尝试均未完成。** 当前18项机制实验中，4项已有同seed科学配置的历史性能结果，14项尚无匹配控制。另有20份后验权重/Fisher存档已完成CPU分析，可直接复用历史机制诊断。

本轮只核验和整理历史证据，未新增训练、未修改当前队列、未把历史结果登记成新队列完成。23:17 CEST 实查：GPU0/1均99%占用，两个队列控制进程存活；第二轮短测20项仍pending，正式18项仍pending。此状态是时间点记录。

## 1. 反查参考结果与实际配置

原日志：[s3_damp_c3_alpha02_int20_s2_gpu3.log]({reference['log_path']})。

| 指标 | 数值 |
|---|---:|
| 最终任务宏平均 FAA | {reference['faa_macro']:.3f} |
| 逐阶段宏平均再平均 AAA | {reference['aaa_macro']:.6f} |
| 最终按样本加权 CNN top1 | {reference['sample_final']:.2f} |

三种统计量不能混用。FAA为最后一行10个任务准确率的等权均值；AAA先计算每阶段已见任务的等权均值，再对10阶段等权平均。未来任务矩阵格中的NaN是结构占位，不视为训练非有限值。

| 参数组 | 原运行设置 |
|---|---|
| 数据与训练预算 | ImageNet-A，200类，10×20类，class_shuffle=false，seed2；每任务50epochs |
| 模型 | ViT-B/16，LoRA rank10，memory_size=0 |
| 优化 | GAM，base SGD；batch128；首任务LR=.04，后续=.02；cosine；momentum=0，weight_decay=0 |
| GAM | lambda_flat=1；beta1/2/3=1；gamma_G=.1；rho=.05；r=.2 |
| EWC | lambda=2000；gamma_F=.9；damping alpha=.2；任务/累积Fisher归一化均开启 |
| 权重融合 | A开启，从Task0开始；omega=.3；快照interval20，即epoch20/40；start_ratio=0；CE加权beta=3 |
| 估计与评估上限 | Fisher及路径CE最多100batches；实际分支WeightMergeThenFisher，在融合后权重处计算Fisher |

实际运行日志证实本参考不是旧路径Fisher混合分支。配置中`approximate_fisher_in_task=true`不能单独证明做了任务内Fisher快照；在该分支中没有raw Fisher snapshot执行记录。

已保存[原样运行参数](reference_config_logged.json)、[指标与文件哈希](reference_result.json)。原样参数包含当时输出路径和设备表示，供溯源，不是直接覆盖当前输出目录的启动配置。

## 2. 历史结果能覆盖哪些实验

| 历史组 | 完成情况与结果 | 适用范围 |
|---|---|---|
| S3 damp与endpoint merge | seed0 FAA52.397 / 49.637 | 两个历史构造的比较；endpoint是任务起点/终点权重混合，同时改变阻尼和Fisher构造，不能当纯Fisher位置消融 |
| seed2组件 | EWC40.025；GAM+EWC52.293；SGD+EWC+A40.127 | 与其他目录补齐历史四格；各组都保留EWC |
| S1h endpoint阻尼 | alpha=0/.1/.2：FAA50.546/50.562/50.490 | 固定lambda40000、A关闭、seed0；不能代替当前lambda2000、A开启的阻尼扫描 |
| 共同omega扫描 | .05/.1/.2/.3/.4/.5：FAA53.036/53.128/53.112/53.179/52.741/52.531 | 六点均完整；.1从相邻目录找回；同时改变Fisher和A/B权重混合，属于旧联合超参曲线 |
| 额外ΔW惩罚 | RΔ=.05，kappa2时lambdaΔ20/50/100：49.734/49.972/49.716；lambdaΔ50时kappa1/0：49.807/50.516 | 有非零惩罚与梯度证据；kappa0时半径失去作用 |
| 额外ΔW边界配对 | kappa1、lambdaΔ50固定，RΔ=.5→.05：FAA52.815→49.807 | 有效的附加惩罚边界对照；GAM训练rho=.05、r=.2未变化 |
| truefisher三个指定分支 | smoke仅启动；普通分支完成Task0/1后止于Task2 epoch1；split止于Task0 epoch18 | 无完整最终FAA，只能复用配置与局部执行证据 |

omega=.1和RΔ=.5锚点来自指定8子目录之外，未计入18条完整结果。另找到一个独立S2E完整MC Fisher分支FAA36.049，但它关闭A、改变训练/Fisher拆分及其他设置，不可用于单变量Fisher比较。

三个truefisher分支实际采用模型标签MC Fisher，每输入从模型分布抽一个标签，并非逐样本真实标签经验Fisher。旧alpha=.001，且late/CE权重作用于Fisher快照，A/B权重路径仍均匀；普通分支late筛选只留下epoch40一个Fisher快照。这些名字不能解释成当前CE加权权重融合的完整实验。

重要文件冲突：June25的S3 damp旧日志FAA52.397对应输出JSON和checkpoint已被June29重跑覆盖；现存JSON的52.316属于后者。旧结果已从原控制台55个已观测矩阵格恢复，不能绑定覆盖后的checkpoint。上述18条完整结果中，17条有与原控制台一致的结果JSON，1条采用这一控制台恢复结果。

逐运行矩阵、配置差异和源码证据见[s3_components.md](s3_components.md)、[omega_delta.md](omega_delta.md)、[truefisher.md](truefisher.md)。

## 3. 权重融合：已有三seed证据及其边界

历史GAM×A四格已补齐seed0/1/2，共12次完整运行；以下±为跨seed样本标准差，n=3。

| 配置 | FAA mean±SD | AAA mean±SD |
|---|---:|---:|
{grid_rows}

GAM背景下A的配对FAA增量依次为+1.947、−1.754、+1.018个百分点，均值 **+0.404±1.925**，2/3种子为正；配对AAA增量 **−0.163±0.488**。EWC背景下A的配对FAA增量 **−0.036±1.272**。

因此历史结果支持GAM带来主要收益，但不足以支持“权重融合稳定提升最终准确率”。旧Aoff还跳过快照质量CE前向，会改变随机数消费，四格估计的是整个A模块开关的历史效果；它不能替代保留这些执行、只令权重不融合的匹配对照。原始逐seed数据与配对统计见[historical_four_grid.json](historical_four_grid.json)。

20份后验存档的CPU分析和权重身份核验见[posterior_diagnostics.md](posterior_diagnostics.md)与[20行数据](posterior_diagnostics.csv)。全部有限，全部与v2日志中的Fisher统计及对应A/B checkpoint匹配。融合组平均因子相对位移约0.1258%，有效LoRA ΔW相对位移约0.5152%；已有当前任务局部评估10/10任务准确率不变，这不是完整已见类别精度不变的证明。

历史诊断基线（同时关闭融合与阻尼）10/10任务的endpoint和retained权重逐元素相同，但两次Fisher平均cosine仅0.6388、相对L2差均值0.8689；融合组平均cosine0.6522。旧额外endpoint Fisher过程未保护RNG，两次估计使用不同随机样本/批次，故Fisher差异不能直接归因于融合改变曲率。存档没有epoch20/40路径因子，不能恢复路径内部的交叉项或逐快照贡献。

## 4. 当前18项任务的复用映射

| 当前任务 | 已有同seed科学配置FAA | 是否首次获得该配置的性能 |
|---|---:|---|
| center_s1 | 50.585 | 否 |
| center_s3 | 51.968 | 否 |
| a_off_s2 | 52.293 | 否 |
| sgd_ewc_s2（A开启） | 40.127 | 否 |

四项可直接供历史性能表使用；若只为该表重复训练，没有新增配置覆盖。当前重跑的新增价值是冻结执行源码/环境记录以及融合前endpoint、epoch20/40路径因子等机制产物。历史运行没有完整冻结执行记录，不能标作本轮同一冻结协议已完成，也不能宣称位级复现。

其余14项尚未找到同配置历史结果：三seed保留CE的no-merge、GAM总方向减半、固定融合模型下的Fisher endpoint位置、均匀/末次路径、alpha0/1、Fisher gamma0/1、任务归一化关闭、GAM投影关闭、rho=.025。相近名字或不同seed/预算/分支的实验不抵扣。逐项匹配条件、源码默认值和差异字段见[current_job_reuse_map.md](current_job_reuse_map.md)。

## 5. 完整索引与复用方式

logs_try共222个.log/.out文件，其中205条实验记录已重新检查原日志哈希；绑定原始矩阵的结果已重新计算。与9月18日索引对照，日志/矩阵哈希及重新计算指标不一致数为0。另17个文件属于启动器、空文件或未解析记录；指定的空truefisher smoke已在本次21条明细中单列。

| 全目录索引状态 | 记录数 |
|---|---:|
| complete_verified | 134 |
| complete_log_only | 3 |
| no_complete_training_result | 30 |
| nonfinite_training | 8 |
| partial_or_missing_final | 29 |
| partial_verified_matrix | 1 |

这些全目录数量跨数据集、任务数和epoch预算，不等于134次当前主线完整实验。各状态保留在[205条反查索引CSV](reverse_index.csv)和[JSON](reverse_index.json)中；本次重点24条含21条指定目录尝试及3条外部补充，见[selected_runs.csv](selected_runs.csv)。205份原样运行参数位于[logged_configs](logged_configs)。

按已知最终成绩反查配置：

```bash
python query_result.py --value 53.311 --metric faa_macro --dataset imageneta --seed 2
```

若手头数字来自CNN top1，改用`--metric sample_final`。默认容差0.005，可用`--tolerance`调整；返回所有命中的日志、指标和原始参数，不把相近分数当作相同配置。

复用顺序：先使用完整且协议匹配的已有结果；历史联合超参扫描单独报告；缺少控制的机制继续补齐。从最终测试成绩反查设置属于探索性审计，不能把测试集最高点重新包装成预先指定的确认性配置，也不按结果筛选种子。
'''
    (HERE / 'README.md').write_text(report)
    # File hashes make the published collection reviewable; archive tensors stay in place.
    paths = sorted(p for p in HERE.rglob('*') if p.is_file() and p.name not in ['BUNDLE_SHA256.json', 'verification.json'])
    hashes = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (HERE / 'BUNDLE_SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({'index_records': 205, 'target_attempts': len(target), 'target_complete': len(completed),
                      'same_science_current_jobs': 4, 'unmatched_current_jobs': 14,
                      'files_hashed': len(hashes), 'report': str(HERE / 'README.md')}))


if __name__ == '__main__':
    main()
