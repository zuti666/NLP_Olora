# 补齐用户六行ImageNet-A表：最小实验清单

状态：只完成现有结果整理与配置设计；以下新训练未启动、未入队。六个目标中4个已有三seed完整结果，2个未执行，即12/18个目标×seed单元已完成、6/18缺失，已有结果没有发现无效单元。

| 缺失行 | 第一任务父起点 | 第二任务 | seeds | 新训练量 |
|---|---|---|---|---|
| Shaping only | 已完成GAM anchor | GAM，lambda_flat=1，EWC λ=0 | 0/1/2 | 3段×50轮 |
| Task-2 shaping | 已完成SGD anchor | GAM，lambda_flat=1，EWC λ=2000 | 0/1/2 | 3段×50轮 |

总计6段第二任务续训、300个task-epochs。第一任务起点均可复用；第一任务没有历史保护损失，现有anchor配置中的λ2000不会使其变成“第一任务带保护”。无需重新训练第一任务或完整十任务序列。

现有GAM→GAM＋EWC三seed来自正式sequence_gam的task01。用户表需要的第一任务batch Fisher proxy、a11/a21/a22/BWT和有效QV移动都已有，因此这行无需重训或新增GPU测量。该分支没有B3的固定train256 endpoint逐样本raw Fisher，但用户表不包含这一列；若将来新增该列，再单独评估三个已有checkpoint即可。

**执行前需要的实现适配**

1. 新建版本化续训入口，显式区分anchor_method与future_optimizer，并核对父manifest/seed/SHA。旧continuation无论method参数为何都固定未来SGD；旧sequence虽然能运行未来GAM，但跨SGD父起点时会把anchor_method错标为gam。不能只改配置名称就认为身份正确。冻结训练算法本身已有GAM与EWC支持。
2. 六行汇总按实际两阶段目标和λ映射。原B3_B为GAM→SGD无EWC，不能改名冒充Shaping only；原B3_D为Task-1 shaping，不能改名冒充FS-EWCLoRA。
3. 训练数据、类别顺序、父状态、初始化、workers4、batch128、lr .02、50轮、FP32/TF32关闭及rq2_shared_task_rng_v1与已完成分支一致。新head、训练pass顺序和已有RNG收据按seed配对核验。
4. 每个新run保存完整50轮与数据遍历收据、task01 checkpoint、Fisher、任务准确率矩阵和配置哈希。表中a21/a22必须取两任务已见40类下的分组准确率，不取固定旧20类head的local accuracy。位移可在CPU从父/子checkpoint有效Q/V权重求差，也可用新版endpoint入口统一复评；不能直接套用旧入口的future-SGD断言和硬编码标签。

这六行可以比较：EWCLoRA→Task-1 shaping、EWCLoRA→Task-2 shaping、Shaping only→FS-EWCLoRA，以及四条带EWC分支的两阶段shaping交互。后两类关键对照目前因两行缺失不能完整估计。

六行并非完整2×2×2八格设计。已有额外GAM→SGD/λ0一行；若另需完整八格，还要新增SGD→GAM/λ0的3个seed，但这不是补齐用户六行表的必要实验。

具体六个计划单元、父checkpoint/Fisher/metrics路径和SHA见[MISSING_TRAINING_PLAN.json](MISSING_TRAINING_PLAN.json)。训练实现审查见[CONFIG_IMPLEMENTATION_AUDIT.md](CONFIG_IMPLEMENTATION_AUDIT.md)。
