# Full 六行旧实验：历史参数快照与按层位移补查

2026-09-20。只读检查六个旧final checkpoint的metadata、六份原日志、六份metrics JSON、各自exp_run目录及当前保存代码。没有使用GPU、启动训练、修改原仓库，也没有重复将六份权重实体读入CPU内存。

**结论：现存六个 `final_model.pt` 不含旧backbone快照；本轮限定的原日志/JSON/同run目录也没有独立的backbone或逐层drift，因此不能绕过被覆盖的原Task1 anchor，精确重建历史有效权重差向量或共同K/V子空间位移。** 已有“历史总能量减新头能量”仍只能用于带条件的backbone总范数分解，不能恢复差向量、各层运动或原旧模型。

## 1. 六份final文件直接检查

检查对象就是 [full_sources.json](full_sources.json) 中 `table_selected=true` 的六条：unified `factored_full`、`factored_s1`、`factored_s2`、`control`，以及old `baseline`、`gam_only`。

独立检查采用 `torch.load(path, map_location='meta', weights_only=True, mmap=True)`，只恢复结构和shape，不物化tensor数据。六份结果完全一致：

- 顶层只有 `tasks` 和 `model_state_dict`；`tasks=1`。
- `model_state_dict` 有152个tensor、85,829,416个标量坐标，key前缀只有 `backbone` 和 `fc`。
- `fc.weight` 为40×768，`fc.bias` 为40。
- 没有 `theta_star`、`old_params`、`model_old`、`means`、`checkpoint_params`、旧state dict或snapshot容器。
- 每个对应 `exp_run` 子树仅发现一个 `.pt/.pth/.pkl/.npy/.npz/.safetensors` 资产，即343,380,210字节的 `checkpoints/final_model.pt`。

完整逐文件路径及metadata已保存到 [full_final_metadata.json](full_final_metadata.json)。此处是现存文件的直接证据，不依赖对历史代码版本的假设。final文件与原日志的身份边界仍沿用 [FULL_PROVENANCE.md](FULL_PROVENANCE.md)。

## 2. 保存代码为何没有保留旧参考

当前 [src/trainer.py:2047](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:2047) 在结束时只保存 `tasks` 和当前网络 `state_dict()`，与六份实际文件结构吻合。此前eval之前的保存虽然附带method/dataset/seed等metadata，[1216行](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:1216) 仍只写当前network，并不包含learner中的 `_checkpoint`；结尾再次写同一final路径。

staged文件是另一种格式：[EWCon_Probe.save_task_checkpoint:625](/data/140-1/users/liying/CopyFlatCV/models_CL/EWCon_Probe.py:625) 会写 `network_state`、`fisher`、`checkpoint_params`。但不能因此推断Task2保存中含Task1参考：

1. [after_task:424](/data/140-1/users/liying/CopyFlatCV/models_CL/EWCon_Probe.py:424) 用旧 `_checkpoint` 计算drift。
2. [after_task:551](/data/140-1/users/liying/CopyFlatCV/models_CL/EWCon_Probe.py:551) 随后把 `_checkpoint` 全部更新为当前任务结束参数。
3. [trainer:1224](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:1224) 先调用after_task，到 [1266行](/data/140-1/users/liying/CopyFlatCV/src/trainer.py:1266) 才调用可选的save_task_checkpoint。

所以在当前调用逻辑中，即使找到Task2的 `task_01.pt`，其中 `checkpoint_params` 通常也是Task2 endpoint，而非旧Task1 backbone。正确的 `task_00.pt` 才足够恢复旧参考；之前已确认同路径现存Task0文件被后续运行覆盖。本次不将当前保存代码当作已绑定到每次旧运行SHA的证据，六份final的“没有旧快照”结论由实际文件metadata独立成立。

## 3. 原记录没有现成的分层分解

对六份metrics JSON递归检查，并搜索六份原日志中的 `delta_norm_backbone`、`backbone_delta/drift`、`head_delta/drift`、`delta_norm_fc`、layer/drift组合、`theta_star`、`old_params`、`model_old`、snapshot等模式；未发现这些记录。

Task2现存指标只有全参数 `delta_norm`、`param_delta_norm`、相应Fisher加权量和top-1%整体比例，以及这些量的训练均值。`param_delta_norm` 与 `delta_norm` 是同一个量的别名，并非backbone分项。代码 [1134行](/data/140-1/users/liying/CopyFlatCV/models_CL/EWCon_Probe.py:1134) 将全部命名参数的delta收成列表交给总量helper；[1180行](/data/140-1/users/liying/CopyFlatCV/models_CL/EWCon_Probe.py:1180) 只是复制总量到别名。top-1%比例不编码按层分配，训练均值也不能确定最终分层运动。

逐run相关字段、匹配结果、二进制资产列表保存于 [full_history_field_scan.json](full_history_field_scan.json)。结论范围限于已匹配六个run及同目录资产，未声称整个磁盘绝无备份。

## 4. 可保留的条件分解与仍缺的信息

若现存final确实对应原日志、旧分类头在Task2全程保持不变、历史头按零扩展，则

\[
E_{backbone}=E_{logged,total}-\|W^{final}_{head,new}\|_F^2-\|b^{final}_{head,new}\|_2^2.
\]

已有CPU结果满足这一路径的条件证据；本次没有重算或提高其身份确信等级。这给出一个标量平方范数，**不能反推出原backbone、K/V分项、位移方向、路径积分或旧任务loss**。需要正确Task1备份，或原先已保存的完整参数差向量/分层分解，才能完成这些精确复算；本次未在目标六run发现后者。
