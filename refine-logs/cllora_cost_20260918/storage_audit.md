# CL-LoRA 实际状态与存储核算

只读核验 frozen attempt2 源码及正式配置，并使用 Pilot_new 的 PyTorch 在 meta device 实例化精确计数。禁用了CUDA可见性，未读取训练数据、未训练、未修改实验。FP32 payload 按4 bytes/参数，MiB=2^20 bytes，不包含序列化元数据、激活、梯度或框架缓存。

## 当前实验是否真的保存了模型

六项正式 CL-LoRA 均 `save_task_checkpoints=false`，没有开启 `save_final_model`（trainer 默认 false）。实际 `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/outputs/logs_inc_lora/cllora_aligned` 只有6个log和6个metrics JSON，总660429 bytes，无 `.pt/.pth/.pkl/.safetensors` 等训练后权重。配置、receipt、源码快照另存于实验根目录，不是训练后模型。

所以本轮保留了结果和复现所需的配置/源码证据，**没有可直接加载的已训练 CL-LoRA checkpoint**。下面“state_dict大小”都是依据实际架构核算的理论权重payload，不能称为已保存文件大小。输入的预训练safetensors是既有基础权重，不是本轮训练结果。同一outputs里的GAM-EWC产生了若干pt文件，不能误算成CL-LoRA的checkpoint。

`save_old_shared_lora()` 的“save”只表示把当前shared参数复制到内存中的old分支，没有任何磁盘写入。

## 每部分保存什么

实际配置 d=768，r=16，Q/V，shared blocks 0–5，specific blocks 6–11。单个六层Q/V adapter的A+B参数数为6×2×2×768×16=294912（1.125 MiB）。

- 当前shared A+B：294912；shared A冻结，仅B可训练。
- 旧shared A+B：另294912，供KD teacher使用，是一份滚动快照而非每任务一个teacher。
- specific A+B：每任务294912；历史specific冻结。全部 `nb_tasks` 组在模型初始化时通过ModuleList一次性预分配，包括未来未训练任务，初始就占存储。
- block_weights：每任务6×3=18 scalar；所有任务也一次预分配。虽然只启用Q/V，K权重列仍被注册并参与完整向量的正交项；纯推理中K列不使用。
- proxy：当前任务类别数 c_t ×768，加1个cosine sigma。每任务重新创建、替换旧proxy；不是保存所有历史proxy头。
- fc：累计类别 C × (768×已见任务数 T)，加1个cosine sigma，整个fc被冻结，值由当前训练图片原型均值写入。新类会在所有已见branch上计算原型，填入对应列块；旧类新列保持0。
- 实际 `interface()` 推理只读取task i类别行与task i特征列的对角块，总共C×768原型。其余dense fc元素被保存/部分计算，但本接口不用于预测。

因此当前存储复杂度是specific的O(Tdr)加dense原型O(CTd)；每任务固定新增类数时C与T同增，dense原型为O(T²)。若实现紧凑推理导出，可把原型降为O(Cd)。此优化本次没有实现。

## CUB与Het5精确计数

|部分|CUB：T=10,C=200|Het5：T=5,C=635|
|---|---:|---:|
|冻结基础ViT，无分类头|85,798,656|85,798,656|
|当前shared A+B|294,912|294,912|
|旧shared A+B|294,912|294,912|
|全部specific|2,949,120|1,474,560|
|全部block weights|180|90|
|dense fc weight|1,536,000（200×7680）|2,438,400（635×3840）|
|fc sigma|1|1|
|最终proxy weight|15,360（20×768）|28,416（37×768）|
|proxy sigma|1|1|
|基础模型以外合计|5,090,486|4,531,292|
|完整state_dict参数合计|90,889,142|90,329,948|
|完整FP32 state权重payload|346.7146 MiB|344.5814 MiB|
|其中基础模型以外payload|19.4187 MiB|17.2855 MiB|

冻结基础ViT本身为327.2959 MiB。CUB初始化（还没有head）就有89,337,780个参数；Het5初始化就有87,863,130个。未来specific已经预分配，所以初始存储并非只含task0。

meta实测state_dict：CUB452个tensor key，363556568 bytes；Het5 327个key，361319792 bytes；dtype均float32。数量包含完整冻结基础权重。如果只保存adapter与头且引用锁定的基础模型SHA，则不需要每个任务重复保存基础ViT。

## 每阶段实际可训练参数

冻结规则只开放shared B、当前specific A+B、当前block weights及当前proxy：

P_train = 147456 + 294912 + 18 + (768 c_t + 1) = 442387 + 768 c_t。

CUB每任务457747。Het5依次595987、519187、592915、520723、470803。meta统计与正式训练log打印值逐项一致。整个dense原型头不通过梯度训练。SGD momentum=0不需要momentum状态，但计算图激活和梯度的显存不能由权重payload推断。

## 可精简的纯推理保存（尚未实现）

纯推理不调用KD或proxy，只需一份当前shared、全部已见specific、block weights、C×768对角原型与fc sigma，加基础ViT及任务边界。

|理论紧凑推理内容|CUB|Het5|
|---|---:|---:|
|dense fc原型现存payload|5.8594 MiB|9.3018 MiB|
|仅对角原型payload|0.5859 MiB|1.8604 MiB|
|紧凑额外参数数（保留18T个block weights）|3,397,813|2,257,243|
|紧凑额外FP32 payload|12.9616 MiB|8.6107 MiB|
|加基础ViT的总payload|340.2575 MiB|335.9066 MiB|

这是可行的部署结构推导，不是已有导出文件或已验证的恢复流程。训练继续需要old shared等状态，不能把纯推理裁剪结果直接当训练恢复包。

## state_dict不能单独恢复任务边界

`task_sizes`是普通Python list、`_cur_task`是普通整数，没有register_buffer或get_extra_state；meta得到的state_dict不包含这些key。新网络初始fc=None、没有proxy，直接load最终state_dict也不会自动创建头或task_sizes。

应额外保存显式task_sizes（CUB=[20]×10，Het5=[200,100,196,102,37]）、当前任务索引、已知/累计类数、实际class order、配置和预训练SHA；恢复时先用任务边界重建fc/proxy形状，再加载权重。仅C=635、T=5不能推断635类如何分到五任务。完整训练恢复还涉及epoch、scheduler与RNG等状态，裸state_dict不包含这些。当前generic final checkpoint只包tasks/model_state_dict，未补CL-LoRA专用恢复元数据，而且本轮未开启保存。

## 证据

- [CLLoRANet初始化/普通task_sizes](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_aligned.py:45)
- [proxy与dense fc扩展](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_aligned.py:78)
- [对角原型推理](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_aligned.py:107)
- [所有已见branch的原型计算](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_aligned.py:119)
- [shared/old与specific预分配](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_archived_vit.py:253)
- [block weights预分配](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_archived_vit.py:449)
- [可训练参数冻结规则](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/models_LoRAbasedCL/cllora_aligned.py:25)
- [条件最终保存](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/src/trainer.py:1204)
- [CUB正式配置](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/configs/cllora_cub200_s0.json)
- [Het5正式配置](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/configs/cllora_het5_s0.json)
