# Data and checkpoint audit, 2026-09-18

结论：可以将新 CL-LoRA 填入“archived configurations”结果汇总，但现有证据不足以称为逐样本、逐权重完全一致的严格对照。历史日志支持相同 backbone **模型 ID**、类顺序与任务规模；只有新运行固定了 checkpoint SHA 和样本路径/标签清单。没有证据可以将 Flowers 97.82 相对于旧 53.05/60.06/86.75 的差距归因于数据修复。

## 已确认及证据边界

| 项目 | 核查结果 | 对可比性的含义 |
|---|---|---|
| 预训练模型名称 | protocol_alignment 源清单内的 34 个相关历史日志均明确加载 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`；新 CL-LoRA 同 ID | ID 一致 |
| 预训练权重字节 | 34 个历史日志均未记录 SHA256 或新运行固定的 revision；新运行固定 revision `063c6c38a5d8510b2e57df480445e94b231dad2c`，SHA256 `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`，本次重新计算匹配 | 历史与新运行的字节同一性未证实；不能把同 ID 写成已证明同 SHA |
| 类顺序 | 历史日志检出的 54 个 DataManager 顺序记录均为连续自然顺序；新配置 `class_shuffle=false` | 类顺序的日志证据匹配 |
| 新数据分割 | 下表所列路径/标签清单及 SHA 与 preflight 匹配；本次逐项检查 train/test 的规范化真实路径交集均为 0 | 新运行的路径及标签已冻结；未做逐图像内容 hash 审计 |
| 历史样本同一性 | `data_path_check.json` 明确 `historical_path_identity_proven=false`，旧默认根目录 `/data/140-0/datasets` 不存在；没有历史逐图像清单 | 不能由现有迁移目录证明历史 train/test 成员完全相同 |
| 变换实现 | 当前 Flatness_CV 与新冻结源码中 `_build_cub_transforms`、`_build_aircraft_transforms`、`_build_cars196_transforms` 的 AST 完全相同；Pets/Aircraft loader AST 亦相同 | 当前源码层面匹配；现存旧代码不是运行时不可变快照，历史实际实现仍有证据缺口 |
| Flowers loader | 当前旧代码使用 `sorted(os.listdir(jpg_dir))[id-1]`；新代码使用 `image_{id:05d}.jpg` | 实现有差异，但当前目录上输出无差异，不能断言旧实验标签错配 |

`legacy_alignment.json` 的 `checkpoint_ids_match=true` 只表示模型 ID 匹配。该文件明确限定比较范围为训练骨架，排除未记录的历史 RNG/source 状态。`legacy_logged_config_records.json` 不是完整数据来源记录，不能用缺省值补成历史事实。

## 新运行实际数据和变换

| 数据集 | Train | Test | 任务类数 |
|---|---:|---:|---|
| CUB-200 | 9430 | 2358 | 10 个任务，每任务 20 类 |
| Cars196 | 8144 | 8041 | 16 + 9×20 |
| Aircraft | 6667 | 3333 | 10×10 |
| Flowers102 | 2040 | 6149 | 12 + 9×10 |
| Pets | 3680 | 3669 | 5 + 8×4 |
| 5Datasets | 29961 | 23550 | CUB 200 → Aircraft 100 → Cars 196 → Flowers 102 → Pets 37 |

CUB 直接读取迁移目录的 `cub200/train` 和 `cub200/test`，不是运行时生成新随机分割；必须披露实际 9430/2358 分割，不可默认为某一公开标准 split。旧根目录不存在，无法核验旧 CUB 成员与本次逐图像相同。

CUB/Flowers/Pets：train 为 resize 到 256×256（bicubic）→ random crop 224 → horizontal flip；test 为 resize 256×256 → center crop 224；ImageNet mean/std。Aircraft 几何变换相同，但走 torchvision tensor loader 和 `ConvertImageDtype`。Cars 单数据集为按短边 resize 224 → center crop 224，train 增加 flip，mean=0/std=1。5Datasets 统一使用 CUB 变换，故其中 Cars 的变换与单数据集 Cars 不同；不能把单数据集与 5Datasets 指标混当同一任务。

## Flowers 的专项排查

在本次只读检查时，`/data/115-2/users/liying/datasets/flowers-102/jpg` 恰好有 8189 个文件，全部为 `image_00001.jpg` 至 `image_08189.jpg`；没有 thumbnail 或额外文件，没有缺失规范文件。对新清单的 2040 个 train 样本与 6149 个 test 样本逐项应用旧位置索引算法，**0 个路径差异**。

因此：新代码确实更健壮，但不能据此认定历史 Flowers 分数使用错误标签。历史目录当时是否有辅助文件、当时源码是否和现存 Flatness_CV 文件相同，均未知。Flowers 的巨大差距需要结合 CL-LoRA 原型分类器/多分支机制与同数据清单重新运行的控制组解释，当前不能给出因果归因。

## 可定位证据

- 冻结源码：[data.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/utils/data.py:29)，SHA256 `e162c8d2d2f50c14b4481e3b153d489a53ff7647ca96fa668e43c48449b0e3f9`；与当前 CopyFlatCV 的该文件相同。
- 现存旧源码：[Flatness_CV data.py](/data/140-1/users/liying/Flatness_CV/utils/data.py:29)，SHA256 `2c02ac940c77ae898ac18fe9d4e5b797fdfa7d75c22a638c92256382bc6f185a`。这是当前可读文件，不能据其 hash 证明三月/四月运行源码。
- [preflight.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/preflight.json)：六个 manifest 路径与 SHA；零 adapter 对齐特征误差约 5.25e-6。该检查证明本次 port 与同权重 timm 的特征接近，不证明其与历史运行逐权重相同。
- [CL-LoRA checkpoint loader](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_aligned.py:54)：严格检查 SHA，仅允许新增 LoRA/block-weights 缺失键。
- [data_path_check.json](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/protocol_alignment_20260918/data_path_check.json)、[code_comparison.json](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/protocol_alignment_20260918/code_comparison.json)、[历史 source manifest](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/protocol_alignment_20260918/source_manifest.json)。
- 历史 Flowers 示例：[Vanilla 日志](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/seqlora/sgd/flowers/0/seqlora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2/exp_run/10/seqlora_inr_sgd_flowers_ep40_lr00025_t20_rank16_eval2_vit_base_patch16_224.log:79)；[GAM 日志](/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/flowers/0/paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0/exp_run/10/paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0_vit_base_patch16_224.log:60)。均只记录模型 ID 和 main 下载请求，未固定权重 hash。

本任务只读审计实验和数据，未训练、未联网、未修改原实验。
