# ImageNet-A 全200类 LoRA / FT 五方法运行说明

本目录执行2026-10-10的五方法计划。2026-10-11用户授权使用物理GPU4、5；最多两项并行，只在无其他compute进程、显存<500MiB、利用率<5%时按index和UUID共享锁接入。其他GPU及已有队列不修改。

新增7组seed2正式训练：LoRA镜像/完整GAM，FT SGD/SAM/镜像/g2/完整GAM。LoRA SGD/SAM/g2直接复用已验收结果。10种模型/算法单元分别进行1epoch/2batch短测，全部及配对门通过后自动启动正式训练。

正式作业优先运行FT SGD与g2的关键对照，再补齐其余五项；不根据中途测试准确率改变配置。

## 环境与只读演练

软件环境复用 `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`，指纹`4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8`，不安装新包。保留190份只读source与原LoRA121份执行pins。每个实际训练worker必须完成独立seed2 CUDA前向/反向witness。

在实际主机命名空间运行以下命令（dry-run不会分配GPU或创建训练输出）：

```bash
CUDA_VISIBLE_DEVICES='' /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/suite.json --dry-run
```

单项配置演练示例：

```bash
CUDA_VISIBLE_DEVICES='' /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/trainer/train_single200.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/formal_full_ft_g2_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/outputs/formal_full_ft_g2_s2 --dry-run
```

真实队列启动由当前root代理在验收后进行，用户或后续代理读取`runtime/controller.json`和`runtime/queue_state.json`核对PID starttime后再决定是否恢复，不重复启动相同manifest。

## 算法与预算

相同TRAIN batch，普通autograd(create_graph=false)，每次虚拟读取恢复全部状态与R0；真实SGD在原参数点进行一次，仅推进基点g0的R1。

N(v)=v/(globalnorm(v)+1e-12)，rho=.05、r=1。沿用已完成LoRA切片的FP64全局范数累积与FP32 `add_(g,alpha=scale)` 构造探针，保持同协议可复用性。GAM内在投影按原gam.py的native dtype norm/cosine/epsilon计算；这是原版GAM组合在统一strict探测协议中的实现，不声明与历史减法恢复执行器逐位同轨迹。

- SGD：U=g0，1梯度。
- SAM：g1+=g(theta+.05N(g0))，U=g1+，2梯度。
- 镜像：g1-=g(theta-.05N(g0))，U=g0+(g0-g1-)，2梯度。
- g2：U=g(theta+1*N(g1+-g0))，3梯度；外点从原参数theta出发。
- GAM：另读g3=g(theta2+.05N(g2))，h-=g0+g2，h+=g1++g3，U=h+-.1*(h-的h+垂直分量)，4梯度；beta全1，无SGV。

50epochs、batch128、2350更新；FT全部骨干+FC200可训练、无LoRA；LoRA冻结骨干，仅Q/V rank10 A/B+同FC200。LoRA LR .04->.004，FT .001->.0001，均零动量/权重衰减的SGD+cosine。相同全5981train/1519test，固定200类。关闭EWC/Fisher/权重融合/历史缓存/replay/HVP/最终匹配/裁剪；只有完整GAM执行其定义内的gamma投影。

新增正式16450 SGD更新/42300训练梯度；短测20 SGD更新/48训练梯度；被动postloss与测试前向另计，不进入训练梯度预算。状态恢复、相同输入/head/初始化及LoRA三行同协议前两步必须通过门控。

## 日志、结果与停止条件

- `runtime/queue_state.json`：18项作业状态，10短测+1CPU配对+7正式。
- `runtime/jobs/<id>/`：真实PID身份、GPU UUID、exit/完成票据与日志。
- `outputs/<id>/`：完整训练曲线、逐步trace、最终checkpoint、CUDAwitness与验收summary。outputs指向专用cache/tmp目录，避免项目磁盘不足。
- `RESULTS.csv`、`RESULTS.json`与`RESULTS.md`：仅已验收固定epoch50结果，复用来源标清，报告sample Top1、test CE、原十个20类分组macro及梯度预算。单任务BWT/forgetting为null。总耗时另见`outputs/<id>/final_summary.json`的elapsed_seconds；每epoch耗时和峰值显存见epoch_test_curve.json，显存记录包含被动诊断/测试期间分配，不能称为隔离测量的纯训练峰值。
- `runtime/PAUSE`存在时只暂停新作业接入；不停止正在训练的作业。失败停止后续接入，让已运行作业完成，不静默改batch/半径/学习率后重试。

检验g2的主要对照是同范围SGD与SAM；每个范围内学习率统一，不以跨LoRA/FT绝对差值推断因果。单seed自然步长结果仍包含更新幅度差异，不作显著性或Hessian机制证明。
