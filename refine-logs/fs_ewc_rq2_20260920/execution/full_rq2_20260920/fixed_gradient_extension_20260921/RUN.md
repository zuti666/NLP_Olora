# 固定Task0正梯度上升方向：执行说明

最终方向为 **正梯度**：`u = +grad(mean Task0 CE) / norm(grad)`，从各自Task0锚点按`theta(r) = theta0 + r*u`扰动。目的是使足够小r下旧任务损失增加。方向只计算一次，所有r共用；不是迭代更新，不使用负梯度。

## 配置

沿用正式三seed模型、Task0数据与有效Q/V坐标，6组（3 seeds × EWCLoRA/FS-EWCLoRA）。每组用全部1259训练样本计算Task0 CE梯度；每个半径评估1259训练/307测试样本的损失及准确率，并在固定256训练样本上测raw empirical Fisher trace。梯度不含EWC/GAM项，Fisher不乘lambda、无额外归一化。

93个半径覆盖原有0–20步长0.25、25/30/35/40/45/50，以及0.001/0.005/0.01/0.025/0.05/0.1。正式扫描前必须验证实际方向与梯度内积为正、小步长CE增加、有限差分与解析方向导数一致。错误符号禁止通过完成门槛。

## 执行顺序与GPU

`wait_radius50 → fixed_gradient → publish`。先等待正在运行的随机方向`radius50_extension_20260921`全部12组测量及CPU发布完成并校验输出哈希，再在物理GPU1/4/6上最多并行3组。共享GPU UUID锁、显存/利用率/compute进程检查和重复启动保护沿用已验证队列。等待过程不占用GPU，不改动前驱代码或结果。

模型/数据处理与环境复用`Pilot_new@e94a148f`；无需重建或安装依赖。每组获得GPU锁后先做FP32、TF32关闭的CUDA前向/反向检查，再做真实梯度方向验证。

## 只读验证

在任意目录运行：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/fixed_gradient_extension_20260921/queue_fixed_gradient.py
```

预期：`status=passed`、`dry_run=true`、6个GPU作业+2个CPU作业、最大并发3、`gradient_task=0`、`direction_sign=1`、93点/组。

## 状态与输出

队列通过宿主机上的`launch.py`独立后台启动；已有启动凭据时拒绝重复运行。状态见[campaign/STATUS.md](campaign/STATUS.md)，前驱等待详情见`campaign/predecessor_wait.json`。`queued`表示等待，不能视为已经测量。

全部6组通过后自动输出三子图：有符号训练损失变化、训练准确率、raw Fisher。另存0–2局部图，均保留Times New Roman和原方法颜色/线型。原始结果包含完整CE、有符号变化和正损伤。最终完成记录为`report/fixed_gradient_complete.json`。

单组93点约需7–10分钟，三卡两批约15–25分钟，另加前驱等待时间。大r下曲线不保证单调；所有方向和结果按实测保存。
