# 环境复用与验证

使用 cudahpc40.cvc.uab.es 的真实 host PID namespace pid:[4026531836]；绝对 Python `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`。复用 `.aris/compute/local.md` 中已有环境，canonical environment_spec.json SHA256 为 4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8，没有安装或升级包。环境 spec 内旧资源范围是来源记录；本次实际 GPU 授权以 suite.json 的 [4,5,6] 为准。

Python3.11.15、PyTorch2.4.1+cu121、Torchvision0.19.1+cu121、timm1.0.9；FP32，AMP/TF32关闭。ImageNet-A现有 train5981/test1519、0…199固定顺序，原离线ViT预训练缓存； source_manifest 190文件 SHA e8a827ae67768a73a59106914477905d051da0646fcb0ffb7c3a13b23d337ab2。native和four-probe分别保留原各自随机数与探针语义。

独立验证者读取运行技能、provider ledger及RUN.md后，只执行其中 documented_check 唯一调用。需要真实 host/GPU访问（exec_command require_escalated），工具自动审核。调用核验全部配置、冻结代码及依赖，然后在GPU4的索引和UUID共享锁下执行固定seed2的32×32 CUDA内核测试；此seed仅用于环境测试，没有任何训练更新。实际训练seed严格来自0/1配置。若GPU4忙则失败并等待空闲，不抢占。每个训练worker还会独立执行CUDA witness。
