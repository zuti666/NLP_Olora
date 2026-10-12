# r=1 Stage1 环境复用

执行host为 cudahpc40.cvc.uab.es，真实宿主PID namespace为 pid:[4026531836]。复用绝对Python `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`，未安装/升级依赖。沿用已通过文档调用与CUDA witness的 r=.25 环境spec：environment_spec.json canonical SHA256=4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8；原provider记录与说明位于前轮 ENVIRONMENT.md / gates/DOCUMENTED_INVOCATION_WITNESS.json。软件栈未改变；新suite仅将TMP/TEMP/TMPDIR改为本轮专属/tmp目录。

版本：Python3.11.15，Torch2.4.1+cu121/CUDA12.1/cuDNN90100，Torchvision runtime0.19.1+cu121（distribution0.19.1）、timm1.0.9。模型pretrained加载使用现有HF离线缓存 augreg2_in21k_ft_in1k。source_root沿用 `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source`，source_manifest SHA e8a827ae67768a73a59106914477905d051da0646fcb0ffb7c3a13b23d337ab2，190份源码/配置逐文件验证。

suite.environment保留PYTHONHASHSEED0、CUBLAS_WORKSPACE_CONFIG=:4096:8、OMP/MKL/OPENBLAS threads8、HF/TORCH_HOME旧缓存、HF离线标志、数据根 `/data/140-1/datasets`。仅允许物理GPU2/3，UUID绑定后逻辑CUDA0；FP32，AMP/TF32关闭。真实模型原生attention，无二阶反传要求。

新文档验证由独立agent执行 RUN.md 中精确 documented_check 调用；它先dry-run验证冻结配置/代码/旧baseline证据，再在共享锁下执行seed2小CUDA forward/backward sentinel，optimizer更新0。执行命令需宿主GPU访问（tools.exec_command require_escalated），以便真实/proc身份与CUDA检查；这是用户授权训练的短验证，不是远程SSH任务。
