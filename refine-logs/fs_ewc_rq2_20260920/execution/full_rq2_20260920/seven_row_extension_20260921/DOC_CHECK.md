# 15分钟GPU轮询修改核验

PASS。独立核对当前runner与旧冻结runner，差异仅新增 `GPU_POLL_SECONDS = 15 * 60` 和三处资源等待使用该常量；不存在原30秒周期资源扫描路径。FIRST_COMPLETED仍可响应任务完成事件。

当前plan的全部stages（包含jobs、训练参数、命令与依赖）与归档旧plan完全一致；新runner已纳入code_files且出现在dry-run指纹中。

按RUN.md所列只读命令执行一次，退出码 0；dry_run=true，10 stages、10 jobs、max_parallel=6。未启动训练或GPU作业。

完整命令、七项核验和 219 项指纹记录于 [DOC_CHECK.json](DOC_CHECK.json)。
