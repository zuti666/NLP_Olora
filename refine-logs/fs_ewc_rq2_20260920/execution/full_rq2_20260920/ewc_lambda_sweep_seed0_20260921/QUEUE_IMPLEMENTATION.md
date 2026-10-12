# 全卡空闲后一次性准入

本目录 `campaign_runner.py` 从 `../seven_row_extension_20260921/campaign_runner.py` 复制；旧扫描代码、freeze 和状态文件保持原样。本实现没有启动控制器或真实 GPU 作业。它适用于搜索前的 Fisher preparation 和随后的独立 λ trials；具体科学参数由计划及 trial wrapper 决定。

## 计划配置

```json
{
  "all_gpu_start_gate": {
    "required": true,
    "gpus": [0, 1, 2, 3, 4, 5, 6],
    "poll_seconds": 900
  },
  "start_after": {
    "gpu_uuids": {
      "0": "<physical GPU 0 UUID>",
      "1": "<physical GPU 1 UUID>",
      "2": "<physical GPU 2 UUID>",
      "3": "<physical GPU 3 UUID>",
      "4": "<physical GPU 4 UUID>",
      "5": "<physical GPU 5 UUID>",
      "6": "<physical GPU 6 UUID>"
    }
  }
}
```

UUID 也可放在 `all_gpu_start_gate.gpu_uuids`；两处同时提供时不得冲突。必需 gate 强制包含 GPU 0–6、七个唯一预期 UUID、900 秒轮询间隔。未配置 gate 的计划继续使用旧逻辑，便于已有队列测试和旧计划兼容。作业可使用哪些卡仍由 `gpus` / `gpu_pool` 限定。

CLI 保持 `--plan PLAN`（dry-run）和 `--plan PLAN --execute [--resume]`。dry-run 只校验计划与代码哈希，不查询 GPU，不创建状态、锁或准入收据。

## 准入与持久状态

在调度任何作业前，runner 先检查 gate。等待期间 `pipeline_state.json.status` 为 `waiting_for_all_gpus`。`pipeline_state.json.all_gpu_start_gate` 保存：

- 全部七张卡的 `gpu_statuses`、预期 UUID、各卡 `blocked_reasons` 和共享锁状态；查询失败也保留该卡和错误。
- `blocked_gpus`、`checked_utc`、`next_poll_utc`、`poll_seconds=900`。
- 初始快照 `initial_gpu_statuses`、获得全部锁的时间 `all_locks_acquired_utc`。

`STATUS.md` 同步显示等待状态、每张卡的指标、阻塞原因和下次检查时间。等待中的控制器重启后沿用持久化的 `next_poll_utc`，不会立即加密轮询。

每轮先观察七张卡。七张卡都必须满足：显存使用 **< 500 MiB**、GPU 利用率 **< 5%**、**没有 compute 进程**、物理卡 UUID 与计划一致。任何一项失败，本轮不会保留 GPU 锁，900 秒后重查。

首次快照全部合格后，runner 尝试以非阻塞方式获取共享 `gpu_lock_dir/<UUID>.lock` 的七个排他锁。一张卡的锁失败也不会准入；该轮已取得的其他锁全部释放。七个锁同时取得后，在锁仍全部持有时再次检查七张卡。如果二次检查仍通过，先原子写入 `run_dir/ADMISSION.json`，再保存状态中的收据路径、SHA-256、`status=admitted` 和准入时间。收据包含配置、代码/计划 fingerprint、全部快照及七个锁同时取得的证据。

只有这些持久记录写入成功才返回正常调度。准入临时锁随后全部释放，未运行作业的卡不会被扫描霸占。一次性准入成功后，GPU 再次有负载不会让整个扫描重新等待七张卡；每个作业仍须自行通过空闲检查和锁检查。

## 作业启动与竞态边界

正常调度继续采用原来的两层检查：调度线程观察候选卡并取得共享 UUID 锁，工作线程持有该锁再检查 UUID、显存、利用率和 compute 进程；通过后才执行 CUDA preflight / job。锁描述符传给子进程，控制器退出后仍存活的作业继续持锁，避免不安全的重复恢复。运行阶段沿用 900 秒等待/heartbeat。

全卡准入证明的是：一次准入期间，七张卡的指标都合格，且七个协作锁被同时持有。`nvidia-smi` 的多次读取不是硬件原子快照，其他不遵守这些共享锁的进程仍可能在任意检查后启动。释放临时锁与具体作业取得锁之间也存在间隔；逐作业的锁及空闲复查会处理这一间隔内可见的占用。这套机制不会终止、抢占或暂停他人的进程，也不能对非协作进程提供永久独占保证。

## 恢复与 trial 终态

已准入的 `--resume` 必须验证 ADMISSION 收据哈希、计划/代码 fingerprint、run 目录、gate 配置和七张卡合格证据。收据缺失或改变会拒绝恢复；不会静默重新生成收据绕过完整性检查。未准入时如在写收据和保存状态之间崩溃，尚未调度任何作业，恢复会重新执行准入。旧的完成作业收据和进程存活检查保持不变。

Scheduler 继续要求进程退出码为 0 且 `receipt_require` 全部匹配。Sweep trial 可将 `receipt_require` 配置为 `{"status":"observed"}`：wrapper 将原生训练成功或显式科学发散的结果包装为 `TRIAL_RESULT.json`，其中 `outcome=complete|diverged`。此时 scheduler 的 `complete` 表示作业执行完毕且结果已经观察到，**不表示训练收敛或成功**；科学结论必须读取 `outcome`。scheduler 不放宽原生 COMPLETE 的语义，也不吞掉异常、OOM、非零退出码或无效收据。这些情况仍会触发旧有停止后续调度并等待已启动作业结束的失败规则。

## 已运行验证

`python test_queue_gate.py`：7 项 fake-GPU 测试通过，覆盖四类阈值/身份阻塞和查询错误、共享锁占用、全锁内复查遇到新负载、准入收据先于作业、观察到发散的正常 receipt、完成后恢复和收据篡改、等待期限恢复、dry-run 无 GPU 查询、配置约束。

另将 `../test_queue.py` 的 4 项既有回归测试加载到本目录 runner 后运行：全部通过，包括无 gate 时的动态选卡、共享锁继承、依赖释放、损坏收据和 freeze 拒绝恢复、失败阻止后续作业。测试只运行 Python 写 JSON 的短进程，所有 GPU 状态均来自 mock。
