# 真实GPU smoke与正式启动前独立审查

结论：12/12真实GPU smoke已完成并通过独立只读审计，可由root释放正式阶段。审查者未启动GPU、未修改deploy文件。审计时间 2026-10-01T18:44:21.382971+00:00，当时formal任务尚未启动。

| 检查 | 结果 |
|---|---|
| 部署文件pins | 59/59匹配 |
| 原生source files | 190/190匹配 |
| completed receipt ID/exit0/退出原记录/各输出hash | 12/12匹配 |
| root GPU gate当前suite与receipt hashes | 匹配 |
| seed2八arms epoch0真实LoRA/FC张量 | exact相同 |
| seed2八arms初始完整model/class/data/augmentation identity | exact相同 |
| Sum/Mean2有效BA+FC差 | 0.000196788052951161 |
| Sum从epoch0到epoch1的实际有效步长 | 0.415903828187773 |
| Sum/Mean2差相对实际更新 | 0.000473157589841455 < 1e-3 |
| Sum/Mean2最大参数绝对差 | 6.23189262114465e-05 |
| formation Gmatch实际delta匹配独立S误差 | 3.08554505131423e-08 |
| future两臂origin/newhead/data/LR/K0 | exact配对 |
| future Gmatch实际delta匹配独立S误差 | 1.01388586344135e-08 |

该检验是实际一个更新的GPU集成与配对检查，不是50epoch数值轨迹等价或formation机制已得到支持的声明。Sum与Mean2存在非零浮点差，按运行前固定的1e-3门槛通过，不能称bitwise一致。实际根匹配使用post-native-unperturb基点，roundtrip漂移另列在每份smoke记录中。

新增脚本静态审查：check_gpu_smokes.py的跨arm比较方法合理；start_background.py正式门单独只校status/count，未重新绑定suite/receipt。当前fresh门缺口由本次独立preformal检查补齐（全部hash已核，root获知后自行决定launch）；不能把launcher本身描述成具有完整stale-gate防护。没有改任何已pin脚本。

证据：GPU_SMOKES_INDEPENDENT_AUDIT.json包含12receipt SHA、每任务actual delta、roundtrip及完成时间；audit_gpu_smokes_readonly.py为本次可重复CPU只读审计脚本；SMOKE_LAUNCH_GATE_REVIEW.json绑定新增两脚本hash。
