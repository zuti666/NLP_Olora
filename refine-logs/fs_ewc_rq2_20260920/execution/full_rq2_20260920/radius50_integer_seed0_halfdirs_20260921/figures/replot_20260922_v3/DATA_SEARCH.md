# 半径21、22、23、24来源核查

独立只读核查：/root/find_r21_24_sources；日期2026-09-22。没有修改源文件，没有启动GPU。

- 遍历 refine-logs/fs_ewc_rq2_20260920 内176个 states.jsonl，共63,476行；全部成功解析，point.radius等于21、22、23、24的行数为0。
- 检查233个CSV的radius/r/radius_value列，没有21–24记录；检查其他JSON、配置、日志及说明，未找到对应测量。
- 随机方向原始dense网格0:0.25:20；粗扩展25/30/35/40/45/50；整数扩展填齐25–50的缺失整数。WORKER_PROTOCOL.md及README明确记录21–24未测。
- fixed_gradient_extension_20260921与task1_gradient_extension_20260921也从20跳到25，且使用不同方向协议；gam_radius1_direction_20260922只覆盖0–5。
- /data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c 与 sgd_gam_s1c_fullmodel 共65文件，6个测量JSON及3个CSV已核对：stage2_r覆盖0–3，stage2_r-2覆盖0:0.25:20。fullmodel目录没有可合入的radius测量JSON/CSV，且历史数据为ImageNet-R。
- mechanism_next_20260920/radius_extension_20260921的补点只到19，且旧锚点checkpoint不同。

结论仅限上述已核对来源：没有可合入当前seed0/50公共随机方向曲线的21–24实测结果。图上20到25之间的连线不是额外的数据点。
