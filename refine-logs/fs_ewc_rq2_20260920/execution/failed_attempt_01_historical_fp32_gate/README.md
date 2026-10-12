原sanity按FP32直接复现历史loss，在两方法Task0处失败。原始失败日志与raw r0完整保留。
后续受控patch-embedding TF32精度探针复现四个历史端点(loss误差约1e-6且准确率相同)，见../investigation/tf32_conv_emulation_witness.json。
正式新诊断继续统一FP32；历史TF32兼容检查单列，全部当前端点重新评估。该目录不计正式完成结果。
