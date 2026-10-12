# ImageNet-A末端中心差分及联合后向替换结果

seed2/rank10/rho=.05/r=1；当前学习率、自然SGD及严格CUDA确定性协议保持固定。新增CPLUS/CMINUS每步5训练CE，BF/BB每步4训练CE，均无额外梯度诊断。BB不读取任何局部上升侧梯度。关闭Fisher、实际权重融合、投影、HVP及最终步长匹配。

FF/FB为已验收v2成绩的显式复用，保留原始来源；不是本轮新增训练。新旧FF/FB同状态点、梯度及实际更新回归检查已通过。旧v2每任务有一次隔离反侧诊断，新v3没有；完整状态/RNG恢复防止该诊断进入训练更新。

| ID | 最终方向 | 训练CE/步 | 状态 | AAA | FAA | 来源 |
|---|---|---:|---|---:|---:|---|
| FF | g0+g3Fplus-g2F | 4 | reused_reference | 65.176 | 47.584 | 已验收旧基线复用 |
| FB | g0+g2F-g3Fminus | 4 | reused_reference | 65.385 | 48.632 | 已验收旧基线复用 |
| CPLUS | g0+(g3Fplus-g3Fminus)/2 | 5 | completed | 65.197 | 48.170 | 新独立训练 |
| CMINUS | g0-(g3Fplus-g3Fminus)/2 | 5 | completed | 2.064 | 0.651 | 新独立训练 |
| BF | g0+g3Bplus-g2B | 4 | completed | 64.608 | 47.339 | 新独立训练 |
| BB | g0+g2B-g3Bminus | 4 | completed | 65.468 | 48.935 | 新独立训练 |

AAA/FAA只填写完整验收终点，部分任务观察独立写入CSV的progress字段；没有将短测当作正式结果。所有比较为单seed，样本标准差不可估计；自然步长未控制更新幅度，中心组与单侧组的梯度调用预算也不同。

中心差分检验删除偶对称项后的符号作用；BF/BB与FF/FB形成第一定位估计×第二残差估计的联合对照。BB成功仅支持当前分量版本的两次局部侧别联合替换，不证明完整GAM等价或计算加速。
