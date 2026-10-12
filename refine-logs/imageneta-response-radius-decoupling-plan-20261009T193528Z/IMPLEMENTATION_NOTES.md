# 新变体实现时的补充要求

独立数学审查已通过，原三方法和旧代码未改变。

1. 三次梯度指正常非fallback步；fallback应按实际梯度/forward/SGD调用核算，保留明确原因。
2. FixedInner不能继续用旧h=eta*norm(g0)<=eps判断下降探针是否有效。应分别检查g0方向有效性、独立半径、真实FP32内位移q与bitwise是否可分辨；FixedOuter也要检查独立外位移。耦合分支继续保留原行为。
3. 同theta/minibatch/R0，在FixedBoth里只改变eta应保持两探针位置与最终g2（正常步）一致，只改变真实提交步长；这是实现是否意外仍耦合的针对性检查。
4. 内外实际参数位移、rho/R/theoretical与FP32actual均写日志。固定几何不代表有历史时的融合强度也解耦；本轮新对照没有历史。
5. 正式实现需要独立branch/module和明确configschema；原legacy分支、旧序列和cached source定义保持原样，验证同state前若干步的legacy等价。禁止在原radius_mode仍为sgd_step时静默忽略独立字段。
6. 先实现/CPU检查，再在用户指定空闲GPU5/6/7做短测，最后按完整50epochs×10tasks训练。当前只完成计划，任何新步骤均尚未执行。
