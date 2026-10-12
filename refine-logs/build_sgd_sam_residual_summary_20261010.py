"""Derive a comparison report from read-only audits; do not mutate training."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path('/data/140-1/users/liying/O-LoRA')
OUT = ROOT/'refine-logs/imageneta-sgd-sam-residual-comparison-20261010'
OUT.mkdir(exist_ok=True)
raw = Path('/tmp/sgd_sam_latest_results_audit_1010.json').read_text()
d = json.loads(raw)
(OUT/'SOURCE_AUDIT.json').write_text(raw)
a = {r['condition']: r for r in d['multi_seed_aggregates']}
labels = {
 'N05': ('SGD', r'g_0'), 'N03': ('SAM', r'g_1'),
 'N08': ('SGD + cross-location residual', r'g_0+(g_2-g_1)'),
 'N07': ('SGD + second local residual', r'g_0+(g_3-g_2)'),
 'N06': ('Outer gradient', r'g_2=g_1+(g_2-g_1)=g_0+(g_2-g_0)'),
 'P01': ('SGD', r'g_0'), 'P02': ('SAM', r'g_1'),
 'P07': ('SGD + cross-location residual', r'g_0+(g_2-g_1)'),
 'P04': ('Outer gradient', r'g_2=g_1+(g_2-g_1)=g_0+(g_2-g_0)'),
 'P12': ('SGD + cross-location residual, matched step', r'g_0+(g_2-g_1)'),
 'P16': ('SGD, matched step', r'g_0'),
}
table_rows = []
for c,(method,formula) in labels.items():
 r=a[c]
 for metric in ('AAA','FAA','BWT'):
  vals=[float(r[f'{metric}_seed{s}']) for s in (0,1,2)]
  assert abs(statistics.mean(vals)-float(r[f'{metric}_mean']))<1e-10
  assert abs(statistics.stdev(vals)-float(r[f'{metric}_std']))<1e-10
 table_rows.append({'condition':c,'protocol':r['family'],'method':method,'gradient':formula,
   'n':3,'seeds':'0;1;2',**{k:r[k] for k in ('AAA_mean','AAA_std','FAA_mean','FAA_std','BWT_mean','BWT_std')}})
for r in d['restored_extended_aggregates']:
 if r['formula']!='g1+(g3-g2)':continue
 table_rows.append({'condition':'D32','protocol':'positive_four_probe_exact_restore_rng_replay',
  'method':'SAM + second local residual','gradient':r'g_1+(g_3-g_2)','n':3,'seeds':'0;1;2',
  **{f'{m}_mean':r['metrics'][m]['mean'] for m in ('AAA','FAA','BWT')},
  **{f'{m}_std':r['metrics'][m]['sample_sd'] for m in ('AAA','FAA','BWT')}})
u3=next(r for r in d['restored_extended_rows'] if r['job']=='formal_sum12minus0_r1_k10_s2')
table_rows.append({'condition':'U3','protocol':u3['cohort'],'method':'SAM + total-location residual',
 'gradient':r'g_1+(g_2-g_0)','n':1,'seeds':'2',**{f'{m}_mean':float(u3[m]) for m in ('AAA','FAA','BWT')},
 **{f'{m}_std':None for m in ('AAA','FAA','BWT')}})
by_id={r['condition']:r for r in table_rows}
with (OUT/'METHOD_RESULTS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,list(table_rows[0]));w.writeheader();w.writerows(table_rows)
(OUT/'PAIRED_CONTRASTS.json').write_text(json.dumps(d['paired_contrasts'],ensure_ascii=False,indent=2)+'\n')

def fmt(row,m):
 sd=row[f'{m}_std'];v=float(row[f'{m}_mean'])
 return f'{v:.3f}±{float(sd):.3f}' if sd is not None else f'{v:.3f}'

def table(ids):
 lines=['| Method | Final direction | Seeds | FAA (%) | AAA (%) | BWT (pp) |',
        '|---|---|---:|---:|---:|---:|']
 for c in ids:
  r=by_id[c];n=str(r['n'])+('*' if c in ('P01','P02','P16') else '')
  lines.append('| '+' | '.join([r['method'],'$'+r['gradient']+'$',n]+[fmt(r,m) for m in ('FAA','AAA','BWT')])+' |')
 return '\n'.join(lines)

source_rows=[]
for key in ('multi_seed_rows','r025_restored_positive_single_seed_rows','prior_r025_natural_matched_separate_protocol',
            'strict_latest_B0_B1_single_seed_rows','restored_extended_rows'):
 for r in d[key]:
  rec=r.get('source_receipt',r.get('receipt_path'))
  assert rec and Path(rec).exists()
  source_rows.append({'audit_group':key,'job':r.get('job',r.get('condition')),'seed':r['seed'],
   'protocol':r.get('family',r.get('protocol',r.get('cohort',''))),
   'AAA':float(r['AAA']),'FAA':float(r['FAA']),'BWT':float(r['BWT']),
   'source_config':r.get('source_config',r.get('config')),
   'source_receipt':rec,'source_summary':r.get('source_summary',r.get('summary_path')),
   'source_matrix':r.get('source_matrix',r.get('metrics_path',''))})
  assert r.get('independent_raw_matrix_metrics')
  for m in ('AAA','FAA','BWT'):
   assert abs(float(r[m])-r['independent_raw_matrix_metrics'][m])<1e-9
with (OUT/'SOURCE_ROWS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,list(source_rows[0]));w.writeheader();w.writerows(source_rows)
unique_receipts=len({r['source_receipt'] for r in source_rows})
strict_lines=['| r | Method | Direction | FAA | AAA | BWT |','|---:|---|---|---:|---:|---:|']
for r in sorted(d['strict_latest_B0_B1_single_seed_rows'],key=lambda r:(r['radius'],r['variant'])):
 strict_lines.append(f"| {r['radius']} | {'SGD + GAM' if r['variant']=='B0' else 'SAM + GAM'} | ${r['formula']}$ | {r['FAA']:.3f} | {r['AAA']:.3f} | {r['BWT']:.3f} |")
report=rf'''# ImageNet-A：以SGD和SAM为基点的梯度残差对比

核验时间UTC：{d['audited_at_utc']}。本次只读取实验证据、生成派生结果，没有修改训练代码、运行队列或启动实验。

## 数据集和训练设置

ImageNet-A本地train5981/test1519，200类固定顺序0…199，10任务×20类。每个方法独立从相同seed初始化重新训练整个10任务序列，依次学习各任务；训练使用当前任务样本，测试在所有已见类别中联合分类。任务数不是独立重复；seed0/1/2才是三次训练重复。

ViT-B/16 augreg2_in21k_ft_in1k，Q/V LoRA rank10，冻结骨干、训练LoRA及分类头；每任务50epochs，共500epochs/2550更新，batch128；SGD momentum/weight_decay=0，Task0 LR=.04、随后=.02，cosine最低倍率.1。内半径rho=.05，外半径r=.25或1。EWC/Fisher及其计算、实际权重融合、replay关闭；下面自然步长方向组不做最终梯度投影、正交修正或范数匹配。匹配步长组独立列出。

FAA为最终10任务等权均值，AAA为10阶段的已见任务等权均值再平均，BWT为前9任务最终准确率减各自刚学完准确率的均值。FAA/AAA单位%，BWT与差值单位百分点。±为seed0/1/2样本SD(ddof=1)；n1不填SD。BWT越高越好；本表负值越接近0表示平均退化越少，但低学习准确率下的低BWT不是良好保持。

## 先纠正公式

SAM最终方向是g1=g0+(g1−g0)。g0+(g2−g1)是独立的跨点残差实验A，不等于g1。

定义c10=g1−g0，c21=g2−g1，c20=g2−g0，c32=g3−g2。正向探针为：

$$g_0=g(\theta),\quad g_1=g(\theta+\rho N(g_0)),\quad g_2=g(\theta+rN(g_1-g_0)),\quad g_3=g(\theta+rN(g_1-g_0)+\rho N(g_2)).$$

外点从原theta出发，不是从第一探针位置theta1继续移动。探测结束恢复theta，才使用最终U执行一次theta←theta−eta U。g0/g1方法是相同探针预算下的SGD/SAM方向参考，不代表原生算法运行成本。

## 分组问题：同一残差分别加到两个基点

| 加入的分量 | SGD基点g0 | SAM基点g1 | 对照问题 |
|---|---|---|---|
| 无残差 | g0 | g1 | 基准 |
| 跨点残差c21 | g0+(g2−g1)，A | g1+(g2−g1)=g2 | 同一跨点差分能否帮助两个基点？ |
| 第二局部响应c32 | g0+(g3−g2)，UG | g1+(g3−g2)，D32 | 外点局部探针的响应能否迁移回两个基点？ |
| 整体位置响应c20 | g0+(g2−g0)=g2 | g1+(g2−g0)，U3 | 外点位置改变是否有效，是否与第一次SAM响应互补？ |

g2是同一个实验，同时对应SAM+c21与SGD+c20，不能重复计数。U3=g1+g2−g0，不能用已有g1+g2成绩替换。

同一模型状态下，A=g2−c10、g2=g2、U3=g2+c10；它们比较是否应减去或加回第一次SAM响应。但三个独立训练轨迹不会一直保持同一参数状态，终点差不是逐步同状态因果测量。所有涉及g2/c32的构造仍用c10选择外点，没有删除首次探针。

## r=1：精确恢复与RNG重放族

以下P族与新正向组合使用共同R0重放和最终状态恢复规则；P族四读取为g0/g1+/g1−/g2，新D32/U3为g0/g1/g2/g3。不用的探针不会改变最终提交状态/RNG，但计算内容不同。完整三seed与单seed分别标记。

{table(['P01','P02','P07','P04','D32','U3'])}

*P01/P02的seed2基准来自r=.25，因最终g0/g1不依赖外半径、额外探针全状态/RNG恢复而在r=1身份复用；不是新增训练，也不是所有三个JSON均literal r=1。若要求每份JSON都r=1，这两个基准仅seed0/1实跑r=1。所有其他表中方法的r=1成员使用实际r=1配置。

该执行族尚缺非严格确定性的g0+c32完整对照；下面原生/严格确定性的同公式结果独立列，不能直接拼入此族计算基线增益。

配对结论（候选−各自基线）：A−SGD FAA+9.300±.994、AAA+4.791±.703、BWT+5.265±1.255均3/3；g2−SAM FAA+4.319±1.173、AAA+1.389±1.172均3/3，BWT+2.867±2.850为2/3；D32−SAM FAA+4.235±1.469、AAA+2.162±.202、BWT+3.077±.506均3/3。

D32−g2的配对FAA−.084±2.270，仅1/3胜出，因此不能称第二局部残差优于直接提交g2。U3只有seed2：相对同seed SAM，FAA+2.295/AAA+1.542/BWT+1.914；相对g2则FAA−3.377/AAA−.960/BWT−3.050，暂不支持第一SAM响应加回g2一定有帮助。

## r=1：原生global-RNG五CE族

每步原生四探针加一次clean CE，随机状态自然推进，减扰动恢复；与上节执行协议分开。5方法×3seed=15/15完整。

{table(['N05','N03','N08','N07','N06'])}

A−SGD的配对FAA+10.565±1.280，3/3；UG−SGD为+6.859±2.216，3/3；g2−SAM为+2.891±2.815，只有2/3改善（seed0−.300），所以不能把“每个协议每个seed均帮助SAM”作为结论。跨点残差加回SGD比末端局部残差FAA+3.706±3.328，3/3，但仍是自然步长。

## 已完成的有效步长匹配对照

{table(['P16','P12'])}

P16的seed2是同样不依赖外半径的已登记身份参考复用；seed0/1正式训练已完成，旧大队列60/60完成、30种原队列条件三seed齐全。matched按同候选当前模型状态下正向g1的LoRA BA与seen FC联合位移匹配，不是简单固定乘常数。

匹配后的A−匹配g0：FAA+6.366±2.805、3/3胜出；匹配A−SAM：FAA+1.113±1.841、2/3胜出。自然A−匹配A的FAA+3.386，3/3。幅度有贡献，匹配g0对照仍保留收益；相对SAM的额外优势减弱。不同训练轨迹不能据此认定已测得纯Hessian因果作用。

## 最新严格确定性v2：同残差、换基点

4/4完整，全部seed2、自然步长；严格CUDA deterministic/native FP32、TF32/AMP关闭。缺同严格协议的g0/g1零残差完整基线，故不计算相对旧基线的正式增益。

{chr(10).join(strict_lines)}

r=1，SAM基点比SGD基点AAA+.159，但FAA−.022；r=.25为AAA+.454/FAA+.475。当前不能概括SAM基点永远更好，也不能把单seed很小的差值当成统计等价。r1 B1的BWT已按原矩阵确认−7.999。

## 当前可支持的结论

1. 跨点残差c21加到SGD有一致跨seed收益；在精确恢复协议下，加到SAM也有3/3的FAA与AAA收益。原生协议对SAM只2/3，泛化措辞应限定协议。
2. c32可以在不同执行族分别帮助SGD或SAM；缺单一协议下四个完整基准格，不宣称已建立全协议统一的SGD/SAM×有无c32因果交互。
3. g2既可以视为外点梯度，也可以视为SAM加c21。其有效不意味着g1+c20必然更好；U3仅单seed且落后同seed g2。
4. 新严格确定性基点比较没有显示SAM基点在两个半径均有FAA优势；局部响应符号与有效更新幅度仍重要。
5. 固定r实验仍不能证明Hessian机制、第二段独立于首次探针，或跨协议/数据集均有效。论文正文应以同残差换基点的分组呈现；先补同执行协议零残差与c32基点格，再扩seed。这里只记录缺口，没有启动训练。

10月9日动态eta||g0||预测器响应实验，以及Task0→Task1最小fork改变半径/任务/算法定义，不填入本固定rho/r十任务表。ImageNet-R也不并入ImageNet-A统计。

## 完整性与来源

相关原始结果引用{len(source_rows)}条，按完成收据去重为{unique_receipts}次完整训练。原生15/15；精确恢复主表16/16；步长匹配6/6；严格v2本次涉及4/4。已登记结果无失败或非有限终点；缺同协议方法和未登记设置不会填0或冒充已完成。独立原矩阵重算、来源存在与严格v2完成哈希核验通过；各自执行协议单列。

[各方法结果CSV]({OUT}/METHOD_RESULTS.csv) · [逐条原始来源]({OUT}/SOURCE_ROWS.csv) · [配对差值]({OUT}/PAIRED_CONTRASTS.json) · [完整只读审计]({OUT}/SOURCE_AUDIT.json)。

探针代码：[direction_step.py]({ROOT}/refine-logs/imageneta-r1-g12-mean-d32-multiseed-20261006T151850Z/trainer/direction_step.py:161)。U3代码：[direction_step.py]({ROOT}/refine-logs/imageneta-positive-g012-combinations-20261005T185032Z/trainer/direction_step.py:66)。
'''
(OUT/'REPORT.md').write_text(report)
manifest={'read_only_training':True,'audit_time_utc':d['audited_at_utc'],'source_reference_count':len(source_rows),
 'distinct_completed_receipts':unique_receipts,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}}
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(OUT)
print(json.dumps({k:v for k,v in manifest.items() if k!='files'},ensure_ascii=False))
