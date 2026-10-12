"""Build derived reports only; never touch training, configurations, or queues."""
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/data/140-1/users/liying/O-LoRA')
PREVIOUS = ROOT / 'refine-logs/imageneta-gam-cumulative-results-20261006T150045Z'
now = datetime.now(timezone.utc)
OUT = ROOT / f'refine-logs/imageneta-gam-cumulative-results-{now:%Y%m%dT%H%M%SZ}'
OUT.mkdir()
(OUT / 'evidence').mkdir()

def read_csv(path):
    with path.open() as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)

def write_csv(name, fields, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def load_audit(name):
    p = Path('/tmp') / name
    shutil.copy2(p, OUT / 'evidence' / name)
    return json.loads(p.read_text())

g12 = load_audit('g12_complete_audit_20261007.json')
rad = load_audit('radius4_complete_audit_20261007.json')
old = load_audit('seed01_increment_audit_20261007.json')
host = load_audit('imageneta_host_status_20261007.json')
fields, previous_rows = read_csv(PREVIOUS / 'ALL_UNIQUE_10TASK_RESULTS.csv')
new_rows = list(old['rows_new'])
for r in g12['new6_rows']:
    new_rows.append({**r, 'cohort': 'four_probe_exact_restore_rng_replay',
        'lr_multiplier': 1, 'note': 'New seed0/1 training; paired realized inputs and initial parameters across all three directions verified'})
for r in rad['formal_rows']:
    sources = r['sources']
    metrics = r['metrics']
    new_rows.append({
        'job': r['job'], 'cohort': 'four_probe_exact_restore_rng_replay',
        'radius': r['radius'], 'seed': r['seed'], 'rank': r['rank'], 'rho': r['rho'],
        'direction': r['direction'], 'branch': 'P', 'mode': 'natural', 'backend': r['backend'],
        'tasks': r['completed_tasks'], 'epochs': r['completed_epochs'], 'updates': r['completed_steps'],
        'lr_multiplier': 1, 'AAA': metrics['AAA_task_macro'], 'FAA': metrics['FAA_task_macro'],
        'LA': metrics['LA'], 'BWT': metrics['BWT'], 'forgetting': metrics['forgetting'],
        'status': 'validated', 'source_config': sources['input_config'],
        'source_matrix': sources['matrix'], 'source_summary': sources['summary'],
        'source_receipt': sources['completion_receipt'],
        'note': 'New complete seed2 radius experiment; mean12 r1 also supplies the external seed2 cell of the new three-seed campaign; counted once',
    })
assert len(new_rows) == 30
all_rows = previous_rows + new_rows
assert len(all_rows) == 144
receipts = [str(Path(r['source_receipt']).resolve()) for r in all_rows]
assert len(set(receipts)) == 144

sources_manifest = {}
for r in all_rows:
    assert int(r['tasks']) == 10 and int(r['epochs']) == 500 and int(r['updates']) == 2550
    assert int(r['rank']) == 10 and math.isclose(float(r['rho']), .05)
    for field in ('source_config', 'source_matrix', 'source_summary', 'source_receipt'):
        p = Path(r[field])
        assert p.is_file(), p
        sources_manifest[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    raw = json.loads(Path(r['source_matrix']).read_text())
    if 'matrix' in raw:
        matrix = raw['matrix']
    else:
        matrix = raw['cnn']['matrices']['t09']
        matrix = [m[:i + 1] for i, m in enumerate(matrix)]
    assert len(matrix) == 10 and all(len(m) == i + 1 for i, m in enumerate(matrix))
    final = matrix[-1]
    computed = {
        'AAA': statistics.mean(statistics.mean(m) for m in matrix),
        'FAA': statistics.mean(final),
        'LA': statistics.mean(matrix[i][i] for i in range(10)),
        'BWT': statistics.mean(final[i] - matrix[i][i] for i in range(9)),
        'forgetting': statistics.mean(max(matrix[t][i] for t in range(i, 10)) - final[i] for i in range(9)),
    }
    for metric, value in computed.items():
        assert math.isfinite(float(r[metric]))
        assert math.isclose(float(r[metric]), value, abs_tol=1e-8), (r['job'], metric, value, r[metric])
    assert math.isclose(float(r['FAA']), float(r['LA']) + .9 * float(r['BWT']), abs_tol=1e-8)

assert Counter(float(r['radius']) for r in all_rows) == {0.25: 44, 1.0: 100}
assert Counter(int(r['seed']) for r in all_rows) == {0: 32, 1: 32, 2: 80}
write_csv('ALL_UNIQUE_10TASK_RESULTS.csv', fields, all_rows)
write_csv('NEW_30_COMPLETE_RESULTS.csv', fields, new_rows)
for radius, name in ((.25, 'R025_ALL_44_RESULTS.csv'), (1, 'R1_ALL_100_RESULTS.csv')):
    write_csv(name, fields, [r for r in all_rows if float(r['radius']) == radius])

aggregate_fields, _ = read_csv(PREVIOUS / 'THREE_SEED_COMPLETE_19_CONDITIONS.csv')
aggregates = list(old['aggregates_complete_29'])
metric_fields = ('AAA', 'FAA', 'LA', 'BWT', 'forgetting')
for a in g12['aggregates']:
    ar = {'condition': 'G12_' + a['direction'], 'family': 'positive_four_probe_g012_g3_exact_restore_rng_replay',
          'label': a['label'], 'expected_seeds': '0;1;2', 'completed_seeds': '0;1;2',
          'n_expected': 3, 'n_complete': 3, 'aggregate_status': 'complete_3_seeds', 'std_ddof': 1}
    rows = sorted([r for r in g12['rows'] if r['direction'] == a['direction']], key=lambda r: r['seed'])
    assert [r['seed'] for r in rows] == [0, 1, 2]
    for r in rows:
        assert r['source_receipt'] in receipts
        ar[f'seed{r["seed"]}_status'] = 'validated_external' if r['external'] else 'validated'
        for metric in metric_fields:
            ar[f'{metric}_seed{r["seed"]}'] = r[metric]
    for metric in metric_fields:
        vals = [r[metric] for r in rows]
        ar[f'{metric}_mean'] = statistics.mean(vals)
        ar[f'{metric}_std'] = statistics.stdev(vals)
        assert math.isclose(ar[f'{metric}_mean'], a['metrics'][metric]['mean'], abs_tol=1e-8)
        assert math.isclose(ar[f'{metric}_std'], a['metrics'][metric]['sample_sd'], abs_tol=1e-8)
    aggregates.append(ar)
assert len(aggregates) == 32
for a in aggregates:
    for m in metric_fields:
        v = [float(a[f'{m}_seed{s}']) for s in (0, 1, 2)]
        assert math.isclose(float(a[f'{m}_mean']), statistics.mean(v), abs_tol=1e-8)
        assert math.isclose(float(a[f'{m}_std']), statistics.stdev(v), abs_tol=1e-8)
write_csv('THREE_SEED_COMPLETE_32_CONDITIONS.csv', aggregate_fields, aggregates)
write_csv('G12_THREE_SEED_9_CELLS.csv', fields, [{**r, 'cohort': 'four_probe_exact_restore_rng_replay', 'lr_multiplier': 1} for r in g12['rows']])
(OUT / 'G12_PAIRED_CONTRASTS.json').write_text(json.dumps(g12['contrasts'], ensure_ascii=False, indent=2) + '\n')
(OUT / 'FOUR_BRANCH_PAIRED_CONTRASTS.json').write_text(json.dumps(old['paired_contrasts'], ensure_ascii=False, indent=2) + '\n')
shutil.copy2(PREVIOUS / 'REPORT.md', OUT / 'evidence/previous_report_20261006.md')

def fmt(a, m):
    return f"{float(a[m + '_mean']):.3f}±{float(a[m + '_std']):.3f}"

new_aggs = aggregates[-3:]
rows_3seed = '\n'.join(f"| {a['label']} | {fmt(a, 'AAA')} | {fmt(a, 'FAA')} | {fmt(a, 'LA')} | {fmt(a, 'forgetting')} |" for a in new_aggs)
rows_seed = '\n'.join(f"| {r['label']} | {r['seed']} | {r['AAA']:.3f} | {r['FAA']:.3f} | {r['LA']:.3f} | {r['forgetting']:.3f} |" for r in sorted(g12['rows'], key=lambda r: (r['direction'], r['seed'])))
rows_radius = '\n'.join(f"| {r['formula']} | {r['radius']} | {r['metrics']['AAA_task_macro']:.3f} | {r['metrics']['FAA_task_macro']:.3f} | {r['vs_sum12']['delta']['FAA_task_macro']:+.3f} |" for r in rad['formal_rows'])
rows_old_native = '\n'.join(f"| {a['condition']}：{a['label']} | {fmt(a, 'AAA')} | {fmt(a, 'FAA')} | {fmt(a, 'LA')} | {fmt(a, 'forgetting')} |" for a in aggregates[:14])
rows_old_four = '\n'.join(f"| {a['condition']}：{a['label']} | {fmt(a, 'AAA')} | {fmt(a, 'FAA')} | {fmt(a, 'LA')} | {fmt(a, 'forgetting')} |" for a in aggregates[14:29])
report = f'''# ImageNet-A纯GAM累计进度与结果：2026-10-07

报告生成时间：{now.astimezone(ZoneInfo('Europe/Paris')).isoformat()}。真实GPU宿主机只读快照：{host['checked_at']}。本次只核验结果、更新派生汇总；未启动、停止、修改或重启训练队列。

最新Sum12、Mean12与D32三个方向的seed0/1/2已全部完成、验收通过。新增正式训练6/6完成，另复用三项seed2，合计9个有效实验格；最后一项新增正式训练结束于{g12['last_new_formal_completed_paris']}。该队列13/13完成包含6正式、6短测和1配对检查，不能写成13次正式训练。

累计144条独立完整十任务训练（此前114+本次30）：r=.25共44条、r=1共100条；seed0/1各32条、seed2共80条。32种条件三seed完整（原队列29种+本轮三方向3种）。以完整完成收据绝对路径去重，Mean12 r=1/seed2同时出现在半和队列和本轮三seed表，但只计一次。身份参考、smoke、诊断、重复草案及运行中的P16不计入144条。

## 当前进程

本轮三方向队列、半和双半径队列、g0+g1+g2双半径队列全部完成，控制器退出、没有活动worker。旧seed0/1主队列正式58/60完成、2运行、0等待、0失败；60项smoke全部完成。

| GPU | 运行项目 | 当前任务（从1开始） | epoch/50 | 已完成更新/2550 |
|---|---|---:|---:|---:|
| 5 | P16，匹配步长g0，seed0 | 5/10 | 27 | 1658 |
| 6 | P16，匹配步长g0，seed1 | 3/10 | 16 | 1023 |

上述进程PID及启动身份与实际宿主机匹配，最近loss有限，trace正在更新。GPU0–4在快照时空闲（每卡显存2MiB、利用率0）；未新增实验。P16两项尚无完整终点，不并入三seed统计。

## 本轮统一设置和结果口径

ImageNet-A本地train5981/test1519；200类固定顺序、10×20类；ViT-B/16 augreg2_in21k_ft_in1k，LoRA rank10，scale=1、无alpha/rank缩放。seed0/1/2；r=gam_grad_norm_rho=1，ρ=gam_grad_rho=.05。每任务50 epochs、总500 epochs/2550更新；batch128，SGD，momentum/weight decay=0，Task0 LR=.04、后续=.02，cosine最低比例.1。

每步四次正向梯度探测：g0=g(φ)，g1=g(φ+ρN(g0))，g2=g(φ+rN(g1−g0))，g3=g(φ+rN(g1−g0)+ρN(g2))。N仅用于构造扰动位置的单位化。各次探测全状态恢复，R0重放并仅推进一次g0/R1；同seed三种方向初始参数哈希及全部2550步实际输入/标签/索引/LR序列一致。每步一次原学习率SGD；Mean仅对最终方向乘.5一次，没有另减半LR。g3每组计算，只有D32使用。

EWC/Fisher、实际权重融合、replay、最终梯度正交修正/投影、HVP/create_graph及最终更新额外范数匹配全部关闭；trace和计数器确认上述工作为0。关闭的生命周期空hook不等于权重融合更新。

AAA为10阶段各阶段已见任务等权准确率的平均，FAA为末阶段10任务等权均值；LA为对角均值，BWT和遗忘按前9任务。FAA=LA+.9BWT。数值单位%，差值和遗忘单位百分点。±为3seed样本标准差（ddof=1）；不混入逐样本准确率，不作为显著性检验。

## 新三方向完整三seed

| 最终方向 | AAA均值±SD | FAA均值±SD | LA均值±SD | 遗忘均值±SD |
|---|---:|---:|---:|---:|
{rows_3seed}

| 最终方向 | seed | AAA | FAA | LA | 遗忘 |
|---|---:|---:|---:|---:|---:|
{rows_seed}

同seed配对：Sum12−Mean12的AAA为+1.772±.354、FAA为+4.686±.527，3/3均胜出（FAA差值4.451、4.318、5.290）。Sum12−D32的AAA为+2.624±.479、FAA为+6.301±1.107，3/3均胜出。Mean12−D32的FAA为+1.615±1.030，3/3均胜出。

本轮支持Sum12在这三个方向中表现最好，并复制了三个seed。Sum与Mean同最终方向、原LR相同，仅幅度相差2倍，半和的FAA稳定降低，说明更新幅度/有效学习率有实质影响；不能称Sum优势完全独立于步长，也不能判为完全由步长解释。Sum−Mean的LA+4.057±.617、Sum−D32的LA+5.332±.633均为3/3胜出；遗忘差分别−.791±1.040和−.815±1.020，仅2/3改善。因此收益同时伴随更强的新任务学习，不能主要归因于一致降低遗忘。

D32在本轮低于Sum和Mean，不能推广为该残差在所有基点、半径和匹配步长下均无效。当前结果不能证明Hessian机制。

## 另外两批seed2双半径：4/4正式完成

| 最终方向 | r | AAA | FAA | 相对同半径Sum12的FAA差 |
|---|---:|---:|---:|---:|
{rows_radius}

g0+g1+g2在r=.25和1均未超过同seed/同半径g1+g2；不能据此说额外保留g0普遍有帮助。Mean12从r=.25到1仅提升.643AAA/.657FAA；Sum12对应提升1.740AAA/4.116FAA，半径影响依赖组合与幅度。这四项是单seed结果，其中Mean12 r=1已作为本轮三seed的seed2成员统计。

## 原生协议族：14种完整三seed（独立列出）

原生GAM五CE、全局自然RNG推进、原生减扰动恢复，和本轮四探针/全状态恢复/RNG重放在执行协议上不同。不能直接把本轮55.911与原版GAM54.977的差称为严格方法提升；不得将不同协议族拼成一个受控排名。

| 条件 | AAA均值±SD | FAA均值±SD | LA均值±SD | 遗忘均值±SD |
|---|---:|---:|---:|---:|
{rows_old_native}

## 四分支协议族：15种完整三seed

该族每步g0/g1+/g1−/g2四探针；P外方向q=g1+−g0，D:q=g1−−g0，C:q=g0−g1−，R:q=g0−g1+。O表示使用g2；A表示g0+(g2−对应分支g1)。matched为同候选当前模型状态下正向g1+的有效LoRA BA及seen FC位移匹配；不是本轮Mean的固定.5倍率。

| 条件 | AAA均值±SD | FAA均值±SD | LA均值±SD | 遗忘均值±SD |
|---|---:|---:|---:|---:|
{rows_old_four}

新增跨seed证据：A_P自然步长相对g1+的配对FAA+4.499±.451、3/3胜出；A_C自然+3.085±1.740、3/3胜出。匹配后A_P为+1.113±1.841、2/3胜出；A_C为−1.564±2.997、1/3胜出；A_D为−1.962±1.241、0/3胜出。因此自然步长下差分帮助不能直接推广到匹配步长，r=1的保持幅度证据弱于此前r=.25单seed。O_C−O_P的FAA+.378±1.528、2/3胜出，不能认定反向探针加符号修正稳定优于正向。A_R三seed接近随机/崩溃，尽管数值有限且验收通过，低遗忘不能解释为保持良好。

## 来源与核验

所有144条结果的原始10×三角任务矩阵独立重算AAA/FAA/LA/BWT/遗忘，完整500epochs/2550更新、rank10/ρ=.05及全部来源路径检查通过。累计CSV按收据去重；32种三seed均值与样本SD独立重算通过。本轮三方向全部validator、冻结文件哈希、源码哈希及配对输入/初始状态检查通过。执行文件、配置、队列及进程没有修改。

- [144条独立十任务CSV]({OUT}/ALL_UNIQUE_10TASK_RESULTS.csv)
- [32种完整三seedCSV]({OUT}/THREE_SEED_COMPLETE_32_CONDITIONS.csv)
- [本轮9格逐seedCSV]({OUT}/G12_THREE_SEED_9_CELLS.csv)
- [三方向配对差值]({OUT}/G12_PAIRED_CONTRASTS.json)
- [旧四分支配对差值]({OUT}/FOUR_BRANCH_PAIRED_CONTRASTS.json)
- [前版完整历史讨论，2026-10-06]({OUT}/evidence/previous_report_20261006.md)（历史快照，当时未完成的条件已由本报告更新）
- [宿主机只读进程/GPU快照]({OUT}/evidence/imageneta_host_status_20261007.json)

各行配置、收据、summary和matrix绝对路径保存在CSV；manifest记录这些来源与报告文件SHA256。本轮审计和完整验证计数保存在evidence目录。
'''
(OUT / 'REPORT.md').write_text(report)
file_hashes = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()}
manifest = {'created_at_utc': now.isoformat(), 'previous_package': str(PREVIOUS),
    'read_only_training_audit': True, 'count_unique_complete_runs': 144, 'count_new_complete_runs': 30,
    'count_radius': {'0.25': 44, '1.0': 100}, 'count_seed': {'0': 32, '1': 32, '2': 80},
    'count_complete_three_seed_conditions': 32, 'old_queue_formal_counts': old['formal_counts'],
    'raw_matrix_recomputation_passed': True, 'all_three_seed_recomputation_passed': True,
    'source_file_sha256': sources_manifest, 'derived_file_sha256': file_hashes}
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
(ROOT / 'refine-logs/IMAGENETA_GAM_RESULTS_CURRENT.md').write_text(f'''# ImageNet-A纯GAM最新累计结果

更新时间：{now.astimezone(ZoneInfo('Europe/Paris')).isoformat()}。

[完整汇总与结论]({OUT}/REPORT.md) · [144条独立十任务CSV]({OUT}/ALL_UNIQUE_10TASK_RESULTS.csv) · [32种完整三seed统计]({OUT}/THREE_SEED_COMPLETE_32_CONDITIONS.csv)。

Sum12、Mean12、g1+(g3−g2)三方向seed0/1/2均完成：AAA/FAA分别68.609±.353/55.911±1.033，66.837±.262/51.225±1.559，65.985±.382/49.610±1.746。半和及g0+g1+g2双半径4/4正式完成。

累计r=.25为44条、r=1为100条。旧seed0/1队列58/60正式完成、GPU5/6上仅P16的seed0/1仍运行；GPU0–4在{host['checked_at']}快照时空闲。本次只读核验并更新派生报告，没有新增/重启/修改训练。
''')
new_campaign = Path(g12['campaign'])
(new_campaign / 'RESULT_ANALYSIS_20261007.md').write_text(f'''# 三方向三seed已完成

最后新增正式训练结束：{g12['last_new_formal_completed_paris']}。新增6/6正式训练与三个外部seed2都验收通过，共9格；同seed输入序列与初始状态核对一致。

| 方向 | AAA均值±样本SD | FAA均值±样本SD | LA均值±样本SD | 遗忘均值±样本SD |
|---|---:|---:|---:|---:|
{rows_3seed}

Sum−Mean配对FAA+4.686±.527，Sum−D32 +6.301±1.107，均3/3胜出。更新幅度有实质影响，不能把优势全部归因于方向；LA提高而遗忘仅2/3改善。本表属于全状态恢复/四探针/RNG重放协议，与旧原生五CE协议分开解释。

[完整逐seed、对照与累计144结果]({OUT}/REPORT.md) · [三方向9格CSV]({OUT}/G12_THREE_SEED_9_CELLS.csv)。
''')
print(OUT)
print(json.dumps({k: v for k, v in manifest.items() if k not in ('source_file_sha256', 'derived_file_sha256')}, ensure_ascii=False))
