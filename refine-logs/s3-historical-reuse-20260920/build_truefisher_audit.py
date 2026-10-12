"""Read-only historical inputs; materialize a source-linked reuse audit."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import math
import re
import subprocess
import yaml

ROOT = Path('/data/140-1/users/liying/CopyFlatCV')
OUT = Path(__file__).parent
BASE = ROOT / 'refine-logs/s3-mechanisms-20260920/base_config.json'
SUITE = ROOT / 'refine-logs/s3-mechanisms-20260920/suite.json'
base = json.loads(BASE.read_text())
all_files = [Path(p) for p in subprocess.check_output(['rg', '--files', '--no-ignore', str(ROOT)], text=True).splitlines()]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def sanitized(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, list):
        return [sanitized(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): sanitized(v) for k,v in obj.items()}
    return obj

def parsed_printed_config(text):
    result = {}
    for line in text.splitlines():
        m = re.search(r'\[trainer\.py\] => ([a-zA-Z_][a-zA-Z_0-9]*): (.*)', line)
        if not m:
            continue
        k, v = m.groups()
        if k in {'CNN', 'NME'}:
            continue
        try:
            result[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            result[k] = v
    return result

specs = [
    ('truefisher_smoke', 'logs_step2d_weightedpath_damp_truefisher_smoke'),
    ('truefisher_omega030', 'logs_step2d_weightedpath_damp_truefisher_omega030'),
    ('truefisher_fishersplit_omega030', 'logs_step2d_weightedpath_damp_truefisher_fishersplit_omega030'),
    ('related_s2e_truefisher_fsplit', 'logs_step2e_truefisher_valsplit_intask'),
]
rows = []
for rid, dirname in specs:
    folder = ROOT / 'logs_try' / dirname
    log = next(folder.glob('*.log'))
    text = log.read_text(errors='replace')
    yamls = list(folder.glob('*.yaml'))
    config_path = next((p for p in yamls if p.name.endswith('_run.yaml')), yamls[0])
    config = yaml.safe_load(config_path.read_text())
    prefix = config['prefix']
    tag = config['optimizer_tag_override']
    output_files = [p for p in all_files if str(ROOT / 'outputs_logs') in str(p) and prefix in str(p) and tag in str(p)]
    internal_log = next((p for p in output_files if p.suffix == '.log'), None)
    internal_text = internal_log.read_text(errors='replace') if internal_log else ''
    printed = parsed_printed_config(text or internal_text)
    metrics_path = next((p for p in output_files if p.name.endswith('_cl_metrics.json')), None)
    metrics = json.loads(metrics_path.read_text()) if metrics_path else {}
    matrices = metrics.get('cnn', {}).get('matrices', {})
    steps = metrics.get('mechanism', {}).get('steps', {})
    completed_tasks = len([k for k in matrices if re.fullmatch('t[0-9]+', k)])
    epoch_tuples = [(int(a), int(b), int(c)) for a,b,c in re.findall(r'Task (\d+), Epoch (\d+)/(\d+)', text + '\n' + internal_text)]
    epoch_max = {str(t): max(e for t2,e,n in epoch_tuples if t2 == t) for t in sorted({t for t,e,n in epoch_tuples})}
    error_lines = [{'source': str(p), 'line': i+1, 'text': l} for p,s in [(log,text),(internal_log,internal_text)] if p for i,l in enumerate(s.splitlines()) if re.search(r'Traceback|(?:^|\W)(?:\w*Error|Exception)(?::|\W)|out of memory|Killed',l)]
    nonfinite_losses = [{'source': str(p), 'line': i+1, 'text': l} for p,s in [(log,text),(internal_log,internal_text)] if p for i,l in enumerate(s.splitlines()) if re.search(r'(?:Loss|CE|GAM) (?:nan|[+-]?inf)(?:\W|$)',l,re.I)]
    evidence = [{'source': str(p), 'line': i+1, 'text': l} for p,s in [(log,text),(internal_log,internal_text)] if p for i,l in enumerate(s.splitlines()) if any(token in l for token in ['fisher_mode=true_sample', '[FisherSplit]', '[WeightMerge]', '[PathWeight]', 'CNN top1 curve:', 'Forgetting (CNN):'])]
    # Completion requires actual ten-stage matrix, final report and ten final epochs.
    m = matrices.get('final')
    complete = bool(m and len(m) == 10 and all(len(r) == 10 for r in m) and all(math.isfinite(m[i][j]) for i in range(10) for j in range(i+1)) and completed_tasks == 10 and all(epoch_max.get(str(t)) == 50 for t in range(10)))
    faa = sum(m[-1]) / 10 if complete else None
    aaa = sum(sum(r[:i+1])/(i+1) for i,r in enumerate(m))/10 if complete else None
    curves = re.findall(r'CNN top1 curve: (\[[^\n]+\])', text or internal_text)
    curve = ast.literal_eval(re.sub(r'np\.float64\(([^)]+)\)', r'\1', curves[-1])) if curves else []
    partial = None
    if matrices and not complete:
        last_key = sorted(matrices)[-1]
        pm = matrices[last_key]
        partial = {'tasks': len(pm), 'faa_over_seen_tasks_only': sum(pm[-1])/len(pm), 'aaa_over_seen_stages_only': sum(sum(r[:i+1])/(i+1) for i,r in enumerate(pm))/len(pm), 'sample_curve': curve, 'not_a_ten_task_result': True}
    ckpt_tasks = sorted({int(match.group(1)) for p in output_files if (match := re.search(r'lora_w_a_(\d+)\.pt$', str(p)))})
    diffs = {}
    ignore = {'outputs_root','prefix','optimizer_tag_override','device','data_root'}
    for k,v in config.items():
        if k in ignore: continue
        bv = base.get(k, '<not explicitly configured>')
        if k == 'seed': bv = [int(s) for s in bv]
        if bv != v: diffs[k] = {'historical_yaml': v, 'current_center': bv}
    common_differences = [
        'seed0；当前主要成对干预seed2，中心补seed1/3。',
        '运行时true_sample：模型分布逐样本MC标签Fisher（每输入1标签）；当前batch_empirical是批均值梯度平方代理。',
        '历史任务内计算多次Fisher并混合路径/最终Fisher；当前WeightMergeThenFisher只在权重融合后计算一次任务Fisher，额外前向/随机数轨迹不同。',
        '未发现历史运行时源码哈希、环境和RNG收据；六月23提交仅作源码佐证，不能证明当时工作树逐字一致。',
    ]
    if rid.startswith('related_'):
        differences = common_differences + ['S2E为旧intask类，权重融合关闭，未加damp类；ω=.2、interval10、Fisher留出集。', 'fisher_min_per_class=20使实际留出比例远高于20%，各任务训练样本预算与当前不同。']
        tier = 'historical_complete_separate_protocol'
        status = 'complete_10_tasks'
    else:
        differences = common_differences + ['历史damp父类只对Fisher快照做late/CE加权；权重路径仍为全部快照的均匀A/B因子平均。当前S3对权重路径做CE加权。', 'alpha=.001而当前=.2；path_start_ratio=.6而当前=0。']
        if 'fishersplit' in rid:
            differences += ['FisherSplit=true，min20/class，训练/Fisher数据拆分；interval10；不限Fisher和path-CE batches。']
        if 'smoke' in rid:
            differences += ['仅配置1epoch/task、interval1、Fisher/path-CE最多1batch；不能作为50epoch正式结果。']
        tier = 'partial_execution_evidence_only' if completed_tasks else 'configuration_or_start_evidence_only'
        status = 'incomplete_truncated_unknown_cause' if epoch_tuples or completed_tasks else 'empty_console_internal_start_only'
    row = {
        'id': rid, 'log_path': str(log), 'config_path': str(config_path), 'metrics_path': str(metrics_path) if metrics_path else None,
        'seed': config['seed'][0], 'tasks_completed': completed_tasks, 'epochs': {'configured_init_epoch': config['init_epoch'], 'configured_epochs': config['epochs'], 'max_observed_per_task': epoch_max},
        'completion_status': status, 'faa_macro': faa, 'aaa_macro': aaa, 'sample_final': curve[-1] if complete else None,
        'config': config, 'printed_config': printed, 'config_source': 'archived_yaml_and_runtime_printout',
        'contrasts': ['模型标签MC Fisher / Fisher与权重路径区别 / 留出集协议'], 'reuse_tier': tier, 'matching_current_jobs': [],
        'protocol_differences': differences, 'explicit_config_differences_from_s3_center': diffs,
        'partial_metrics': partial, 'sample_curve': curve, 'checkpoint_tasks': ckpt_tasks,
        'errors_found': error_lines, 'nonfinite_training_losses_found': nonfinite_losses,
        'failure_cause': None, 'failure_cause_note': '未见异常栈、OOM或明确终止原因；缺尾/无完整输出不能推断具体失败原因。' if not complete else None,
        'evidence': evidence, 'mechanism_steps': sanitized(steps),
        'source_files': [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in [log,config_path,internal_log,metrics_path] if p],
        'runtime_source_uncertainty': 'Exact historical working tree and environment not preserved in run receipt. Related git snapshots are corroboration only.'
    }
    rows.append(row)

source_evidence = []
for rev,name in [('7aba039','models_LoRAbasedCL/as1_normfisher_gam.py'),('ead1768','utils/factory.py'),('ead1768','models_LoRAbasedCL/as1_normfisher_gam_intask.py'),('ead1768','models_LoRAbasedCL/as1_normfisher_gam_intask_damp.py')]:
    blob = subprocess.check_output(['git','show',rev+':'+name],cwd=ROOT)
    source_evidence.append({'git_revision': rev, 'repository': str(ROOT), 'path': name, 'sha256_of_git_blob_bytes': hashlib.sha256(blob).hexdigest()})
report = {
    'audit_time': datetime.datetime.now().astimezone().isoformat(),
    'scope': '三个指定truefisher目录；全CopyFlatCV按相同前缀查后续输出；关联S2E分支只单列历史完整结果。',
    'baseline_path': str(BASE), 'baseline_sha256': sha(BASE), 'suite_path': str(SUITE), 'suite_sha256': sha(SUITE),
    'summary': {'requested_runs': 3, 'requested_complete_runs': 0, 'new_queue_jobs_replaceable': 0, 'related_complete_runs': 1},
    'metric_definitions': {'faa_macro': '10-task final matrix row mean', 'aaa_macro': 'mean of 10 lower-prefix row means', 'sample_final': 'last CNN top1 curve item (sample weighted)', 'structural_nan': 'Unobserved upper-triangle/FWT NaN is not training divergence.'},
    'selection_warning': '从已有最终测试成绩回看配置只能作为探索/复用审计；不能据最高测试值重新定义确认性中心或据此挑选种子。',
    'runs': rows, 'code_evidence': source_evidence,
    'search_outcome': '同名前缀仅找到上述原始日志/输出，没有同配置完整续跑。S2E为不同设置的新运行，不是原omega030/fishersplit的续跑。S1C truefisher控制台也为0字节，内部仅Task0启动，无完整矩阵。'
}
(OUT/'truefisher.json').write_text(json.dumps(sanitized(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
md = '''# 历史 truefisher 结果复用核验

三个指定目录中 **0/3 有完整十任务结果，不能替代当前18项机制实验**。已按原配置前缀回查整个 CopyFlatCV 的内部日志、原始矩阵和checkpoint，没有找回同配置完整续跑。

|目录|实际进度|可复用内容|
|---|---|---|
|weightedpath_damp_truefisher_smoke|控制台0字节；内部日志只到Task0启动，无完成epoch/矩阵|仅配置与启动记录，不能称短测通过|
|weightedpath_damp_truefisher_omega030|Task0/1各50epochs完成，止于Task2 epoch1；矩阵仅t00/t01，2组checkpoint|前两任务机制执行记录，不能记作最终FAA|
|weightedpath_damp_truefisher_fishersplit_omega030|止于Task0 epoch18/50；仅epoch10的Fisher/CE快照，无任务最终矩阵或checkpoint|拆分与Fisher估计确实执行的记录|

三个目录均未找到明确异常栈、OOM或非有限训练Loss。缺尾原因未知，不能把未完成归因为OOM，也不能把未见报错误当成成功。

## “truefisher”与“weighted path”的实际含义

运行时打印 `fisher_mode=true_sample`，原始机制字段写明 `model_distribution_fisher`、`per_sample_cross_entropy`、`mean_of_per_sample_squared_delta_w_gradients`。六月23源码逐输入前向并从模型softmax分布抽1个标签后反传。因此这里确实是 **模型标签MC Fisher**，不是batch-gradient代理，也不是使用真实数据标签的逐样本经验Fisher。当前S3中心使用 `batch_empirical`，两者不能混为一组。

历史 `as1_normfisher_gam_intask_damp` 只将late筛选/CE权重用于 **Fisher快照**；LoRA A/B权重路径继承父类，仍均匀平均全部快照。omega030日志中权重融合用epochs20/40共2个快照，而Fisher路径在tau=.6筛选后只剩epoch40一个，`weight_min=weight_max=1, effective_n=1`。这不提供多快照CE加权优于均匀权重融合的证据。当前S3为 `WeightMergeThenFisher`，对权重路径CE加权，再在保留的融合权重处计算一次Fisher；旧运行还混合路径Fisher与最终Fisher，随机数/计算预算也不同。

共有设置为ImageNet-A、10×20类、ViT-B/16、LoRA rank10、seed0、GAM rho=.05/r=.2、lambda_flat=1、EWC lambda=2000/gamma=.9。旧damping alpha=.001（当前=.2），tau=.6（当前=0），omega=.3。普通正式分支50epochs/task、interval20、Fisher/CE上限100batches；smoke仅1epoch、interval1、1batch。split分支interval10、Fisher/CE不截断，并且从训练集留出Fisher样本，min20/class。

普通分支在前两任务的CNN sample曲线为87.95/82.00；t01宏平均为81.850。这些是仅见40类的中间成绩，不是200类最终性能。

## 找到的关联完整分支

`logs_step2e_truefisher_valsplit_intask/s2e_truefisher_fsplit_r020_min20_interval10_omega02_gpu7.log` 是后续独立分支，原始矩阵/日志与10组checkpoint均完整，seed0：**FAA=36.049、AAA=56.260739、sample-weighted最终CNN=48.39**。AAA为逐阶段宏平均再平均；原始JSON `cnn.final.AAA=47.776727` 使用另一种聚合，不能代替此值。

该分支关闭权重融合，使用旧intask类、omega=.2、interval10、模型标签MC Fisher和留出集，没有damp子类。它可复用为历史分支记录/训练集拆分诊断，不能作为当前S3的A-off或单变量Fisher对照。

`fisher_min_per_class=20` 使实际留出比例远大于声明的20%：Task0 fit859/fisher400；Task5 fit33/fisher227；Task9 fit120/fisher258。低性能不能单独归因于Fisher类型；训练样本预算等多项同时变化。未重训。

## 可追溯性与复用边界

完整路径、各文件SHA256、原样YAML配置、运行时打印、逐项与S3中心差异、矩阵、机制字段和带行号证据已写入 [truefisher.json](truefisher.json)。未发现历史源码/环境/RNG完整收据；相关Git快照只作机制语义佐证，不声明与运行时工作树逐字一致。

从最终测试成绩回看已有配置属于探索性审计。应复用已有完整且协议匹配的结果；不能据最高测试分数反向指定“确认性”中心或挑选种子。本批指定truefisher旧日志没有可直接替代当前18项的结果。
'''
(OUT/'truefisher.md').write_text(md)
print(json.dumps({'output':str(OUT),'summary':report['summary'],'runs':[{k:r[k] for k in ['id','completion_status','tasks_completed','faa_macro','aaa_macro','sample_final']} for r in rows]},ensure_ascii=False,indent=2))
