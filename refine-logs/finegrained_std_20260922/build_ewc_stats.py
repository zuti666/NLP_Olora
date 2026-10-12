"""Recompute seed statistics without mixing the selected EWC configurations."""
import ast
import hashlib
import json
import math
import re
import statistics
import yaml
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = Path('/data/140-1/users/liying/CopyFlatCV')
selected_path = BASE / 'finegrained_complete_20260919/ewc_best_selection.json'
candidates_path = BASE / 'ewclora_best_audit_20260919/extra_candidates.json'
selected = json.loads(selected_path.read_text())['selected_runs']
candidates = json.loads(candidates_path.read_text())['candidates']
ignored = {'outputs_root', 'prefix', 'seed', 'device', 'config', 'filepath', 'CNN', 'NME', 'Warning'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def record(item):
    source, log = Path(item['source']), Path(item['log'])
    data = json.loads(source.read_text())
    txt = log.read_text(errors='replace')
    params = dict(re.findall(r'=> ([a-zA-Z_][a-zA-Z_0-9]*): ([^\n]*)', txt))
    matrix = data['cnn']['matrices']['final']
    n_tasks = int(params['nb_tasks'])
    assert len(matrix) == n_tasks and all(len(row) == n_tasks for row in matrix)
    assert all(math.isfinite(matrix[t][j]) for t in range(n_tasks) for j in range(t+1))
    assert all(math.isnan(matrix[t][j]) for t in range(n_tasks) for j in range(t+1,n_tasks))
    cnn_lines = re.findall(r'=> CNN: (\{[^\n]*\})',txt)[-n_tasks:]
    assert len(cnn_lines) == n_tasks
    for t, line in enumerate(cnn_lines):
        cleaned = re.sub(r'np\.float64\(([^()]*)\)',r'\1',line)
        values = [float(v) for v in re.findall(r"'\d+-\d+': ([0-9.]+)",cleaned)]
        assert len(values) == t + 1
        assert all(abs(a-b)<1e-9 for a,b in zip(values,matrix[t][:t+1]))
    assert not re.search(r'Loss\s+(?:nan|inf)|loss[:=]\s*(?:nan|inf)',txt,re.I)
    stages = [statistics.mean(row[:t+1]) for t,row in enumerate(matrix)]
    faa, aaa = statistics.mean(matrix[-1]), statistics.mean(stages)
    assert abs(faa-item['faa'])<1e-8
    assert abs(aaa-item['aaa_stage'])<1e-8
    assert sha(source) == item['metrics_sha256']
    assert sha(log) == item['log_sha256']
    config = ROOT / params['config']
    return {'seed':int(params['seed']), 'faa':faa, 'aaa_stage':aaa,
            'stage_means':stages, 'source':str(source), 'log':str(log),
            'metrics_sha256':sha(source),'log_sha256':sha(log),
            'config':str(config),'config_sha256':sha(config) if config.exists() else None,
            'logged_params':params, 'full_lower_triangle_matches_log':True,
            'hash_matches_prior_audit':True}

rows=[]
for item in selected:
    dataset = item['dataset']
    if dataset == 'pets':
        continue
    items=[item]
    if dataset in ('cars196','flowers'):
        items=[r for r in candidates if r['dataset']==dataset
               and r['family']=='alpha0_wrapper'
               and r['ewc_lambda']==item['ewc_lambda']
               and r['params'].get('fs_preserve_probe_rng')=='True'
               and r['params'].get('ewc_preserve_fisher_rng')=='True'
               and r['eligible_for_descriptive_comparison']]
    runs=sorted([record(r) for r in items],key=lambda r:r['seed'])
    assert len({r['seed'] for r in runs}) == len(runs)
    baseline=runs[0]['logged_params']
    diffs=[]
    for r in runs[1:]:
        p=r['logged_params']
        dd={k:[baseline.get(k),p.get(k)] for k in baseline.keys()|p.keys() if baseline.get(k)!=p.get(k)}
        assert not (dd.keys()-ignored), dd
        diffs.append({'seed':r['seed'],'differences':dd})
    yaml_diffs=[]
    first_yaml=yaml.safe_load(Path(runs[0]['config']).read_text())
    for r in runs[1:]:
        other_yaml=yaml.safe_load(Path(r['config']).read_text())
        dd={k:[first_yaml.get(k),other_yaml.get(k)] for k in first_yaml.keys()|other_yaml.keys() if first_yaml.get(k)!=other_yaml.get(k)}
        assert not (dd.keys()-{'prefix','seed'}),dd
        yaml_diffs.append({'seed':r['seed'],'differences':dd})
    stats={}
    for metric in ('faa','aaa_stage'):
        values=[r[metric] for r in runs]
        stats[metric]={'mean':statistics.mean(values),'sample_sd':statistics.stdev(values) if len(values)>1 else None,
                       'sample_variance':statistics.variance(values) if len(values)>1 else None,'values':values}
    rows.append({'method':'EWCLoRA','dataset':dataset,'seeds':[r['seed'] for r in runs],'n':len(runs),
                 'ewc_lambda':item['ewc_lambda'],'family':item['family'],
                 'selected_configuration_source':item['source'],
                 'fs_preserve_probe_rng':item['params'].get('fs_preserve_probe_rng','not_logged'),
                 'ewc_preserve_fisher_rng':item['params'].get('ewc_preserve_fisher_rng','not_logged'),
                 'stats':stats,'config_differences':diffs,'yaml_config_differences':yaml_diffs,'runs':runs})

out={'created_utc':datetime.now(timezone.utc).isoformat(),
     'selection':'Hold the configuration of each current best-FAA row fixed; aggregate all eligible independent seeds from that same campaign/configuration.',
     'sd_definition':'Sample standard deviation across seeds (ddof=1); undefined for n=1, never replaced with zero.',
     'metric_definition':'FAA = mean of last CNN task-matrix row; AAA_stage = mean of lower-triangular row means; not triangular average or across-task SD.',
     'search':{'roots':['/data/140-1/users/liying/CopyFlatCV','/data/140-1/users/liying/Flatness_CV'],
               'tool':'rg --files -uuu -g *_cl_metrics.json', 'total_metrics_files':3149,
               'five_dataset_positive_ewc_native_or_alpha0_candidate_paths':56,
               'new_candidate_paths_since_20260919_audit':0},
     'excluded_protocols':['Aircraft/CUB dual-RNG alpha0 runs differ from selected historical alpha0 configuration.',
                           'Cars historical seed 0 lacks dual RNG isolation; native seed 0 is nonfinite.',
                           'Flowers historical/fisher-only/damped seed 0 differs from selected dual-RNG beta=0 configuration.',
                           'Het5 alpha0 wrapper seeds 0,1,2 differ from selected native EWC seed 2.',
                           'Het5 lambda=100000 and 200000 records have concurrent-output overwrite evidence and remain excluded.'],
     'input_hashes':{str(selected_path):sha(selected_path),str(candidates_path):sha(candidates_path)},'rows':rows}
(HERE/'ewc.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
lines=['# EWCLoRA 同配置跨 seed 统计','',
       '固定当前表中每个数据集 best-FAA 运行的配置，加入该配置可核实的其他 seed。FAA 和 AAA 均先按一次完整运行计算，再跨 seed 聚合。sample SD 与 variance 使用 n−1 分母；n=1 为不可估计。','',
       '| 数据集 | seeds | n | FAA mean | FAA SD | AAA mean | AAA SD |',
       '|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    f,a=r['stats']['faa'],r['stats']['aaa_stage']
    fmt=lambda x:'--' if x is None else f'{x:.6f}'
    lines.append(f"| {r['dataset']} | {r['seeds']} | {r['n']} | {fmt(f['mean'])} | {fmt(f['sample_sd'])} | {fmt(a['mean'])} | {fmt(a['sample_sd'])} |")
lines+=['','Cars 和 Flowers 的 seed 1、2 完整日志参数除 seed、输出名称/路径、评估结果外完全一致。两组完整 CNN 矩阵与原始逐阶段日志一致，原始日志和指标 SHA256 与前次审计相同。','',
        '其他三个数据集虽有相同 λ 的其他实验，但 RNG 隔离或 native/wrapper 执行方式不同，不能作为当前配置的重复 seed。完整源路径、哈希、配置差异和样本方差见 ewc.json。','',
        '同配置搜索覆盖 CopyFlatCV 和 Flatness_CV 全部 3149 份 *_cl_metrics.json，候选五数据集 native/alpha0 EWC 路径 56 个，与 9 月 19 日审计相比无新增候选。']
(HERE/'ewc.md').write_text('\n'.join(lines)+'\n')
for r in rows:
    print(r['dataset'],r['seeds'],r['stats'])
