from pathlib import Path
import ast, datetime, hashlib, json, math, re, subprocess
import yaml

B = Path('/data/140-1/users/liying/CopyFlatCV')
OUT = Path(__file__).resolve().parent
CAM = B / 'refine-logs/s3-mechanisms-20260920'
SOURCE = CAM / 'source'
BASE = json.loads((CAM / 'base_config.json').read_text())
DIRS = ['logs_s3_damp_vs_endpoint_merge', 'logs_s3_component_ablation_seed2', 'logs_s1h_endpoint_damp_ablation']
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v, float) and not math.isfinite(v): return None
    if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v, (tuple,list)): return [clean(x) for x in v]
    return v
def parse_config(t):
    c={}
    for l in t.splitlines():
        m=re.search(r'\[trainer.py\] => ([A-Za-z_][A-Za-z_0-9]*): (.*)$',l)
        if not m: continue
        k,v=m.groups()
        try:v=ast.literal_eval(v)
        except (ValueError,SyntaxError):pass
        c[k]=v
        if k=='feature_flat_save_path':break
    return c
def curves(t):
    rows=[]; samples=[]
    for line in re.findall(r'=> CNN: (.*)',t):
        parsed={k:float(v) for k,v in re.findall(r"'([^']+)': np\.float64\(([-+0-9.eE]+)\)",line)}
        row=[parsed.get(f'{i*20:02d}-{i*20+19}') for i in range(10)]
        rows.append(row);samples.append(parsed['total'])
    return rows,samples
def scores(rows):
    return sum(rows[-1])/len(rows[-1]), sum(sum(r[:i+1])/(i+1) for i,r in enumerate(rows))/len(rows)
def linesshow(t,patterns):
    found=[]
    for no,line in enumerate(t.splitlines(),1):
        if any(re.search(p,line) for p in patterns) and 'Epoch' not in line:
            found.append({'normalized_line':no,'text':line})
    return found
rows=[]
for d in DIRS:
  for p in sorted((B/'logs_try'/d).glob('*.log')):
    t=p.read_text(errors='replace').replace('\r','\n');c=parse_config(t)
    cfg=B/c['config'];saved=yaml.safe_load(cfg.read_text());exp=(B/c['filepath']).parent
    metric=next(exp.rglob('*_cl_metrics.json'));m=json.loads(metric.read_text())
    outlog=next(exp.glob('20/*.log'));ot=outlog.read_text(errors='replace').replace('\r','\n')
    cr,samples=curves(t);mr=clean(m['cnn']['matrices']['final']);faa,aaa=scores(cr)
    consistent=cr==mr
    eps={str(task):sorted(set(int(ep) for task2,ep,den in re.findall(r'Task (\d+), Epoch (\d+)/(\d+)',t) if int(task2)==task)) for task in range(10)}
    errors=[l for l in t.splitlines() if re.search(r'Traceback|CUDA out of memory|RuntimeError|AssertionError',l)]
    nonfinite=[l for l in t.splitlines() if 'Epoch' in l and re.search(r'(?:Loss|CE|GAM|EWC\([^)]*\))\s+[-+]?(?:nan|inf)\b',l,re.I)]
    configdiff={}
    for k,v in saved.items():
      if k not in c:continue
      a=c[k]
      if k=='seed':v=v[0] if isinstance(v,list) else v
      if k=='device':continue
      if a!=v:configdiff[k]={'printed':a,'file':v}
    diff={k:{'historical':v,'current':BASE.get(k)} for k,v in c.items() if k in BASE and k not in ['outputs_root','prefix','device','optimizer_tag_override','seed'] and v!=BASE[k]}
    ck=exp/'checkpoints'
    groups={str(i):{f:(ck/f'{f}_{i}.pt').exists() for f in ['lora_w_a','lora_w_b','fc_state']} for i in range(10)}
    runtime=linesshow(t,[r'WMergeThenFisher.*enabled',r'\[AS1-NormFisher-InTask\] in_task=',r'\[AS1-NormFisher-Damp\] ewc_',r'Saved raw Fisher snapshot',r'\[EndpointMerge\]',r'Using endpoint Fisher only'])
    r={'id':p.stem,'log_path':str(p),'log_sha256':sha(p),'config_path':str(cfg),'config_sha256':sha(cfg),'printed_config_sha256':hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),'metrics_path':str(metric) if consistent else None,'metrics_sha256':sha(metric) if consistent else None,'result_source':'raw_metrics_and_crosschecked_console' if consistent else 'original_console_CNN_per_task_rows_recovered','candidate_metrics_path':str(metric),'candidate_metrics_sha256':sha(metric),'candidate_metrics_matches_console':consistent,'candidate_metrics_faa':m['cnn']['final']['FAA'],'candidate_artifact_log_path':str(outlog),'candidate_artifact_log_sha256':sha(outlog),'seed':c['seed'],'tasks_completed':len(cr),'epochs':{'first_task':c['init_epoch'],'subsequent_task':c['epochs'],'observed_epochs_by_task':eps},'completion_status':'complete' if len(cr)==10 and all(len(v)==50 for v in eps.values()) and not errors and not nonfinite else 'needs_review','faa_macro':faa,'aaa_macro':aaa,'sample_final':samples[-1],'sample_curve':samples,'sample_aia':sum(samples)/len(samples),'cnn_matrix':cr,'config':c,'saved_config':saved,'config_file_mismatches':configdiff,'explicit_config_differences_from_current_center':diff,'checkpoint_root':str(ck),'checkpoint_groups':groups,'checkpoints_belong_to_this_log':consistent,'errors':errors,'nonfinite_training_lines':nonfinite,'evidence':runtime,'contrasts':[],'matching_current_jobs':[],'source_receipt_available':False,'source_provenance':'Historical source execution not hash-pinned; runtime identity and dated git snapshots supply bounded semantic evidence. Current frozen source is a comparison reference, not proof of the historical executable.'}
    if p.stem=='s3_damp_c3_alpha02_int20_gpu3':
      r.update(reuse_tier='supporting_different_protocol_original_console_recovered',protocol_differences=['Original June25 learner is Damp parent with trajectory Fisher snapshots AND weight fusion; current alias now maps to WeightMergeThenFisher with weight snapshots only.','Current JSON and checkpoints were overwritten by June29 seed0 reference rerun; never attribute those files to June25 result.','Seed0 differs from current seed2 center.'],actual_implementation='as1_normfisher_gam_intask_damp.Learner (June25 parent, runtime confirms raw Fisher snapshots)',contrasts=['Joint weighted trajectory Fisher + path factor fusion versus task-start/end factor fusion; multiple mechanisms differ.'])
    elif p.stem=='s3_endpoint_wmerge_omega030_gpu4':
      r.update(reuse_tier='supporting_different_protocol',protocol_differences=['Factor interpolation is 0.7*task_end + 0.3*task_start, not epochs20/40 path fusion.','No trajectory Fisher snapshots, no damping, seed0.','The endpoint name denotes weight-fusion endpoints; Fisher runs after the interpolation and is not the current premerge Fisher-location ablation.'],actual_implementation='as1_normfisher_gam_intask.Learner; EndpointMerge runtime confirms all10 tasks',contrasts=['Alternative task-start/end factor-fusion construction; usable as historical exploratory arm.'])
    elif p.stem.startswith('component_'):
      matches={'component_gam_gpu2':['s3_mech_a_off_s2'],'component_merge_gpu6':['s3_mech_sgd_ewc_s2'],'component_ewc_gpu1':[]}
      r.update(reuse_tier='historical_same_scientific_protocol_component' if matches[p.stem] else 'supporting_missing_factorial_corner',matching_current_jobs=matches[p.stem],actual_implementation='as1_normfisher_gam_intask_damp_weightmergethenfisher.Learner (runtime explicitly confirmed)',protocol_differences=['No immutable historical source/environment/RNG receipt; source-level equivalence is bounded by dated git inspection, not bitwise provenance.','No new task-boundary endpoint/path sidecars.']+(['Whole A-module removal also removes snapshot CE forwards and changes native RNG consumption; cannot replace matched_no_merge_start_task10.','approximate_fisher_in_task=false differs textually from current Aoff config but is a legacy no-op in WeightMergeThenFisher (Fisher snapshots disabled in either case).'] if p.stem!='component_merge_gpu6' else []),contrasts=['GAM on/off at fixed A-module state; historical factorial reuse.'])
    else:
      r.update(reuse_tier='supporting_different_protocol',matching_current_jobs=[],actual_implementation='as1_normfisher_gam_intask.Learner' if 'equiv' in p.stem else 'as1_normfisher_gam_intask_damp.Learner (historical parent)',protocol_differences=['Seed0, EWC lambda40000, weight fusion off; current damping axis uses seed2/lambda2000/Aon.','No path snapshots; interval10/omega0.5/path settings are inactive, not tested fusion hyperparameters.'],contrasts=['Historical endpoint-Fisher damping alpha0/0.1/0.2 at fixed lambda40000 and Aoff.'])
    if consistent:
      r['mechanism_steps']=clean(m.get('mechanism',{}).get('steps',{}))
    else:
      r['artifact_collision']={'console_start':t.splitlines()[0][:23],'console_end':t.splitlines()[-1][:23],'artifact_log_last_date':ot.splitlines()[-1][:23],'replacement_reference_log':str(B/'logs_try/logs_s3_damp_c3_multiseed/s3_damp_c3_alpha02_int20_s0_gpu1.log'),'original_console_faa':faa,'overwritten_json_faa':m['cnn']['final']['FAA']}
    rows.append(r)

source=[]
for rev in ['df683e1','47bea95','7c6aaf2','0a7ac7f']:
  fs=[]
  for f in ['backbone/lora.py','models_LoRAbasedCL/as1_normfisher_gam_intask.py','models_LoRAbasedCL/as1_normfisher_gam_intask_damp.py','models_LoRAbasedCL/as1_normfisher_gam_intask_damp_weightmergethenfisher.py','optimer_PerturabtionType/gam.py','utils/factory.py']:
    ret=subprocess.run(['git','show',f'{rev}:{f}'],cwd=B,capture_output=True)
    fs.append({'path':f,'exists':ret.returncode==0,'historical_sha256':hashlib.sha256(ret.stdout).hexdigest() if ret.returncode==0 else None,'frozen_sha256':sha(SOURCE/f),'byte_identical':ret.returncode==0 and ret.stdout==(SOURCE/f).read_bytes()})
  source.append({'git_revision':subprocess.check_output(['git','rev-parse',rev],cwd=B,text=True).strip(),'commit_date':subprocess.check_output(['git','show','-s','--format=%cI',rev],cwd=B,text=True).strip(),'files':fs})
data={'audit_date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':DIRS,'metric_definition':{'FAA':'mean(final CNN task row)','AAA':'mean(stage-wise seen-task means); not raw cnn.final.AAA which weights all55 cells equally','sample_final':'sample-weighted CNN total; not FAA','structural_nulls':'unseen task entries only; training nonfinite scan separated'},'baseline_config_path':str(CAM/'base_config.json'),'baseline_config_sha256':sha(CAM/'base_config.json'),'runs':rows,'source_comparison':source,'notes':['All8 logs completed10tasks x50epochs and contain finite55 seen-task accuracy cells.','7 original JSONs crosschecked, 1 overwritten JSON rejected; original result reconstructed from log.','Two configuration-equivalent component arms can support historical reuse: component_gam -> s3_mech_a_off_s2, component_merge -> s3_mech_sgd_ewc_s2. Exact frozen-campaign receipt equivalence not established; do not assign completed.json to these historical runs.','Model alias alone cannot determine June24/25 implementation; factory mapping changed June26.','No result in this slice substitutes for the CE/RNG-budget-matched no-merge three-seed comparison or new boundary sidecars.','Reverse tracing recovers existing configurations and evidence; it does not justify selecting new hyperparameters on final test FAA.']}
(OUT/'s3_components.json').write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
md=['# 指定 S3 组件、融合与 S1h 阻尼历史结果核验','', '8/8 均完成10任务×50epoch，55个已见任务精度均有限；7条原始JSON与控制台逐格一致，1条同名JSON已被后续运行覆盖，已从原控制台恢复。FAA、AAA均按任务宏平均计算，Sample Final为样本加权精度。','', '| 原始日志 | seed | FAA | AAA | Sample Final | 复用范围 |','|---|---:|---:|---:|---:|---|']
for r in rows:md.append(f"| {r['id']} | {r['seed']} | {r['faa_macro']:.3f} | {r['aaa_macro']:.3f} | {r['sample_final']:.2f} | {r['reuse_tier']} |")
md += ['', '## 能复用的核心结论','', '- July4历史组件：EWC为40.025，GAM+EWC为52.293，SGD+EWC+A为40.127 FAA。对应GAM增量+12.268个百分点；在SGD背景下打开A模块仅+0.102。二者都保留EWC，不是GAM-only或merge-only。','- `component_gam_gpu2`在科学配置上对应当前`s3_mech_a_off_s2`，`component_merge_gpu6`对应`s3_mech_sgd_ewc_s2`。可作为历史同协议组件结果复用；缺少历史启动源代码/环境哈希，不能伪装为新冻结队列产物，也不能提供新边界sidecars。','- A模块关闭会跳过快照CE前向，改变历史原生RNG消费；因此不能替代当前三个seed的`a_nomerge_matched`，后者保留快照和CE，仅跳过融合。','- S1h是λ=40000、A关闭、seed0的阻尼切片：α=0/0.1/0.2 FAA分别50.546/50.562/50.490，增量+0.016/−0.056。可以复用为这一背景下的阴性探索证据，不能拼到当前λ=2000、A打开、seed2阻尼曲线上。','', '## 必须修正的来源混淆','', '- June25 `s3_damp_c3_alpha02_int20_gpu3.log`对应旧父类，原日志明确记录epoch20/40同时保存raw Fisher与LoRA权重。其输出目录被June29的WMergeThenFisher seed0重跑复用；当前JSON的FAA52.316属于June29，并非June25。原日志恢复结果见上表；该条checkpoint也不能再归属于June25。','- `s3_endpoint_wmerge_omega030_gpu4`是每个任务结束执行`0.7*θ_end+0.3*θ_start`，运行日志证实10次各48个LoRA因子tensor有非零shift。它不是“在原路径融合上把Fisher估计位置移到premerge端点”；两组还同时改变阻尼、Fisher快照与额外前向。','- 同名`ewclora_normfisher_gam_intask_damp`在June24/25映射Damp父类，June26起映射WeightMergeThenFisher；不能用当前factory反向认定所有旧日志实现。','', '## 源码证据的边界','', 'July3 commit `0a7ac7f`中GAM、LoRA forward、Intask与Damp四个核心文件与当前冻结源逐字一致。WMergeThenFisher差异仅在本批旧组件默认关闭的anchor/future诊断class_slice修复。更低层变动含默认关闭的Fisher RNG隔离与来源收据、当前双归一化开启时等价的归一化重构。以上支持科学配置复用，不构成历史执行代码完整哈希证明。详细哈希、55格CNN矩阵、500个epoch覆盖、原配置与指标来源、每条运行机制统计见`s3_components.json`。','']
for r in rows:
  md += [f"## {r['id']}",'',f"- 原日志：[{Path(r['log_path']).name}]({r['log_path']})",f"- 配置：[{Path(r['config_path']).name}]({r['config_path']})；SHA256 `{r['config_sha256']}`",f"- 有效结果来源：{r['result_source']}；原始JSON路径：{r['metrics_path'] or '拒绝使用：同名输出已覆盖'}",f"- 当前队列科学配置对应：{', '.join(r['matching_current_jobs']) or '无'}"]
  md += [f'- {v}' for v in r['protocol_differences']]
  md.append('')
(OUT/'s3_components.md').write_text('\n'.join(md))
for r in rows:print(r['id'],r['completion_status'],round(r['faa_macro'],3),round(r['aaa_macro'],3),r['sample_final'],'JSONmatch',r['candidate_metrics_matches_console'],'CFGdiff',r['config_file_mismatches'],'matchjobs',r['matching_current_jobs'])
