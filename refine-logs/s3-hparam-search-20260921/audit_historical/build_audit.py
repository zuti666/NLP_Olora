#!/usr/bin/env python3
"""Read-only audit of original S3 reference neighbors and historical axes."""
import ast
import collections
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

BASE = Path('/data/140-1/users/liying/CopyFlatCV')
OUT = Path(__file__).resolve().parent
OLD = BASE / 'refine-logs/s3-historical-reuse-20260920'
SOURCE = BASE / 'refine-logs/s3-reference-hyperparams-20260920/source/models_LoRAbasedCL'
IGNORE = {'prefix','seed','device','optimizer_tag_override','config','filepath','feature_flat_save_path','outputs_root','data_root'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v, float) and not math.isfinite(v): return None
    if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v, list): return [clean(x) for x in v]
    return v
def save(name, data): (OUT/name).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def parse_console(p):
    text = Path(p).read_text(errors='replace')
    curves = re.findall(r'CNN top1 curve: (\[[^\r\n]+\])', text)
    curve = []
    if curves:
        s = re.sub(r'np\.float\d+\(([^)]+)\)', r'\1', curves[-1])
        try: curve = ast.literal_eval(s)
        except (SyntaxError,ValueError): pass
    epochs = sorted(set(tuple(map(int,m)) for m in re.findall(r'Task (\d+), Epoch (\d+)/(\d+)', text)))
    matrix = []
    mats = re.findall(r'Accuracy Matrix \(CNN Evaluation \| time_by_task\):\s*(\[\[.*?\]\])', text, re.S)
    if mats:
        matrix = [[float(v) for v in re.findall(r'nan|[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', line)] for line in mats[-1].splitlines()]
    finite = len(matrix)==10 and all(len(r)==10 for r in matrix) and all(math.isfinite(matrix[i][j]) for i in range(10) for j in range(i+1))
    return {'sample_curve':curve, 'observed_task_count':len(curve), 'last_training_task_epoch':epochs[-1] if epochs else None,
            'unique_epoch_records':len(epochs), 'complete_10x50_epoch_coverage':all((t,e,50) in epochs for t in range(10) for e in range(1,51)),
            'final_10x10_matrix_finite_observed':finite, 'console_faa_macro':sum(matrix[-1])/10 if finite else None,
            'console_aaa_macro':sum(sum(r[:i+1])/(i+1) for i,r in enumerate(matrix))/10 if finite else None,
            'has_nonfinite_loss':bool(re.search(r'Loss\s+(?:nan|inf)',text,re.I)),
            'has_traceback':'Traceback (most recent call last)' in text,
            'runtime_weight_merge_then_fisher':'[AS1-NormFisher-Damp-WMergeThenFisher]' in text,
            'runtime_raw_fisher_snapshot_count':text.count('Saved raw Fisher snapshot')}

old=json.loads((OLD/'reverse_index.json').read_text())
ref=json.loads((OLD/'reference_config_logged.json').read_text())
incremental=json.loads((OUT/'incremental_imageneta_logs.json').read_text())
allpaths=subprocess.check_output(['rg','--no-ignore','--files',str(BASE/'logs_try'),'-g','*.log','-g','*.out'],text=True).splitlines()
known={r['log_path'] for r in old['runs']} | set(old['summary']['auxiliary_or_unparsed_files'])
hash_checks=[]
for r in old['runs']:
    check={'id':r['id'],'log_path':r['log_path'],'log_sha256':sha(r['log_path'])}
    check['log_unchanged']=check['log_sha256']==r['log_sha256']
    if r.get('metrics_path') and r.get('metrics_sha256'):
        check['metrics_sha256']=sha(r['metrics_path'])
        check['metrics_unchanged']=check['metrics_sha256']==r['metrics_sha256']
    hash_checks.append(check)
assert all(x['log_unchanged'] and x.get('metrics_unchanged',True) for x in hash_checks)
save('prior_index_hash_revalidation.json',hash_checks)

families={
 'logs_s3_damp_c3_multiseed': '同中心六seed；可复用历史中心，实际执行未冻结环境，不能冒充新运行收据。',
 'logs_s1_lam_ablation':'旧Intask、A关闭；λ4万/10万/20万未完成，2千万非有限；不是当前A开启λ扫描。',
 'logs_euc_lambda':'旧Intask、A关闭下欧氏信赖域或软惩罚R=.5/1/2；GAM r恒.2。完整性能可保留，但未证明早期约束激活，不据此声称半径机制。',
 'logs_intask_normaccum_ablation':'旧Intask、20epochs下路径Fisher及归一化控制；可复用旧背景组件表，不是当前50epochs/WMergeThenFisher超参曲线。',
 'logs_s5_phase1_lambda_damp':'A关闭；λ5000/10000时α=0，α扫描时λ2000；均未完整，不可补最终性能。',
 'logs_step1_training_ablation':'旧Intask训练预算/LR/Fisher搜索；含未完成true_sample λ点；r=.1点同时20epochs，不能作训练r单轴。',
 'logs_step2d_damp_batchfisher_ablation':'旧Damp父类同时混合路径Fisher；C3参数文字同中心但运行分支不同，不与新WMergeThenFisher合并。',
 'logs_s1h_endpoint_damp_ablation':'seed0，λ40000、A关闭下α0/.1/.2可作该背景阻尼三点；不可补当前λ2000/A开启曲线。',
 'logs_s4a_wmergethenfisher_AB':'seed0 WMergeThenFisher；α=.001、uniform、interval10，ω0/1及τ0/.4/.6；可作该旧背景路径窗口实验。',
 'logs_s6_ablation':'seed0 WMergeThenFisher，ω.5但interval10/uniform；与当前间隔20/β3同时改变，不是纯ω对照。',
 'logs_step2d_weightmerge_omega005_020_030_040_050_gpu12345':'旧Intask联合Fisher/权重ω轴；五完整点，不能替代当前纯权重ω轴。',
 'logs_step2d_weightmerge_sameomega':'补齐旧联合轴ω.1完整点；其余重试未完整，ω.15无完整结果。',
 'logs_step2_intask_ablation':'旧路径Fisher interval/ω轴，不是当前WMergeThenFisher；文件名interval5实际打印25。',
 'logs_s3_component_ablation_seed2':'seed2三组件完整；A关闭可作模块关闭性能，但不能作保留快照CE的纯ω=0控制。',
 'logs_s3_multiseed_verification':'seed0/1历史三组件完整；可作系统组件消融，不是连续ω扫描。',
 'logs_s3_damp_vs_endpoint_merge':'seed0旧Damp路径混合与任务起点/终点融合；多机制变化，且Damp原JSON被后续覆盖，只用原console。',
 'logs_s2d_delta_r005_ablation':'额外ΔW惩罚半径R=.05及惩罚系数；GAM r恒.2，不作GAM训练r轴。',
 'logs_step2d_delta_penalty_ablation':'额外ΔW惩罚半径R=.5及惩罚系数；与R=.05可作匹配约束对照，不是GAM训练r。',
 'logs_step2d_weightedpath_damp_truefisher_omega030':'旧模型标签MC Fisher，α=.001、τ=.6；仅两任务完成，不能补最终FAA。',
 'logs_step2d_weightedpath_damp_truefisher_fishersplit_omega030':'旧模型标签MC Fisher+拆分；Task0尚未完成，不作性能。',
 'logs_step2e_truefisher_valsplit_intask':'旧Intask、A关闭、heldout Fisher；完成但不同数据预算/估计器背景。',
}
selected=[]
for r in old['runs']:
    family=Path(r['log_path']).parent.name
    if r['dataset']!='imageneta' or family not in families:continue
    c=r['config']
    rr={k:r.get(k) for k in ('id','log_path','log_sha256','metrics_path','metrics_sha256','status','faa_macro','aaa_macro','seed','config','runtime_branch_signals')}
    rr['family']=family;rr['reuse_scope']=families[family]
    rr['config_diff_from_reference_excluding_seed_and_paths']={k:{'reference':ref.get(k),'candidate':c.get(k)} for k in sorted(ref.keys()|c.keys()) if k not in IGNORE and ref.get(k)!=c.get(k)}
    rr['console_checks']=parse_console(r['log_path'])
    selected.append(rr)

# Incremental paths have both shell logs and nested copies; keep the original
# shell logs in the review table and explicitly retain all 102 screening rows.
for r in incremental['imageneta_rows']:
    checks=parse_console(r['path'])
    r.update(checks)
    r.pop('has_final_curve',None)
    r['screening_status']='complete_10x50_console' if checks['complete_10x50_epoch_coverage'] and checks['final_10x10_matrix_finite_observed'] and len(checks['sample_curve'])==10 else 'screening_only_not_formally_classified'
    r.pop('sample_curves',None)
    if '/logs_merge/' not in r['path'] and '/logs-merge/run_logs/' not in r['path']:continue
    c=r['config']; family=str(Path(r['path']).parent.relative_to(BASE))
    rr={'id':Path(r['path']).stem,'log_path':r['path'],'log_sha256':r['sha256'],'family':family,'config':c,'seed':c['seed'],
        'status':r['screening_status'],'faa_macro':checks['console_faa_macro'],'aaa_macro':checks['console_aaa_macro'],'console_checks':checks,
        'config_diff_from_reference_excluding_seed_and_paths':{k:{'reference':ref.get(k),'candidate':c.get(k)} for k in sorted(ref.keys()|c.keys()) if k not in IGNORE and ref.get(k)!=c.get(k)}}
    rr['reuse_scope']='seed1六个均匀窗口；ω=1、β=0、interval/τ改变，完整性能可作窗口附录，不能补当前ω轴。' if 's7_' in family else 'seed2原始F/归一化/阻尼+A关闭的组件控制；机制统计也关闭，不作当前连续α或ω曲线。'
    selected.append(rr)
save('incremental_imageneta_logs.json',incremental)
save('historical_candidates.json',selected)

# Static, executable extraction of the exact snapshot gate (no torch imports,
# models, GPU work, or modifications to the frozen source).
src=SOURCE/'as1_normfisher_gam_intask_damp_weightmergethenfisher.py'
tree=ast.parse(src.read_text()); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
gate=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_should_capture_in_task_weight')
ns={};exec(compile(ast.Module(body=[gate],type_ignores=[]),str(src),'exec'),ns)
from types import SimpleNamespace
observed=[]
for label,omega,start in [('center',.3,0),('native_omega0',0.,0),('matched_no_merge',.3,10)]:
    dummy=SimpleNamespace(_in_task_weight_merge=True,_in_task_weight_checkpoint_interval=20,_in_task_weight_merge_omega=omega,_in_task_weight_merge_start_task=start)
    captures=[e for e in range(1,51) if ns[gate.name](dummy,e)]
    observed.append({'case':label,'omega':omega,'merge_start_task':start,'snapshot_epochs_50_epochs':captures,'snapshot_CE_enabled_at_captured_epochs':True})
assert [r['snapshot_epochs_50_epochs'] for r in observed]==[[20,40],[],[20,40]]
omega={'source':str(src),'source_sha256':sha(src),'snapshot_gate_lines':[166,173],'snapshot_ce_lines':[190,197],
       'parent_source':str(SOURCE/'as1_normfisher_gam_intask.py'),'parent_sha256':sha(SOURCE/'as1_normfisher_gam_intask.py'),
       'parent_start_task_guard_lines':[880,888],'convex_merge_lines':[915,920],'executable_gate_witness':observed,
       'native_omega0_equivalent_to_matched_no_merge':False,
       'conclusion':'omega=0 disables snapshot/CE calls before task-end; start_task=10 retains snapshot/CE RNG consumption, only skips task-end merge. Use matched-no-merge as effective_omega=0 for controlled weight-only axis, and keep native omega0/module-off separate.'}
save('omega_zero_source_audit.json',omega)

summary={'reference_logs':[str(BASE/'logs_try/logs_s3_damp_c3_multiseed'/f's3_damp_c3_alpha02_int20_s{s}_gpu{s+1}.log') for s in (2,3)],
 'scanned_roots':[str(BASE/s) for s in ('logs_try','logs','logs_merge','logs-merge')],
 'reference_directory_and_siblings':[str(p) for p in sorted((BASE/'logs_try').iterdir()) if p.is_dir()],
 'logs_try_current_log_count':len(allpaths),'logs_try_prior_index_rows_rehashed':len(hash_checks),'logs_try_new_unindexed_logs':sorted(set(allpaths)-known),
 'incremental_log_files_keyword_screened':incremental['scanned_matching_logs'],'incremental_imageneta_logs_including_duplicates':len(incremental['imageneta_rows']),
 'historical_candidate_rows':len(selected),'candidate_families':dict(collections.Counter(r['family'] for r in selected)),
 'not_exhaustively_scanned_other_top_level_log_dirs':[str(p) for p in sorted(BASE.iterdir()) if p.is_dir() and 'log' in p.name and p.name not in ('logs_try','logs','logs_merge','logs-merge')],
 'mainline_seed2_3_noncenter_historical_lambda_r_alpha_or_pure_omega_points_found':0,
 'caution':'No same-center off-axis historical points means searched roots only. Nonidentical scientific backgrounds remain reusable within their own comparisons, including seeds0/1. Current frozen sweep is separate and already contains seed2 lambda/r points.'}
save('coverage_and_summary.json',summary)

fields=['id','family','seed','status','ewc_lambda','gam_grad_norm_rho','ewc_fisher_damping_alpha','in_task_fisher_omega','in_task_weight_merge','in_task_fisher_interval','in_task_path_start_ratio','in_task_path_beta','faa_macro','aaa_macro','tasks_completed_console','last_task_epoch','reuse_scope','log_path','config_diff_json']
with (OUT/'historical_candidates.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in selected:
        c=r['config'];v={k:r.get(k) for k in fields};v.update({k:c.get(k) for k in fields if k in c});v['tasks_completed_console']=r['console_checks']['observed_task_count'];v['last_task_epoch']=r['console_checks']['last_training_task_epoch'];v['config_diff_json']=json.dumps(r['config_diff_from_reference_excluding_seed_and_paths'],ensure_ascii=False);w.writerow(v)

lines=['# S3 历史超参路径复核（2026-09-21）','',
'本审计从两份原日志所在目录及同级目录开始（logs_try共44个子目录），重新核验旧索引 205 条日志及已绑定结果文件的哈希；另扫描 logs / logs_merge / logs-merge 中 226 份含超参键的日志，保留 102 条 ImageNet-A 记录（含控制台/内部日志镜像，不能当作102次独立实验）。没有发现新增未索引的 logs_try 日志。原项目只读，未启停训练。','',
'参考中心是 WMergeThenFisher，λ=2000、GAM训练 r=.2（内层ρ=.05）、α=.2、ω=.3、interval20、β3、10×50 epochs。seed2 FAA=53.311，seed3=51.968。历史中心可以复用；本审计根中没有与此中心同背景的非中心 λ/r/α/纯权重ω单轴点。seed0/1 的旧实验仍可在各自固定背景下复用，不因seed不同被丢弃。','',
'最新 frozen s3-reference-hyperparams-20260920 已完成 seed2 λ={0,200,2000,20000}、r={.1,.2,.5}，其中中心重跑FAA=53.419；原日志FAA=53.311应单列历史参照，不能把二者择优混为同一次实验。','',
'## 核心结论','',
'- S5 λ/阻尼共12条尝试均未完成；A关闭、λ扫描时α=0，不能填当前曲线。S1 λ4万/10万/20万未完成，2千万发生非有限训练。','- S1h α0/.1/.2三点完整，但λ40000/A关闭。旧Damp C1/C2/C3/b0四点完整，含路径Fisher；C3即使配置文字同中心也不是当前实现。','- 旧共同ω={.05,.1,.2,.3,.4,.5}六点完整，同时改变Fisher和权重混合。S4四点、S6一点、S7六种窗口也完整，但有明确背景差异，见下表。','- GAM训练r和额外ΔW边界R、LoRA rank、训练后诊断半径必须分开；旧r=.1点同时改变epoch，不是当前训练r控制。','- 原生omega=0会禁用快照CE，不能代替保留CE的matched-no-merge。后者适合作为纯权重融合曲线的effective_omega=0点。','',
'## 每条候选及完成情况','', '| 日志 | seed | 状态 / 已完成任务 | λ / r / α / ω | FAA | 与参考的科学配置差异 |','|---|---:|---|---|---:|---|']
for r in selected:
    c=r['config'];delta=', '.join(f"{k}: {v['candidate']}" for k,v in r['config_diff_from_reference_excluding_seed_and_paths'].items()) or '打印配置同参考；仍须检查运行分支'
    faa='—' if r['faa_macro'] is None else f"{r['faa_macro']:.3f}"
    vals=' / '.join(str(c.get(k,'default/not logged')) for k in ['ewc_lambda','gam_grad_norm_rho','ewc_fisher_damping_alpha','in_task_fisher_omega'])
    lines.append(f"| [{Path(r['log_path']).name}]({r['log_path']}) | {r['seed']} | {r['status']} / {r['console_checks']['observed_task_count']} | {vals} | {faa} | {delta} |")
lines+=['','## 各组可复用范围','']
for family in summary['candidate_families']:
    scope=next(r['reuse_scope'] for r in selected if r['family']==family);lines.append(f'- `{family}`：{scope}')
lines+=['','## omega=0 的源码证据','',f'- [{src.name}:166]({src}:166)：omega≤0 时 _should_capture_in_task_weight 返回False；冻结源码AST实际调用证据为中心[20,40]、native omega0[]、matched-no-merge[20,40]。',f'- [{src.name}:190]({src}:190)：仅进入快照后执行CE质量评估。',f'- [as1_normfisher_gam_intask.py:883]({SOURCE}/as1_normfisher_gam_intask.py:883)：start_task检查仅在任务末融合函数，未禁用前面的快照CE。','',
'## 扫描路径与边界','']
lines += [f'- `{p}`' for p in summary['scanned_roots']]
lines += ['','原日志所在目录及同级目录：','']+[f'- `{p}`' for p in summary['reference_directory_and_siblings']]
lines += ['','其余尚未全文审计的顶层日志目录：','']+[f'- `{p}`' for p in summary['not_exhaustively_scanned_other_top_level_log_dirs']]
lines += ['','原始行级参数、完成进度、配置差异及哈希见 `historical_candidates.json/.csv`、`prior_index_hash_revalidation.json`；增量筛选含无关模型的候选见 `incremental_imageneta_logs.json`。FAA按最后一行任务均值，AAA按逐阶段行均值再平均；未来任务上三角NaN不视为失败。']
(OUT/'HISTORICAL_HPARAM_AUDIT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
