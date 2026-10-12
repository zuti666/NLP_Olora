"""Read-only historical artifact audit; only writes this audit directory."""
import ast
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import yaml

ROOT = Path('/data/140-1/users/liying/CopyFlatCV')
OUT = Path(__file__).resolve().parent
CURRENT = ROOT/'refine-logs/s3-mechanisms-20260920'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def scalar(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s

def safe(s):
    s = re.sub(r'np\.float64\(([^)]+)\)',r'\1',s)
    s = re.sub(r'\bnan\b','None',s)
    return ast.literal_eval(s)

def summary(values):
    values=[float(v) for v in values if isinstance(v,(float,int)) and math.isfinite(v)]
    return {'n':len(values),'min':min(values),'max':max(values),'mean':statistics.mean(values)} if values else {'n':0}

def audit(path,group):
    text=path.read_text()
    config={k:scalar(v) for k,v in re.findall(r'\[trainer.py\] => (\w+): (.*)',text)}
    # CNN and NME lines are also logger key/value text; only the initial config preamble is a config.
    preamble=text.split('[data_manager.py]')[0]
    config={k:scalar(v) for k,v in re.findall(r'\[trainer.py\] => (\w+): (.*)',preamble)}
    cfgp=ROOT/config['config']
    yf=yaml.safe_load(cfgp.read_text()) if cfgp.exists() else None
    yaml_differences={}
    if yf is not None:
        for k,v in yf.items():
            if k=='seed' and isinstance(v,list):v=v[0]
            if k!='device' and config.get(k)!=v:
                yaml_differences[k]={'yaml':v,'log':config.get(k)}
    results_dir=(ROOT/config['filepath']).parent
    choices=list(results_dir.rglob('*_cl_metrics.json'))
    assert len(choices)==1,(path,choices)
    mp=choices[0]
    j=json.loads(mp.read_text())
    matrix=j['cnn']['matrices']['final']
    n=len(matrix)
    lower=[matrix[t][i] for t in range(n) for i in range(t+1)]
    complete=(n==10 and all(len(row)==10 for row in matrix) and all(math.isfinite(v) for v in lower))
    logged_cnn=[safe(s) for s in re.findall(r'=> CNN: (.*)',text)]
    console_matrix=[[row[f'{20*i:02d}-{20*i+19:02d}'] for i in range(t+1)] for t,row in enumerate(logged_cnn)]
    match=(len(console_matrix)==10 and all(console_matrix[t]==matrix[t][:t+1] for t in range(10)))
    epochs=[(int(t),int(e),int(b)) for t,e,b in re.findall(r'Task (\d+), Epoch (\d+)/(\d+)',text)]
    per_task={str(t):sorted(set(e for tt,e,_ in epochs if tt==t)) for t in range(10)}
    full_budget=all(per_task[str(t)]==list(range(1,51)) for t in range(10))
    badloss=re.findall(r'\b(?:Loss|CE|GAM|EWC|DWpen)\s+[-+]?(?:nan|inf)\b',text,re.I)
    errors=re.findall(r'(?:Traceback \(most recent call last\)|CUDA out of memory|RuntimeError:.*)',text)
    steps=j['mechanism']['steps']
    logged_mechanism={}
    for record in re.findall(r'\[Mechanism\] (\{.*\})',text):
        parsed=safe(record)
        logged_mechanism[str(parsed['task'])]=parsed
    common_mechanism_mismatches={}
    for task,data in steps.items():
        observed=logged_mechanism.get(task,{})
        diff={key:[value,observed[key]] for key,value in data.items() if key in observed and value!=observed[key] and not(isinstance(value,float) and isinstance(observed[key],float) and math.isnan(value) and math.isnan(observed[key]))}
        if diff:common_mechanism_mismatches[task]=diff
    selected={k:{f:v for f,v in d.items() if f.startswith(('in_task_', 'train_delta_w_', 'grad_delta_w_')) or f in ['task','delta_norm','grad_fisher_to_clean_norm_ratio_mean','train_ewc_mean','fisher_past_trace_mass']} for k,d in steps.items()}
    values=list(steps.values())
    noninitial=[d for k,d in steps.items() if int(k)>0]
    fields=['train_delta_w_penalty_active_mean','train_delta_w_penalty_mean','grad_delta_w_penalty_norm_mean','grad_delta_w_penalty_to_clean_norm_ratio_mean','train_delta_w_penalty_drift_norm_mean','train_delta_w_penalty_warn_radius_mean','in_task_weight_merge_shift_l2','grad_fisher_to_clean_norm_ratio_mean']
    stats={f:summary([d.get(f) for d in noninitial]) for f in fields}
    evidence={
        'sha256':{str(path):sha(path),str(mp):sha(mp)},
        'yaml_log_differences_excluding_device':yaml_differences,
        'full_10x10_finite_lower_triangle':complete,
        'finite_lower_triangle_cells':sum(math.isfinite(v) for v in lower),
        'raw_json_matches_all_10_console_task_rows':match,
        'epochs_seen':per_task,
        'every_task_has_epochs_1_through_50':full_budget,
        'nonfinite_training_losses':badloss,
        'error_markers':errors,
        'mechanism_task_count':len(steps),
        'console_mechanism_task_count':len(logged_mechanism),
        'common_mechanism_field_mismatches_between_json_and_log':common_mechanism_mismatches,
        'weight_merge_applied_tasks':sum(d.get('in_task_weight_merge_applied')==1 for d in values),
        'fisher_path_used_tasks':sum(bool(d.get('in_task_fisher_used_path')) for d in values),
        'weight_snapshot_epochs':sorted(set((d.get('in_task_weight_merge_first_epoch'),d.get('in_task_weight_merge_last_epoch')) for d in values)),
        'noninitial_task_mechanism_summary':stats,
        'mechanism_steps':selected,
        'log_start':text.splitlines()[0][:23],
        'log_end':next(line[:23] for line in reversed(text.splitlines()) if re.match(r'20\d\d-',line)),
        'historical_execution_source_hash_proven':False,
        'historical_source_limit':'现存源码可解释日志中的已执行分支；没有逐运行冻结源码/提交凭据，不能证明历史代码与当前源码逐字一致。',
    }
    if cfgp.exists():evidence['sha256'][str(cfgp)]=sha(cfgp)
    base_diffs=['seed0（当前主配对 seed1/2/3）','基础类 as1_normfisher_gam_intask；当前别名映射 WMergeThenFisher','旧版每任务 20/40 epoch 估计路径 Fisher 并混入最终 Fisher，当前仅混权重后估计一次 Fisher','旧版路径 A/B 权重均匀平均；当前训练 CE 权重 beta=3','旧版无当前 alpha=.2 阻尼分支','历史训练源码与预训练文件身份未逐运行冻结']
    if group.startswith('delta'):
        base_diffs.append('额外 ΔW 惩罚开启；当前中心没有该惩罚，不能解释为 GAM 训练半径扫描')
    raw_resume={k:v for k,v in config.items() if any(x in k.lower() for x in ['resume','load_path','checkpoint','pretrain'])}
    resume_lines=[line for line in text.splitlines() if re.search(r'resume|restoring|load(?:ing)? .*checkpoint',line,re.I)]
    contrasts={'group':group,'joint_omega':config['in_task_fisher_omega'],'gam_rho':config['gam_grad_rho'],'gam_r':config['gam_grad_norm_rho']}
    if group.startswith('delta'):
        contrasts.update({'delta_lambda':config['ewc_delta_penalty_lambda'],'kappa':config['ewc_delta_penalty_warn_kappa'],'R_delta':values[-1]['delta_w_penalty_radius'],'effective_boundary':values[-1]['delta_w_penalty_warn_radius']})
    return {
        'id':path.stem,'log_path':str(path),'config_path':str(cfgp),'metrics_path':str(mp),'seed':config['seed'],
        'tasks_completed':n,'epochs':config['epochs'],'init_epochs':config['init_epoch'],
        'completion_status':'complete_verified' if complete and match and full_budget and not badloss and not errors else 'needs_review',
        'faa_macro':statistics.mean(matrix[-1]),'aaa_macro':statistics.mean(statistics.mean(row[:t+1]) for t,row in enumerate(matrix)),
        'sample_final':logged_cnn[-1]['total'],'sample_curve':[r['total'] for r in logged_cnn],
        'bwt_final_avg':j['cnn']['final']['BWT_final_avg'],'native_json_AAA_triangle_mean':j['cnn']['final']['AAA'],
        'config':config,'yaml_config':yf,'contrasts':contrasts,
        'prior_parent_state':{'resume_config_fields':raw_resume,'resume_log_evidence':resume_lines,'interpretation':'完整从 Task0 训练；fromgam 表示边界从 GAM rho 取值，不是从 GAM checkpoint 恢复。' if group.startswith('delta') else '完整从 Task0 训练，未见任务 checkpoint 恢复配置或执行记录。'},
        'reuse_tier':'supporting_different_protocol','matching_current_jobs':[],
        'protocol_differences':base_diffs,'evidence':evidence,
    }

def main():
    paths=[(p,'joint_omega') for p in sorted((ROOT/'logs_try/logs_step2d_weightmerge_omega005_020_030_040_050_gpu12345').glob('*.log'))]
    paths.append((ROOT/'logs_try/logs_step2d_weightmerge_sameomega/s2d_wmerge_sameomega_omega010_gpu6.log','joint_omega'))
    paths.extend((p,'delta_r005') for p in sorted((ROOT/'logs_try/logs_s2d_delta_r005_ablation').glob('f*.log')))
    paths.append((ROOT/'logs_try/logs_step2d_delta_penalty_ablation/e2_warnk100_r05_lam50_gpu2.log','delta_r05_auxiliary_anchor'))
    rows=[audit(*p) for p in paths]
    # All contrasts are determined from resolved logs, and metrics from the final matrices.
    exclude={'outputs_root','prefix','device','config','filepath','feature_flat_save_path','optimizer_tag_override','in_task_fisher_omega'}
    omegas=sorted([r for r in rows if r['contrasts']['group']=='joint_omega'],key=lambda r:r['contrasts']['joint_omega'])
    center=omegas[3]
    for r in omegas:
        r['contrasts']['all_log_config_differences_vs_omega030']={k:{'this':r['config'].get(k),'omega030':center['config'].get(k)} for k in set(r['config'])|set(center['config']) if r['config'].get(k)!=center['config'].get(k)}
        r['contrasts']['other_training_field_differences_vs_omega030']={k:v for k,v in r['contrasts']['all_log_config_differences_vs_omega030'].items() if k not in exclude}
    delta_center=next(r for r in rows if r['id']=='f2_fromgam_k200_lam50_gpu2')
    delta_skip=exclude|{'ewc_delta_penalty_lambda','ewc_delta_penalty_warn_kappa','ewc_delta_penalty_radius','ewc_delta_penalty_radius_source','ewc_delta_penalty_radius_scale','ewc_delta_penalty_radius_from_gam'}
    for r in rows:
        if not r['contrasts']['group'].startswith('delta'):continue
        diff={k:{'this':r['config'].get(k),'f2':delta_center['config'].get(k)} for k in set(r['config'])|set(delta_center['config']) if r['config'].get(k)!=delta_center['config'].get(k)}
        r['contrasts']['all_log_config_differences_vs_f2']=diff
        r['contrasts']['other_training_field_differences_vs_f2']={k:v for k,v in diff.items() if k not in delta_skip}
    source_paths=[CURRENT/'source/models_LoRAbasedCL/as1_normfisher_gam_intask.py',CURRENT/'source/models_LoRAbasedCL/as1_normfisher_gam_intask_damp_weightmergethenfisher.py',CURRENT/'source/utils/factory.py',ROOT/'scripts/run_s2d_delta_r005_ablation.sh',ROOT/'logs_try/logs_s2d_delta_r005_ablation/launcher.log',CURRENT/'base_config.json',CURRENT/'suite.json']
    result={'audit_scope':'指定两目录的10次完成运行 + 找回的 omega=.10 完成运行 + R_delta=.5 的外部配对锚点；共12个完整来源。','runs':rows,'source_explanation_sha256':{str(p):sha(p) for p in source_paths},'direct_current_job_replacements':[],
      'notes':['支持复用旧协议的6点共同 omega 曲线和5点附加漂移惩罚曲线；不能抵扣当前18项同协议机制控制。','FAA=最终行任务均值；AAA=逐阶段任务均值的平均。sample_final 是样本加权最终 CNN 准确率；不混用日志 Average Accuracy 或 JSON 的三角均值 AAA。','固定 seed0 的历史探索不能获得跨 seed 误差条；结果导向挑参应标为探索并保留全部点。','kappa=0 时惩罚为 lambda*||DeltaW-DeltaWref||²，R_delta不改变该目标；不能解释为有效半径对照。']}
    (OUT/'omega_delta.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 历史 omega 与 ΔW 惩罚复用核验','',result['audit_scope'],'','**本组可复用 11 次旧协议结果，外加 1 次边界配对锚点；直接替代当前同协议机制实验 0 次。**','',
      '全部 12 次均 seed0、ImageNet-A 10×20 类、rank10、batch128、每任务 50 epochs、LR .04/.02、λEWC=2000、GAM ρ=.05/r=.2。日志覆盖每任务 epoch1–50；10×10 原始 CNN 下三角 55 项有限并逐行吻合控制台；无非有限训练 Loss/CE/GAM/EWC/DWpen 或 traceback。未见 checkpoint 恢复，`fromgam` 指从 GAM ρ 读取附加边界。','',
      '## 共同 omega 六点','',
      '| omega | FAA | AAA | 样本加权最终 CNN | 完成 |','|---|---:|---:|---:|---|']
    for r in omegas:lines.append(f"| {r['contrasts']['joint_omega']:.2f} | {r['faa_macro']:.3f} | {r['aaa_macro']:.3f} | {r['sample_final']:.2f} | {r['completion_status']} |")
    lines += ['', '六点除 omega 和输出/设备元数据外，日志训练字段一致。omega=.10 的完整控制台是 `logs_step2d_weightmerge_sameomega/s2d_wmerge_sameomega_omega010_gpu6.log`，同目录其它 gpu1 的 .10 仅训练到 Task0 epoch2；不能把中断日志计作独立完成复现。该目录 omega=.15 的 gpu7 记录为 CUDA 不可用失败，gpu2 只到 Task0 epoch2，不能算第七个完整点。','',
      '原 YAML 明示同一 omega 控制 `F=(1−ω)N(F_final)+ωN(mean(F20,F40))` 与 `θ=(1−ω)θ_final+ω mean(θ20,θ40)`。每条日志的10任务均有路径 Fisher 使用和48个 A/B 因子的实际融合。现存基础类代码吻合此执行证据。当前 WMergeThenFisher 则先以 CE 权重融合 A/B，之后只在保留权重估计一次 Fisher，并带 alpha=.2 阻尼。故旧曲线不能称为当前方案的纯权重 omega 扫描，也不能替换当前均匀融合/关闭融合控制。FAA最优旧点为 omega=.3，AAA最优旧点为 omega=.5；这是单seed探索，不据此改当前中心。','',
      '## 额外 ΔW 软惩罚','',
      '训练额外目标为 `λΔ max(||ΔW−ΔWref||−κRΔ,0)²`；κ=0 时精确退化为无阈值 L2。五点的 RΔ=.05 来自固定 GAM ρ=.05；GAM 外半径 r=.2 未变。Task0不启用惩罚，Task1–9保留非零惩罚和非零惩罚梯度记录。','',
      '| RΔ | κ | λΔ | 有效阈值 κRΔ | FAA | AAA | 样本最终 CNN | Task1–9 激活比例范围 |','|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if not r['contrasts']['group'].startswith('delta'):continue
        c=r['contrasts']; a=r['evidence']['noninitial_task_mechanism_summary']['train_delta_w_penalty_active_mean']
        lines.append(f"| {c['R_delta']:.2f} | {c['kappa']:.0f} | {c['delta_lambda']:.0f} | {c['effective_boundary']:.2f} | {r['faa_macro']:.3f} | {r['aaa_macro']:.3f} | {r['sample_final']:.2f} | {a['min']:.3f}–{a['max']:.3f} |")
    lines+=['','可复用 κ=2 下 λΔ=20/50/100 三点，以及 λΔ=50 下 κ=0/1/2 三点。外部 RΔ=.5、κ=1、λΔ=50 对照 RΔ=.05 的字段受控比较给出 FAA 52.815→49.807（−3.008pp），但属于附加约束，不能填入 GAM ρ/r 曲线。','',
      '## 来源和限制','',
      '原始日志、YAML、metrics JSON及现存源码 SHA256、逐任务机制字段、全量配置、矩阵/训练预算检查保存在 `omega_delta.json`。原始JSON的FWT/AFG及上三角NaN为未定义结构项，不作为训练失败；报告JSON只保留有限派生结果。历史执行源码和预训练文件未逐运行冻结，因此不声称其与当前 frozen source 逐字一致。','',
      '| ID | 原始日志 | 原始矩阵 |','|---|---|---|']
    for r in rows:lines.append(f"| {r['id']} | [log]({r['log_path']}) | [metrics]({r['metrics_path']}) |")
    (OUT/'omega_delta.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{'id':r['id'],'FAA':r['faa_macro'],'AAA':r['aaa_macro'],'sample':r['sample_final'],'status':r['completion_status'],'yaml_diffs':r['evidence']['yaml_log_differences_excluding_device'],'other_omega_diffs':r['contrasts'].get('other_training_field_differences_vs_omega030'),'activation':r['evidence']['noninitial_task_mechanism_summary']['train_delta_w_penalty_active_mean']} for r in rows],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
