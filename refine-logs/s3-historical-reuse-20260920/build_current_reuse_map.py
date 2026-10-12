from pathlib import Path
import datetime, hashlib, json

OUT=Path(__file__).resolve().parent
CAM=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-mechanisms-20260920')
INDEX=json.loads((OUT/'reverse_index.json').read_text())
RUNS=INDEX['runs']
SUITE=json.loads((CAM/'suite.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
DEFAULTS={'train_num_workers':8,'eval_num_workers':8,'fisher_mode':'batch_empirical','fisher_split':False,'ewc_preserve_fisher_rng':False,'ewc_skip_task_fisher':False,'data_root':'/data/140-1/datasets','s3_fisher_location':'retained'}
IGNORE={'outputs_root','prefix','device','optimizer_tag_override','s3_boundary_capture','approximate_fisher_in_task'}
def diff(r,c):
    historical={**DEFAULTS,**r['config']}
    return {k:{'historical':historical.get(k),'current':v} for k,v in c.items() if k not in IGNORE and (historical.get(k) if k!='seed' else [historical.get(k)])!=v}
def full(r):return r.get('status')=='complete_verified' and r.get('scientific_budget_full_10x50') and r.get('dataset')=='imageneta'
def ref(r,c):
    return {k:r.get(k) for k in ['id','log_path','log_sha256','config_path','metrics_path','metrics_sha256','seed','faa_macro','aaa_macro','sample_final','status','runtime_branch_signals']}|{'config_sha256':sha(r['config_path']) if r.get('config_path') and Path(r['config_path']).exists() else None,'current_config_differences':diff(r,c)}

reasons={
 'gam_total_direction_scale':'索引中完成ImageNet-A实验的GAM β1/β3均为1，未见完整GAM方向减半而EWC保持不变的实验。',
 'gam_projection':'索引中完成ImageNet-A实验的GAM gamma均为0.1，未见gamma=0投影消融。',
 'gam_first_radius':'索引中完成ImageNet-A实验的rho为0.05或其他方法的0.02，未见当前rho=0.025。',
 'fisher_history':'完成的相关EWC实验gamma均为0.9，未见gamma=0或1的相应历史消融。',
 'fisher_task_normalization':'未见同一S3分支中仅关闭任务Fisher归一化的完整实验；NCL/KFAC的不归一化配置是另一方法。',
 'fisher_anchor_location':'旧endpoint权重融合改变task-start/end权重，不是固定融合模型下只将Fisher估计移到premerge端点。',
 'weight_fusion_weights':'历史快照加权/窗口有相关结果，但seed、omega、间隔、Fisher版本或额外CE执行预算不同。',
 'fisher_damping':'已有旧背景阻尼结果，但不能替代当前seed2、lambda2000、Aon的单轴控制。',
}
mapping=[]
for j in SUITE['jobs']:
    c=json.loads(Path(j['config']).read_text())
    exact=[r for r in RUNS if full(r) and r.get('runtime_branch_signals',{}).get('weight_merge_then_fisher') and not diff(r,c)]
    related=[]
    reason=reasons.get(j['axis'],'')
    if j['value']=='matched_no_merge':
        related=[r for r in RUNS if full(r) and 'component_gam' in r['log_path'] and int(r['seed'])==j['seed']]
        reason='未见in_task_weight_merge_start_task=10且仍保存两次快照、执行质量CE的历史训练。Aoff跳过CE并改变RNG，不能替代。'
    elif j['axis']=='fisher_anchor_location':
        related=[r for r in RUNS if full(r) and ('s3_endpoint_wmerge_omega030' in r['log_path'] or 's1h_endpoint_' in r['log_path'])]
    elif j['axis']=='weight_fusion_weights':
        key='in_task_path_beta' if j['value']=='uniform' else 'in_task_path_start_ratio'; val=c[key]
        related=[r for r in RUNS if full(r) and r['config'].get(key)==val and r['config'].get('in_task_weight_merge')]
    elif j['axis']=='fisher_damping' and j['value']=='alpha0':
        related=[r for r in RUNS if full(r) and (r['config'].get('ewc_fisher_damping_alpha')==0 or 's1h_endpoint_' in r['log_path'])]
    elif j['axis']=='fisher_damping':
        reason='未见alpha=1的trace匹配各向同性Fisher控制；旧alpha0/0.1/0.2只能提供相关背景。'
        related=[r for r in RUNS if full(r) and 's1h_endpoint_' in r['log_path']]
    if exact:
        reason='同seed、相同科学配置与运行分支的历史结果已经存在；可复用性能及已有机制统计。新运行的增量是冻结源/环境收据与任务边界endpoint/path状态，不是首次覆盖该科学配置。'
    level='same_seed_scientific_equivalent' if exact else 'related_different_control' if related else 'no_matching_historical_control'
    mapping.append({'current_job_id':j['id'],'axis':j['axis'],'value':j['value'],'seed':j['seed'],'current_config_path':j['config'],'current_config_sha256':sha(j['config']),'reuse_class':level,'exact_same_seed_scientific_candidates':[ref(r,c) for r in exact],'related_different_control_candidates':[ref(r,c) for r in related],'reason':reason,'can_replace_historical_accuracy_table_cell':bool(exact),'can_claim_new_frozen_queue_completion':False,'can_replace_new_boundary_sidecars':False,'cannot_replace':['冻结源和环境哈希绑定的当次完成收据','新边界状态：融合前endpoint和epoch20/40路径LoRA因子','与新批次matched-no-merge对照一致的执行来源'] if exact else ['该新控制的受控因果比较'],'queue_action':'none; audit only; running/planned jobs unchanged'})

assert len(mapping)==18
assert sum(bool(x['exact_same_seed_scientific_candidates']) for x in mapping)==4
grid=[r for r in RUNS if full(r) and int(r['seed']) in [0,1,2] and ('logs_s3_component_ablation_seed2/' in r['log_path'] or 'logs_s3_multiseed_verification/' in r['log_path'] or 'logs_s3_damp_c3_multiseed/' in r['log_path'])]
assert len(grid)==12
default_evidence=[
 {'field':'train_num_workers','value':8,'source':'source/models_LoRAbasedCL/seqlora.py:402,428','historical_revisions':['7c6aaf2','0a7ac7f'],'note':'源码默认8；历史日志未显式覆盖。'},
 {'field':'eval_num_workers','value':8,'source':'source/models_LoRAbasedCL/baseLoRA.py:897','historical_revisions':['7c6aaf2:427','0a7ac7f:427'],'note':'旧源码相同默认8；Fisher split关闭。'},
 {'field':'fisher_split','value':False,'source':'source/models_LoRAbasedCL/seqlora.py:218-233','historical_revisions':['7c6aaf2','0a7ac7f'],'note':'缺省false，候选4条亦未设置ratio/per_class隐式开启项。'},
 {'field':'fisher_mode','value':'batch_empirical','source':'source/models_LoRAbasedCL/as1_normfisher_gam.py:129-130','note':'源代码默认；4条候选运行日志也明确打印batch_empirical。'},
 {'field':'ewc_preserve_fisher_rng','value':False,'source':'source/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:89-91','note':'历史实现没有隔离Fisher RNG；当前新增开关默认false，与历史执行语义相符。'},
 {'field':'ewc_skip_task_fisher','value':False,'source':'source/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:83','note':'历史任务后计算Fisher；当前新增开关默认false。'},
 {'field':'compute_nme_class_means','value':True,'source':'source/models_LoRAbasedCL/seqlora.py:459','note':'历史及当前均默认true，旧日志有NME结果；没有为加速关闭额外前向。'},
 {'field':'data_root','value':'/data/140-1/datasets','source':'7c6aaf2:utils/data_manager.py:22; 0a7ac7f:utils/data_manager.py:22','note':'历史硬编码默认路径；当前显式相同路径，但历史数据内容没有哈希收据。'},
 {'field':'approximate_fisher_in_task','value':'inactive legacy switch in WeightMergeThenFisher','source':'source/models_LoRAbasedCL/as1_normfisher_gam_intask_damp_weightmergethenfisher.py','note':'仅在runtime确认WeightMergeThenFisher后允许忽略：无轨迹Fisher快照，weight checkpoint由独立逻辑控制。不能在June25旧父类上忽略。'},
 {'field':'s3_boundary_capture','value':'observational sidecars','source':'source/utils/s3_mechanism_controls.py','note':'比较科学配置时忽略，但其新增产物不可用旧结果冒充；同样不能生成伪完成收据。'}
]
result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reverse_index_path':str(OUT/'reverse_index.json'),'reverse_index_sha256':sha(OUT/'reverse_index.json'),'suite_path':str(CAM/'suite.json'),'suite_sha256':sha(CAM/'suite.json'),'current_jobs':18,'same_seed_science_equivalent_jobs':4,'not_historical_matched_jobs':14,'new_queue_completions_created':0,'queue_modified':False,'matching_definition':'same seed, complete10x50, runtime-proven WeightMergeThenFisher branch, all current scientific fields equal after narrowly documented defaults; source/environment bitwise execution equivalence is NOT established','default_evidence':default_evidence,'jobs':mapping,'historical_four_grid':{'seeds':[0,1,2],'runs':12,'reuse_scope':'完整GAM×A模块历史四格可复用；Aoff是模块整体删除而非CE/RNG匹配融合控制。','records':[{'id':r['id'],'log_path':r['log_path'],'seed':r['seed'],'faa_macro':r['faa_macro']} for r in grid]},'limitations':['未对新队列发停止、跳过、注册完成等操作。','4条等价候选减少的是纯性能重复需求；严格新冻结配对和新边界机制证据仍有新增价值。','历史center seed1/3不能替代新source冻结并与matched-no-merge同批控制的来源要求。','203+历史索引中任何损失NaN、未完成或协议不同记录均未提升为同配置已完成。']}
(OUT/'current_job_reuse_map.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
md=['# 当前18项实验与历史结果逐项复用映射','', '在205条历史索引中，核验到 **4项同seed、同科学配置的已有结果**：center seed1/3、GAM+EWC且A模块关闭seed2、SGD+EWC+A seed2。这里的“已有”指历史结果可复用，**不是新增4项队列完成，也未修改或停止当前队列**。','', '| 当前任务 | 历史同配置结果FAA | 复用结论 |','|---|---:|---|']
for x in mapping:
  vals='；'.join(f"{r['faa_macro']:.3f}" for r in x['exact_same_seed_scientific_candidates']) or '—'
  label={'same_seed_scientific_equivalent':'已有同seed科学配置','related_different_control':'仅有不同控制的相关证据','no_matching_historical_control':'未见匹配的历史控制'}[x['reuse_class']]
  md.append(f"| `{x['current_job_id']}` | {vals} | {label} |")
md+=['', '历史GAM×A模块四格12次（seed0/1/2）可完整复用，适用于历史组件表。它不替代本轮保持快照CE执行和RNG消费的matched-no-merge三seed对照。','', '四项已有同配置实验如果仅为补齐性能表，属于重复训练；若用于生成新冻结源/环境绑定证据以及premerge endpoint、epoch20/40路径因子，则仍有明确新增产物。历史checkpoint仅保存融合后状态，不能反推出这些边界状态。','', '源码默认核验：train/eval workers=8、batch_empirical Fisher、fisher_split=false、Fisher RNG未隔离、任务末Fisher未跳过、NME均开启；历史数据根目录默认/data/140-1/datasets。ignore approximate_fisher_in_task仅限runtime已证实WMergeThenFisher的候选。无历史全代码/环境/数据收据，故不宣称位级复现。','']
for x in mapping:
  md += [f"## {x['current_job_id']}",'',x['reason']]
  for r in x['exact_same_seed_scientific_candidates']:
    md += [f"- 原日志：[{Path(r['log_path']).name}]({r['log_path']})；FAA {r['faa_macro']:.3f}，AAA {r['aaa_macro']:.3f}，Sample Final {r['sample_final']:.2f}。"]
  for r in x['related_different_control_candidates']:
    md += [f"- 相关但不同控制：[{Path(r['log_path']).name}]({r['log_path']})；与当前不同字段：{', '.join(r['current_config_differences'])}。"]
  md.append('')
(OUT/'current_job_reuse_map.md').write_text('\n'.join(md))
print(json.dumps({'jobs':len(mapping),'science_equivalent':[(x['current_job_id'],x['exact_same_seed_scientific_candidates'][0]['faa_macro']) for x in mapping if x['exact_same_seed_scientific_candidates']],'four_grid_runs':len(grid)},ensure_ascii=False,indent=2))
