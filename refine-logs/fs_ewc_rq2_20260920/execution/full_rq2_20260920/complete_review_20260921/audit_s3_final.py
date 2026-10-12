"""Read-only S3 raw-result audit; output is confined to this O-LoRA directory."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json, hashlib, math, statistics as st, re, datetime, collections
import torch
torch.set_num_threads(2)
P = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-mechanisms-20260920')
O = P.parent / 's3-reference-hyperparams-20260920'
N = Path(__file__).resolve().parent
issues = []; checks = collections.Counter(); hash_cache = {}
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    p = Path(p)
    stat = p.stat(); k = (str(p), stat.st_mtime_ns, stat.st_size)
    if k not in hash_cache:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
        hash_cache[k] = h.hexdigest()
    return hash_cache[k]
def check(ok, category, detail):
    checks[category] += 1
    if not ok: issues.append({'category': category, 'detail': detail})
def eq(a,b): return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)
def stats(v):
    return {'n': len(v), 'mean': st.mean(v), 'min': min(v), 'max': max(v), 'median': st.median(v)} if v else None
def provenance(p):
    p = Path(p); s = p.stat()
    return {'path': str(p), 'sha256': sha(p), 'mtime_utc': datetime.datetime.fromtimestamp(s.st_mtime, datetime.timezone.utc).isoformat(), 'bytes': s.st_size}
plan = read(P/'REVIEWER_MECHANISM_PLAN.json'); base = read(P/'base_config.json')
plans = {x['id']:x for x in plan['jobs']}; records = []; boundary_records = []
administrative = {'outputs_root','prefix','seed','optimizer_tag_override','save_task_checkpoint_dir'}
mechanism_keys = ['grad_fisher_to_clean_norm_ratio_mean','grad_cos_clean_gam_mean','grad_flat_to_clean_norm_ratio_mean','current_fisher_raw_trace','current_fisher_trace','fisher_past_trace_mass','delta_norm','fisher_weighted_drift','in_task_weight_merge_shift_l2','in_task_path_weight_effective_n']
for root, origin in [(O,'reused_native_sweep'),(P,'new_mechanisms')]:
    manifest = read(root/'source_manifest.json'); suite = read(root/'suite.json')
    jobs = {x['id']:x for x in suite['jobs']}
    for rel, digest in manifest['files'].items(): check(sha(root/'source'/rel)==digest,'source_files',str(root/'source'/rel))
    for row in read(root/'results.json')['jobs']:
        ident = row['id']; job = jobs[ident]; receipt_path = root/'runtime/jobs'/ident/'completed.json'; receipt = read(receipt_path)
        config = read(job['config']); raw = read(row['metrics_path']); mat = raw['cnn']['matrices']['final']
        check(row['status']==receipt['status']=='completed','completion_status',ident)
        check(len(mat)==10 and all(len(r)==10 for r in mat),'matrix_shape',ident)
        check(all(math.isfinite(mat[t][j]) and 0<=mat[t][j]<=100 for t in range(10) for j in range(t+1)),'matrix_lower_finite',ident)
        check(sha(row['metrics_path'])==row['metrics_sha256']==receipt['result']['metrics_sha256'],'metrics_hash',ident)
        check(sha(job['config'])==job['config_sha256']==row['config_sha256']==receipt['config_sha256'],'config_hash',ident)
        check(sha(root/'source_manifest.json')==receipt['source_manifest_sha256'],'manifest_hash',ident)
        check(receipt['source_files']==manifest['files'],'receipt_source_mapping',ident)
        check(sha(root/'suite.json')==receipt['suite_sha256'],'suite_hash',ident)
        for rel,digest in receipt['runtime_sha256'].items(): check(sha(root/rel)==digest,'runtime_hash',ident+':'+rel)
        for t in range(10):
            earlier = raw['cnn']['matrices'][f't{t:02d}']
            check(all(eq(earlier[i][j],mat[i][j]) for i in range(t+1) for j in range(i+1)),'intermediate_matrix_consistency',ident+':'+str(t))
        val = {'faa_task_macro':st.mean(mat[-1]),'aaa_task_macro':st.mean(st.mean(mat[t][:t+1]) for t in range(10)),
               'forgetting':st.mean(max(mat[t][j] for t in range(j,10))-mat[-1][j] for j in range(9)),
               'bwt':st.mean(mat[-1][j]-mat[j][j] for j in range(9))}
        for key,v in val.items(): check(eq(v,row[key]) and eq(v,receipt['result'][key]),'raw_metric_recompute',ident+':'+key)
        log_path=root/'runtime/jobs'/ident/'console.log'; log=log_path.read_text(errors='replace')
        curve_text=re.findall(r'CNN top1 curve:\s*(\[[^\r\n]*\])',log)[-1]
        curve=json.loads(re.sub(r'np\.float64\(([^()]*)\)',r'\1',curve_text))
        check(curve==row['cnn_top1_curve'] and eq(curve[-1],row['faa_sample_weighted']) and eq(st.mean(curve),row['aaa_sample_weighted']),'sample_weighted_curve',ident)
        check(eq(st.mean(mat[i][j] for i in range(10) for j in range(i+1)),raw['cnn']['final']['AAA']),'native_aaa_definition',ident)
        epochs={t:sorted({int(e) for tt,e in re.findall(r'Task (\d+), Epoch (\d+)/50',log) if int(tt)==t}) for t in range(10)}
        check(all(es==list(range(1,51)) for es in epochs.values()),'fifty_epoch_per_task',ident)
        check(not re.search(r'(?:Loss|CE|GAM)\s+(?:nan|[+-]?inf)(?:[,\s]|$)',log,re.I),'finite_logged_loss',ident)
        steps=raw['mechanism']['steps']; check(set(map(int,steps))==set(range(10)),'mechanism_task_coverage',ident)
        summaries={key:stats([float(s[key]) for k,s in steps.items() if int(k)>0 and s.get(key) is not None]) for key in mechanism_keys}
        diff={k:{'center':base.get(k),'actual':config.get(k)} for k in sorted(set(base)|set(config)) if base.get(k)!=config.get(k) and k not in administrative}
        if root==P:
            expected=dict(base,**plans[ident]['overrides']); expected['seed']=[plans[ident]['seed']]
            check(all(expected.get(k)==config.get(k) for k in set(expected)|set(config) if k not in administrative),'planned_scientific_config',ident)
        check(config['seed']==[row['seed']],'seed_identity',ident)
        records.append({'id':ident,'origin':origin,'seed':row['seed'],'status':row['status'], **val,
            'faa_sample_weighted':row['faa_sample_weighted'],'aaa_sample_weighted':row['aaa_sample_weighted'],
            'native_stored_aaa':raw['cnn']['final']['AAA'],'scientific_diff_vs_center':diff,'task1_to9_mechanism':summaries,
            'fisher_sources':sorted({str(s.get('current_fisher_source')) for s in steps.values()}),
            'fisher_estimator_family':sorted({str(s.get('fisher_estimator_family')) for s in steps.values()}),
            'metrics':provenance(row['metrics_path']),'receipt':provenance(receipt_path),'config':provenance(job['config']),
            'completed_50_epochs_tasks':len([t for t,v in epochs.items() if v==list(range(1,51))]),
            'source_manifest_sha256':receipt['source_manifest_sha256'],'environment':receipt['environment'],'native_flags':receipt['native_flags_before_main'],
            'torch_version':receipt['torch_version'],'cuda_version':receipt['cuda_version'],'gpu_name':receipt['gpu_name']})
        if root!=P: continue
        boundary_path=P/'boundary_analysis'/f'{ident}.json'; ba=read(boundary_path); hashes=read(P/'boundary_analysis'/f'{ident}.hashes.json')
        check(ba['passed'] and ba['boundary_count']==10 and ba['layer_row_count']==240,'boundary_analysis_receipt',ident)
        check(sha(hashes['utility'])==hashes['utility_sha256'],'boundary_utility_hash',ident)
        for f,d in hashes['outputs'].items(): check(sha(f)==d,'boundary_output_hash',f)
        for b in ba['boundaries']:
            f=b['source']; digest=sha(f)
            check(digest==b['input_sha256']==hashes['input_hashes'][f]==receipt['result']['boundary_validation']['files'][f],'boundary_input_hash',f)
            payload=torch.load(f,map_location='cpu',weights_only=False)
            endpoint=payload['endpoint_factors']; retained=payload['retained_factors']; selected=payload['selected_path_weights']
            applied=bool(payload['merge_metrics'].get('in_task_weight_merge_applied',False)); omega=b['omega']
            snapshots={int(s['index']):s for s in payload['snapshots']}; reconstructed=endpoint
            if applied:
                total=sum(x['weight'] for x in selected); path={}
                for k in endpoint:
                    value=snapshots[int(selected[0]['index'])]['factors'][k]*selected[0]['weight']
                    for x in selected[1:]: value=value+snapshots[int(x['index'])]['factors'][k]*x['weight']
                    path[k]=value/total
                reconstructed={k:v*(1-omega)+path[k]*omega for k,v in endpoint.items()}
            check(all(torch.equal(reconstructed[k],retained[k]) for k in endpoint),'independent_native_factor_reconstruction',f)
            check(payload['rng_before_boundary_sha256']==payload['rng_after_boundary_sha256'],'boundary_rng_unchanged',f)
            expected_merge=config['in_task_weight_merge'] and b['task']>=config['in_task_weight_merge_start_task']
            check(applied==expected_merge,'boundary_intervention',f)
            boundary_records.append({'id':ident,'task':b['task'],'sha256':digest,'bytes':Path(f).stat().st_size,'applied':applied,
                'selected_path_weights':b['selected_path_weights'],'effective_n':b['path_weight_entropy_effective_n'],
                'algebra_report_passed':b['cross_term_identities_passed'],'aggregate':b['aggregate']})
        print('audited',ident,flush=True)
lookup={r['id']:r for r in records}; centers={1:'s3_mech_center_s1',2:'s3_center_s2',3:'s3_mech_center_s3'}
pairs=[]
for seed,on_id in centers.items():
    off_id=f's3_mech_a_nomerge_matched_s{seed}'; on=lookup[on_id]; off=lookup[off_id]
    pairs.append({'seed':seed,'on_id':on_id,'off_id':off_id,'delta_faa':on['faa_task_macro']-off['faa_task_macro'],'delta_aaa':on['aaa_task_macro']-off['aaa_task_macro']})
def paired_summary(rows):
    n=len(rows); critical={2:12.706204736432095,3:4.302652729911275}[n]; result={'n':n,'df':n-1,'rows':rows}
    for key,margin in [('delta_faa',.5),('delta_aaa',.3)]:
        v=[r[key] for r in rows]; mean=st.mean(v); sd=st.stdev(v); half=critical*sd/math.sqrt(n); ci=[mean-half,mean+half]
        result[key]={'mean':mean,'sample_sd':sd,'paired_t95':ci,'margin_pp':margin,'all_positive':all(x>0 for x in v),'positive_supported':all(x>0 for x in v) and ci[0]>0,'equivalence_by_preregistered_95ci_rule':ci[0]>=-margin and ci[1]<=margin}
    return result
center=lookup['s3_center_s2']; contrasts=[]
for r in records:
    if r['seed']!=2: continue
    g=r['task1_to9_mechanism']['grad_fisher_to_clean_norm_ratio_mean']; cg=center['task1_to9_mechanism']['grad_fisher_to_clean_norm_ratio_mean']['mean']
    ratio=g['mean']/cg if g else None
    contrasts.append({'id':r['id'],'delta_faa':r['faa_task_macro']-center['faa_task_macro'],'delta_aaa':r['aaa_task_macro']-center['aaa_task_macro'],'ewc_clean_gradient_ratio_relative_to_center':ratio})
boundary_summary={}
for ident in plans:
    rows=[b for b in boundary_records if b['id']==ident]
    keys=['factor_retained_endpoint_drift_relative','dw_retained_endpoint_drift_relative','total_gap_relative_to_linear_dw','total_gap_relative_to_retained_drift','total_identity_error_l2']
    boundary_summary[ident]={'boundaries':len(rows),'applied':sum(b['applied'] for b in rows),'effective_n':stats([b['effective_n'] for b in rows if b['effective_n'] is not None]),
        **{k:stats([b['aggregate'][k] for b in rows if b['aggregate'][k] is not None]) for k in keys},
        'selected_epoch_weight_examples':rows[0]['selected_path_weights']}
man_new=read(P/'source_manifest.json'); man_old=read(O/'source_manifest.json')
check(all(man_new['files'].get(k)==v for k,v in man_old['files'].items()),'parent_source_unchanged','all old source files preserved')
smoke=read(P/'preflight/smoke_review.json'); witness=read(P/'preflight/NATIVE_BOUNDARY_WITNESS.json')
check(smoke['passed'] and smoke['verified_smokes']==20,'smoke20','20 native smokes')
check(sha(P/'preflight/NATIVE_BOUNDARY_WITNESS.json')==smoke['observer_neutrality']['native_boundary_witness_sha256'],'witness_hash','same-state native witness')
check(witness['passed'] and witness['extra_forward_count']==witness['extra_fisher_count']==0,'same_state_witness','no added calls')
backup=P/'analysis_corrections/baseline_identity_20260921T150248Z'
provenance_paths=[P/'REVIEWER_MECHANISM_PLAN.md',P/'REVIEWER_MECHANISM_PLAN.json',P/'suite.json',P/'base_config.json',P/'source_manifest.json',O/'source_manifest.json',P/'summarize_mechanisms.py',P/'mechanism_summary.json',P/'MECHANISM_RESULTS.md',P/'SUMMARY_CORRECTION.json',backup/'summarize_mechanisms.py',backup/'mechanism_summary.json',P/'preflight/smoke_review.json',P/'preflight/NATIVE_BOUNDARY_WITNESS.json',P/'PIPELINE_STATUS.json']
result={'audit_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only_sources':True,'new_gpu_jobs':0,
    'passed_raw_integrity':not issues,'checks':dict(checks),'issues':issues,'counts':{'new_completed':18,'reuse_completed':8,'new_failed':0,'new_running':0,'new_pending':0,'boundary_archives':len(boundary_records),'boundary_bytes':sum(b['bytes'] for b in boundary_records)},
    'protocol':base,'records':records,'seed2_contrasts':contrasts,'preregistered_fusion_three_seeds':paired_summary(pairs),
    'unregistered_two_seed_sensitivity':paired_summary([p for p in pairs if p['seed'] in (2,3)]),
    'boundary_summary':boundary_summary,'boundary_raw_reconstruction_checks':boundary_records,
    'observer_neutrality':smoke['observer_neutrality'],'witness_scope_limits':witness['scope_limits'],
    'source_provenance':[provenance(p) for p in provenance_paths if p.exists()],
    'aggregation_correction':{'old_bug':'All s3_ overrides were dropped; the endpoint Fisher intervention was misidentified as center and overwrote seed2 on_id.',
        'correct_center_ids':centers,'current_source_has_explicit_mapping':True,
        'cohort_warning':'Current summarizer labels seeds2/3 primary and seed1 supplementary, whereas unchanged preregistration specifies seeds1/2/3 primary. No validity-exclusion rationale found in correction note; three-seed analysis is retained.'},
    'optional_confirmation':{'isotropic_alpha1_gradient_ratio_change_factor':1/next(c['ewc_clean_gradient_ratio_relative_to_center'] for c in contrasts if c['id']=='s3_mech_fisher_alpha1_s2'),
        'damping_replicates_triggered_if_twofold_means_bidirectional_factor_change':['alpha0_seed1','alpha0_seed3','alpha1_seed1','alpha1_seed3'],
        'not_launched_by_audit':True},
    'limits':['Only fusion has three paired seeds; other contrasts are exploratory seed2.',
        'Accuracy inputs are stored to 0.01pp per task; derived three-decimal differences and t intervals inherit this rounding.',
        'CPU audit independently reconstructs all native factors; dense cross-term algebra identities are checked via immutable offline-analysis receipts, not all independently recomputed here.',
        'Cross-run bitwise equality failed in smoke; same-state observer neutrality and all formal boundaries passed.',
        'No paired endpoint/retained Fisher cosine, best-rank10 SVD residual, or fixed-batch counterfactual logit difference was collected.',
        'Fisher is batch-gradient delta-W proxy in ImageNet-A native protocol; do not pool with the new RQ2 FP32/per-sample-Fisher protocol, the old ImageNet-R table, or the separate 20260918 sample_empirical studies.',
        'No raw source, training, configuration, receipt or CopyFlatCV summary was modified by this audit.']}
center_boundaries=[b for b in boundary_records if b['id'] in ['s3_mech_center_s1','s3_mech_center_s3']]
result['quality_weight_diagnostics']={
    'fresh_center_boundary_count':len(center_boundaries),
    'largest_weight_epoch_counts':dict(collections.Counter(max(b['selected_path_weights'],key=lambda x:x['weight'])['epoch'] for b in center_boundaries)),
    'standardized_two_snapshot_weight_rule':'For two unequal losses, population-standardized z scores approach +/-1, so beta=3 yields softmax(-3,+3) ~= (0.002472623,0.997527377), regardless of nonzero loss-gap magnitude.',
    'center_effective_n':stats([b['effective_n'] for b in center_boundaries]),
    'center_cross_term_relative_to_linear_dw':stats([b['aggregate']['total_gap_relative_to_linear_dw'] for b in center_boundaries]),
    'center_cross_term_relative_to_retained_drift':stats([b['aggregate']['total_gap_relative_to_retained_drift'] for b in center_boundaries]),
    'near_tie_flip':[b for b in center_boundaries if max(b['selected_path_weights'],key=lambda x:x['weight'])['epoch']==20],
    'seed2_old_center_has_observer_archives':False}
(N/'S3_FINAL_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
lines=[
    '# S3 机制实验：独立最终审查', '',
    f"审查时间：{result['audit_time_utc']}。审查只读取 CopyFlatCV，写入本 O-LoRA 目录；未启动 GPU，未改源代码/配置/运行/原结果。",'',
    '**18 个新增核心实验、8 个复用实验均完整完成，原始结果完整性检查通过。** 三 seed 的融合增益较小，尚不能声称其 FAA 正贡献或 FAA 实用等效；AAA 按预设区间标准等效。GAM 主训练路径贡献明显的证据来自 seed2，Fisher 多个准确率消融接近中心。另发现已有汇总器修正 seed2 误配时把主 cohort 改成 seeds2/3；该缩减没有预注册依据或 seed1 失效凭据，本审查保留原 seeds1/2/3 主分析。','',
    '## 完整性与口径','',
    '- 26 份 raw 10×10 CNN 矩阵、260 份中间阶段矩阵与最终矩阵一致；FAA、AAA、BWT、forgetting 共104个量从 raw 独立重算。26条 sample-weighted 曲线另从训练日志核验。',
    '- 26份 completed receipt、config、suite、source manifest 及122个运行文件哈希引用均通过；190份原生代码完整保留，新source仅增加observer/control模块，共191份。没有 seed1 特有冻结协议或环境不一致。',
    '- 26作业均有 task0…9 × epoch1…50 的完整日志，无非有限训练 loss；不是只凭最后checkpoint判定完成。20/20预检通过，正式18/18完成，0运行/排队/失败。',
    '- 新18作业的180份边界archive共1,043,964,880字节全部核SHA，独立按原dtype/顺序重构48个因子张量，逐张量bitwise相等；180/180 observer前后RNG不变、干预开关符合配置。dense代数报告的哈希/身份/240×18层行通过，未在本审查重复全部dense乘法。',
    '- 同状态 native observer witness 六个案例通过，0额外 forward/Fisher；跨两次GPU执行的完整轨迹bitwise相等曾失败，原因未解决，不能把同状态中性证明表述成跨运行完全可复现。',
    '- FAA=最终10任务准确率均值；AAA=10个阶段各自已见任务准确率均值再平均。原raw `cnn.final.AAA` 是55个下三角cell直接平均（阶段权重不同），不能和本表AAA混写。中心seed2原字段60.624，本表阶段等权AAA66.745541。',
    '- 此处是 ImageNet-A、200类、10×20任务、ViT-B/16、LoRA rank10、50轮、batch128、8workers、初始/后续lr .04/.02、无replay；Fisher为dense ΔW空间的batch-gradient平方proxy，原生增强/loader/RNG。与RQ2新FP32/逐样本Fisher协议、旧ImageNet-R表及20260918 sample-empirical/batch32/RNG隔离协议分开。','',
    '## 原预注册的三 seed 融合配对','',
    '`center − matched_no_merge`。后者保留20/40轮snapshot和质量CE前向预算，将merge start设为task10，十任务内不融合。差值单位均为百分点；±为样本SD。','',
    '| Seed | Center FAA | Matched no-merge FAA | ΔFAA | ΔAAA |', '| --- | ---: | ---: | ---: | ---: |']
for p in pairs:
    lines.append(f"| {p['seed']} | {lookup[p['on_id']]['faa_task_macro']:.3f} | {lookup[p['off_id']]['faa_task_macro']:.3f} | {p['delta_faa']:+.3f} | {p['delta_aaa']:+.6f} |")
lines += ['', '| 量 | 配对均值 ± SD | 95% paired t CI, df=2 | 预设等效界 | 判定 |','| --- | ---: | --- | --- | --- |']
for key,label,verdict in [('delta_faa','ΔFAA','正贡献证据不足；等效未成立'),('delta_aaa','ΔAAA','满足预设实用等效区间')]:
    s=result['preregistered_fusion_three_seeds'][key]
    lines.append(f"| {label} | {s['mean']:+.6f} ± {s['sample_sd']:.6f} | [{s['paired_t95'][0]:+.6f}, {s['paired_t95'][1]:+.6f}] | ±{s['margin_pp']:.1f} | {verdict} |")
lines += ['', '三组FAA均为正，但95%区间跨0，未满足预注册正贡献条件。FAA区间上限0.506702略超过0.5，不能四舍五入后宣布等效。AAA区间完全位于±0.3内。只有3个配对种子，且raw每任务准确率存到0.01pp；区间依赖小样本t模型，应保留这些限制。','',
    '## 汇总错误与未经说明的 cohort 变动','',
    '原汇总器 `summarize_mechanisms.py` 丢弃所有 `s3_` 前缀override，连科学干预 `s3_fisher_location=endpoint` 也被丢掉。因此endpoint分支被误认作无override中心并覆盖seed2的on_id。正确seed2中心是预注册复用 `s3_center_s2`（FAA53.419），不是endpoint-Fisher（53.454）。错误影响融合聚合，未污染26份训练raw。',
    '原错误三seed ΔFAA=+0.172±0.159603，正确为+0.160333±0.139432；错误AAA区间曾越过+0.3，正确AAA区间在±0.3内，等效判断会改变。',
    '当前CopyFlatCV汇总器于2026-09-21 15:01:34 UTC已被其他流程改为显式ID，`SUMMARY_CORRECTION.json` 时间15:02:48 UTC。该说明只解释误配，没有说明为何另将seeds2/3改称primary、seed1称legacy/supplementary。原计划（mtime 2026-09-20 20:35:18 UTC）仍明确seeds1/2/3；全部seed1 config/source/runtime/receipt检查正常，未找到基于有效性的排除证据。本审查没有改该文件，不能推断未知修改者动机，也不能把新的结果后cohort标签当预注册。',
    '保留2/3作为敏感性展示：ΔFAA=+0.196000±0.176777，df1 t95=[−1.392276,+1.784276]；ΔAAA=+0.056698±0.004772，t95=[+0.013824,+0.099571]。删去seed1会改变AAA的符号/区间解释，故主结论仍采用预注册三seed。当前与备份代码SHA、mtime、计划及receipt路径见JSON `source_provenance`。','',
    '## Seed2 完整结果与机制对照','',
    '以下全部是单seed探索性控制，Δ均相对同协议中心53.419/66.745541；不与三seed均值混为一列。EWC/clean为task1…9的每任务平均梯度范数比再平均，百分比表示。SGD路径没有记录这一GAM专属梯度统计，记为未测。','',
    '| 实验ID | FAA | AAA阶段均值 | ΔFAA | ΔAAA | EWC/clean % |','| --- | ---: | ---: | ---: | ---: | ---: |']
for r in records:
    if r['seed']!=2: continue
    g=r['task1_to9_mechanism']['grad_fisher_to_clean_norm_ratio_mean']; gs=f"{100*g['mean']:.6f}" if g else '未测'
    lines.append(f"| {r['id']} | {r['faa_task_macro']:.3f} | {r['aaa_task_macro']:.6f} | {r['faa_task_macro']-center['faa_task_macro']:+.3f} | {r['aaa_task_macro']-center['aaa_task_macro']:+.6f} | {gs} |")
lines += ['', '另外四份非seed2新训练（均完成）：','', '| 实验ID | Seed | FAA | AAA阶段均值 |','| --- | ---: | ---: | ---: |']
for r in records:
    if r['seed']!=2: lines.append(f"| {r['id']} | {r['seed']} | {r['faa_task_macro']:.3f} | {r['aaa_task_macro']:.6f} |")
lines += ['',
    'GAM：true single-forward SGD+EWC为40.123，中心高13.296pp；半完整GAM方向为51.834，低中心1.585pp，但高SGD11.711pp。支持该seed下完整GAM训练路径承担主要准确率差异，不能凭它证明泛化上的普适机制。旧lambda_flat=0仍执行GAM探测，只令组合 `g_clean + lambda_flat(g_GAM−g_clean)` 回到clean方向，结果38.036，不等于trueSGD；lambda_flat=.5也不等于将整个GAM方向乘.5。projection关闭只降.158pp，第一扰动rho减半降.553pp；第二扰动radius=.5的+1.559pp只是旧单seed敏感性结果，不能升级为选优baseline。',
    'Fisher：lambda从0到20000的FAA落在53.405…53.500；阻尼α0/1、history γ0/1、task normalization off、endpoint-Fisher的FAA相对中心均在−.189…+.155pp，未支持该工作点下EWC必需或某个Fisher设计提高准确率。中心平均EWC/clean仅.144108%，clean/GAM余弦.952869，flat/clean范数比1.877472。α1虽准确率近似，却将EWC/clean降到.002803%，说明trace匹配并不匹配沿实际更新方向的梯度约束，不能写成Fisher几何完全无变化。',
    '融合执行：whole-module-off相对matched-no-merge，FAA−.948pp、AAA+.471690pp。这项比较同时改变snapshot/质量前向与RNG消费；不能把其差值当纯权重融合贡献。','',
    '## Fisher、参数位移与质量权重到底测了什么','',
    '各任务raw `mechanism.steps` 已记录：current_fisher_raw_trace、归一化current_fisher_trace、累计fisher_past_trace_mass、ΔW delta_norm、fisher_weighted_drift、EWC penalty与梯度比、因子融合shift、snapshot CE与权重/effective_n。中心seed2 task1…9 raw Fisher trace均值17.644410（10.387028…28.287242），归一化trace≈1，累计trace mass从1.9到6.513216；ΔW范数均值.221211，Fisher二次位移均值1.884603e−7，因子merge shift均值.010363。这些分别是不同量，不能将归一化trace、累计mass或因子移动当作raw Fisher trace/新任务有效ΔW范数。',
    'Observer另保存endpoint/retained因子、20/40轮snapshot、实际选择权重。新center seeds1/3共20个边界中，19次主要选epoch40；seed3 task2的训练CE=.5078422/.5080230，几乎打平但epoch20更低，于是其权重约.997527。这说明不能说权重总是选末个snapshot。',
    '代码对两个CE先以population std标准化，再softmax(−βz)，β=3。只要差值非零且大于epsilon，两项z近似±1，权重便约.0024726/.9975274，effective_n≈1.017462，与CE差距大小几乎无关。uniform有效数2，latest有效数1；这是质量权重非常尖锐的直接实现/数值证据，不是其带来准确率收益的证明。','',
    '| 边界集合 | median 因子融合相对移动 % | median ΔW相对移动 % | median 交叉项/线性ΔW % | max 交叉项/线性ΔW % | median 交叉项/实际移动 % |',
    '| --- | ---: | ---: | ---: | ---: | ---: |']
for ident in ['s3_mech_center_s1','s3_mech_center_s3','s3_mech_a_uniform_s2','s3_mech_a_latest_s2']:
    b=boundary_summary[ident]
    vals=[b['factor_retained_endpoint_drift_relative']['median'],b['dw_retained_endpoint_drift_relative']['median'],b['total_gap_relative_to_linear_dw']['median'],b['total_gap_relative_to_linear_dw']['max'],b['total_gap_relative_to_retained_drift']['median']]
    lines.append('| '+ident+' | '+' | '.join(f'{100*v:.6f}' for v in vals)+' |')
lines += ['',
    '边界代数证明因子平均并不严格等于ΔW平均：`B̄Ā−ΣwᵢBᵢAᵢ = −Σᵢ<ⱼwᵢwⱼ(Bᵢ−Bⱼ)(Aᵢ−Aⱼ)`；endpoint/path因子插值还增加 `−ω(1−ω)(B̄−Bₑ)(Ā−Aₑ)`，并有原dtype舍入残差。现有报告的恒等式残差约1e−16。上表分母均明确，交叉项相对完整ΔW很小，但相对实际位移可更大。旧seed2中心没有新observer archive，不能把endpoint干预seed2的archive冒充中心，也不能声称三seed中心均做了这个dense边界审查。',
    '未测：同一checkpoint/loader的endpoint与retained配对Fisher cosine；best rank10 SVD残差；固定batch的counterfactual logit差；改成ΔW/SVD融合后的训练收益；未测GAM与Fisher所有多seed交互。observer实际不额外估Fisher、不额外forward，因此上述不能从当前archive报告凭空得到。','',
    '## 尚未执行的可选确认与结论边界','',
    '- uniform/latest没有达到|ΔFAA|≥.5或|ΔAAA|≥.3的复测阈值；endpoint-Fisher为+.035/+.104686pp，平均下一任务EWC/clean相对变化+1.203541%，未到10%阈值。',
    '- Fisher准确率阈值未触发。但α1的EWC/clean相对中心降低51.419119倍：按“changes … by at least 2x”的双向倍数变化解释，触发damping两端α0/α1各补seed1/3，共4个可选确认。若原规则仅指增加而不含降低，必须明确说明这一口径；不能事后默默选择。本审查未启动这些作业。',
    '- 中心交叉项/线性ΔW的median远低于1%，SVD训练触发条件第一项未满足；SVD残差与logit门槛本身也未测。',
    '- optional accumulated-normalization-off/alpha0未跑；它是历史相关scale变化，不是可由一项固定lambda替代的no-op。radius=.5复现和lambda0多seed只在要主张相应改善/必要性时按原计划追加，核心18完成不能写成所有可选已完成。',
    '- 可支持：指定协议/seed下GAM训练路径是主要准确率来源；融合配对FAA效应小且区间不确定、AAA按预设实用等效；Fisher消融在当前λ2000工作点准确率变化小；两snapshot权重几乎二选一、因子融合存在但较小的非线性ΔW交叉项。',
    '- 尚不支持：权重融合显著提高FAA，融合FAA已证明等效，Fisher普遍无效，shaping通过降低Fisher而必然降低遗忘，quality weighting优于uniform/latest，SVD/ΔW融合会更好，或与RQ2另一训练/测量协议或旧ImageNet-R数据拼表得出跨设置结论。','',
    '复核入口：本目录 `audit_s3_final.py` 可仅用CPU重跑；`S3_FINAL_AUDIT.json` 保留26条原始路径、receipt/config/source SHA、全部统计、180个边界重构记录及源文件mtime。']
(N/'S3_FINAL_AUDIT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'checks':dict(checks),'issues':issues,'paired':result['preregistered_fusion_three_seeds'],'two_seed':result['unregistered_two_seed_sensitivity'],'optional':result['optional_confirmation']},indent=2))
