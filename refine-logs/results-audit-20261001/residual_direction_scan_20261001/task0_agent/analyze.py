#!/usr/bin/env python3
"""Read-only source audit and within-anchor residual/interpolation scan tables."""
import csv, hashlib, json, math, statistics
from datetime import datetime, timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SRC=ROOT/'task0_diagnostics/final'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def writecsv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
d=read(SRC/'compact.json')
rows=list(csv.DictReader((SRC/'candidate_batches.csv').open()))
assert d['status']=='completed' and d['validated_anchors']==d['expected_anchors']==5
assert d['schema_mix']=={'schema1':4,'schema2':1} and len(rows)==500
anchors=d['anchors'];validation=[]
for a,l in zip(anchors,d['ledger']):
    raw=read(l['report']);assert sha(l['report'])==l['report_sha256']==a['report_sha256']
    assert raw['schema_version']==a['schema_version'] and raw['status']=='completed'
    assert raw['batch_count']==4 and raw['all_matches_successful'] and raw['finite_check']['ok']
    assert all(raw['final_restore_check'].values())
    assert a['candidate_count']==a['matched_count']==100
    rr=[r for r in rows if r['anchor']==a['anchor']]
    assert len(rr)==100 and {int(r['batch']) for r in rr}=={0,1,2,3}
    for b in raw['batches']:
        assert len(b['counterfactuals'])==25 and all(b['restore_check'].values())
        for c in b['counterfactuals']:
            assert c['match']['ok'] and c['match']['actual_relative_error']<=raw['arguments']['match_rtol']
            assert c['loss_change']['old_task_macro'] is None
            row=next(r for r in rr if int(r['batch'])==b['batch'] and r['candidate']==c['label'])
            assert float(row['current_delta_ce'])==c['loss_change']['current']['task_local_ce']
    validation.append({'anchor':a['anchor'],'schema_version':a['schema_version'],
        'status':'PASS','matched':100,'expected':100,'report':l['report'],'sha256':sha(l['report'])})
families={'g1_plus_residual_lambda_':('g1_residual',[0,.5,1,2]),
          'sum_family_lambda_':('sum_family',[0,.5,1,2]),
          'unit_interpolation_alpha_':('unit_interpolation',[0,.25,.5,.75,1])}
eps=1e-7
curves=[];batchbest=[];family_summary=[]
for a in anchors:
    aa=a['anchor'];ar=[r for r in rows if r['anchor']==aa]
    for prefix,(family,grid) in families.items():
        zs={int(r['batch']):float(r['current_delta_ce']) for r in ar if r['candidate']==prefix+'0'}
        cr=[]
        for coeff in grid:
            selected=[r for r in ar if r['candidate']==prefix+format(coeff,'g')]
            assert len(selected)==4
            actual=[float(r['current_delta_ce']) for r in selected]
            differences=[float(r['paired_delta_vs_g1']) for r in selected]
            versus0=[float(r['current_delta_ce'])-zs[int(r['batch'])] for r in selected]
            assert all(math.isfinite(x) for x in actual+differences+versus0)
            row={'anchor':aa,'schema_version':a['schema_version'],'family':family,'coefficient':coeff,
                'probe_batches':4,'mean_delta_ce':statistics.mean(actual),
                'mean_vs_g1_ce':statistics.mean(differences),'min_vs_g1_ce':min(differences),'max_vs_g1_ce':max(differences),
                'better_batches_vs_g1':sum(x < -eps for x in differences),
                'tied_batches_vs_g1':sum(abs(x)<=eps for x in differences),
                'worse_batches_vs_g1':sum(x > eps for x in differences),
                'mean_vs_family_zero_ce':statistics.mean(versus0),
                'better_batches_vs_family_zero':sum(x < -eps for x in versus0)}
            curves.append(row);cr.append(row)
            ref=a['candidates'][prefix+format(coeff,'g')]
            assert math.isclose(row['mean_vs_g1_ce'],ref['paired_vs_g1']['mean'],abs_tol=1e-12)
        best=min(cr,key=lambda r:r['mean_delta_ce'])
        perbatch=[]
        for b in range(4):
            br=[r for r in ar if int(r['batch'])==b and r['candidate'].startswith(prefix)]
            br.sort(key=lambda r:float(r['candidate'][len(prefix):]))
            winner=min(br,key=lambda r:float(r['current_delta_ce']))
            coef=float(winner['candidate'][len(prefix):])
            strict_interior=coef not in (grid[0],grid[-1]) and float(winner['current_delta_ce'])<min(float(br[0]['current_delta_ce']),float(br[-1]['current_delta_ce']))-eps
            rec={'anchor':aa,'schema_version':a['schema_version'],'family':family,'batch':b,
                 'grid_minimizer':coef,'minimizer_vs_g1_ce':float(winner['paired_delta_vs_g1']),
                 'strict_interior_grid_minimum':strict_interior}
            batchbest.append(rec);perbatch.append(rec)
        family_summary.append({'anchor':aa,'schema_version':a['schema_version'],'family':family,
             'mean_curve_best_coefficient':best['coefficient'],
             'mean_curve_monotone_nondecreasing':all(cr[i+1]['mean_delta_ce'] >= cr[i]['mean_delta_ce']-eps for i in range(len(cr)-1)),
             'nonzero_mean_improvement_over_g1':any(r['coefficient']>0 and r['mean_vs_g1_ce'] < -eps for r in cr),
             'strict_interior_batch_count':sum(r['strict_interior_grid_minimum'] for r in perbatch),
             'batch_grid_minimizers':[r['grid_minimizer'] for r in perbatch]})
zero_noise=max(abs(float(r['paired_delta_vs_g1'])) for r in rows if r['candidate'] in ('sum_family_lambda_0','unit_interpolation_alpha_0'))
selected=[r for r in rows if any(r['candidate'].startswith(prefix) for prefix in families)]
assert len(selected)==260 and len(curves)==65
writecsv('curve_summary.csv',curves);writecsv('candidate_batches.csv',selected);writecsv('batch_grid_minima.csv',batchbest)
result={'status':'PASS','audited_at_utc':datetime.now(timezone.utc).isoformat(),'validated_anchors':5,'expected_anchors':5,
    'all_matched_candidates':500,'all_expected_candidates':500,'selected_scan_candidates':260,
    'schema_mix':d['schema_mix'],'validation':validation,'input_sha256':{str(SRC/f):sha(SRC/f) for f in ['compact.json','candidate_batches.csv']},
    'training_seed':2,'old_task_metrics':'NA','no_pooled_anchor_mean':True,'batch_is_not_seed':True,
    'win_loss_tolerance_ce':eps,'collinear_zero_control_maxabs_ce_difference':zero_noise,
    'family_summary':family_summary,'curves':curves,'batch_minima':batchbest,
    'limitations':['offline one-step counterfactual, not independent training sweep','virtual eta .02; eval subset 128, not full 307',
      'Mean schema2 has actual-probe self-spec and GPU instrumentation sensitivity up to .413%; other four schema1',
      'tested positive coefficient grid only; no global optimum or uncertainty estimate']}
(OUT/'compact.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
lines=['# Task0 残差系数与方向插值：已有离线结果', '',
    '审计 PASS：5/5 起点有效，全部 500/500 候选实际 BA+FC 步幅匹配成功；本报告提取其中 260 个残差/插值候选（65 个起点×系数单元，每单元4个固定probe batch）。不是新增独立训练的超参数扫描。', '',
    '四个 schema1 起点为 S0、S50、G50、Sum50；Mean50 使用恢复 schema2 自身探针/CE/操作校验。旧 Mean partial 被排除。CPU 对照逐位相同，但 GPU 插桩 g3 相对差可达约0.413%；保持版本分层，不把五个起点混成均值。只使用 seed2，batch 不是 seed，所有旧任务指标均 NA。', '',
    '## 扫描定义', '',
    '在每个checkpoint、每个batch重新计算 g3=a g1+q，其中 q 与 g1 正交。所有候选均缩放到 −.02 g1 所对应的实际 concat(ΔBA,ΔFC) 更新范数；.02 是虚拟诊断步长。当前CE在同一个128样本评测子集上测量，不是最终训练表的全部307样本。', '',
    '- A：g1+λq，λ∈{0,.5,1,2}。λ=1 不等于 g3；g3=a g1+q，a随batch变化。',
    '- B：(1+a)g1+λq，同一λ网格。λ=1 才是 g1+g3，λ=0与g1正向共线。A/B的同一λ不是相同夹角或残差比例。',
    '- C：u((1−α)u(g1)+αu(g3))，α∈{0,.25,.5,.75,1}，u表示单位化。α=0是g1方向，α=1是g3方向。', '',
    '## 逐起点精确比较', '',
    '**表格单位为 (候选ΔCE−g1ΔCE)×1000，不是准确率百分点。负值表示候选更好。括号为 better/tie/worse batch 数，阈值±1e−7 CE；仅描述固定批次，不是统计显著性。**', '',
    f'λ/α=0共线控制也会因数值求根及浮点运算出现微小差异：逐batch最大绝对CE差 {zero_noise:.9g}。零系数处的微小“胜负”不能当方向收益。', '']
for family,title in [('g1_residual','A：g1+λq'),('sum_family','B：(1+a)g1+λq'),('unit_interpolation','C：单位方向插值')]:
    grid=next(v[1] for v in families.values() if v[0]==family)
    lines += ['### '+title,'','| 起点 / schema | '+' | '.join(format(x,'g') for x in grid)+' |','|---|'+'---:|'*len(grid)]
    for a in anchors:
        cells=[]
        for coeff in grid:
            r=next(r for r in curves if r['anchor']==a['anchor'] and r['family']==family and r['coefficient']==coeff)
            cells.append(f"{1000*r['mean_vs_g1_ce']:+.6f} ({r['better_batches_vs_g1']}/{r['tied_batches_vs_g1']}/{r['worse_batches_vs_g1']})")
        lines.append(f"| {a['anchor']} / v{a['schema_version']} | "+' | '.join(cells)+' |')
    lines.append('')
lines += ['## 最优点与batch异质性', '', '| 起点 | 家族 | 平均曲线最优系数 | batch0/1/2/3的网格最优 | 严格内部最优batch数 |', '|---|---|---:|---|---:|']
for r in family_summary:
    lines.append(f"| {r['anchor']} | {r['family']} | {r['mean_curve_best_coefficient']:g} | "+'/'.join(format(x,'g') for x in r['batch_grid_minimizers'])+f" | {r['strict_interior_batch_count']} |")
lines += ['', '## 可支持与不能支持', '',
    '1. 在已测正系数网格、每个起点自己的四batch均值上，三个家族都以系数0最好；没有任一非零系数改善平均CE。单位插值在五个起点均随α增大而平均CE变差；这不意味着每个batch都单调。',
    '2. 不能说残差始终有害。S50 的 A家族 λ=2 在3/4个batch优于g1，但均值仍更差：少数batch的损害可以超过多数batch的改善。S0的B家族λ=.5及插值α=.25在batch3出现内部网格最优；S50的A家族在batch3以λ=1最好。它们不是跨seed最优，也不是可凭当前评测集选定的训练超参数。',
    '3. 不能声称存在起点平均意义的内部最优，也不能由这组正系数网格推断连续全局最优或负残差方向。S50的A家族均值不单调：λ=2比λ=.5/1损失小，但仍不如λ=0。',
    '4. 这些是末端/初始化状态的一步、等有效步幅反事实；不能推出残差在完整50epoch训练无用，不能用来解释全部长期收益，也没有旧任务遗忘/冲突结论。它仅不支持在这些起点中添加正残差具有普遍局部平均优势。',
    '5. 固定原始g1/g3方向后的线性组合，与重新以不同r形成g3，是不同控制；本报告不把两者合并。', '',
    '追溯：curve_summary.csv保留未缩放原始CE、相对g1及相对各家族零系数的差值；candidate_batches.csv为260个选中候选逐batch数据；batch_grid_minima.csv记录各batch网格最优；compact.json记录原结果哈希和有效性。未训练、未修改旧输出。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':'PASS','anchors':5,'matched':500,'selected_scan_batches':260,'zero_control_ce_noise_max':zero_noise,'output':str(OUT)}))
