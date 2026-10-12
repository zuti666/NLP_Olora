"""CPU-only analysis of immutable state-formation records; no model execution."""
from pathlib import Path
import argparse, collections, csv, hashlib, json, math, statistics, sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OLD = Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z')
HOME = Path(__file__).resolve().parent
OUT = HOME / 'results/state_analysis'
SEEDS = [0, 1, 2]
BRANCHES = ['S', 'R']
EPS = 1e-30

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def write_json(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,keys); w.writeheader(); w.writerows(rows)
def summary(v):
    return {'n':len(v),'values':v,'mean':statistics.mean(v),'sample_sd':statistics.stdev(v) if len(v)>1 else None}
def fs(v,d=4):return f"{v['mean']:.{d}g}±{v['sample_sd']:.{d}g}" if v['sample_sd'] is not None else str(v['mean'])
def median(v):return statistics.median(v)
def stage(e):return 'probe_1_25' if e<=25 else 'common_sgd_26_50'
def check_finite(d,path='root'):
    if isinstance(d,float): assert math.isfinite(d),path
    elif isinstance(d,dict):
        for k,v in d.items():check_finite(v,path+'.'+k)
    elif isinstance(d,(list,tuple)):
        for i,v in enumerate(d):check_finite(v,path+f'[{i}]')

def signed_projection_fractions(net, first, second):
    """Signed projections onto the net, not additive norm shares or causal effects."""
    if net<=EPS:return None,None
    firstfrac=(net*net+first*first-second*second)/(2*net*net)
    return firstfrac,1-firstfrac

def lora_product_inner(b,a,d,c):
    return float(((b.T@d)*(a@c.T)).sum())
def lora_difference_norm(bs,as_,br,ar):
    sq=lora_product_inner(bs,as_,bs,as_)+lora_product_inner(br,ar,br,ar)-2*lora_product_inner(bs,as_,br,ar)
    return math.sqrt(max(0,sq))

def main():
    torch.set_num_threads(2); OUT.mkdir(parents=True,exist_ok=True)
    manifest={}; ledger=[]; curves={}; factors={}; traces={}; decomps={}; steps=[]; layers=[]; epoch_states=[]
    def input_file(p):
        p=Path(p);manifest[str(p)]=sha(p);return p
    state=read(input_file(OLD/'runtime/queue_state.json'))
    for seed in SEEDS:
        pair={}
        for branch in BRANCHES:
            job=f'formal_{branch}_s{seed}'; folder=OLD/'outputs'/job
            summ=read(input_file(folder/'final_summary.json'))
            assert state['jobs'][job]['status']=='completed'
            assert summ['status']=='completed' and summ['smoke'] is False and summ['branch']==branch and summ['seed']==seed
            assert summ['completed_epochs']==50 and summ['completed_steps']==400
            assert summ['epochs_per_task']==50 and summ['switch_after_epoch']==25
            assert summ['counters']['fisher_estimation_passes']==summ['counters']['weight_merge_updates']==summ['counters']['training_hvp_passes']==0
            trace=[json.loads(s) for s in input_file(folder/'step_trace.jsonl').read_text().splitlines()]
            ff=read(input_file(folder/'factor_state_curve.json'))['epochs']
            dd=read(input_file(folder/'factor_update_decomposition.json'))['steps']
            assert len(trace)==400 and len(ff)==50 and len(dd)==100
            assert set(x['epoch'] for x in ff)==set(range(1,51))
            assert all(len(x['layers'])==24 and set(y['layer'] for y in x['layers'])==set(range(24)) for x in ff+dd)
            for obj in [summ,trace,ff,dd]:check_finite(obj)
            for x in trace:
                assert x['active_method']==(('sam' if branch=='S' else 'sf_g2') if x['epoch']<=25 else 'sgd')
                assert x['natural_sgd_exact'] is True
                assert x['hvp_calls']==x['projection_calls']==x['fisher_estimation_passes']==x['weight_merge_updates']==0
            pair[branch]=summ
            ledger.append(dict(seed=seed,branch=branch,validated=True,epochs=len(ff),updates=len(trace),observed_update_batches=len(dd),all_layer_update_records=len(dd)*24,all_layer_state_records=len(ff)*24))
            curves[seed,branch]={x['epoch']:x for x in summ['retention_curve']}
            factors[seed,branch]={x['epoch']:x for x in ff}
            traces[seed,branch]=trace;decomps[seed,branch]=dd
            tmap={(t['epoch'],t['batch']):t for t in trace}
            for x in dd:
                tm=tmap[x['epoch'],x['batch']]
                lr=tm['current_lr']
                derived=[]
                for a in x['layers']:
                    na,nb,nc=a['T_A_norm'],a['T_B_norm'],a['T_cross_norm']
                    first=math.sqrt(max(0,na*na+nb*nb+2*na*nb*(a['T_A_T_B_cosine'] or 0)))
                    q1,q2=signed_projection_fractions(a['delta_q_fro'],a['delta_q_first_fro'],a['delta_q_second_fro'])
                    w1,wc=signed_projection_fractions(a['actual_delta_W_norm'],first,nc)
                    z=dict(seed=seed,branch=branch,stage=stage(x['epoch']),epoch=x['epoch'],batch=x['batch'],lr=lr,**a,
                           Q_first_over_second=a['delta_q_first_fro']/(a['delta_q_second_fro']+EPS),
                           Q_first_signed_net_projection=q1,Q_second_signed_net_projection=q2,
                           W_linear_norm=first,W_cross_over_net=nc/(a['actual_delta_W_norm']+EPS),
                           W_linear_signed_net_projection=w1,W_cross_signed_net_projection=wc,
                           W_linear_over_parameter_step=first/(tm['actual_parameter_step_norm']+EPS))
                    layers.append(z);derived.append(z)
                act=x['actual_update']
                st=dict(seed=seed,branch=branch,stage=stage(x['epoch']),epoch=x['epoch'],batch=x['batch'],lr=lr,
                        lora_effective_norm=act['lora_norm'],head_norm=act['fc_norm'],joint_norm=act['joint_norm'],parameter_norm=act['parameter_norm'],
                        head_squared_share=act['fc_norm']**2/(act['joint_norm']**2+EPS),
                        head_over_lora=act['fc_norm']/(act['lora_norm']+EPS),
                        final_to_base_gradient_ratio=tm['u_g_ratio'],
                        delta_q_identity_state_scale_error_max=max(y['delta_q_identity_state_scale_error'] for y in derived))
                for k in ['pi_A','T_A_T_B_cosine','delta_q_first_fro','delta_q_second_fro','Q_first_over_second','Q_first_signed_net_projection','Q_second_signed_net_projection','W_cross_over_net','W_cross_signed_net_projection']:
                    vals=[z[k] for z in derived if z[k] is not None];st['layer_median_'+k]=median(vals);st['layer_mean_'+k]=statistics.mean(vals)
                steps.append(st)
            for f in ff:
                act=f['effective_displacement']; epoch_states.append(dict(seed=seed,branch=branch,epoch=f['epoch'],stage=stage(f['epoch']),clean_training_ce=f['clean_training_ce'],
                    qhat_median=f['cross_layer_qhat']['median'],qhat_change_median=f['cross_layer_qhat_change']['median'],qhat_q25=f['cross_layer_qhat']['q25'],qhat_q75=f['cross_layer_qhat']['q75'],
                    q_change_fro_median=f['cross_layer_q_change_fro']['median'],lora_displacement=act['lora_norm'],head_displacement=act['fc_norm'],
                    head_squared_share=act['fc_norm']**2/(act['joint_norm']**2+EPS)))
        assert pair['S']['common_start_sha256']==pair['R']['common_start_sha256']
        assert all(a['identity']==b['identity'] for a,b in zip(traces[seed,'S'],traces[seed,'R']))
    # Seed is the independent unit: median across 24 layers per batch, then mean
    # across two fixed batches and 25 epochs; no repeated-layer significance test.
    seed_stage=[]
    metrics=[k for k in steps[0] if k not in ['seed','branch','stage','epoch','batch']]
    for seed in SEEDS:
        for branch in BRANCHES:
            for ph in ['probe_1_25','common_sgd_26_50']:
                rr=[r for r in steps if r['seed']==seed and r['branch']==branch and r['stage']==ph]
                assert len(rr)==50
                seed_stage.append(dict(seed=seed,branch=branch,stage=ph,sampled_updates=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in metrics}))
    stage_stats=[]
    for ph in ['probe_1_25','common_sgd_26_50']:
        for k in metrics:
            vals={b:[next(r[k] for r in seed_stage if r['seed']==s and r['branch']==b and r['stage']==ph) for s in SEEDS] for b in BRANCHES}
            deltas=[r-s for r,s in zip(vals['R'],vals['S'])]
            stage_stats.append(dict(stage=ph,metric=k,S=summary(vals['S']),R=summary(vals['R']),R_minus_S=summary(deltas),R_greater_count=sum(x>0 for x in deltas)))
    # All 400 updates: effective coordinate magnitudes, including final short batch.
    all_step=[]
    for seed in SEEDS:
        for branch in BRANCHES:
            for ph in ['probe_1_25','common_sgd_26_50']:
                ts=[t for t in traces[seed,branch] if stage(t['epoch'])==ph]; assert len(ts)==200
                rr=[{'lora_norm':t['actual_effective_delta']['lora_norm'],'fc_norm':t['actual_effective_delta']['fc_norm'],
                     'joint_norm':t['actual_effective_delta']['joint_norm'],'parameter_norm':t['actual_parameter_step_norm'],
                     'head_squared_share':t['actual_effective_delta']['fc_norm']**2/(t['actual_effective_delta']['joint_norm']**2+EPS),
                     'final_to_base_gradient_ratio':t['u_g_ratio'],'cosine_final_to_base':t['g_U_cosine'],
                     'lr':t['current_lr']} for t in ts]
                row=dict(seed=seed,branch=branch,stage=ph,updates=200)
                for k in rr[0]: row[k+'_step_mean']=statistics.mean(x[k] for x in rr)
                for k in ['lora_norm','fc_norm','joint_norm','parameter_norm']:row[k+'_path_length']=sum(x[k] for x in rr)
                all_step.append(row)
    # Exact checkpoint R-S differences, effective weights and old/new FC rows.
    snapshot_rows=[];snapshot_layers=[]
    for seed in SEEDS:
        for ep in [1,10,25,50]:
            ck={b:torch.load(input_file(OLD/'outputs'/f'formal_{b}_s{seed}'/'checkpoints'/f'epoch_{ep:03d}.pt'),map_location='cpu',weights_only=False) for b in BRANCHES}
            sd={b:ck[b]['trainable_state'] for b in BRANCHES}
            names=[n for n in sd['S'] if '.linear_a_' in n];assert len(names)==24
            ds=[]
            for i,n in enumerate(names):
                bn=n.replace('.linear_a_','.linear_b_');as_,bs=sd['S'][n].double(),sd['S'][bn].double();ar,br=sd['R'][n].double(),sd['R'][bn].double()
                dw=lora_difference_norm(bs,as_,br,ar);ds.append(dw)
                snapshot_layers.append(dict(seed=seed,epoch=ep,layer=i,name=n,delta_effective_W_R_S=dw,delta_A_R_S=float((ar-as_).norm()),delta_B_R_S=float((br-bs).norm())))
            fc_names=[n for n in sd['S'] if n.startswith('fc.')];assert set(fc_names)=={'fc.weight','fc.bias'}
            head_d=math.sqrt(sum(float((sd['R'][n].double()-sd['S'][n].double()).square().sum()) for n in fc_names))
            old_equal=all(torch.equal(sd['R'][n][:20],sd['S'][n][:20]) for n in fc_names)
            old_d=math.sqrt(sum(float((sd['R'][n][:20].double()-sd['S'][n][:20].double()).square().sum()) for n in fc_names))
            snapshot_rows.append(dict(seed=seed,epoch=ep,lora_R_S_norm=math.sqrt(sum(x*x for x in ds)),new_fc_R_S_norm=head_d,old_fc_R_S_norm=old_d,old_fc_rows_bitwise_equal=old_equal,
                qhat_R_S=factors[seed,'R'][ep]['cross_layer_qhat']['median']-factors[seed,'S'][ep]['cross_layer_qhat']['median'],
                clean_training_ce_R_S=curves[seed,'R'][ep]['clean_training']['ce']-curves[seed,'S'][ep]['clean_training']['ce'],
                S_lr=ck['S']['epoch_lr_used'],R_lr=ck['R']['epoch_lr_used']))
            del ck,sd
    # Match training fit within declared fixed bins. Separate pre-switch and
    # full-50 selections explicitly, retaining epochs and LR differences.
    matches=[];fit_seed=[];fit_stats=[]
    for selection,maxep in [('probe_only_1_25',25),('full_1_50',50)]:
        for seed in SEEDS:
            first={}
            for b in BRANCHES:
                for e in range(1,maxep+1):
                    ce=curves[seed,b][e]['clean_training']['ce']
                    if 0<=ce<5:first.setdefault((b,math.floor(ce/.05)),e)
            bins=sorted(set(k[1] for k in first if k[0]=='S')&set(k[1] for k in first if k[0]=='R'))
            sr=[]
            for bi in bins:
                se,re=first['S',bi],first['R',bi];s,r=curves[seed,'S'][se],curves[seed,'R'][re]
                slr=next(t['current_lr'] for t in traces[seed,'S'] if t['epoch']==se);rlr=next(t['current_lr'] for t in traces[seed,'R'] if t['epoch']==re)
                row=dict(selection=selection,seed=seed,ce_bin_low=bi*.05,ce_bin_high=(bi+1)*.05,S_epoch=se,R_epoch=re,S_lr=slr,R_lr=rlr,
                         same_training_phase=stage(se)==stage(re),S_ce=s['clean_training']['ce'],R_ce=r['clean_training']['ce'],ce_R_minus_S=r['clean_training']['ce']-s['clean_training']['ce'],
                         old_accuracy_R_minus_S=r['old_test_allseen40']['accuracy']-s['old_test_allseen40']['accuracy'],
                         new_accuracy_R_minus_S=r['current_test_allseen40']['accuracy']-s['current_test_allseen40']['accuracy'],
                         new_conditional20_accuracy_R_minus_S=r['current_test_conditional20']['accuracy']-s['current_test_conditional20']['accuracy'],
                         new_conditional20_ce_R_minus_S=r['current_test_conditional20']['ce']-s['current_test_conditional20']['ce'],
                         median_qhat_R_minus_S=factors[seed,'R'][re]['cross_layer_qhat']['median']-factors[seed,'S'][se]['cross_layer_qhat']['median'])
                matches.append(row);sr.append(row)
            keys=['ce_R_minus_S','old_accuracy_R_minus_S','new_accuracy_R_minus_S','new_conditional20_accuracy_R_minus_S','new_conditional20_ce_R_minus_S','median_qhat_R_minus_S']
            assert sr
            fit_seed.append(dict(selection=selection,seed=seed,shared_bins=len(sr),cross_phase_bins=sum(not x['same_training_phase'] for x in sr),
                epoch_gap_mean=statistics.mean(x['R_epoch']-x['S_epoch'] for x in sr),lr_gap_mean=statistics.mean(x['R_lr']-x['S_lr'] for x in sr),
                **{k:statistics.mean(x[k] for x in sr) for k in keys}))
        rr=[r for r in fit_seed if r['selection']==selection]
        fit_stats.append(dict(selection=selection,independent_seeds=SEEDS,total_shared_bins=sum(r['shared_bins'] for r in rr),
            **{k:summary([r[k] for r in rr]) for k in keys+['epoch_gap_mean','lr_gap_mean']}))
    write_csv(OUT/'VALIDITY_LEDGER.csv',ledger)
    for name,rows in [('UPDATE_ALL_LAYERS.csv',layers),('UPDATE_BATCH_MEDIANS.csv',steps),('UPDATE_SEED_STAGE.csv',seed_stage),('ALL_STEPS_SEED_STAGE.csv',all_step),('EPOCH_STATE_SEED.csv',epoch_states),('CHECKPOINT_BRANCH_DIFFERENCES.csv',snapshot_rows),('CHECKPOINT_ALL_LAYERS.csv',snapshot_layers),('FIT_MATCHED_ROWS.csv',matches),('FIT_MATCHED_SEED_SUMMARY.csv',fit_seed)]:write_csv(OUT/name,rows)
    details={'protocol':'CPU-only existing six formal branches; original records immutable','validated_expected':'6/6','independent_unit':'seed; n=3, same-seed R minus S','source_input_sha256':manifest,
        'state_formation_analysis_sha256':sha(__file__),'record_counts':{'formal_updates':2400,'sampled_factor_updates':600,'sampled_factor_layers':14400,'epoch_layer_states':7200},
        'aggregation':'24-layer median per fixed first/middle batch, then mean over 25 epochs and 2 batches within each seed, then 3-seed paired statistics; all-step traces separately include all 400 updates',
        'norm_warning':'Component norms do not add. W_cross/net is a magnitude ratio, not a contribution. Signed net projection is geometric and may be negative; not a causal attribution. FC squared norm share uses concatenated BA/FC Euclidean coordinates and does not prove causal dominance.',
        'stage_stats':stage_stats,'fit_matching':fit_stats,'checkpoint_differences':snapshot_rows,'all_steps_seed_stage':all_step}
    write_json(OUT/'STATE_FORMATION_ANALYSIS.json',details)
    def stat(ph,k):return next(r for r in stage_stats if r['stage']==ph and r['metric']==k)
    text=['# 已有状态形成记录的补充分析','',
          '本次只读取原始六条正式分支，不新增训练、不执行模型forward/backward，不修改旧代码或门槛。验收6/6：seed0/1/2，每分支400次真实SGD提交；600个预先固定的首/中间批记录、14,400条层记录全部保留。独立单位为三个seed，层和批不能作为独立重复。','',
          'S：SAM前25轮→SGD后25轮；R：原生S_F g2前25轮→SGD后25轮。二者共同Task0父模型、一次扩头、数据增强次序、50轮cosine相同。','',
          '## 实际更新的因子分配与有限步幅','',
          '以下先在每批计算24层中位数，再在各seed的50个固定批记录中求均值；表中是三个seed均值±样本SD。这是各层等权的描述统计，不是因果贡献。','',
          '| 阶段 | 分支 | πA | TA/TB cosine | 一阶ΔQ范数 | 二阶ΔQ范数 | ΔW交叉项/净ΔW |','|---|---|---:|---:|---:|---:|---:|']
    for ph in ['probe_1_25','common_sgd_26_50']:
        for b in BRANCHES:
            text.append('| '+' | '.join([ph,b]+[fs(stat(ph,k)[b]) for k in ['layer_median_pi_A','layer_median_T_A_T_B_cosine','layer_median_delta_q_first_fro','layer_median_delta_q_second_fro','layer_median_W_cross_over_net']])+' |')
    text+=['','πA=||BΔA||/(||BΔA||+||ΔBA||)仅描述两个一阶通道的大小分配，不可解释为可加贡献。ΔW=BΔA+ΔBA+ΔBΔA；交叉项/净值为范数比，方向抵消时可以超过1。原始每层范数、角度和相对于净更新的带符号投影均保留。','',
        '对依赖BA的损失、相同LR且无动量/WD的基点SGD，设G=∂L/∂W，gA=BᵀG、gB=GAᵀ，则 ΔQ一阶=−η(BᵀgB+gBᵀB−gAAᵀ−AgAᵀ)=0。本轮记录来自真实FP32提交因子，再以FP64复测；SGD尾段的残余一阶量可与FP32提交误差相容，不能将其当成额外机制。探针点梯度提交回基点时该抵消条件不成立，SAM/S_F一阶ΔQ不必为零。','',
        '## 所有真实步骤的幅度与头部','', '| 阶段 | 分支 | 有效LoRA步长 | FC步长 | FC平方范数占比 | 总有效路径长度 |','|---|---|---:|---:|---:|---:|']
    for ph in ['probe_1_25','common_sgd_26_50']:
        for b in BRANCHES:
            rr=[r for r in all_step if r['branch']==b and r['stage']==ph]
            text.append('| '+' | '.join([ph,b]+[fs(summary([r[k] for r in rr])) for k in ['lora_norm_step_mean','fc_norm_step_mean','head_squared_share_step_mean','joint_norm_path_length']])+' |')
    text+=['','这里按200次SGD提交等权汇总，不将路径长度当成终点位移。FC和BA处于不同功能位置；拼接坐标下FC范数较大并不能证明收益由分类头造成。两分支经历不同梯度轨迹，后段虽都用相同SGD规则和LR，其更新大小仍可以不同，离线同一步幅不能排除长期学习率/轨迹解释。','',
        '## 形成的有效状态及分类头','', '| Epoch | ||BA_R−BA_S|| | ||FC_new,R−FC_new,S|| | old FC rows exactly equal |','|---:|---:|---:|---|']
    for ep in [1,10,25,50]:
        rr=[r for r in snapshot_rows if r['epoch']==ep]
        text.append('| '+' | '.join([str(ep),fs(summary([r['lora_R_S_norm'] for r in rr])),fs(summary([r['new_fc_R_S_norm'] for r in rr])),str(all(r['old_fc_rows_bitwise_equal'] for r in rr))])+' |')
    text+=['','有效BA差异不是仅由因子坐标的等价重参数化造成。旧分类头权重/偏置在两个分支逐位一致，而新分类头与BA均不同；因此旧任务结果差异可以来自表征及与新类竞争，但无法凭范数区分头与骨干的因果作用。','',
        '## 固定拟合箱的补充比较','',
        '仍用预注册clean训练CE宽度0.05、区间[0,5)，只选各分支首次进入的共同箱，不看测试挑选、不插值。先对每seed共同箱等权平均，再报告三个seed。前25轮限制避免混合方法阶段；全50轮作为描述性补充，保留跨阶段、epoch和LR差。','',
        '| 选择 | 共同箱总数 | 40类新准确率 R−S | 20类条件新准确率 R−S | 20类条件CE R−S | 旧准确率 R−S | R−S平均epoch差 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in fit_stats:
        text.append('| '+' | '.join([r['selection'],str(r['total_shared_bins'])]+[fs(r[k]) for k in ['new_accuracy_R_minus_S','new_conditional20_accuracy_R_minus_S','new_conditional20_ce_R_minus_S','old_accuracy_R_minus_S','epoch_gap_mean']])+' |')
    text+=['','同箱CE仍非完全相同；R−S的剩余CE差、LR差、所有选中epochs已保存。相近拟合下的测试改进可强化泛化描述，但不能单独排除不同训练时间、日程位置或长期更新大小的影响。共同箱并非独立实验重复。','',
        '## 可支持的结论与下一步','',
        '1. 更新规则形成了不同的有效LoRA状态和新分类头状态，差异延续到共同SGD后段。','2. 一阶ΔQ、因子通道分配和有限步幅均可从已有日志直接测量；指标与FAA同向不构成因果证明，不能把ΔQ当作已证实的收益来源。','3. 真正要排除纯头部解释，可离线交换同seed e25/e50两个分支的新FC与LoRA，保持旧FC和冻结骨干不动，比较SS/SR/RS/RR四个评估状态。这不新增训练，但需明确是分布外组件交换诊断；不是完整因果中介估计。','4. 长期学习率/步长解释若需因果排除，仍需训练级对照。本次按用户要求不扩展训练矩阵，只完整报告既有幅度证据与局限。','',
        '所有输入SHA256见STATE_FORMATION_ANALYSIS.json；全部层/批和匹配点均以CSV保留。']
    (OUT/'RESULT_ANALYSIS.md').write_text('\n'.join(text)+'\n')
    # Standard scientific plots with seed means, sample SD ribbons.
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    specs=[('layer_median_pi_A','A-channel magnitude allocation'),('layer_median_T_A_T_B_cosine','cos(T_A,T_B)'),('layer_median_W_cross_over_net','||cross|| / ||net W update||'),('layer_median_delta_q_first_fro','First-order imbalance update'),('layer_median_delta_q_second_fro','Second-order imbalance update'),('head_squared_share','FC squared share (coordinate metric)')]
    for a,(k,label) in zip(ax.flat,specs):
        for b,c in [('S','#2464A0'),('R','#DA6B20')]:
            byseed=[[statistics.mean(r[k] for r in steps if r['seed']==s and r['branch']==b and r['epoch']==e) for e in range(1,51)] for s in SEEDS]
            means=[statistics.mean(x[i] for x in byseed) for i in range(50)];sds=[statistics.stdev(x[i] for x in byseed) for i in range(50)]
            a.plot(range(1,51),means,color=c,label='SAM→SGD' if b=='S' else 'S_F g2→SGD');a.fill_between(range(1,51),[m-s for m,s in zip(means,sds)],[m+s for m,s in zip(means,sds)],color=c,alpha=.17)
        a.axvline(25.5,color='.4',linestyle='--',lw=.8);a.set(title=label,xlabel='Task1 epoch');a.grid(alpha=.15)
        if 'q_' in k:a.set_yscale('log')
    ax.flat[0].legend(frameon=False);fig.suptitle('Existing state formation records: 3 paired seeds; all 24 LoRA layers')
    fig.savefig(OUT/'STATE_FORMATION.pdf');fig.savefig(OUT/'STATE_FORMATION.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for a,maxep in zip(ax,[25,50]):
        for b,c in [('S','#2464A0'),('R','#DA6B20')]:
            for s in SEEDS:
                a.plot([curves[s,b][e]['clean_training']['ce'] for e in range(1,maxep+1)],[curves[s,b][e]['current_test_conditional20']['accuracy'] for e in range(1,maxep+1)],color=c,alpha=.55,lw=.8)
        a.set(xlabel='Clean train CE (new 20 classes)',ylabel='Conditional new-20 accuracy (%)',title=f'Epoch 1–{maxep}');a.grid(alpha=.15)
    fig.savefig(OUT/'FIT_CONDITIONAL_TEST.pdf');fig.savefig(OUT/'FIT_CONDITIONAL_TEST.png',dpi=180);plt.close(fig)
    print(json.dumps({'out':str(OUT),'validated':'6/6','stage_results':[r for r in stage_stats if r['metric'] in ['layer_median_pi_A','layer_median_delta_q_first_fro','layer_median_delta_q_second_fro','layer_median_W_cross_over_net']],'fit_results':fit_stats},ensure_ascii=False))

if __name__=='__main__': main()
