"""Nine-direction native-r1 table; incomplete cells never contribute scores."""
import csv
import io
from common import HERE,atomic_json,atomic_text,read_json,now,verify_reuse


def write_nine(suite,document):
    reused=verify_reuse(suite);rows=[]
    labels={'s3_puregam_original_r1_k10_s2':'原版 GAM，γ=0.1',
            's3_puregam_gamma0_sum_r1_k10_s2':'g1+g3，γ=0',
            's3_puregam_gamma0_g1_r1_k10_s2':'仅 g1',
            's3_puregam_gamma0_g3_r1_k10_s2':'仅 g3',
            'formal_g0_r1_k10_s2':'仅 g0','formal_g2_r1_k10_s2':'仅 g2',
            'formal_ug_r1_k10_s2':'g0+(g3-g2)'}
    order=list(labels)
    for previous in reused['jobs']:
        row=dict(previous)
        row.update(status='completed',origin='verified_completed_r1',direction=labels[row['job_id']],
                   learning_accuracy=row['learning_accuracy_task_macro'],bwt=row['bwt_old9'])
        rows.append(row)
    rows.sort(key=lambda r:order.index(r['job_id']))
    records={row['id']:row for row in document['jobs']}
    for selector,label in [('c21_base','A：g0+(g2-g1)'),('c21_perp','B：g1+(g2-g1)⊥g1')]:
        row=dict(records['formal_'+selector+'_r1_k10_s2']);row.update(origin='new_native_r1_AB',direction=label);rows.append(row)
    result={'updated_at':now(),'dataset':'imageneta','seed':2,'rank':10,'outer_r':1.,'inner_rho':.05,
            'protocol':'native historical global RNG / native DataLoader / mechanism flags preserved',
            'expected':9,'completed':sum(r['status']=='completed' for r in rows),'rows':rows}
    comparisons=[]
    for a,b in [(7,4),(7,5),(8,2),(8,5)]:
        if rows[a]['status']==rows[b]['status']=='completed':
            comparisons.append({'direction':rows[a]['direction'],'reference':rows[b]['direction'],
                'AAA_gain_pp':rows[a]['aaa_task_macro']-rows[b]['aaa_task_macro'],
                'FAA_gain_pp':rows[a]['faa_task_macro']-rows[b]['faa_task_macro']})
    result['AB_comparisons']=comparisons
    atomic_json(HERE/'PURE_GAM_R1_NINE.json',result)
    fields=['direction','status','origin','aaa_task_macro','faa_task_macro','learning_accuracy','bwt',
            'forgetting','aaa_sample_weighted','faa_sample_weighted','metrics_path','metrics_sha256']
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    atomic_text(HERE/'PURE_GAM_R1_NINE.csv',stream.getvalue())
    lines=['# ImageNet-A：原生 r=1 九方向对照','',f"更新 {result['updated_at']}；完整结果 {result['completed']}/9。",
        '', '统一 seed2、rank10、r=1、rho=.05、10×20类、每任务50epochs、无EWC/Fisher/权重融合。',
        '保留原生全局RNG/DataLoader与历史机制评估设置；AAA/FAA为任务等权口径。',
        '', '| 方向 | AAA (%) | FAA (%) | LA (%) | BWT (pp) | 遗忘 (pp) | 状态 | 来源 |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for row in rows:
        vals=[f'{row[k]:.3f}' if row['status']=='completed' and row.get(k) is not None else '—' for k in ['aaa_task_macro','faa_task_macro','learning_accuracy','bwt','forgetting']]
        lines.append('| '+' | '.join([row['direction'],*vals,row['status'],row['origin']])+' |')
    lines+=['','A回加跨探测点残差；B在全部可训练LoRA A/B与完整seen FC坐标上统一去掉残差的平行分量。',
        'A/B均关闭原GAM γ正交修正，使用名义学习率，不做方向或有效步长范数匹配；B的投影是本次有意干预。',
        '每组保留四次GAM探针与一次clean CE，逐batch验收真实方向、原FP32projector、扰动还原后的实际SGD。',
        '记录投影系数、正交比例、夹角、参数与有效BA+FC更新范数；只观察步长，不改变优化器更新。',
        '未完成格不填准确率。原七行来自完成凭据/原始矩阵核验，r=.25结果不进入本表。',
        '方向与幅度共同变化，单seed不能证明统计显著、等价或普遍必要；跨GPU轨迹不宣称逐位一致。']
    if comparisons:
        lines+=['','A/B相对指定对照的增益：']
        for c in comparisons:lines.append(f"- {c['direction']} 相对 {c['reference']}：AAA {c['AAA_gain_pp']:+.3f}、FAA {c['FAA_gain_pp']:+.3f} pp。")
    atomic_text(HERE/'PURE_GAM_R1_NINE.md','\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    from common import load_suite
    from collect_results import collect
    state=HERE/'runtime/state.json';states=read_json(state).get('jobs',{}) if state.exists() else {}
    collect(load_suite(),states)
