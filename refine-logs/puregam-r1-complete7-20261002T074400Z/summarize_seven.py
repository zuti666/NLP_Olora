"""Combine hash-verified historical four directions with three native extensions."""
import csv
import io
from common import HERE, atomic_json, atomic_text, now, read_json, verify_reuse


def write_seven(suite, document):
    reused=verify_reuse(suite)
    rows=[]
    for historical in reused['jobs']:
        row=dict(historical)
        labels={'s3_puregam_original_r1_k10_s2':'原版 GAM，γ=0.1',
                's3_puregam_gamma0_sum_r1_k10_s2':'g1+g3，γ=0',
                's3_puregam_gamma0_g1_r1_k10_s2':'仅 g1',
                's3_puregam_gamma0_g3_r1_k10_s2':'仅 g3'}
        row.update(status='completed',origin='historical_verified',direction=labels[row['job_id']])
        rows.append(row)
    ordering={'s3_puregam_original_r1_k10_s2':0,'s3_puregam_gamma0_sum_r1_k10_s2':1,
              's3_puregam_gamma0_g1_r1_k10_s2':2,'s3_puregam_gamma0_g3_r1_k10_s2':3}
    rows.sort(key=lambda r:ordering[r['job_id']])
    records={row['id']:row for row in document['jobs']}
    for selector,label in [('g0','g0'),('g2','g2'),('ug','g0+(g3-g2)')]:
        row=dict(records['formal_'+selector+'_r1_k10_s2'])
        row.update(origin='new_native_extension',direction=label)
        rows.append(row)
    baseline=rows[0]
    for row in rows:
        for metric in ['aaa_task_macro','faa_task_macro']:
            row['delta_'+metric]=row[metric]-baseline[metric] if row['status']=='completed' else None
    output={'updated_at':now(),'dataset':'imageneta','seed':2,'rank':10,'outer_r':1.,'inner_rho':.05,
            'protocol':'historical native global RNG; same five CE passes per training batch',
            'completed':sum(row['status']=='completed' for row in rows),'expected':7,'rows':rows}
    atomic_json(HERE/'PURE_GAM_R1_SEVEN.json',output)
    fields=['direction','status','origin','aaa_task_macro','faa_task_macro','delta_aaa_task_macro',
            'delta_faa_task_macro','aaa_sample_weighted','faa_sample_weighted','metrics_path','metrics_sha256']
    text=io.StringIO();writer=csv.DictWriter(text,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    atomic_text(HERE/'PURE_GAM_R1_SEVEN.csv',text.getvalue())
    lines=['# ImageNet-A: r=1 七方向对照','',f"更新 {output['updated_at']}；完成 {output['completed']}/7。",
           '', '统一 seed2、LoRA rank10、外半径 r=1、内层 rho=.05、10×20类、每任务50 epochs。',
           'AAA/FAA 为任务等权口径；EWC/Fisher/权重融合关闭。保留历史原生数据加载及机制评估。',
           '', '| 方向 | AAA (%) | FAA (%) | 状态 | 来源 |','|---|---:|---:|---|---|']
    for row in rows:
        scores=[f'{row[k]:.3f}' if row.get(k) is not None and row['status']=='completed' else '—' for k in ['aaa_task_macro','faa_task_macro']]
        lines.append('| '+' | '.join([row['direction'],*scores,row['status'],row['origin']])+' |')
    lines+=['','原版/双分支/单分支包含方向和更新幅度的共同影响；未做额外范数匹配。',
            'g0 保留四次探测，因此是同探测流程下的 SGD 方向对照；浮点扰动还原与纯单次前向 SGD 可能存在微小差别。',
            '新三组仅在训练 optimizer 实例上选择最终方向，不改变机制评估中的 GAM。',
            '单 seed 结果属于当前配置的消融证据，不能直接声称统计显著或普遍必要。',
            '尚未完成的格不填分数；短测和 r=.25 结果不进入本表。']
    atomic_text(HERE/'PURE_GAM_R1_SEVEN.md','\n'.join(lines)+'\n')
    return output


if __name__=='__main__':
    from collect_results import collect
    from common import load_suite
    state=HERE/'runtime/state.json'
    states=read_json(state).get('jobs',{}) if state.exists() else {}
    collect(load_suite(),states)
