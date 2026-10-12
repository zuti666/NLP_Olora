"""Eight-arm full-task comparison with old Mean shown separately."""
import csv
import io
from common import HERE,atomic_json,atomic_text,now,read_json,verify_reuse


def write_controls(suite,document):
    reused=verify_reuse(suite)
    old={row['arm']:row for row in reused['jobs']}
    jobs={row['id']:row for row in document['jobs']}
    formal={j['arm']:j for j in suite['jobs'] if j['phase']=='formal'}
    direction={'S':'g1','S1.5':'g1','S2':'g1','Sum':'g1+g3','Mean2':'(g1+g3)/2',
               'Graw':'g3','Gmatch':'g3，匹配独立S实际有效步长','G025':'g3，r=.25'}
    rows=[]
    for arm in suite['primary_arms']:
        if arm in formal:
            row=dict(jobs[formal[arm]['id']]);row.update(origin='new_full10_native',arm=arm,
                                                      lr_multiplier=formal[arm]['lr_multiplier'])
            if row['status']=='completed':
                proof=read_json(HERE/'runtime/jobs'/row['id']/'completed.json')['direction_proof']
                for k in ['matched_steps','matching_relative_error_max','matching_relative_error_mean']:
                    if k in proof:row[k]=proof[k]
        else:
            row=dict(old[arm]);row.update(status='completed',origin='verified_historical_full10',arm=arm,
                learning_accuracy=row['learning_accuracy_task_macro'],bwt=row['bwt_old9'],lr_multiplier=1.)
        row['direction']=direction[arm];rows.append(row)
    indexed={r['arm']:r for r in rows};contrasts=[]
    for a,b in [('S1.5','S'),('S2','S'),('Mean2','Sum'),('Gmatch','S'),('Gmatch','Graw'),('G025','Graw')]:
        if indexed[a]['status']==indexed[b]['status']=='completed':
            contrasts.append({'arm':a,'reference':b,
                'AAA_gain_pp':indexed[a]['aaa_task_macro']-indexed[b]['aaa_task_macro'],
                'FAA_gain_pp':indexed[a]['faa_task_macro']-indexed[b]['faa_task_macro']})
    result={'updated_at':now(),'seed':2,'rank':10,'inner_rho':.05,'outer_r':1.,
        'expected_primary_arms':8,'completed_primary_arms':sum(r['status']=='completed' for r in rows),
        'fresh_formal_expected':5,'fresh_formal_completed':sum(r['status']=='completed' and r['origin']=='new_full10_native' for r in rows),
        'rows':rows,'contrasts':contrasts,'secondary_existing_Mean':old['Mean'],
        'historical_S_context_only':old['S'],
        'limits':'one seed, natural LR controls; Gmatch length matched to independent S; old controls lack paired per-step traces; no bitwise Sum/Mean2 trajectory claim'}
    atomic_json(HERE/'LR_CONTROL_TABLE.json',result)
    fields=['arm','direction','lr_multiplier','status','origin','aaa_task_macro','faa_task_macro',
            'learning_accuracy','bwt','forgetting','matched_steps','matching_relative_error_max',
            'metrics_path','metrics_sha256']
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    atomic_text(HERE/'LR_CONTROL_TABLE.csv',stream.getvalue())
    lines=['# ImageNet-A：十任务学习率与有效步长对照','',f"更新 {result['updated_at']}；主对照 {result['completed_primary_arms']}/8，新增正式训练 {result['fresh_formal_completed']}/5。",'',
        '统一seed2、rank10、rho=.05、10×20类、50epochs/task；无EWC/Fisher/融合/replay，保留原生RNG、机制评估和workers8。',
        '外半径r=1，只有G025为.25。LR基础首任务.04、后续.02，cosine floor为起点×.1；标称倍率同时作用于峰值和下限。','',
        '|组|最终方向|LR倍率|AAA (%)|FAA (%)|LA (%)|BWT (pp)|遗忘 (pp)|状态|来源|',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for row in rows:
        values=[f'{row[k]:.3f}' if row['status']=='completed' and row.get(k) is not None else '—'
                for k in ['aaa_task_macro','faa_task_macro','learning_accuracy','bwt','forgetting']]
        lines.append('| '+' | '.join([row['arm'],row['direction'],str(row['lr_multiplier']),*values,row['status'],row['origin']])+' |')
    lines+=['','已有正常LR Mean（次级对照）：AAA %.3f，FAA %.3f；历史S仅作背景，新S完成前不填其分数。'%(old['Mean']['aaa_task_macro'],old['Mean']['faa_task_macro']),
        'Gmatch目标严格来自新S完成轨迹对应(task,step)的真实FP32 SGD有效BA＋完整seen FC位移，不来自G当前状态g1，不复制S模型参数。',
        '每步匹配相对误差≤1e-3；未完成、不匹配、错误batch/RNG、无正根或非有限值停止后续放行。临时SGD scalar恢复后才推进原生scheduler。',
        'Mean2与Sum在实数算术中具有相同名义更新；FP32 projector保留，短测不宣称完整轨迹逐位一致。',
        'AAA/FAA为任务等权口径。已有控制的缺失逐步trace不追溯补造；单seed不作统计显著性结论。']
    for c in contrasts:lines.append(f"- {c['arm']} vs {c['reference']}：AAA {c['AAA_gain_pp']:+.3f}，FAA {c['FAA_gain_pp']:+.3f} pp。")
    atomic_text(HERE/'LR_CONTROL_TABLE.md','\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    from collect_results import collect
    from common import load_suite
    state=HERE/'runtime/state.json'
    collect(load_suite(),read_json(state).get('jobs',{}) if state.exists() else {})
