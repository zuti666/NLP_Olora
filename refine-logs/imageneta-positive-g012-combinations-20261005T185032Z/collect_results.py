"""Collect accepted full endpoints and honest partial progress; controls stay frozen."""
import csv,io,json,statistics
from pathlib import Path
from common import atomic_json,atomic_text,now,sha256
R=Path(__file__).resolve().parent
LABELS={'g1_plus_g2':'g1+g2','g0_plus_perp_g1_g2_residuals':'g0+perp_g0(g1-g0)+perp_g0(g2-g0)','g1_plus_g2_minus_g0':'g1+g2-g0'}

def controls():
    inv=json.loads((R/'historical_inventory.json').read_text())
    for x in inv['reused_controls']:
        if sha256(x['source_summary'])!=x['source_summary_sha256'] or sha256(x['source_matrix'])!=x['source_matrix_sha256']:
            raise RuntimeError('Historical control changed: '+x['job'])
    return inv['reused_controls']

def median(values):
    values=[x for x in values if x is not None]
    return statistics.median(values) if values else None

def main():
    suite=json.loads((R/'suite.json').read_text());old=controls()
    p=R/'runtime/queue_state.json';state=json.loads(p.read_text()) if p.exists() else {'jobs':{}}
    rows=[]
    for job in suite['jobs']:
        if job['kind']!='training' or job['smoke']:continue
        out=Path(job['output_dir']);matrix_path=out/'task_matrix.json'
        m=json.loads(matrix_path.read_text()) if matrix_path.exists() else {}
        status=state['jobs'].get(job['id'],{}).get('status','pending')
        accepted=status=='completed' and len(m.get('matrix',[]))==10
        ref=next(x for x in old if x['radius']==job['radius'] and x['direction']=='g2')
        g1=next(x for x in old if x['direction']=='g1_plus')
        trace=[json.loads(x) for x in (out/'step_trace.jsonl').read_text().splitlines()] if accepted else []
        ratios=[x['norms']['U']/x['norms']['g2'] for x in trace if x['norms']['g2']>0]
        projs=[x['projection'] for x in trace if x.get('projection')]
        rows.append(dict(job=job['id'],direction=job['direction'],label=LABELS[job['direction']],branch='P',radius=job['radius'],seed=2,mode='natural',status=status,tasks=len(m.get('matrix',[])),final_metrics_accepted=accepted,
            AAA=m.get('AAA_task_macro') if accepted else None,FAA=m.get('FAA_task_macro') if accepted else None,LA=m.get('LA') if accepted else None,
            BWT=m.get('BWT') if accepted else None,forgetting=m.get('forgetting') if accepted else None,
            progress_AAA=m.get('AAA_task_macro'),progress_FAA=m.get('FAA_task_macro'),
            g2_reference_AAA=ref['AAA_task_macro'],g2_reference_FAA=ref['FAA_task_macro'],
            delta_vs_g2_AAA=m['AAA_task_macro']-ref['AAA_task_macro'] if accepted else None,
            delta_vs_g2_FAA=m['FAA_task_macro']-ref['FAA_task_macro'] if accepted else None,
            delta_vs_g1_AAA=m['AAA_task_macro']-g1['AAA_task_macro'] if accepted else None,
            delta_vs_g1_FAA=m['FAA_task_macro']-g1['FAA_task_macro'] if accepted else None,
            median_update_over_g2_norm=median(ratios),median_update_g0_cosine=median([x['cosines']['U_g0'] for x in trace]),
            median_update_g2_cosine=median([x['cosines']['U_g2'] for x in trace]),
            median_alpha10=median([x['alpha10'] for x in projs]),median_alpha20=median([x['alpha20'] for x in projs]),
            max_abs_perp10_cosine=max([abs(x['orthogonality_g0_c10']) for x in projs if x['orthogonality_g0_c10'] is not None],default=None),
            max_abs_perp20_cosine=max([abs(x['orthogonality_g0_c20']) for x in projs if x['orthogonality_g0_c20'] is not None],default=None),
            reference_g2_summary=ref['source_summary'],output_dir=str(out)))
    contrasts=[]
    for radius in (.25,1.):
        group={x['direction']:x for x in rows if x['radius']==radius}
        for name,lhs,rhs in [('extra_g0','g1_plus_g2','g1_plus_g2_minus_g0'),('keep_parallel_residuals','g1_plus_g2_minus_g0','g0_plus_perp_g1_g2_residuals')]:
            a,b=group[lhs],group[rhs];ready=a['final_metrics_accepted'] and b['final_metrics_accepted']
            contrasts.append(dict(radius=radius,name=name,lhs=lhs,rhs=rhs,accepted=ready,delta_AAA=a['AAA']-b['AAA'] if ready else None,delta_FAA=a['FAA']-b['FAA'] if ready else None))
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_text(R/'summary.csv',buf.getvalue());atomic_json(R/'results.json',dict(updated_at=now(),new_rows=rows,reused_controls=old,contrasts=contrasts))
    f=lambda v:'—' if v is None else f'{v:.3f}'
    lines=['# ImageNet-A 正向g012组合','',
        '固定seed2/rank10/rho=.05，r=.25/1；正向P几何，g0/g1+/g2/g3四探针。g3计算但不进入最终U；组2以全局g0为参考分别投影两个残差。自然SGD，无最终平均或范数匹配，无Fisher/EWC/融合/HVP。','',
        '| 最终U | r | 状态 | 完成任务 | 最终AAA | 最终FAA | 相对同r的g2 AAA/FAA差值 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for x in rows:lines.append(f"| {x['label']} | {x['radius']} | {x['status']} | {x['tasks']} | {f(x['AAA'])} | {f(x['FAA'])} | {f(x['delta_vs_g2_AAA'])}/{f(x['delta_vs_g2_FAA'])} |")
    lines+=['','主要配对比较（前项减后项，百分点）：','','| 比较 | r | AAA差值 | FAA差值 |','|---|---:|---:|---:|']
    for x in contrasts:lines.append(f"| {LABELS[x['lhs']]} vs {LABELS[x['rhs']]} | {x['radius']} | {f(x['delta_AAA'])} | {f(x['delta_FAA'])} |")
    lines+=['','已完成并核验的历史参照（任务等权%，仅复用）：','','| 方法 | r | AAA | FAA | LA |','|---|---:|---:|---:|---:|']
    for x in old:lines.append(f"| {x.get('label',x['direction'])} | {x['radius'] if x['radius'] is not None else '与r无关'} | {f(x['AAA_task_macro'])} | {f(x['FAA_task_macro'])} | {f(x['LA'])} |")
    lines+=['','所有新组保留相同四探针预算。U1−U3=g0；U3−U2只含残差沿g0的平行分量（在相同模型/batch状态下）。训练轨迹会分开，自然步长结果同时包含方向和幅度影响。范数与余弦只记录，不用于训练匹配。',
        '历史无用第四探针为g1−，本轮为g3；均完整恢复/R0重播/R1单次推进，CPU测试验证g3值不影响U。旧B/g0+B投影参考g1，旧P/A=g0+g2−g1，C/D使用负首探针，都不是本轮三种方向。',
        '完整10任务/500epochs/2550updates且队列日志/结果验收后才填最终指标。单seed不称统计显著或Hessian机制证明；部分任务分数仅作进度。']
    atomic_text(R/'RESULTS.md','\n'.join(lines)+'\n')

if __name__=='__main__':main()
