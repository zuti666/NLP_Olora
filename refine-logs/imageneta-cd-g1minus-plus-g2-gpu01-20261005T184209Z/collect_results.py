"""Compare accepted complete sum endpoints against frozen historical g2 controls."""
import csv,io,json,statistics
from pathlib import Path
from common import atomic_json,atomic_text,now,sha256
R=Path(__file__).resolve().parent

def controls():
    inv=json.loads((R/'historical_inventory.json').read_text())
    for x in inv['reused_controls']:
        if sha256(x['source_summary'])!=x['source_summary_sha256'] or sha256(x['source_matrix'])!=x['source_matrix_sha256']:
            raise RuntimeError('Historical control changed: '+x['job'])
    return inv['reused_controls']

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
        ref=next(x for x in old if x['radius']==job['radius'] and x['branch']==job['branch'] and x['direction']=='g2')
        ac=next((x for x in old if x['radius']==job['radius'] and x['branch']=='C' and x['direction']=='A'),None) if job['branch']=='C' else None
        trace=[json.loads(x) for x in (out/'step_trace.jsonl').read_text().splitlines()] if accepted else []
        norm_ratios=[x['norms']['U']/x['norms']['g2'] for x in trace if x['norms']['g2']>0]
        cosines=[x['cosines']['U_g2'] for x in trace if x['cosines']['U_g2'] is not None]
        rows.append(dict(job=job['id'],branch=job['branch'],radius=job['radius'],seed=2,
            direction=job['direction'],mode='natural',status=status,tasks=len(m.get('matrix',[])),
            final_metrics_accepted=accepted,AAA=m.get('AAA_task_macro') if accepted else None,
            FAA=m.get('FAA_task_macro') if accepted else None,LA=m.get('LA') if accepted else None,
            forgetting=m.get('forgetting') if accepted else None,
            progress_AAA=m.get('AAA_task_macro'),progress_FAA=m.get('FAA_task_macro'),
            g2_reference_AAA=ref['AAA_task_macro'],g2_reference_FAA=ref['FAA_task_macro'],
            delta_vs_g2_AAA=m['AAA_task_macro']-ref['AAA_task_macro'] if accepted else None,
            delta_vs_g2_FAA=m['FAA_task_macro']-ref['FAA_task_macro'] if accepted else None,
            AC_reference_AAA=ac['AAA_task_macro'] if ac else None,AC_reference_FAA=ac['FAA_task_macro'] if ac else None,
            median_update_over_g2_norm=statistics.median(norm_ratios) if norm_ratios else None,
            median_update_g2_cosine=statistics.median(cosines) if cosines else None,
            reference_g2_summary=ref['source_summary'],output_dir=str(out)))
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_text(R/'summary.csv',buf.getvalue());atomic_json(R/'results.json',dict(updated_at=now(),new_rows=rows,reused_controls=old))
    f=lambda v:'—' if v is None else f'{v:.3f}'
    lines=['# ImageNet-A C/D: g1−+g2','',
        '固定seed2/rank10/rho=.05，外r=.25/1；自然相加，不除以2。只改变最终更新，均保留g0/g1+/g1−/g2四探针；不计算g3、Fisher、权重融合、投影或HVP。','',
        '| 新增方向 | r | 状态 | 已完成任务 | 最终AAA | 最终FAA | 相对同分支g2的AAA/FAA差值 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for x in rows:lines.append(f"| {x['branch']}: g1−+g2 | {x['radius']} | {x['status']} | {x['tasks']} | {f(x['AAA'])} | {f(x['FAA'])} | {f(x['delta_vs_g2_AAA'])}/{f(x['delta_vs_g2_FAA'])} |")
    lines+=['','已完成并核验的历史参照（任务等权%，仅复用）：','',
        '| 方法 | r | AAA | FAA | LA |','|---|---:|---:|---:|---:|']
    for x in old:lines.append(f"| {'A_C' if x['direction']=='A' else x['branch']+'/g2'} | {x['radius']} | {f(x['AAA_task_macro'])} | {f(x['FAA_task_macro'])} | {f(x['LA'])} |")
    lines+=['','AC探针末只用g2和C/g2是同一设置；AC探针末用g1−+g2和C分支sum也是同一设置，不重复训练。',
        '所有新组都是自然步长，方向和幅度共同变化；范数比和余弦只是观察，不用于修改训练步长。单seed结果不作为统计显著性或Hessian机制证明。部分任务准确率仅为进度，完整10任务和队列验收后才填最终指标。']
    atomic_text(R/'RESULTS.md','\n'.join(lines)+'\n')

if __name__=='__main__':main()
