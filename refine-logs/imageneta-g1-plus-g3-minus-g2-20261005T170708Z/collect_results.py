"""Accepted final endpoints only; partial task metrics are progress."""
import csv,io,json
from pathlib import Path
from common import atomic_json,atomic_text,now
H=Path(__file__).resolve().parent
def main():
    s=json.loads((H/'suite.json').read_text());p=H/'runtime/queue_state.json'
    st=json.loads(p.read_text()) if p.exists() else {'jobs':{}}
    inv=json.loads((H/'historical_inventory.json').read_text())
    baseline=next(x for x in inv['historical_comparison_controls'] if x['direction']=='g1_plus' and x['mode']=='natural')
    rows=[]
    for j in s['jobs']:
        if j['kind']!='training' or j['smoke']:continue
        out=Path(j['output_dir']);f=out/'task_matrix.json';m=json.loads(f.read_text()) if f.exists() else {}
        status=st['jobs'].get(j['id'],{}).get('status','pending');accepted=status=='completed' and len(m.get('matrix',[]))==10
        rows.append(dict(job=j['id'],radius=j['radius'],seed=2,direction=j['direction'],mode='natural',protocol=s['protocol'],status=status,tasks=len(m.get('matrix',[])),final_metrics_accepted=accepted,
                         AAA=m.get('AAA_task_macro') if accepted else None,FAA=m.get('FAA_task_macro') if accepted else None,
                         LA=m.get('LA') if accepted else None,BWT=m.get('BWT') if accepted else None,forgetting=m.get('forgetting') if accepted else None,
                         progress_AAA=m.get('AAA_task_macro'),progress_FAA=m.get('FAA_task_macro'),
                         g1_reference_AAA=baseline['AAA_task_macro'],g1_reference_FAA=baseline['FAA_task_macro'],
                         delta_vs_g1_AAA=m['AAA_task_macro']-baseline['AAA_task_macro'] if accepted else None,
                         delta_vs_g1_FAA=m['FAA_task_macro']-baseline['FAA_task_macro'] if accepted else None,
                         reference_g1_summary=baseline['source_summary'],output_dir=str(out)))
    b=io.StringIO();w=csv.DictWriter(b,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_text(H/'summary.csv',b.getvalue());atomic_json(H/'results.json',dict(updated_at=now(),rows=rows))
    lines=['# ImageNet-A g1+(g3−g2) 实验','', 'seed2/rank10/rho=.05，r=.25/1；四CE探针、一次自然SGD。只提交g1+(g3−g2)，无投影/Fisher/融合/HVP/最终归一化或匹配。','',
           '| r | 状态 | 已完成任务 | 最终AAA | 最终FAA | 比g1增加AAA/FAA |','|---:|---|---:|---:|---:|---:|']
    f=lambda v:'—' if v is None else f'{v:.3f}'
    for r in rows:lines.append(f"| {r['radius']} | {r['status']} | {r['tasks']} | {f(r['AAA'])} | {f(r['FAA'])} | {f(r['delta_vs_g1_AAA'])}/{f(r['delta_vs_g1_FAA'])} |")
    lines+=['',f"R0-replay g1+自然步长参考：AAA/FAA={baseline['AAA_task_macro']:.3f}/{baseline['FAA_task_macro']:.3f}。参考最终公式不含外半径，rho=.05，但其四探针包含另一侧g1−而未计算g3；计算流程和新方向有区别。",'',
            '既有g0+(g3−g2)及原生g3通常是旧五次CE/native RNG协议，仅作背景，不当同协议严格消融。完整训练轨迹不同，单seed自然步长比较混有方向与更新幅度变化。部分任务指标不作为最终成绩。']
    atomic_text(H/'RESULTS.md','\n'.join(lines)+'\n')
if __name__=='__main__':main()
