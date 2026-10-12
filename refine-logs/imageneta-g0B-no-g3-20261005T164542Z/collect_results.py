"""Report partial progress separately from accepted full task-macro endpoints."""
import csv,io,json
from pathlib import Path
from common import atomic_json,atomic_text,now
H=Path(__file__).resolve().parent
def main():
    suite=json.loads((H/'suite.json').read_text())
    state=json.loads((H/'runtime/queue_state.json').read_text()) if (H/'runtime/queue_state.json').exists() else {'jobs':{}}
    inventory=json.loads((H/'historical_inventory.json').read_text())
    references={(x['radius'],x['branch']):x for x in inventory['historical_comparison_controls'] if x['direction']=='A' and x['mode']=='natural'}
    rows=[]
    for j in suite['jobs']:
        if j['kind']!='training' or j['smoke']:continue
        out=Path(j['output_dir']);m=json.loads((out/'task_matrix.json').read_text()) if (out/'task_matrix.json').exists() else {}
        status=state['jobs'].get(j['id'],{}).get('status','pending')
        accepted=status=='completed' and len(m.get('matrix',[]))==10
        ac=references[(j['radius'],'C')];ap=references[(j['radius'],'P')]
        rows.append(dict(job=j['id'],radius=j['radius'],seed=2,direction=j['direction'],mode='natural',protocol=suite['protocol'],
                         status=status,tasks=len(m.get('matrix',[])),final_metrics_accepted=accepted,
                         AAA=m.get('AAA_task_macro') if accepted else None,FAA=m.get('FAA_task_macro') if accepted else None,
                         LA=m.get('LA') if accepted else None,BWT=m.get('BWT') if accepted else None,forgetting=m.get('forgetting') if accepted else None,
                         progress_AAA=m.get('AAA_task_macro'),progress_FAA=m.get('FAA_task_macro'),
                         historical_AC_AAA=ac['AAA_task_macro'],historical_AC_FAA=ac['FAA_task_macro'],
                         delta_vs_AC_AAA=m['AAA_task_macro']-ac['AAA_task_macro'] if accepted else None,
                         delta_vs_AC_FAA=m['FAA_task_macro']-ac['FAA_task_macro'] if accepted else None,
                         historical_AP_AAA=ap['AAA_task_macro'],historical_AP_FAA=ap['FAA_task_macro'],
                         delta_vs_AP_AAA=m['AAA_task_macro']-ap['AAA_task_macro'] if accepted else None,
                         delta_vs_AP_FAA=m['FAA_task_macro']-ap['FAA_task_macro'] if accepted else None,
                         reference_AC_summary=ac['source_summary'],reference_AP_summary=ap['source_summary'],output_dir=str(out)))
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_text(H/'summary.csv',stream.getvalue());atomic_json(H/'results.json',dict(updated_at=now(),rows=rows))
    lines=['# ImageNet-A g0+B 三探针实验','', '固定seed2/rank10/ρ=.05，r=.25/1。正式训练只有两条g0+B；未验收完整十任务的指标不作最终结果。','',
           '| 实验 | r | 状态 | 已完成任务 | 最终AAA | 最终FAA |','|---|---:|---|---:|---:|---:|']
    for r in rows:
        f=lambda v:'—' if v is None else f'{v:.3f}'
        lines.append(f"| {r['job']} | {r['radius']} | {r['status']} | {r['tasks']} | {f(r['AAA'])} | {f(r['FAA'])} |")
    lines+=['','历史自然步长参考（同seed2/训练配置，四探针；本轮三探针）：','',
            '| r | A_C AAA/FAA | A_P AAA/FAA | g0+B−A_C AAA/FAA | g0+B−A_P AAA/FAA |','|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['radius']} | {r['historical_AC_AAA']:.3f}/{r['historical_AC_FAA']:.3f} | {r['historical_AP_AAA']:.3f}/{r['historical_AP_FAA']:.3f} | {f(r['delta_vs_AC_AAA'])}/{f(r['delta_vs_AC_FAA'])} | {f(r['delta_vs_AP_AAA'])}/{f(r['delta_vs_AP_FAA'])} |")
    lines+=['','g0+B 与 A_C 同时改变第一侧、外点和最终投影，不能单独归因于投影；与 A_P 在同状态相同P端点下，U差为(1−alpha)g1+，涉及平行分量和总更新步幅。完整训练沿不同轨迹进行，单seed只报告描述性差值。','',
            'CPU删去unused探针等价检查通过；这不等于跨进程CUDA位级复现。Fisher/融合调用与3CE/1SGD预算以逐步记录和完成凭据验收。']
    atomic_text(H/'RESULTS.md','\n'.join(lines)+'\n')
if __name__=='__main__':main()
