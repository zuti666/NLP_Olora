"""Separate endpoint metrics from partial training and diagnostic records."""
import csv
import json
from pathlib import Path
from common import HERE,atomic_json,atomic_text,now,read_json

def external_seed2_ready():return True

def main():
    suite=read_json(HERE/'suite.json');state_path=HERE/'runtime/queue_state.json'
    state=read_json(state_path) if state_path.exists() else {'jobs':{}}
    rows=[]
    for j in suite['jobs']:
        if j['kind']!='training' or j['smoke']:continue
        s=state['jobs'].get(j['id'],{'status':'pending'})
        row=dict(job=j['id'],direction=j['direction'],seed=2,status=s['status'],
                 history_weight=j['history_weight'],completed_tasks=0,completed_steps=0,
                 AAA=None,FAA=None,BWT=None,LA=None,forgetting=None)
        p=Path(j['output_dir'])/'final_summary.json'
        if p.exists():
            a=read_json(p);row['completed_tasks']=a.get('completed_tasks',0);row['completed_steps']=a.get('completed_steps',0)
            if s['status']=='completed' and a['status']=='completed':
                m=a['final_metrics']
                row.update(AAA=m['AAA_task_macro'],FAA=m['FAA_task_macro'],BWT=m['BWT'],LA=m['LA'],forgetting=m['forgetting'],history_used_steps=a['history_used_steps'])
        rows.append(row)
    atomic_json(HERE/'results.json',dict(updated_at=now(),protocol=suite['protocol'],rows=rows))
    keys=['job','direction','seed','status','history_weight','completed_tasks','completed_steps','AAA','FAA','BWT','LA','forgetting']
    import io
    text=io.StringIO();w=csv.DictWriter(text,keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    atomic_text(HERE/'summary.csv',text.getvalue())
    lines=['# 历史响应算法首轮结果','', '两组均为seed2；只将通过验收的完整十任务成绩填入FAA/AAA/BWT。','',
           '| 版本 | 状态 | 完成任务 | 完成更新 | FAA | AAA | BWT |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        f=lambda k:'—' if r[k] is None else f'{r[k]:.3f}'
        lines.append(f"| {r['direction']} | {r['status']} | {r['completed_tasks']}/10 | {r['completed_steps']}/2550 | {f('FAA')} | {f('AAA')} | {f('BWT')} |")
    atomic_text(HERE/'RESULTS.md','\n'.join(lines)+'\n')

if __name__=='__main__':main()
