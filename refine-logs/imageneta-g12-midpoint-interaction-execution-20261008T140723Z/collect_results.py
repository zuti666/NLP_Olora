"""Collect accepted training endpoints; never put diagnostic losses in the accuracy table."""
import csv,io
from pathlib import Path
from common import HERE,atomic_json,atomic_text,load_suite,read_json,now
from validate_job import metrics

def main():
    suite=load_suite();state=read_json(HERE/'runtime/queue_state.json') if (HERE/'runtime/queue_state.json').exists() else {'jobs':{}}
    records=[]
    for job in suite['jobs']:
        if job['kind']!='training' or job['smoke']:continue
        record=dict(condition=job['id'],direction=job['direction'],radius=1.,rho=.05,seed=2,rank=10,protocol=suite['protocol'],status=state['jobs'].get(job['id'],{}).get('status','pending'),accepted=False,AAA=None,FAA=None,LA=None,BWT=None,forgetting=None,progress_tasks=0,progress_AAA=None,progress_FAA=None,source_matrix=str(Path(job['output_dir'])/'task_matrix.json'),source_summary=str(Path(job['output_dir'])/'final_summary.json'))
        matrixpath=Path(record['source_matrix'])
        if matrixpath.exists():
            matrix=read_json(matrixpath)['matrix'];values=metrics(matrix);record.update(progress_tasks=len(matrix),progress_AAA=values['AAA_task_macro'],progress_FAA=values['FAA_task_macro'])
            if record['status']=='completed':record.update(accepted=True,AAA=values['AAA_task_macro'],FAA=values['FAA_task_macro'],LA=values['LA'],BWT=values['BWT'],forgetting=values['forgetting'])
        records.append(record)
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records);atomic_text(HERE/'summary.csv',buffer.getvalue())
    atomic_json(HERE/'results.json',dict(updated_at=now(),protocol=suite['protocol'],formal_results=records,diagnostics_separate=True,queue_counts=state.get('counts',{})))
    text='# 实验进度与结果\n\n仅验收完成的十任务成绩进入AAA/FAA；中间阶段分数列为progress，不当作最终结果。\n\n| 方向 | 状态 | 完成任务 | AAA | FAA |\n|---|---|---:|---:|---:|\n'
    for r in records:text+='| '+r['direction']+' | '+r['status']+' | '+str(r['progress_tasks'])+' | '+('—' if r['AAA'] is None else f"{r['AAA']:.3f}")+' | '+('—' if r['FAA'] is None else f"{r['FAA']:.3f}")+' |\n'
    atomic_text(HERE/'RESULTS.md',text)
if __name__=='__main__':main()
