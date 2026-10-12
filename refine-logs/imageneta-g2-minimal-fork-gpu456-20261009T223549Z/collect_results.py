"""Status and accepted-only branch results; never rank partial test performance."""
import json,csv,io
from pathlib import Path
from common import HERE as H,atomic_json,atomic_text,now,read_json,sha256

def main():
    suite=read_json(H/'suite.json');p=H/'runtime/queue_state.json';state=read_json(p) if p.exists() else {'jobs':{}}
    rows=[];pending=[]
    for j in suite['jobs']:
        status=state['jobs'].get(j['id'],{});out=Path(j['output_dir']);row=dict(run=j['id'],kind=j['kind'],smoke=j['smoke'],status=status.get('status','pending'),gpu=status.get('gpu'),condition=j.get('study_condition_id'),trained_task=j.get('task_id'))
        if (out/'final_summary.json').exists():
            a=read_json(out/'final_summary.json');row.update(completed_updates=a.get('completed_steps'),completed_tasks=a.get('completed_tasks'))
        if status.get('status')=='completed' and j['kind']=='training' and not j['smoke']:
            a=read_json(out/'final_summary.json');receipt=read_json(H/'runtime/jobs'/j['id']/'completed.json')
            digest=sha256(out/'final_summary.json')
            if digest!=status['summary_sha256'] or digest!=receipt['summary_sha256'] or receipt['exit']['returncode']!=0:raise RuntimeError('Accepted formal evidence changed: '+j['id'])
            m=a['final_metrics'];row.update(dev_CE=a['endpoint_dev']['dev_current_task_ce'],dev_accuracy=a['endpoint_dev']['dev_current_task_accuracy'],test_current_task_accuracy=m['matrix'][-1][-1],test_old_task_accuracy=m['matrix'][-1][0] if j['task_id']==1 else None,old_task_change_from_parent=m['BWT'],FAA_two_stage=m['FAA_task_macro'],AAA_two_stage=m['AAA_task_macro'],elapsed_seconds=a['elapsed_seconds'],gradient_reads=a['counters']['training_ce_gradient_passes'],summary_sha256=sha256(out/'final_summary.json'))
        rows.append(row)
    atomic_json(H/'results/summary.json',dict(updated_at=now(),run_id=suite['run_id'],selection_rule='epoch50 Task1 held-out-train current-task CE; test is report-only',rows=rows))
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows);atomic_text(H/'results/summary.csv',buf.getvalue())
    comparison=[r for r in rows if r['kind']=='training' and not r['smoke'] and r['trained_task']==1 and r['status']=='completed']
    fields=['run','condition','dev_CE','dev_accuracy','test_current_task_accuracy','test_old_task_accuracy','old_task_change_from_parent','FAA_two_stage','AAA_two_stage','gradient_reads','elapsed_seconds','summary_sha256']
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(comparison);atomic_text(H/'results/comparison.csv',buf.getvalue())
    body='| Run | Condition | Status | GPU | Accepted endpoint dev CE |\n|---|---|---|---|---|\n'+'\n'.join(f"| {r['run']} | {r['condition'] or r['kind']} | {r['status']} | {r['gpu'] if r['gpu'] is not None else '—'} | {r.get('dev_CE','—')} |" for r in rows)
    atomic_text(H/'results/STATUS.md',f"# G2 minimal fork status\n\nUpdated UTC: {now()}\n\n{body}\n\nOnly accepted completed formal endpoints enter comparison. No old full-data scores are reused.\n")
if __name__=='__main__':main()
