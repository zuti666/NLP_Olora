"""Only accepted full runs enter performance tables; bounded smokes are excluded."""
import csv,io,json
from pathlib import Path
from common import HERE,read_json,atomic_json,atomic_text,sha256,now

def external_seed2_ready():
    p=HERE/'gates/ORIGINAL_RESULT_REUSE.json'
    if not p.exists():return False
    gate=read_json(p)
    if gate['status']!='passed' or gate['updates']!=2550 or gate['tasks']!=10:raise RuntimeError('Invalid original reuse gate')
    for path,digest in gate['files'].items():
        if sha256(path)!=digest:raise RuntimeError('Frozen original reuse artifact changed: '+path)
    out=Path(gate['output_dir']);a=read_json(out/'final_summary.json')
    receipt=read_json(Path(gate['run_dir'])/'runtime/jobs'/gate['job_id']/'completed.json')
    if a['status']!='completed' or a['completed_steps']!=2550 or a['completed_tasks']!=10 or not receipt['full_formal_training'] or receipt['exit']['returncode']!=0:
        raise RuntimeError('Original reuse budget/acceptance missing')
    if receipt['summary_sha256']!=sha256(out/'final_summary.json') or a['final_metrics']!=gate['final_metrics']:raise RuntimeError('Original result changed')
    return True

def collect():
    external_seed2_ready();suite=read_json(HERE/'suite.json');p=HERE/'runtime/queue_state.json'
    state=read_json(p) if p.exists() else {'status':'not_launched','jobs':{}}
    gate=read_json(HERE/'gates/ORIGINAL_RESULT_REUSE.json');a=read_json(Path(gate['output_dir'])/'final_summary.json');m=a['final_metrics']
    rows=[dict(Method='CurrentG2-Original',Seed=2,Inner='eta*norm(g0)',Outer='eta*norm(g0)',Status='completed_reused',FAA=m['FAA_task_macro'],AAA=m['AAA_task_macro'],BWT=m['BWT'],Updates=2550,Run=gate['job_id'],Source=gate['output_dir'])]
    for j in suite['jobs']:
        if j['kind']!='training' or j['smoke']:continue
        r=state['jobs'].get(j['id'],{'status':'pending'});out=Path(j['output_dir']);metrics={}
        steps=0
        if r['status']=='completed':
            from validate_job import validate_job
            checked=validate_job(j)
            receipt=read_json(HERE/'runtime/jobs'/j['id']/'completed.json')
            if checked['summary_sha256']!=r['summary_sha256'] or receipt['summary_sha256']!=checked['summary_sha256']:raise RuntimeError('New completion receipt differs')
            metrics=read_json(out/'final_summary.json')['final_metrics'];steps=2550
        elif (out/'step_trace.jsonl').exists():
            with (out/'step_trace.jsonl').open() as handle:steps=sum(1 for line in handle if line.strip())
        rows.append(dict(Method=j['response_probe_variant'],Seed=2,Inner='0.05' if j['inner_probe_radius_mode']=='fixed_l2' else 'eta*norm(g0)',Outer='0.05' if j['outer_probe_radius_mode']=='fixed_l2' else 'eta*norm(g0)',Status=r['status'],FAA=metrics.get('FAA_task_macro'),AAA=metrics.get('AAA_task_macro'),BWT=metrics.get('BWT'),Updates=steps,Run=j['id'],Source=j['output_dir']))
    atomic_json(HERE/'RESULTS.json',dict(updated_at=now(),queue_status=state['status'],protocol='ImageNet-A CL10 seed2 rank10 50 epochs/task, ordinary SGD .04/.02 epoch cosine; no history/Fisher/EWC/merge/projection',results=rows))
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);atomic_text(HERE/'RESULTS.csv',buf.getvalue())
    text=['# Current-G2 radius decoupling results','',f"Queue status: {state['status']}. Updated UTC: {now()}.",'','All performance values are task-equal percentages. Only accepted full 10-task runs enter FAA/AAA/BWT. Original is source-attributed reuse; smoke accuracies are excluded.','','| Method | Inner radius | Outer radius | Status | FAA | AAA | BWT | Updates |','|---|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        fmt=lambda v:'—' if v is None else f'{v:.3f}'
        text.append(f"| {r['Method']} | {r['Inner']} | {r['Outer']} | {r['Status']} | {fmt(r['FAA'])} | {fmt(r['AAA'])} | {fmt(r['BWT'])} | {r['Updates']} / 2550 |")
    text+=['','New variants submit only g2 from BASE; three gradient reads plus one passive post-update loss forward per normal step. Fixed radii are global L2 distances over trainable LoRA and classifier parameters.','']
    atomic_text(HERE/'RESULTS.md','\n'.join(text));return rows
if __name__=='__main__':print(json.dumps(collect()))
