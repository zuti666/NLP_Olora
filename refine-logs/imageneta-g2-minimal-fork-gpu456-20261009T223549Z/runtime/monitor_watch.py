"""Passive host NVML/PID/trace monitor for this authorized GPU456 queue."""
import json,sys,os,time,math
from pathlib import Path
H=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(H))
from common import read_json,atomic_json,now,process_matches,query_gpus,proc_identity
suite=read_json(H/'suite.json');runtime=H/'runtime'
atomic_json(runtime/'monitor_identity.json',dict(process=proc_identity(os.getpid()),started_at=now(),passive=True,interval_seconds=30))
while True:
    state=read_json(runtime/'queue_state.json');jobs={j['id']:j for j in suite['jobs']};records={}
    for jid,j in jobs.items():
        q=state['jobs'][jid];r={k:q[k] for k in ('status','gpu','gpu_uuid','process','error') if k in q};out=Path(j['output_dir']);trace=out/'step_trace.jsonl'
        if trace.exists():
            rows=[]
            with trace.open() as f:
                for line in f:
                    try:rows.append(json.loads(line))
                    except json.JSONDecodeError:pass
            if rows:
                last=rows[-1];r.update(actual_updates=len(rows),expected_updates=j['expected_steps'],last={k:last.get(k) for k in ('task','epoch','batch','loss0','ordinary_gradient_calls','sgd_updates','nominal_lr','inner_probe_radius_theoretical','outer_probe_radius_theoretical','actual_parameter_step_norm')},finite_losses=all(math.isfinite(x['loss0']) for x in rows),exact_sgd=all(x['natural_sgd_exact'] and x['sgd_updates']==1 for x in rows))
        records[jid]=r
    gpus={str(i):g for i,g in query_gpus().items() if i in {4,5,6}}
    alive=process_matches(state.get('controller'));report=dict(at=now(),controller_alive=alive,queue_status=state['status'],gpus=gpus,jobs=records,passive=True)
    atomic_json(runtime/'monitor_snapshot.json',report)
    with (runtime/'monitor_history.jsonl').open('a') as f:f.write(json.dumps(report,allow_nan=False)+'\n')
    import subprocess
    subprocess.run([suite['python'],str(H/'collect_results.py')],cwd=H,check=True)
    if not alive or state['status'] in ('completed','halted_failure'):break
    time.sleep(30)
