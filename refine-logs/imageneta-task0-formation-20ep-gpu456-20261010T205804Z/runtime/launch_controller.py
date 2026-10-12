#!/usr/bin/env python3
"""Actual-host durable launch of this authorized idle-GPU campaign."""
import hashlib,json,os,subprocess,socket,time
from pathlib import Path
H=Path(__file__).resolve().parents[1]
P='/home-local/liying/.conda/envs/Pilot_new_local/bin/python'
suite=json.loads((H/'suite.json').read_text())
if socket.gethostname()!=suite['execution_hostname'] or os.readlink('/proc/self/ns/pid')!=suite['execution_pid_namespace']:raise RuntimeError('Actual pinned host namespace required')
check=subprocess.run([P,str(H/'run_queue.py'),'--suite',str(H/'suite.json'),'--dry-run'],cwd=H,capture_output=True,text=True,check=True)
(H/'runtime/HOST_DRY_RUN.log').write_text(check.stdout+check.stderr)
for k in ['implementation_acceptance_file','independent_implementation_review_file','fresh_doc_check']:
 d=json.loads(Path(suite[k]).read_text());assert d['status']=='passed' and all(d['checks'].values()),k
 for f,v in d.get('evidence_hashes',{}).items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==v,f
if (H/'runtime/LAUNCH_RECEIPT.json').exists():raise RuntimeError('Controller already launched; inspect recorded PID before considering restart')
log=open(H/'runtime/controller.log','xb')
env=os.environ.copy();env.update(suite['environment'])
try:
 child=subprocess.Popen([P,str(H/'run_queue.py'),'--suite',str(H/'suite.json')],cwd=H,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
finally:log.close()
time.sleep(2)
if child.poll() is not None:raise RuntimeError('Controller exited at startup; inspect controller.log')
s=Path('/proc')/str(child.pid)/'stat';raw=s.read_text();fields=raw[raw.rfind(')')+2:].split()
r={'pid':child.pid,'proc_starttime':fields[19],'hostname':socket.gethostname(),'pid_namespace':os.readlink('/proc/self/ns/pid'),'allowed_gpus':suite['allowed_gpus'],'max_parallel':suite['max_parallel'],'gpu_policy':'idle only; no preemption','command':[P,str(H/'run_queue.py'),'--suite',str(H/'suite.json')],'dry_run_stdout':check.stdout.strip(),'suite_sha256':hashlib.sha256((H/'suite.json').read_bytes()).hexdigest()}
(H/'runtime/LAUNCH_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
