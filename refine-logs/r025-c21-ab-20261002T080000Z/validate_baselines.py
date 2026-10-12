"""Read-only revalidation of completed reference training, never unrelated diagnostics."""
import json,sys,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
g=json.loads((H/'handoff_guard.json').read_text());P=Path(g['parent_dir'])
sys.path.insert(0,str(P))
from validate_job import validate_job,receipt_matches
from common import sha256,atomic_json,now
for name,key in [('suite.json','parent_suite_sha256'),('RUN_PINS.json','parent_run_pins_sha256')]:
 if sha256(P/name)!=g[key]:raise RuntimeError('reference pin changed '+name)
for name,digest in json.loads((P/'RUN_PINS.json').read_text())['files'].items():
 if sha256(P/name)!=digest:raise RuntimeError('reference code changed '+name)
s=json.loads((P/'suite.json').read_text());jobs={j['id']:j for j in s['jobs']};rows=[];files={}
for jid in g['baseline_jobs']:
 j=jobs[jid];rp=P/'runtime/jobs'/jid/'completed.json';receipt=json.loads(rp.read_text());v=validate_job(j)
 if receipt['job_id']!=jid or receipt['exit']['returncode']!=0 or not receipt_matches(receipt,v):raise RuntimeError('invalid baseline '+jid)
 files[str(rp)]=sha256(rp)
 out=Path(j['output_dir'])
 for name in ['final_summary.json','step_trace.jsonl','task_identities.json','effective_config.json','task_matrix.json','task_checkpoints.json']:
  if (out/name).exists():files[str(out/name)]=sha256(out/name)
 for checkpoint in v['checkpoints']:
  for name,digest in checkpoint['files_sha256'].items():
   files[str(out/'checkpoints'/f"task_{checkpoint['task']:03d}"/name)]=digest
 rows.append(dict(job_id=jid,rule=j['rule'],scale=j['scale'],smoke=j['smoke'],summary=v['summary'],receipt_sha256=sha256(rp)))
atomic_json(H/'runtime/BASELINES_VALIDATED.json',dict(status='validated',at=now(),guard_sha256=sha256(H/'handoff_guard.json'),files=files,baselines=rows))
print(json.dumps(dict(baselines_validated=len(rows),formal_training=sum(not x['smoke'] for x in rows))))
