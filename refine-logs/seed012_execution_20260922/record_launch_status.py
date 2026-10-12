"""Write a timestamped human-readable snapshot of the live campaign."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'state.json').read_text());result=json.loads((ROOT/'results.json').read_text())
now=datetime.now(timezone.utc).isoformat()
lines=['# Seed012 campaign launch status','',f'Snapshot: {now}. Live state is state.json; this report is not a live progress indicator.','',
 f'Controller PID: {s["pid"]}; status: {s["status"]}. Physical GPUs: 2, 3, 4, 5, at most one job per device.','',
 'Joost multi-seed inventory: no eligible Aircraft/Cars/CUB/Flowers/Het5 run; available multi-seed datasets are ImageNet-A/R, CIFAR100 and DomainNet.', '',
 'Cohort: 99 newly queued full runs + 6 validated archived seed0 results = 105 results, 35 groups × seeds0/1/2. Nine extra two-task one-epoch smoke runs are excluded from statistics.','',
 '| Stage | Status | Count |','|---|---|---:|']
for key,n in sorted(Counter((j['stage'],j['status'])for j in s['jobs'].values()).items()):lines.append(f'| {key[0]} | {key[1]} | {n} |')
lines+=['','| GPU | PID | Current job |','|---:|---:|---|']
for name,j in s['jobs'].items():
 if j['status']=='running':lines.append(f'| {j["gpu"]["index"]} | {j["pid"]} | {name} |')
lines+=['',f'Collector verified: {result["counts"]}.','',
 'Environment and dependency witnesses, CPU data/weight checks and independent documented-command dry-run passed. Full-run metrics require finite complete task matrices; errors park the affected group instead of changing hyperparameters.','',
 'Files: README.md (policy/reproduction), suite.json (all frozen jobs/configs), queue.log, logs/<job>.log, receipts/<job>.json, SUMMARY.md, results.json and results.csv.','',
 'Limitations remain documented: source recovery does not establish binding to every old run; the new campaign does not pool unbound historical seed0 values. Six reused formal runs have source/config/data evidence, but historical image-byte hashes and GAM-Het5 historical checkpoint-byte hash are unavailable.','']
(ROOT/'LAUNCH_REPORT.md').write_text('\n'.join(lines))
print('\n'.join(lines[:20]))
