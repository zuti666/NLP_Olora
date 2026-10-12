from pathlib import Path
import json,math,re,hashlib,time
root=Path(__file__).resolve().parent
launch=json.loads((root/'preflight/smoke_launch.json').read_text());pid=launch['pid']
stat=Path(f'/proc/{pid}/stat')
if stat.exists():
 fields=stat.read_text().rsplit(')',1)[1].split();assert fields[0]=='Z','Smoke still running'
p=root/'preflight/smoke.log';s=p.read_text(errors='replace').replace('\r','\n')
ms=list((root/'outputs/s3_smoke_s2').rglob('*_cl_metrics.json'));assert len(ms)==1
d=json.loads(ms[0].read_text());a=d['cnn']['matrices']['final'];assert len(a)==2
assert all(math.isfinite(a[i][j])for i in range(2)for j in range(i+1))
for t in (0,1):assert f'[WeightMerge] task={t} applied=True' in s
assert re.search(r'\[TaskStart\] task=1 .*has_fisher=True',s)
assert 'Traceback (most recent call last)' not in s
assert not re.search(r'\bLoss\s+(?:nan|[+-]?inf)\b',s,re.I)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=dict(passed=True,scope='integration only; 2tasks x2epochs interval1, not formal performance',pid=pid,finished_check_unix=time.time(),tasks=2,matrix=[r[:i+1]for i,r in enumerate(a)],weight_merge_both_tasks=True,task1_has_fisher=True,finite_training_loss=True,metrics_path=str(ms[0]),metrics_sha256=sha(ms[0]),log_sha256=sha(p),config_sha256=sha(root/'configs/s3_smoke_s2.json'),source_manifest_sha256=sha(root/'source_manifest.json'))
(root/'preflight/smoke_validation.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r))
