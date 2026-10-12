"""Create only two authorized natural-step full runs and three short checks."""
from pathlib import Path
import json,hashlib,copy,shutil
H=Path(__file__).resolve().parent
P=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-component-horizon-recovery-20261001T230000Z')
D=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r025-c21-ab-20261002T080000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
s=json.loads((P/'suite.json').read_text());old=copy.deepcopy(s)
s.update(run_id=D.name,run_dir=str(D),source_manifest=str(D/'source_manifest.json'),parent_gate=str(D/'handoff_guard.json'),max_parallel=3)
for k in ['TMPDIR','TMP','TEMP']:s['environment'][k]='/tmp/r025-c21-ab-20261002T080000Z'
s['jobs']=[]
smokes=[f'smoke_{r}_natural_r025_s2' for r in ['c21_base','c21_perp','us']]
for smoke,rules in [(True,['c21_base','c21_perp','us']),(False,['c21_base','c21_perp'])]:
 for rule in rules:
  template=next(j for j in old['jobs'] if j['kind']=='horizon' and j['rule']=='ug' and j['scale']=='natural' and j['smoke']==smoke)
  j=copy.deepcopy(template);jid=f'{"smoke" if smoke else "screen"}_{rule}_natural_r025_s2'
  j.update(id=jid,rule=rule,config=str(D/'configs'/f'{jid}.json'),output_dir=str(D/'outputs'/jid),depends_on=[] if smoke else smokes)
  c=json.loads(Path(template['config']).read_text());c.update(outputs_root=j['output_dir'],prefix=jid+'_c21ab_20261002T080000Z',component_rule=rule)
  write(H/'configs'/f'{jid}.json',c);j['config_sha256']=sha(H/'configs'/f'{jid}.json')
  cmd=[s['python'],str(D/'trainer/component_horizon.py'),'--source',s['source_root'],'--config',j['config'],'--output-dir',j['output_dir']]
  if smoke:cmd+=['--max-tasks','2','--max-epochs','1','--max-batches','1']
  j['command']=cmd;s['jobs'].append(j)
write(H/'suite.json',s)
baselines=[j['id'] for j in old['jobs'] if j['kind']=='horizon' and not j['smoke']]+['smoke_us_natural_r025_s2']
write(H/'handoff_guard.json',dict(parent_dir=str(P),parent_suite_sha256=sha(P/'suite.json'),parent_run_pins_sha256=sha(P/'RUN_PINS.json'),baseline_jobs=baselines,expected_parent_jobs=len(baselines),execution_pid_namespace='pid:[4026531836]',release_policy='Validate selected completed training baselines only. Independent A/B cohort; no dependency on unrelated curvature diagnostics.',new_phase2_phase3_auto_launch=False))
shutil.copyfile('/home-local/liying/.codex/attachments/fe3092ed-7b02-4d33-9f74-6744661aef9e/Pasted text.txt',H/'USER_PLAN.txt')
print('Created',len(s['jobs']),'jobs: 3 short checks + 2 full natural-step A/B runs')
