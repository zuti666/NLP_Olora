#!/usr/bin/env python3
"""Freeze and deploy this new, isolated campaign without starting any GPU job."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from common import assert_host,now,sha256,verify_source

HERE=Path(__file__).resolve().parent
suite=json.loads((HERE/'suite.json').read_text())
guard=json.loads((HERE/'handoff_guard.json').read_text())
assert_host(suite)
if os.readlink('/proc/self/ns/pid')!=guard['execution_pid_namespace']:
    raise RuntimeError('deployment must run in pinned host PID namespace')
dest=Path(suite['run_dir'])
if dest.exists():raise FileExistsError('deployment requires a fresh campaign directory')
verified=verify_source({**suite,'source_manifest':str(HERE/'source_manifest.json')})
parent=Path(guard['parent_dir'])
for name,key in (('suite.json','parent_suite_sha256'),('RUN_PINS.json','parent_run_pins_sha256')):
    if sha256(parent/name)!=guard[key]:raise RuntimeError('parent pin changed '+name)
spec=Path('/data/140-1/users/liying/CopyFlatCV/.aris/compute/local-pilot-new-local-env-spec.json')
canonical=hashlib.sha256(json.dumps(json.loads(spec.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
if canonical!='574b252ec7b676947b1e431b88231959dcb69e79742d933f03cf354eb8c17798':
    raise RuntimeError('warm environment specification changed')
import torch,torchvision,timm
versions={'python':sys.version.split()[0],'torch':torch.__version__,'torchvision':torchvision.__version__,'timm':timm.__version__}
parent_reuse=json.loads((parent/'ENVIRONMENT_REUSE.json').read_text())
if versions!=parent_reuse['versions'] or canonical!=parent_reuse['canonical_sha256']:
    raise RuntimeError('environment differs from validated parent environment')
witness=parent/'reused_env_witness.json'
if sha256(witness)!=parent_reuse['witness_sha256']:raise RuntimeError('environment witness changed')
reuse={**parent_reuse,'checked_at':now(),'reused_from_campaign':str(parent),'no_GPU_probe_during_deployment':True}
with (HERE/'ENVIRONMENT_REUSE.json').open('x') as f:json.dump(reuse,f,indent=2);f.write('\n')
shutil.copyfile(witness,HERE/'reused_env_witness.json')
files=list(HERE.glob('*.py'))+list((HERE/'trainer').glob('*.py'))+list((HERE/'diagnostics').glob('*.py'))+list((HERE/'configs').glob('*.json'))
files += [HERE/n for n in ('suite.json','data_inventory.json','FROZEN_GATES.json','handoff_guard.json','source_manifest.json','ENVIRONMENT_REUSE.json','reused_env_witness.json','USER_PLAN.txt','PLAN.md')]
files += [p for p in (HERE/'review').iterdir() if p.is_file() and p.suffix in ('.py','.json','.md')]
pins={'created_at':now(),'files':{str(p.relative_to(HERE)):sha256(p) for p in sorted(set(files))},'source_manifest_sha256':verified['source_manifest_sha256']}
with (HERE/'RUN_PINS.json').open('x') as f:json.dump(pins,f,indent=2);f.write('\n')
shutil.copytree(HERE,dest,ignore=shutil.ignore_patterns('__pycache__','runtime','results'))
for name,expected in pins['files'].items():
    if sha256(dest/name)!=expected:raise RuntimeError('deployed bytes differ '+name)
print(json.dumps({'deployed':str(dest),'files_pinned':len(pins['files']),'source_files':len(verified['source_files']),'environment_warm_reused':True,'GPU_jobs_started':0}))
