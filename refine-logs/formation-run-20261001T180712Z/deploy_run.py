#!/usr/bin/env python3
"""Verify pinned warm environment/source, freeze this isolated attempt, then copy."""
import hashlib
import json
import shutil
import socket
from pathlib import Path
from datetime import datetime,timezone
from common import verify_source,sha256

HERE=Path(__file__).resolve().parent
suite=json.loads((HERE/'suite.json').read_text())
dest=Path(suite['run_dir'])
if socket.gethostname()!=suite['execution_hostname']: raise RuntimeError('wrong host')
if dest.exists(): raise FileExistsError('deployment must be a fresh directory')
local_suite={**suite,'source_manifest':str(HERE/'source_manifest.json')}
verified=verify_source(local_suite)
spec=Path('/data/140-1/users/liying/CopyFlatCV/.aris/compute/local-pilot-new-local-env-spec.json')
canonical=hashlib.sha256(json.dumps(json.loads(spec.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
if canonical!='574b252ec7b676947b1e431b88231959dcb69e79742d933f03cf354eb8c17798': raise RuntimeError('environment changed')
import torch,torchvision,timm,sys
versions={'python':sys.version.split()[0],'torch':torch.__version__,'torchvision':torchvision.__version__,'timm':timm.__version__}
if versions!={'python':'3.11.15','torch':'2.4.1+cu121','torchvision':'0.19.1+cu121','timm':'1.0.9'}: raise RuntimeError('warm environment differs '+repr(versions))
witness=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/reused_env_witness.json')
reuse={'unchanged':True,'no_environment_changes':True,'spec_path':str(spec),'canonical_sha256':canonical,'raw_sha256':sha256(spec),
       'historical_validated_witness':str(witness),'witness_sha256':sha256(witness),'versions':versions,
       'checked_at':datetime.now(timezone.utc).isoformat()}
with (HERE/'ENVIRONMENT_REUSE.json').open('x') as f: json.dump(reuse,f,indent=2);f.write('\n')
shutil.copyfile(witness,HERE/'reused_env_witness.json')
files=[p for p in HERE.glob('*.py')]+[p for p in (HERE/'trainer').glob('*.py')]+list((HERE/'configs').glob('*.json'))
files += [HERE/n for n in ('suite.json','selection_split.json','FROZEN_GATES.json','conditional_plan.json','source_manifest.json','ENVIRONMENT_REUSE.json','reused_env_witness.json','USER_PLAN.txt')]
pins={'created_at':datetime.now(timezone.utc).isoformat(),'files':{str(p.relative_to(HERE)):sha256(p) for p in files},'source_manifest_sha256':verified['source_manifest_sha256']}
with (HERE/'RUN_PINS.json').open('x') as f: json.dump(pins,f,indent=2);f.write('\n')
shutil.copytree(HERE,dest,ignore=shutil.ignore_patterns('__pycache__','runtime','results'))
for name,expected in pins['files'].items():
    if sha256(dest/name)!=expected:raise RuntimeError('deploy byte mismatch '+name)
print(json.dumps({'deployed':str(dest),'files_pinned':len(pins['files']),'source_files':len(verified['source_files']),'env_warm_reused':True}))
