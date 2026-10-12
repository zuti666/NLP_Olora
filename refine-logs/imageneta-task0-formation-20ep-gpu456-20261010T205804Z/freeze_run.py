#!/usr/bin/env python3
"""Seal reproducible inputs; deliberately exclude live outputs/review receipts."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
H=Path(__file__).resolve().parent
paths=[]
for pattern in ['*.py','configs/*.json','trainer/*.py']:
    paths.extend(H.glob(pattern))
paths.extend(H/n for n in ['EXPERIMENT_PLAN.md','RUN.md','protocol.json','suite.json','dev_split.json','source_manifest.json','environment_spec.json'])
files={str(p.relative_to(H)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
(H/'RUN_PINS.json').write_text(json.dumps({'sealed_at':datetime.now(timezone.utc).isoformat(),'files':files},indent=2)+'\n')
print(json.dumps({'sealed_files':len(files),'pins':str(H/'RUN_PINS.json')}))
