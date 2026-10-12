#!/usr/bin/env python3
"""Read-only parent receipt validation in an isolated interpreter/import context."""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
parent=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(parent))
import common
import validate_job
if Path(common.__file__).resolve().parent!=parent or Path(validate_job.__file__).resolve().parent!=parent:
    raise RuntimeError('parent validator/common imported from wrong campaign')
suite=json.loads((parent/'suite.json').read_text())
receipts={}
for job in suite['jobs']:
    path=parent/'runtime/jobs'/job['id']/'completed.json'
    receipt=json.loads(path.read_text());actual=validate_job.validate_job(job)
    if receipt['job_id']!=job['id'] or receipt['exit']['returncode']!=0 or not validate_job.receipt_matches(receipt,actual):
        raise RuntimeError('parent completion invalid '+job['id'])
    receipts[job['id']]=common.sha256(path)
if len(receipts)!=len(suite['jobs']):raise RuntimeError('duplicate parent job IDs')
print(json.dumps({'status':'validated','parent_dir':str(parent),'parent_jobs_validated':len(receipts),
    'common_file':str(Path(common.__file__).resolve()),'validator_file':str(Path(validate_job.__file__).resolve()),
    'receipts_sha256':receipts}))
