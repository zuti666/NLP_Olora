#!/usr/bin/env python3
"""Validate prior campaign in its own import context; no GPU operations."""
import json
from pathlib import Path
import sys
parent=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(parent))
import common
import validate_job
suite=json.loads((parent/'suite.json').read_text())
receipts={}
for job in suite['jobs']:
    path=parent/'runtime/jobs'/job['id']/'completed.json'
    receipt=json.loads(path.read_text());actual=validate_job.validate_job(job)
    if receipt['job_id']!=job['id'] or receipt['exit']['returncode']!=0 or not validate_job.receipt_matches(receipt,actual):
        raise RuntimeError('parent completion invalid '+job['id'])
    receipts[job['id']]=common.sha256(path)
print(json.dumps({'status':'validated','parent_dir':str(parent),'parent_jobs_validated':len(receipts),'receipts_sha256':receipts}))
