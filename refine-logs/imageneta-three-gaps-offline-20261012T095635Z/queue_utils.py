"""Small host/GPU lease utilities; no training or network."""
import csv
import io
import json
import math
import os
from pathlib import Path
import hashlib
import subprocess
import tempfile
import time

def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def identity(pid):
    try:
        s=Path('/proc/%d/stat'%pid).read_text();parts=s[s.rfind(')')+2:].split()
        if parts[0]=='Z':return None
        return dict(pid=int(pid),proc_starttime=parts[19])
    except (FileNotFoundError,ProcessLookupError):return None
def alive(record):return bool(record and identity(record['pid'])==record)
def query_gpus():
    data=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=30)
    rows={}
    for row in csv.reader(io.StringIO(data)):
        index,uuid,mem,util=[x.strip() for x in row]
        rows[int(index)]=dict(index=int(index),uuid=uuid,memory_used_mb=int(mem),utilization_percent=int(util),compute_pids=[])
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=30)
    for row in csv.reader(io.StringIO(apps)):
        if len(row)==2:
            uuid,pid=[x.strip() for x in row]
            for gpu in rows.values():
                if gpu['uuid']==uuid:gpu['compute_pids'].append(int(pid))
    return rows
def idle(gpu):return bool(gpu and gpu['memory_used_mb']<500 and gpu['utilization_percent']<5 and not gpu['compute_pids'])
def finite(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite stored output')
    if isinstance(value,dict):
        for v in value.values():finite(v)
    if isinstance(value,list):
        for v in value:finite(v)
def restored(record):
    if not record or not all(value is True for value in record.values()):raise ValueError('State/RNG restoration failed')
def validate_output(job):
    output=Path(job['output_dir']); receipt=read(output/'completion_receipt.json')
    aggregate=read(output/job['report_file']);finite(receipt);finite(aggregate)
    if receipt['submitted_training_updates']!=0 or aggregate['submitted_training_updates']!=0:raise ValueError('Training update in offline job')
    if receipt['results_sha256']!=sha(output/job['report_file']):raise ValueError('Result hash differs from receipt')
    if len(aggregate['results'])!=job['expected_items']:raise ValueError('Incomplete fixed scope')
    if job['kind']=='curvature':
        if receipt['status']!='accepted_execution' or aggregate['status']!='completed_exploratory':raise ValueError('Curvature execution not complete')
        for row in aggregate['results']:
            restored(row['diagnostic']['restoration'])
            if 'fp64_restoration' in row['diagnostic']:restored(row['diagnostic']['fp64_restoration'])
            d=row['diagnostic']
            if not d['candidate_comparisons_permitted'] and d['scales']:raise ValueError('Unverified curvature comparison')
            if d['candidate_comparisons_permitted'] and len(d['scales'])!=2:raise ValueError('Missing declared scales')
    else:
        if receipt['status']!='accepted' or aggregate['status']!='completed':raise ValueError('Diagnostic not complete')
        for row in aggregate['results']:
            restored(row['restoration'])
            if job['kind']=='matching' and row['HVP_calls']!=0:raise ValueError('Unexpected Hessian dependency')
            if job['kind']=='evaluation':
                if row['test_used_for_selection']:raise ValueError('Test selection forbidden')
                if not all(x['accepted'] for x in row['agreement'].values()):raise ValueError('Historical metrics changed')
                if sha(row['logits_file'])!=row['logits_sha256']:raise ValueError('Logits hash changed')
                for data in row['datasets'].values():
                    if data['cross_group_error_count']+data['within_group_error_count']!=data['joint40']['errors']:raise ValueError('Error partition not exact')
    return dict(report_sha256=sha(output/job['report_file']),receipt_sha256=sha(output/'completion_receipt.json'),items=len(aggregate['results']),scientific_outcome=aggregate.get('scientific_outcome'),accepted_matches=aggregate.get('accepted_matches'))
