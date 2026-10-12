#!/usr/bin/env python3
"""Freeze stratified selection/reporting IDs before any new training result."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')
sys.path.insert(0,str(SOURCE))
from utils.data_manager import DataManager

def h(value):
    return hashlib.sha256(value).hexdigest()

cfg=json.loads(Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-stage1-task0-run-20260930T213643Z/configs/task0_s_k10_s2.json').read_text())
dm=DataManager('imageneta',False,0,20,20,cfg)
split={'schema_version':1,'split_seed':1729,'legacy_seed2_exposure':'exploratory; full official test was previously inspected',
       'rule':'canonical image ID SHA256 ordering within class; largest remainder floor(N/3); n>=2 both sides; singleton reporting only'}
for task in (0,1):
    data,targets,_=dm.get_dataset(np.arange(task*20,(task+1)*20),source='test',mode='test',ret_data=True)
    targets=np.asarray(targets,dtype=np.int64)
    ids=[str(x) for x in data]
    if len(set(ids))!=len(ids): raise ValueError('duplicate canonical image IDs')
    groups={int(c):np.flatnonzero(targets==c).tolist() for c in np.unique(targets)}
    count={c:(max(1,min(len(ii)-1,len(ii)//3)) if len(ii)>=2 else 0) for c,ii in groups.items()}
    total=len(targets)//3
    while sum(count.values())<total:
        c=min((c for c in groups if count[c]<len(groups[c])-1),key=lambda c:(-(len(groups[c])/3-count[c]),c))
        count[c]+=1
    while sum(count.values())>total:
        c=min((c for c in groups if count[c]>1),key=lambda c:(-(count[c]-len(groups[c])/3),c))
        count[c]-=1
    diag=[]
    for c,ii in groups.items():
        order=sorted(ii,key=lambda i:(h(f'1729|{task}|{c}|{ids[i]}'.encode()),ids[i]))
        diag.extend(order[:count[c]])
    diag=sorted(diag)
    report=sorted(set(range(len(targets)))-set(diag))
    train=dm.get_dataset(np.arange(task*20,(task+1)*20),source='train',mode='train')
    split[f'task{task}']={'dataset_size':len(targets),'labels_sha256':h(targets.tobytes()),
        'canonical_ids_sha256':h(json.dumps(ids,ensure_ascii=False,separators=(',',':')).encode()),
        'diagnostic_indices':diag,'reporting_indices':report,'diagnostic_n':len(diag),'reporting_n':len(report),
        'per_class_counts':{str(c):len(ii) for c,ii in groups.items()},'diagnostic_per_class_counts':count,
        'train_n':len(train),'train_batches_per_epoch':(len(train)+127)//128,
        'singleton_classes_reporting_only':[c for c,ii in groups.items() if len(ii)==1]}
    assert not set(diag)&set(report) and len(diag)+len(report)==len(targets)
assert split['task0']['dataset_size']==307 and split['task0']['diagnostic_n']==102
assert split['task0']['train_n']==1259 and split['task1']['train_n']==997
path=HERE/'selection_split.json'
with path.open('x') as f: json.dump(split,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:{x:v[x] for x in ('dataset_size','diagnostic_n','reporting_n','train_n','train_batches_per_epoch')} for k,v in split.items() if k.startswith('task')}))
