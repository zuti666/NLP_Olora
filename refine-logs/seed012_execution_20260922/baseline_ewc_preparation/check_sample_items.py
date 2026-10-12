"""CPU-only first-item transform checks for both resolved train and test datasets."""
import argparse
import importlib
import json
import logging
import os
import sys
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
os.environ.update(json.loads((HERE/'environment.json').read_text()))
os.environ['CUDA_VISIBLE_DEVICES']=''
parser=argparse.ArgumentParser();parser.add_argument('--source',choices=['baseline','ewc'],required=True);arg=parser.parse_args()
jobs=json.loads((HERE/'candidate_jobs.json').read_text())['jobs']
jobs=[j for j in jobs if j['source_id']==arg.source and j['stage']=='formal_candidate' and j['seed']==0]
root=Path(jobs[0]['cwd']);sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root))
import numpy as np
import torch
torch.set_num_threads(2)
from utils.data_manager import DataManager
logging.getLogger().setLevel(logging.WARNING)
records=[]
for dataset in ['aircraft','cars196','cub200','flowers','het5']:
    config=json.loads(Path(next(j for j in jobs if j['dataset']==dataset)['config']).read_text())
    torch.random.default_generator.manual_seed(0);np.random.seed(0)
    dm=DataManager(dataset,False,0,config['init_cls'],config['increment'],config)
    classes=sum(dm._increments)
    for split in ['train','test']:
        ds=dm.get_dataset(np.arange(classes),source=split,mode=split)
        idx,image,target=ds[0]
        assert isinstance(image,torch.Tensor)
        assert tuple(image.shape)==(3,224,224),(dataset,split,image.shape)
        assert bool(torch.isfinite(image).all()),(dataset,split)
        assert 0<=int(target)<classes
        records.append({'dataset':dataset,'split':split,'item_index':int(idx),'shape':list(image.shape),'dtype':str(image.dtype),
                        'device':str(image.device),'finite':True,'label':int(target),'label_range':[0,classes-1],
                        'min':float(image.min()),'max':float(image.max()),'dataset_length':len(ds)})
assert not torch.cuda.is_initialized()
out={'checked_utc':datetime.now(timezone.utc).isoformat(),'source_id':arg.source,'source_root':str(root),'pass':True,
     'scope':'one first train item and one first test item per dataset; not a full image-content validation',
     'items':records,'cuda_initialized':False,'training_launched':False}
(HERE/f'sample_items_{arg.source}.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS',arg.source,len(records),'items, all3x224x224 finite and labels in range; CUDA uninitialized')
