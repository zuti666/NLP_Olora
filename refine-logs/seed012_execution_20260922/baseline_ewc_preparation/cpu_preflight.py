"""Import selected learners and resolve data metadata only; never create a model."""
import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
HERE=Path(__file__).resolve().parent
args=argparse.ArgumentParser();args.add_argument('--source',choices=['baseline','ewc'],required=True);ns=args.parse_args()
env=json.loads((HERE/'environment.json').read_text());os.environ.update(env)
jobs=json.loads((HERE/'candidate_jobs.json').read_text())['jobs']
selected=[j for j in jobs if j['source_id']==ns.source and j['stage']=='formal_candidate' and j['seed']==0]
source=Path(selected[0]['cwd']);sys.path.insert(0,str(source/'src'));sys.path.insert(0,str(source))
import torch
import numpy as np
import timm
import torchvision
entry=importlib.import_module('src.main')
modules=['seqlora','inflora','olora','sdlora'] if ns.source=='baseline' else ['as1_normfisher_fishershape']
imports={}
for name in modules:
    module=importlib.import_module('models_LoRAbasedCL.'+name)
    assert Path(module.__file__).resolve().is_relative_to(source.resolve())
    imports[name]=module.__file__
from utils.data_manager import DataManager
data=[]
for dataset in ['aircraft','cars196','cub200','flowers','het5']:
    j=next(j for j in selected if j['dataset']==dataset)
    cfg=json.loads(Path(j['config']).read_text())
    dm=DataManager(dataset,False,0,cfg['init_cls'],cfg['increment'],cfg)
    train=[(str(Path(str(p)).resolve()),int(y)) for p,y in zip(dm._train_data,dm._train_targets)]
    test=[(str(Path(str(p)).resolve()),int(y)) for p,y in zip(dm._test_data,dm._test_targets)]
    trainpaths={p for p,_ in train};testpaths={p for p,_ in test}
    missing=[p for p in trainpaths|testpaths if not Path(p).is_file()]
    assert not missing,(dataset,missing[:5])
    assert not trainpaths&testpaths,dataset
    expect_tasks=5 if dataset=='het5' else 10
    assert dm.nb_tasks==expect_tasks,(dataset,dm.nb_tasks)
    digest=lambda z:hashlib.sha256(json.dumps(z,separators=(',',':')).encode()).hexdigest()
    record={'dataset':dataset,'data_root_config':cfg['data_root'],'task_widths':list(dm._increments),
            'train_count':len(train),'test_count':len(test),'classes':len(set(dm._train_targets)|set(dm._test_targets)),
            'train_paths_labels_sha256':digest(train),'test_paths_labels_sha256':digest(test),
            'train_first':train[0],'test_first':test[0],'missing_files':len(missing),'train_test_realpath_overlap':0}
    manifest=HERE/'data_manifests'/f'{ns.source}_{dataset}.json';manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps({'train':train,'test':test},ensure_ascii=False,separators=(',',':'))+'\n')
    record['manifest']=str(manifest);data.append(record)
    print(dataset,'tasks',record['task_widths'],'n',len(train),len(test),flush=True)
assert not torch.cuda.is_initialized()
out={'created_utc':datetime.now(timezone.utc).isoformat(),'source_id':ns.source,'source_root':str(source),'entrypoint':entry.__file__,
     'imports':imports,'datasets':data,'python':sys.executable,'torch':torch.__version__,'timm':timm.__version__,'torchvision':torchvision.__version__,
     'cuda_initialized':False,'model_created':False,'training_launched':False,'pass':True}
(HERE/f'cpu_preflight_{ns.source}.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('PASS',ns.source)
