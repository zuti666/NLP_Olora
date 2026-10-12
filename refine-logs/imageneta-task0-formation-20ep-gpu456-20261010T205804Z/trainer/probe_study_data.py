"""Deterministic train-only holdout for the shared-parent G2 study.

The frozen source arrays/test set/class partition remain intact. Only training
get_dataset excludes pre-registered train indices. Development uses deterministic
native test transforms on held-out training images, never the test set.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

SPLIT_SEED = 20261010
SPLIT_RATIO = .1
SCHEMA = 'G2StudyTrainOnlySplit_v1'

def _digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _fingerprint(data, targets):
    h = hashlib.sha256()
    for i, (item, label) in enumerate(zip(data, targets)):
        h.update(json.dumps([i,str(item),int(label)],separators=(',',':')).encode())
        h.update(b'\n')
    return h.hexdigest()

def select_indices(targets, seed=SPLIT_SEED, ratio=SPLIT_RATIO):
    targets=np.asarray(targets)
    train=[];dev=[];counts=[]
    for label in sorted(map(int,np.unique(targets))):
        ids=np.flatnonzero(targets==label)
        if len(ids)<2:raise ValueError('Every class needs at least two training samples')
        # A class-local generator makes membership independent of other classes.
        ordered=np.random.default_rng(seed+label).permutation(ids)
        count=min(len(ids)-1,max(1,int(math.floor(len(ids)*ratio))))
        dev.extend(map(int,ordered[:count]));train.extend(map(int,ordered[count:]))
        counts.append(dict(label=label,full=len(ids),train=len(ids)-count,dev=count))
    return sorted(train),sorted(dev),counts

def _source_modules(source):
    source=str(Path(source).resolve())
    if source not in sys.path:sys.path.insert(0,source)
    from utils.data_manager import DataManager, DummyDataset
    return DataManager,DummyDataset

def prepare_split(source,cfg,destination):
    DataManager,_=_source_modules(source)
    dm=DataManager('imageneta',False,2,20,20,cfg)
    train,dev,counts=select_indices(dm._train_targets)
    result=dict(schema=SCHEMA,split_seed=SPLIT_SEED,dev_ratio=SPLIT_RATIO,
                dev_count_rule='floor(n*0.1), at least 1, at most n-1, per class',
                train_indices=train,dev_indices=dev,per_class=counts,
                full_train_samples=len(dm._train_targets),full_test_samples=len(dm._test_targets),
                class_order=list(map(int,dm._class_order)),task_increments=list(map(int,dm._increments)),
                train_source_fingerprint=_fingerprint(dm._train_data,dm._train_targets),
                test_source_fingerprint=_fingerprint(dm._test_data,dm._test_targets),
                test_unchanged=True,selection_data='held-out train only; test is excluded',
                native_train_paths=[str(x) for x in dm._train_data],
                native_train_targets=list(map(int,dm._train_targets)))
    validate_split(dm,result)
    dest=Path(destination)
    if dest.exists():
        if json.loads(dest.read_text())!=result:raise FileExistsError('Refusing to change a frozen split')
    else:
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

def validate_split(dm,split):
    if split.get('schema')!=SCHEMA or split.get('split_seed')!=SPLIT_SEED or split.get('dev_ratio')!=SPLIT_RATIO:
        raise ValueError('Unregistered holdout schema/seed/ratio')
    if len(dm._train_targets)!=5981 or len(dm._test_targets)!=1519:
        raise AssertionError('Frozen native train/test sizes changed')
    if split['train_source_fingerprint']!=_fingerprint(dm._train_data,dm._train_targets):
        raise AssertionError('Training image order/path/label receipt changed')
    if split['test_source_fingerprint']!=_fingerprint(dm._test_data,dm._test_targets):
        raise AssertionError('Test image order/path/label receipt changed')
    train,dev,counts=select_indices(dm._train_targets)
    if split['train_indices']!=train or split['dev_indices']!=dev or split['per_class']!=counts:
        raise AssertionError('Split does not exactly reproduce the pre-registered class-local split')
    if set(train)&set(dev) or sorted(train+dev)!=list(range(5981)):
        raise AssertionError('Train/dev overlap or missing original training samples')
    if split['class_order']!=list(range(200)) or split['task_increments']!=[20]*10:
        raise AssertionError('Complete native class partition changed')
    return True

def build_data_manager(source,cfg,split_path):
    DataManager,DummyDataset=_source_modules(source)
    class StudyDataManager(DataManager):
        def get_dataset(self, indices,source,mode,appendent=None,ret_data=False,m_rate=None):
            if source!='train':return super().get_dataset(indices,source,mode,appendent,ret_data,m_rate)
            if appendent is not None and len(appendent):raise AssertionError('Replay/appendent is forbidden')
            if m_rate is not None:raise AssertionError('Random extra training subsets are forbidden')
            wanted=set(map(int,indices));full_ids=[i for i in self.study_train_indices if int(self._train_targets[i]) in wanted]
            data=self._train_data[full_ids];targets=self._train_targets[full_ids]
            dataset=DummyDataset(data,targets,self._build_transform(mode),self.use_path,loader_mode=self.loader_mode)
            dataset.study_full_train_indices=full_ids;dataset.study_raw_train_indices=full_ids
            return (data,targets,dataset) if ret_data else dataset
        def get_dev_dataset(self,task):
            lo,hi=self.get_task_class_range(task)
            ids=[i for i in self.study_dev_indices if lo<=int(self._train_targets[i])<hi]
            dataset=DummyDataset(self._train_data[ids],self._train_targets[ids],self._build_transform('test'),
                                 self.use_path,loader_mode=self.loader_mode)
            dataset.study_full_train_indices=ids;dataset.study_raw_train_indices=ids
            return dataset
    dm=StudyDataManager('imageneta',False,2,20,20,cfg)
    split=json.loads(Path(split_path).read_text());validate_split(dm,split)
    dm.study_split=split;dm.study_train_indices=tuple(split['train_indices']);dm.study_dev_indices=tuple(split['dev_indices'])
    dm.study_split_sha256=_digest_file(split_path)
    return dm

def build_dev_loader(dm,task,cfg):
    return dm.build_dataloader(dm.get_dev_dataset(task),batch_size=128,shuffle=False,
        num_workers=cfg.get('eval_num_workers',0),generator=torch.Generator().manual_seed(SPLIT_SEED+task),persistent_workers=False)

def split_fingerprint(dm):
    return dict(dev_split_sha256=dm.study_split_sha256,
                full_train_samples=5981,full_test_samples=1519,
                training_samples=len(dm.study_train_indices),development_samples=len(dm.study_dev_indices),
                per_class=dm.study_split['per_class'],train_dev_disjoint=True,test_unchanged=True)

def current_task_train_dataset(dm,task,cfg=None):
    lo,hi=dm.get_task_class_range(task)
    ds=dm.get_dataset(range(lo,hi),source='train',mode='test')
    raw=list(ds.study_raw_train_indices)
    if set(raw)&set(dm.study_dev_indices):raise AssertionError('Held-out development samples leaked into train probes')
    return ds,raw

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for key in ('source','config','output'):p.add_argument('--'+key,required=True)
    c=p.parse_args();result=prepare_split(c.source,json.loads(Path(c.config).read_text()),c.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('native_train_paths','native_train_targets','train_indices','dev_indices','per_class')}))
