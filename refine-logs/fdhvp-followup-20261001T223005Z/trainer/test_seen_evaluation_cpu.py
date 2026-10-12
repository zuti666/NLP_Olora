#!/usr/bin/env python3
"""Regression for actual native full-ten-boundary accuracy at Task0/1/9."""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode=True
import numpy as np
import component_horizon as trainer

SOURCE=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')
spec=importlib.util.spec_from_file_location('native_evaluation_toolkit',SOURCE/'utils/toolkit.py')
toolkit=importlib.util.module_from_spec(spec);spec.loader.exec_module(toolkit)


def native_result(task):
    seen=20*(task+1);labels=np.repeat(np.arange(seen),2);predictions=labels.copy()
    predictions[::2]=(predictions[::2]+1)%seen
    grouped=toolkit.accuracy(predictions,labels,20*task,20,20,
        task_boundaries=[(20*t,20*(t+1)) for t in range(10)])
    return {'grouped':grouped,'top1':grouped['total']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    positive=[];negative=[]
    for task in (0,1,9):
        cnn=native_result(task);row=trainer.seen_accuracy_row(cnn,task,20*(task+1),task)
        assert row==[50.]*(task+1) and cnn['top1']==50.
        keys=['%02d-%02d'%(20*t,20*t+19) for t in range(10)]
        assert all(math.isnan(cnn['grouped'][k]) for k in keys[task+1:])
        assert len([k for k in cnn['grouped'] if k in keys])==10
        positive.append({'task':task,'native_numeric_groups':10,'future_structural_NA':9-task,
            'observed_row':row,'sample_accuracy':float(cnn['top1'])})
        seen_only=copy.deepcopy(cnn)
        for key in keys[task+1:]:seen_only['grouped'].pop(key)
        assert trainer.seen_accuracy_row(seen_only,task,20*(task+1),task)==row
    base=native_result(1)
    def reject(label,mutate,task=1,seen=40,history=1):
        cnn=copy.deepcopy(base);mutate(cnn)
        try:trainer.seen_accuracy_row(cnn,task,seen,history)
        except AssertionError as exc:negative.append({'case':label,'error':str(exc)})
        else:raise AssertionError('Invalid current evaluation accepted: '+label)
    reject('missing_old_seen_group',lambda c:c['grouped'].pop('00-19'))
    reject('missing_current_seen_group',lambda c:c['grouped'].pop('20-39'))
    reject('old_seen_nan',lambda c:c['grouped'].__setitem__('00-19',float('nan')))
    reject('current_seen_nan',lambda c:c['grouped'].__setitem__('20-39',float('nan')))
    reject('current_seen_inf',lambda c:c['grouped'].__setitem__('20-39',float('inf')))
    reject('future_finite_evaluation_scope_contamination',lambda c:c['grouped'].__setitem__('40-59',0.))
    reject('future_inf_is_not_structural_NA',lambda c:c['grouped'].__setitem__('40-59',float('inf')))
    reject('wrong_seen_range',lambda c:c['grouped'].__setitem__('20-38',c['grouped'].pop('20-39')))
    reject('unexpected_numeric_range',lambda c:c['grouped'].__setitem__('40-60',float('nan')))
    reject('noncanonical_numeric_alias',lambda c:c['grouped'].__setitem__('0-19',50.))
    reject('nonfinite_sample_accuracy',lambda c:c.__setitem__('top1',float('nan')))
    reject('accuracy_outside_percent_range',lambda c:c['grouped'].__setitem__('20-39',101.))
    reject('wrong_seen_head',lambda c:None,seen=60)
    reject('wrong_history_length',lambda c:None,history=0)
    reject('task_outside_fixed_ten_task_protocol',lambda c:None,task=10,seen=220,history=10)
    report={'status':'passed','device':'cpu','source_accuracy_sha256':trainer.audit.digest(SOURCE/'utils/toolkit.py'),
        'trainer_sha256':trainer.audit.digest(trainer.__file__),'test_sha256':trainer.audit.digest(__file__),
        'positive':positive,'negative':negative,
        'scope':'Actual frozen accuracy with all ten task boundaries; canonical future NA excluded, seen completeness and finite guards retained/strengthened.'}
    if args.output:trainer.audit.atomic_json(args.output,report)
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
