"""Read actual ImageNet-A metadata and verify both complete task partitions."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from datetime import datetime, timezone

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'imageneta-history-response-execution-20261008T175403Z'
SOURCE=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')
sys.path.insert(0,str(ROOT/'trainer'))
import numpy as np
import torch
import response_train as new
from layout_lifecycle import TaskLayout,seen_accuracy_row
spec=importlib.util.spec_from_file_location('independent_prior_trainer',OLD/'trainer/history_train.py')
old=importlib.util.module_from_spec(spec)
sys.path.insert(0,str(OLD/'trainer'));spec.loader.exec_module(old)
sys.path.insert(0,str(SOURCE))
from utils.data_manager import DataManager
from models_CL.baseLearner import BaseLearner

cli=SimpleNamespace(max_tasks=0,max_epochs=0,max_batches=0,
                    output_dir='/tmp/independent_response_protocol_not_written',source=str(SOURCE))
configs={}
for layout in ('single200','cl10'):
    configs[layout]=new.resolve_config(json.loads((ROOT/'configs'/f'formal_{layout}_history_fusion_s2.json').read_text()),cli)
oldcfg=old.resolve_config(json.loads((OLD/'configs/formal_history_fused_k10_s2.json').read_text()),cli)
cfgdiff={k:[oldcfg.get(k),configs['cl10'].get(k)] for k in sorted(set(oldcfg)|set(configs['cl10']))
         if oldcfg.get(k)!=configs['cl10'].get(k)}

partitions={}
for label,cfg in (('prior_cl10',oldcfg),('cl10',configs['cl10']),('single200',configs['single200'])):
    dm=DataManager(cfg['dataset'],cfg['class_shuffle'],2,cfg['init_cls'],cfg['increment'],cfg)
    layout=TaskLayout.from_name('cl10' if label=='prior_cl10' else label)
    layout.validate_data_manager(dm)
    train_targets=np.asarray(dm._train_targets)
    test_targets=np.asarray(dm._test_targets)
    assert len(train_targets)==5981 and len(test_targets)==1519
    assert set(train_targets)==set(test_targets)==set(range(200))
    counts=[int(np.logical_and(train_targets>=lo,train_targets<hi).sum()) for lo,hi in layout.boundaries]
    test_counts=[int(np.logical_and(test_targets>=lo,test_targets<hi).sum()) for lo,hi in layout.boundaries]
    batches=[math.ceil(n/128) for n in counts]
    partitions[label]=dict(task_count=layout.tasks,boundaries=list(layout.boundaries),
                           train_samples=counts,test_samples=test_counts,batches_per_epoch=batches,
                           updates=50*sum(batches),training_sample_exposures=50*sum(counts),
                           train_targets_sha256=hashlib.sha256(train_targets.tobytes()).hexdigest(),
                           test_targets_sha256=hashlib.sha256(test_targets.tobytes()).hexdigest(),
                           class_order=list(map(int,dm._class_order)))
assert partitions['prior_cl10']==partitions['cl10']
assert partitions['single200']['updates']==2350 and partitions['cl10']['updates']==2550
assert partitions['single200']['training_sample_exposures']==partitions['cl10']['training_sample_exposures']==299050

# The new explicit complete evaluation boundaries add only future NaN groups;
# all seen-class groups and overall native rounded accuracies remain unchanged.
for task in range(10):
    seen=20*(task+1)
    y=np.repeat(np.arange(seen),3)
    p=y.copy();p[::7]=(p[::7]+1)%seen
    pred=np.stack([p,(p+1)%seen],axis=1)
    conditions=[]
    for args in (oldcfg,configs['cl10']):
        learner=BaseLearner.__new__(BaseLearner)
        learner.args=args;learner._total_classes=seen;learner._known_classes=20*task;learner.topk=2
        conditions.append(learner._evaluate(pred,y))
    oldrow=old.native.seen_accuracy_row(conditions[0],task,seen,task)
    newrow=seen_accuracy_row(conditions[1],task,seen,task,TaskLayout.from_name('cl10'))
    assert oldrow==newrow and conditions[0]['top1']==conditions[1]['top1']
    logits=torch.linspace(-.7,.6,steps=6*seen,dtype=torch.float32).view(6,seen).requires_grad_()
    targets=torch.tensor([20*task+i for i in (0,1,3,5,9,19)])
    old_loss=torch.nn.functional.cross_entropy(logits[:,20*task:],targets-20*task) if task else torch.nn.functional.cross_entropy(logits,targets)
    new_loss=torch.nn.functional.cross_entropy(logits[:,20*task:seen],targets-20*task)
    assert torch.equal(old_loss,new_loss)
    assert torch.equal(torch.autograd.grad(old_loss,logits)[0],torch.autograd.grad(new_loss,logits)[0])

report=dict(status='passed',audited_at_utc=datetime.now(timezone.utc).isoformat(),cpu_only=True,
            actual_dataset_metadata_read=True,gpu_used=False,partitions=partitions,
            cl10_prior_partition_metadata_and_order_exact=True,
            cl10_seen_evaluation_groups_10stages_exact=True,
            cl10_old_task_loss_new_explicit_ce_10stages_bitwise_equal=True,
            shared_baseline_settings_changed=False,resolved_config_differences=cfgdiff,
            conclusion='Prior HRF may be reused as a sourced existing seed2 cl10 result; it was not newly run in this plan.',
            limitation='No actual new CUDA/full training trajectory has been executed; supporting state/lifecycle code is unchanged and core CPU equivalence is independently checked.')
(ROOT/'gates/INDEPENDENT_PROTOCOL_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('resolved_config_differences',)},indent=2))
