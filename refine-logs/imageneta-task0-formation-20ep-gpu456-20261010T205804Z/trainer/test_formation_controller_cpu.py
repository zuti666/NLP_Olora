"""Run the actual patched controller/epochloader on small native-shape CPU data.

This verifies high-level method selection, all20epoch schedules, one-SGD
records, epoch0 expanded-head evaluation and the switch receipt. Synthetic
accuracies are never exported as experiment results.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from torch.utils.data import Dataset,DataLoader
import formation_train as train
from formation_step import CONDITIONS

class Data(Dataset):
    def __init__(self,task):self.task=task
    def __len__(self):return 20
    def __getitem__(self,i):return i,torch.tensor([.2+i*.01,.3-i*.02,-.1+i*.03]),20*self.task+i

class DM:
    nb_classes=200;nb_tasks=10;_increments=[20]*10;_class_order=list(range(200))
    study_split_sha256='synthetic_not_scientific_result'
    def get_task_class_range(self,t):return 20*t,20*(t+1)
    def build_dataloader(self,dataset,**kw):return DataLoader(dataset,**kw)
    def get_dataset(self,indices,**kw):return Data(min(indices)//20)

class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__();self.w_As=torch.nn.ModuleList([torch.nn.Linear(3,2,bias=False)])
        self.w_Bs=torch.nn.ModuleList([torch.nn.Linear(2,3,bias=False)]);self.learn_alpha=False;self.wrapped_param=[]
    def forward(self,x):return x+self.w_Bs[0](self.w_As[0](x))

class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.backbone=Backbone();self.fc=torch.nn.Linear(3,20)
    def forward(self,x):return {'logits':self.fc(self.backbone(x))}

class Learner:
    def __init__(self):
        self._network=Model();self._device=torch.device('cpu');self._cur_task=0;self._total_classes=20
        self._ewc_skip_task_fisher=True;self._lambda_ewc=0.;self._known_classes=0
    def _unwrap_network(self):return self._network
    def _build_optimizer(self,*a):raise AssertionError('Native method must be patched')
    def _step_batch_with_log(self,*a):raise AssertionError('Native method must be patched')
    def _task_loss(self,logits,y,class_offset):
        chunk=logits[:,class_offset:self._total_classes];targets=y-class_offset
        return torch.nn.functional.cross_entropy(chunk,targets),chunk,targets
    def _run_regularized_train(self,loader,test_loader,opt,scheduler,*,epochs,class_offset):
        for _ in range(epochs):
            self._network.train()
            for _,x,y in loader:self._step_batch_with_log(opt,x,y,class_offset)
            scheduler.step()
    def incremental_train(self,dm):pass
    def eval_task(self):raise AssertionError('Not needed in thin controller integration')
    def after_task(self):self._known_classes=self._total_classes

class ControllerTests(unittest.TestCase):
    def test_all_six_actual_controllers_run_two20epoch_tasks_and_epoch0_head_expansion(self):
        root=Path(__file__).resolve().parents[1]
        for condition in CONDITIONS:
            with self.subTest(condition=condition),tempfile.TemporaryDirectory() as tmp:
                raw=json.loads((root/'configs'/('formal_'+condition+'_s2.json')).read_text());raw['formation_smoke']=True
                cli=SimpleNamespace(max_batches=1,max_epochs=20,max_tasks=2,output_dir=tmp,source='/tmp/notused')
                cfg=train.resolve_config(raw,cli);cfg.update(train_num_workers=0,eval_num_workers=0)
                torch.manual_seed(2);learner=Learner();c=train.Controller(learner,cfg,cli);c.dm=DM()
                c.evaluate_dev=lambda:dict(task=c.task,epoch=c.epoch,dev_current_task_ce=0.,dev_current_task_accuracy=0.,dev_samples=1)
                c.install()
                try:
                    for task in (0,1):
                        c.task=task;learner._cur_task=task
                        if task==0:c.old_fc={}
                        else:
                            old=learner._network.fc;c.old_fc=copy.deepcopy(old.state_dict())
                            learner._network.fc=torch.nn.Linear(3,40)
                            with torch.no_grad():
                                learner._network.fc.weight[:20].copy_(old.weight);learner._network.fc.bias[:20].copy_(old.bias)
                            learner._total_classes=40
                        opt=learner._build_optimizer([p for p in learner._network.parameters() if p.requires_grad],'init' if task==0 else 'inc')
                        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=(.04 if task==0 else .02)*.1)
                        learner._run_regularized_train(DataLoader(Data(task),batch_size=128),None,opt,scheduler,epochs=20,class_offset=20*task)
                        if task==0:c.task0_endpoint=c.evaluate_test_group(0,20,old_local=True)
                    self.assertEqual(c.steps,40);self.assertEqual(len(c.rows),40);self.assertEqual(len(c.retention_curve),21)
                    self.assertEqual([r['epoch'] for r in c.retention_curve],list(range(21)))
                    self.assertEqual(c.retention_curve[0]['old_test_allseen40']['classification_classes'],40)
                    self.assertEqual(c.task0_endpoint['classification_classes'],20)
                    self.assertEqual(c.counters['training_sgd_updates'],40)
                    self.assertEqual(c.counters['training_ce_gradient_passes'],{'sgd':40,'sam':60,'gam_residual':100,'sf_g2':80,'sf_g2_then_sgd':60,'sgd_then_sf_g2':60}[condition])
                    self.assertEqual(len(c.switch_receipts),1)
                    self.assertTrue(c.switch_receipts[0]['optimizer_same']);self.assertTrue(c.switch_receipts[0]['scheduler_same'])
                    self.assertTrue(c.switch_receipts[0]['rng_unchanged_at_switch'])
                    records=[json.loads(x) for x in c.trace.read_text().splitlines()]
                    self.assertTrue(all(row['active_method']=='sgd' for row in records if row['task']==1))
                    self.assertTrue(all(row['normal_gradient_calls']==row['ordinary_gradient_calls'] for row in records))
                finally:c.handle.close()

if __name__=='__main__':unittest.main(verbosity=2)
