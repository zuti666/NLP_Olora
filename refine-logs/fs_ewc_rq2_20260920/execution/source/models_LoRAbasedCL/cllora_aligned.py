"""CL-LoRA learner for the common DataManager/trainer and arbitrary task sizes."""
import logging
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models_CL.baseLearner import BaseLearner
from backbone.cllora_aligned import CLLoRANet


def orthogonality_loss(previous, current):
    vec = current.flatten() / (current.norm() + 1e-8)
    return sum((vec * (p.flatten() / (p.norm() + 1e-8))).sum().abs() for p in previous) / len(previous)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.topk = 1
        if len(self._multiple_gpus) != 1:
            raise ValueError('CL-LoRA aligned jobs use one device per process')
        self._network = CLLoRANet(args)

    def _freeze(self):
        self._network.requires_grad_(False)
        net = self._network
        for pos in self.args['shared_pos']:
            attn = net.image_encoder.blocks[pos].attn
            for proj, enabled in zip(['q', 'k', 'v'], self.args['msa']):
                if enabled:
                    getattr(attn, 'lora_' + proj).lora_B.requires_grad_(True)
        for pos in self.args['specific_pos']:
            attn = net.image_encoder.blocks[pos].attn
            for proj, enabled in zip(['q', 'k', 'v'], self.args['msa']):
                if enabled:
                    getattr(attn, 'lora_' + proj)[self._cur_task].requires_grad_(True)
        net.image_encoder.block_weights[self._cur_task].requires_grad_(True)
        net.proxy_fc.requires_grad_(True)

    def _loader(self, data, shuffle):
        return DataLoader(data, batch_size=int(self.args['batch_size']), shuffle=shuffle,
                          num_workers=int(self.args.get('train_num_workers', 2)), pin_memory=True)

    @staticmethod
    def _checked_step(loss, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite CL-LoRA loss')
        optimizer.zero_grad()
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in optimizer.param_groups for p in g['params']):
            raise FloatingPointError('Nonfinite CL-LoRA gradient')

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.to(self._device)
        self._network.update_fc(self._total_classes)
        self._freeze()
        current = np.arange(self._known_classes, self._total_classes)
        self.train_loader = self._loader(data_manager.get_dataset(current, source='train', mode='train'), True)
        self.test_loader = self._loader(data_manager.get_dataset(np.arange(self._total_classes), source='test', mode='test'), False)
        proto_loader = self._loader(data_manager.get_dataset(current, source='train', mode='test'), True)
        epochs = int(self.args['init_epoch'] if self._cur_task == 0 else self.args['epochs'])
        lr = float(self.args['init_lr'] if self._cur_task == 0 else self.args['lrate'])
        optimizer = torch.optim.SGD([p for p in self._network.parameters() if p.requires_grad], lr=lr,
                                    momentum=float(self.args.get('momentum', 0)), weight_decay=float(self.args.get('weight_decay', 0)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        logging.info('CL-LoRA task=%d sizes=%s trainable=%d KD=%s temperature=%s orth=%s', self._cur_task,
                     self._network.task_sizes, sum(p.numel() for p in self._network.parameters() if p.requires_grad),
                     self.args['kd_ratio'], self.args['temperature'], self.args['orth_ratio'])
        for epoch in range(epochs):
            self._network.train()
            losses = kd_losses = 0.0
            correct = total = 0
            for _, images, targets in self.train_loader:
                images, labels = images.to(self._device), targets.to(self._device) - self._known_classes
                if self._cur_task > 0:
                    student, teacher = self._network.forward_kd(images)
                    temp = float(self.args['temperature'])
                    kd = -float(self.args['kd_ratio']) * (F.softmax(teacher/temp, dim=1) * F.log_softmax(student/temp, dim=1)).sum(1).mean()
                    self._checked_step(kd, optimizer)
                    with torch.no_grad():
                        for pos in self.args['shared_pos']:
                            attn = self._network.image_encoder.blocks[pos].attn
                            for proj, enabled in zip(['q', 'k', 'v'], self.args['msa']):
                                if not enabled: continue
                                old = getattr(attn, 'lora_' + proj + '_old').lora_B.weight
                                new = getattr(attn, 'lora_' + proj).lora_B.weight
                                scale = old.norm(dim=1)
                                # Degenerate all-zero teacher: neutral gradient scaling.
                                scale = scale * len(scale) / scale.sum() if scale.sum() > 0 else torch.ones_like(scale)
                                if new.grad is not None: new.grad.mul_(scale[:, None])
                    optimizer.step()
                    kd_losses += float(kd.detach())
                # Compute CE after the KD update; do not reuse a stale autograd graph.
                logits = self._network(images, self._cur_task)['logits']
                loss = F.cross_entropy(logits, labels)
                if self._cur_task > 0:
                    weights = self._network.image_encoder.block_weights
                    loss = loss + float(self.args['orth_ratio']) * orthogonality_loss(weights[:self._cur_task], weights[self._cur_task])
                self._checked_step(loss, optimizer)
                optimizer.step()
                losses += float(loss.detach())
                correct += int((logits.argmax(1) == labels).sum())
                total += len(labels)
            scheduler.step()
            logging.info('Task %d, Epoch %d/%d => Loss %.6f, KD %.6f, Train_accy %.2f',
                         self._cur_task, epoch+1, epochs, losses/len(self.train_loader), kd_losses/len(self.train_loader), 100*correct/total)
        self._network.replace_fc(proto_loader)
        if any(not torch.isfinite(p).all() for p in self._network.parameters()):
            raise FloatingPointError('Nonfinite CL-LoRA endpoint parameter')

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.save_old_shared_lora()
