"""CL-LoRA shared/specific adapters with explicit cumulative class boundaries.

Architecture derived from low-rank-cl-new/models/{vit,net}_cllora.py.
Use the common archived checkpoint; retain the native diagonal prototype rule.
"""
import hashlib
import logging
import math

import torch
from torch import nn
from torch.nn import functional as F

from backbone.cllora_archived_vit import VisionTransformer, Attention_LoRA


class Encoder(VisionTransformer):
    def forward(self, x, task_id=None, shared=False, use_new=True):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed[:, :x.size(1), :])
        if shared:
            for pos in self.shared_pos:
                x = self.blocks[pos](x, use_new=use_new)
        else:
            weights = self.block_weights[task_id]
            for pos, block in enumerate(self.blocks):
                weight = weights[self.specific_pos.index(pos)] if pos in self.specific_pos else None
                x = block(x, task_id, weight, use_new)
        return self.norm(x)[:, 0, :]


class CosineLinear(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.in_features, self.out_features = dim, classes
        self.weight = nn.Parameter(torch.empty(classes, dim))
        self.sigma = nn.Parameter(torch.ones(1))
        nn.init.uniform_(self.weight, -1 / math.sqrt(dim), 1 / math.sqrt(dim))

    def forward(self, x):
        return self.sigma * F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))


class CLLoRANet(nn.Module):
    def __init__(self, args, encoder=None):
        super().__init__()
        self.task_sizes = []
        self._cur_task = -1
        self.fc = None
        self.args = args
        self.image_encoder = encoder if encoder is not None else Encoder(
            num_classes=0, n_tasks=int(args['nb_tasks']), rank=int(args['lora_rank']),
            msa=args['msa'], shared_pos=args['shared_pos'], specific_pos=args['specific_pos'])
        if encoder is None:
            from safetensors.torch import load_file
            path = args['cllora_pretrained_path']
            with open(path, 'rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            if digest != args['cllora_pretrained_sha256']:
                raise ValueError('Pretrained checkpoint hash mismatch')
            weights = load_file(path, device='cpu')
            weights = {k: v for k, v in weights.items() if not k.startswith('head.')}
            missing, unexpected = self.image_encoder.load_state_dict(weights, strict=False)
            if unexpected or any('lora_' not in k and not k.startswith('block_weights.') for k in missing):
                raise ValueError((missing, unexpected))
            logging.info('CL-LoRA checkpoint timm/vit_base_patch16_224.augreg2_in21k_ft_in1k SHA256=%s', digest)
        for module in self.image_encoder.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def update_fc(self, total_classes):
        current = int(total_classes) - sum(self.task_sizes)
        if current <= 0:
            raise ValueError('Task must introduce a positive number of classes')
        self.task_sizes.append(current)
        self._cur_task += 1
        device = next(self.image_encoder.parameters()).device
        self.proxy_fc = CosineLinear(self.feature_dim, current).to(device)
        new_fc = CosineLinear(self.feature_dim * len(self.task_sizes), total_classes).to(device)
        with torch.no_grad():
            new_fc.weight.zero_()
            if self.fc is not None:
                new_fc.sigma.copy_(self.fc.sigma)
                new_fc.weight[:self.fc.out_features, :-self.feature_dim].copy_(self.fc.weight)
        new_fc.requires_grad_(False)
        self.fc = new_fc

    def forward(self, x, task_id=None):
        if task_id is not None:
            features = self.image_encoder(x, task_id=task_id)
            return {'logits': self.proxy_fc(features), 'features': features}
        return {'logits': self.interface(x)}

    def forward_kd(self, x):
        new = self.image_encoder(x, shared=True, use_new=True)
        old = self.image_encoder(x, shared=True, use_new=False)
        # Retain the source's shared proxy head on both KD branches.
        return self.proxy_fc(new), self.proxy_fc(old)

    def interface(self, x):
        outputs, start = [], 0
        for task, size in enumerate(self.task_sizes):
            features = self.image_encoder(x, task_id=task)
            weights = self.fc.weight[start:start + size, task*self.feature_dim:(task+1)*self.feature_dim]
            outputs.append(self.fc.sigma * F.linear(F.normalize(features, dim=1), F.normalize(weights, dim=1)))
            start += size
        logits = torch.cat(outputs, dim=1)
        if logits.size(1) != sum(self.task_sizes):
            raise RuntimeError('CL-LoRA global classifier coverage mismatch')
        return logits

    @torch.no_grad()
    def replace_fc(self, loader):
        self.image_encoder.eval()
        device = next(self.parameters()).device
        start = sum(self.task_sizes[:-1])
        size = self.task_sizes[-1]
        # Same prototype averages as the source, accumulated without caching images.
        for branch in range(len(self.task_sizes)):
            sums = torch.zeros(size, self.feature_dim, device=device)
            counts = torch.zeros(size, device=device)
            for _, images, labels in loader:
                local = labels.to(device) - start
                if (local < 0).any() or (local >= size).any():
                    raise ValueError('Prototype loader must contain current training classes only')
                features = self.image_encoder(images.to(device), task_id=branch)
                sums.index_add_(0, local, features)
                counts.index_add_(0, local, torch.ones_like(local, dtype=sums.dtype))
            if not (counts > 0).all():
                raise ValueError('Missing training class while constructing CL-LoRA prototypes')
            self.fc.weight[start:start+size, branch*self.feature_dim:(branch+1)*self.feature_dim].copy_(sums/counts[:, None])

    def save_old_shared_lora(self):
        for module in self.image_encoder.modules():
            if isinstance(module, Attention_LoRA):
                module.save_old_shared_lora()
