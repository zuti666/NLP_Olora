"""One fixed 200-class model; LoRA initialization or a plain full ViT copy."""
import copy
import hashlib
import torch


def parameter_hash(model, frozen_only=False):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if frozen_only and param.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(param.shape)).encode())
        digest.update(param.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


class FullFTClassifier(torch.nn.Module):
    """Plain pretrained ViT features and the exact paired SimpleLinear head."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.fc = head
        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, inputs):
        features = self.backbone(inputs)
        outputs = self.fc(features)
        outputs.update(features=features)
        return outputs


def full_ft_from_lora_origin(model):
    """Use the untouched pretrained backup, not a merged/fresh-random ViT.

    deepcopy makes no random draws. Head and initial backbone weights therefore
    remain the same across methods and tuning scopes; no LoRA module survives.
    """
    origin = model.backbone.base_vit
    return FullFTClassifier(copy.deepcopy(origin), copy.deepcopy(model.fc))


def scope_receipt(model, scope):
    named = list(model.named_parameters())
    trainable = [(name, p) for name, p in named if p.requires_grad]
    frozen = [(name, p) for name, p in named if not p.requires_grad]
    lora_names = [name for name, _ in named if 'linear_a_' in name or 'linear_b_' in name]
    if scope == 'full_ft':
        if frozen or lora_names or any('lora' in type(m).__name__.lower() for m in model.modules()):
            raise AssertionError('Full FT must train a plain ViT and classifier, without LoRA')
    elif scope == 'lora':
        if not lora_names or not frozen:
            raise AssertionError('LoRA requires trainable factors and a frozen backbone')
        if any(not (name.startswith('fc.') or 'linear_a_' in name or 'linear_b_' in name)
               for name, _ in trainable):
            raise AssertionError('Unexpected trainable tensor in LoRA scope')
    else:
        raise ValueError('Unknown training scope')
    if not trainable or tuple(model.fc.weight.shape) != (200, 768):
        raise AssertionError('Missing full 200-class head')
    if any(p.dtype != torch.float32 for _, p in named):
        raise AssertionError('All model parameters must be FP32')
    return dict(train_scope=scope,
        trainable_parameter_count=sum(p.numel() for _, p in trainable),
        frozen_parameter_count=sum(p.numel() for _, p in frozen),
        trainable_parameters=[dict(name=n, shape=list(p.shape), numel=p.numel()) for n,p in trainable],
        frozen_parameter_tensors=len(frozen), lora_factor_tensor_count=len(lora_names),
        all_backbone_and_head_trainable=not frozen,
        no_lora_modules=(scope == 'full_ft'))
