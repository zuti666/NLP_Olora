"""Exact effective-weight empirical-Fisher diagnostics (no training).

For a sample-independent linear map y[b,n,o] = sum_i x[b,n,i] W[o,i],
the per-example gradient is dW[b,o,i] = sum_n dy[b,n,o] x[b,n,i].
Backpropagating sum_b CE_b therefore gives ALL individual gradients with
one ordinary backward; we reduce each block before releasing it. No batch
gradient square, diagonal directional approximation, normalization or floor.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/data/140-1/users/liying/CopyFlatCV')
T0 = ORIGINAL / 'outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50'
T9 = ORIGINAL / 'outputs_logs/sgd_gam_s1c/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50'


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_hash(tensors):
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().cpu().contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        # Feed contiguous CPU storage directly; a full train cache is ~750 MB.
        h.update(memoryview(t.numpy()).cast('B'))
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def checkpoint_path(method, task, config):
    paths = config.get('checkpoints', {}).get(method, {})
    if str(task) in paths:
        return Path(paths[str(task)])
    return (T0 / method / 'seed0/checkpoints/task_00.pt' if task == 0 else
            T9 / method / f'T10/seed0/checkpoints/task_{task:02d}.pt')


def read_state(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)['network_state']


def effective_weights(state):
    out = []
    for layer in range(12):
        for channel in ('q', 'v'):
            stem = f'backbone.lora_vit.blocks.{layer}.attn.qkv.linear_'
            a = state[f'{stem}a_{channel}.weight']
            b = state[f'{stem}b_{channel}.weight']
            assert a.shape == (10, 768) and b.shape == (768, 10)
            out.append(b @ a)
    return out


def coordinates():
    return [f'block.{i}.effective_{c}[out,in]' for i in range(12) for c in ('q', 'v')]


class RestoredNetwork(nn.Module):
    def __init__(self, backbone, classes):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(768, classes)

    def forward(self, x):
        return self.fc(self.backbone(x))[:, :20]


def load_network(path, source_path, device):
    """Strictly restore every manifest tensor before any diagnostic surgery."""
    sys.path.insert(0, str(source_path))
    import timm
    from backbone.lora import LoRA_ViT_timm, IdentityScale
    state = read_state(path)
    assert not any('scaling' in k or 'wrapped_param' in k for k in state)
    base = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    backbone = LoRA_ViT_timm(base.eval(), r=10, num_classes=0, increment=20,
                            filepath=str(ROOT / 'unused_no_sidecars') + '/', learn_alpha=False)
    model = RestoredNetwork(backbone, state['fc.weight'].shape[0])
    model.load_state_dict(state, strict=True)
    for block in model.backbone.lora_vit.blocks:
        qkv = block.attn.qkv
        assert qkv.task_id == 0 and not qkv.saved_A and not qkv.saved_B
        assert len(qkv.scaling_factor) == 1 and isinstance(qkv.scaling_factor[0], IdentityScale)
    # base_vit is an unused saved copy. Strict restore above verified its keys.
    del model.backbone.base_vit
    model.requires_grad_(False).eval().to(device)
    return model


class EffectiveQKV(nn.Module):
    def __init__(self, original, collector, offset):
        super().__init__()
        self.qkv = original.qkv
        self.effective_q = nn.Parameter((original.linear_b_q.weight @ original.linear_a_q.weight).detach())
        self.effective_v = nn.Parameter((original.linear_b_v.weight @ original.linear_a_v.weight).detach())
        self.collector = collector
        self.offset = offset

    def forward(self, x):
        q = F.linear(x, self.effective_q)
        v = F.linear(x, self.effective_v)
        if self.collector.active and torch.is_grad_enabled():
            xd = x.detach()
            q.register_hook(lambda adj: self.collector.add(self.offset, xd, adj))
            v.register_hook(lambda adj: self.collector.add(self.offset + 1, xd, adj))
        base_q, base_k, base_v = self.qkv(x).chunk(3, dim=-1)
        return torch.cat((base_q + q, base_k, base_v + v), dim=-1)


class PerExampleCollector:
    def __init__(self):
        self.active = False

    def start(self, count, device, directions):
        self.active = True
        self.trace = torch.zeros(count, device=device, dtype=torch.float64)
        self.dots = {name: torch.zeros_like(self.trace) for name in directions}
        self.directions = directions
        self.blocks_seen = set()

    def add(self, index, x, adj):
        # adj is d(sum CE)/dy, hence the unscaled per-example adjoint.
        with torch.no_grad():
            g = torch.bmm(adj.detach().transpose(1, 2), x)
            self.trace.add_(g.square().sum(dim=(1, 2), dtype=torch.float64))
            for name, direction in self.directions.items():
                self.dots[name].add_((g * direction[index]).sum(dim=(1, 2), dtype=torch.float64))
            self.blocks_seen.add(index)
        return adj

    def finish(self):
        self.active = False
        assert self.blocks_seen == set(range(24)), self.blocks_seen
        return self.trace.detach(), {name: dot.detach() for name, dot in self.dots.items()}


def convert_effective(model):
    collector = PerExampleCollector()
    for i, block in enumerate(model.backbone.lora_vit.blocks):
        block.attn.qkv = EffectiveQKV(block.attn.qkv, collector, i * 2)
    params = [p for block in model.backbone.lora_vit.blocks
              for p in (block.attn.qkv.effective_q, block.attn.qkv.effective_v)]
    assert len(params) == 24 and sum(p.numel() for p in params) == 14155776
    return collector, params


def set_point(params, anchor, direction=None, amount=0.0):
    with torch.no_grad():
        for i, (p, a) in enumerate(zip(params, anchor)):
            p.copy_(a)
            if direction is not None and amount != 0:
                p.add_(direction[i], alpha=float(amount))


def norm_sq(values):
    return sum(float(v.double().square().sum()) for v in values)


def unit_direction(values):
    norm = norm_sq(values) ** 0.5
    assert norm > 0
    return [v / norm for v in values], norm


def random_direction(index, seed, device):
    # Generation stays on CPU to reproduce directions across GPU models.
    gen = torch.Generator(device='cpu').manual_seed(int(seed) + 104729 * int(index))
    values = [torch.randn(768, 768, generator=gen, dtype=torch.float32) for _ in range(24)]
    direction, _ = unit_direction(values)
    meta = {'index': index, 'seed': int(seed) + 104729 * int(index),
            'sha256': tensor_hash(direction), 'norm': norm_sq(direction) ** 0.5}
    return [x.to(device) for x in direction], meta


def prepare_data(config, output):
    from torchvision import datasets, transforms
    import torchvision
    from PIL import Image, features
    from torch.utils.data import DataLoader, Subset
    data_root = Path(config.get('data_root', '/data/115-2/users/liying/datasets/imagenet-a'))
    transform = transforms.Compose([transforms.Resize(256, interpolation=3),
                                    transforms.CenterCrop(224), transforms.ToTensor()])
    tensors, labels, records = {}, {}, {}
    class_names = None
    for split, expected in (('train', 1259), ('test', 307)):
        dataset = datasets.ImageFolder(data_root / split, transform=transform)
        if class_names is None:
            class_names = dataset.classes[:20]
        assert dataset.classes[:20] == class_names
        indices = [i for i, (_, y) in enumerate(dataset.samples) if y < 20]
        assert len(indices) == expected, (split, len(indices), expected)
        records[split] = [{'id': str(Path(dataset.samples[i][0]).relative_to(data_root)),
                           'label': dataset.samples[i][1], 'sha256': sha_file(dataset.samples[i][0])}
                          for i in indices]
        loader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=False,
                            num_workers=config.get('workers', 4))
        xx, yy = [], []
        for x, y in loader:
            xx.append(x); yy.append(y)
        tensors[split] = torch.cat(xx)
        labels[split] = torch.cat(yy)
    # Hamilton proportional allocation; stable lower-class-index tie break.
    counts = torch.bincount(labels['train'], minlength=20)
    target = int(config.get('fisher_samples', 256))
    raw = counts.double() * target / len(labels['train'])
    allocated = raw.floor().long()
    remainders = sorted(range(20), key=lambda i: (-(float(raw[i]) - int(allocated[i])), i))
    for i in remainders[:target - int(allocated.sum())]:
        allocated[i] += 1
    gen = torch.Generator().manual_seed(int(config.get('subset_seed', 20260920)))
    selected = []
    for cls in range(20):
        candidates = torch.where(labels['train'] == cls)[0]
        selected.extend(candidates[torch.randperm(len(candidates), generator=gen)[:allocated[cls]]].tolist())
    selected.sort()
    assert len(selected) == target
    tensors['subset'] = tensors['train'][selected]
    labels['subset'] = labels['train'][selected]
    manifest = {'data_root': str(data_root), 'class_names': class_names, 'classes': list(range(20)),
                'transform': 'PIL RGB; Resize(256,bicubic=3); CenterCrop(224); ToTensor; NO normalization',
                'records': records, 'counts': counts.tolist(), 'subset_counts': allocated.tolist(),
                'subset_seed': int(config.get('subset_seed', 20260920)), 'subset_indices': selected,
                'subset_ids': [records['train'][i]['id'] for i in selected],
                'subset_image_tensor_sha256': tensor_hash([tensors['subset']]),
                'decoded_tensor_sha256': {split: tensor_hash([tensors[split]]) for split in ('train','test')},
                'decoder': {'Pillow':Image.__version__, 'jpeg_codec':features.version_codec('jpg'),
                            'libjpeg_turbo':features.version_feature('libjpeg_turbo')},
                'torchvision_version':torchvision.__version__,
                'dataset_records_sha256': canonical_hash(records)}
    manifest['manifest_sha256'] = canonical_hash(manifest)
    dest = Path(output) / 'data_manifest.json'
    if dest.exists():
        assert json.loads(dest.read_text()) == manifest, 'Existing data manifest differs; use a new output directory'
    else:
        atomic_json(dest, manifest)
    return tensors, labels, manifest


@torch.no_grad()
def evaluate(model, x, y, device, batch_size=64):
    loss, correct = 0.0, 0
    for i in range(0, len(y), batch_size):
        logits = model(x[i:i+batch_size].to(device))
        yi = y[i:i+batch_size].to(device)
        loss += float(F.cross_entropy(logits, yi, reduction='none').double().sum())
        correct += int((logits.argmax(1) == yi).sum())
    return {'loss': loss / len(y), 'accuracy': 100.0 * correct / len(y), 'samples': len(y)}


@torch.no_grad()
def evaluate_local_and_seen(model, x, y, device, batch_size=64):
    """Old-task examples under fixed local20 and current seen-class heads."""
    losses={'local':0.0,'seen':0.0}
    correct={'local':0,'seen':0}
    for i in range(0,len(y),batch_size):
        logits=model.fc(model.backbone(x[i:i+batch_size].to(device)))
        yi=y[i:i+batch_size].to(device)
        for scope,values in (('local',logits[:,:20]),('seen',logits)):
            losses[scope]+=float(F.cross_entropy(values,yi,reduction='none').double().sum())
            correct[scope]+=int((values.argmax(1)==yi).sum())
    return {scope:{'loss':losses[scope]/len(y),'accuracy':100.0*correct[scope]/len(y),
                   'samples':len(y),'classes':20 if scope=='local' else model.fc.out_features}
            for scope in ('local','seen')}


def tf32_round_reference(value):
    """Round FP32 operands to TF32 mantissas (round-to-nearest, ties-to-even).

    This is exclusively a historical convolution precision witness. It is NOT
    used to produce any Fisher, neighborhood, bridge, path or trajectory data.
    """
    assert value.dtype==torch.float32
    bits=value.contiguous().view(torch.int32)
    return ((bits+4095+((bits>>13)&1)) & -8192).view(torch.float32)


@torch.no_grad()
def historical_tf32_conv_reference(model,x,y,device,batch_size=64):
    """Temporarily emulate historical TF32 patch convolution, then restore."""
    proj=model.backbone.lora_vit.patch_embed.proj
    original=proj.weight.detach().clone()
    handle=None
    try:
        proj.weight.copy_(tf32_round_reference(original))
        handle=proj.register_forward_pre_hook(lambda module,args:(tf32_round_reference(args[0]),))
        result=evaluate(model,x,y,device,batch_size)
    finally:
        if handle is not None:
            handle.remove()
        proj.weight.copy_(original)
        assert torch.equal(proj.weight,original)
    result['precision']='reference only: TF32-rounded patch-convolution operands, FP32 accumulation; no Fisher measured'
    result['weights_restored_exactly']=True
    return result


def fisher(model, collector, params, x, y, directions, device, batch_size=8, individual=False):
    started = time.monotonic()
    trace_sum = loss_sum = 0.0
    q_sums = {name: 0.0 for name in directions}
    sample_trace, sample_dots = [], {name: [] for name in directions}
    for i in range(0, len(y), batch_size):
        yi = y[i:i+batch_size].to(device)
        model.zero_grad(set_to_none=True)
        collector.start(len(yi), device, directions)
        losses = F.cross_entropy(model(x[i:i+batch_size].to(device)), yi, reduction='none')
        losses.sum().backward()
        trace, dots = collector.finish()
        trace_sum += float(trace.sum())
        loss_sum += float(losses.detach().double().sum())
        for name, dot in dots.items():
            q_sums[name] += float(dot.square().sum())
            if individual:
                sample_dots[name].extend(dot.cpu().tolist())
        if individual:
            sample_trace.extend(trace.cpu().tolist())
    model.zero_grad(set_to_none=True)
    elapsed = time.monotonic() - started
    out = {'raw_trace': trace_sum / len(y), 'subset_loss': loss_sum / len(y),
           'q': {name: total / len(y) for name, total in q_sums.items()},
           'samples': len(y), 'seconds': elapsed, 'samples_per_second': len(y)/elapsed,
           'gradient_batch_size': batch_size, 'estimator': 'mean per-example CE gradient outer product',
           'coordinate': '24 Q/V effective BA matrices, scaling=1; fixed Task0 20-class head',
           'normalized': False, 'floor': None, 'regularizer': None}
    if individual:
        out.update(sample_trace=sample_trace, sample_dots=sample_dots)
    assert out['raw_trace'] >= 0 and all(v >= 0 for v in out['q'].values())
    return out
