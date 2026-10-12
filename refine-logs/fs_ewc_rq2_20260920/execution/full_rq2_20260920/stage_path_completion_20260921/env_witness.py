"""Seeded FP32 CUDA forward/backward witness after the queue claims one GPU."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

import torch
import torchvision
import timm

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.__version__ == '2.4.1+cu121'
assert torchvision.__version__ == '0.19.1+cu121'
assert timm.__version__ == '1.0.9'
assert os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
x = torch.randn(8, 8, device='cuda', dtype=torch.float32, requires_grad=True)
loss = (x @ x).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()
assert x.grad.norm().item() > 0
result = dict(status='passed', utc=datetime.now(timezone.utc).isoformat(),
              python=platform.python_version(), torch=torch.__version__,
              torchvision=torchvision.__version__, timm=timm.__version__,
              gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
              gpu_name=torch.cuda.get_device_name(0),
              precision='FP32, TF32 off', seed=0, loss=loss.item(),
              gradient_norm=x.grad.norm().item())
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_suffix('.tmp')
temporary.write_text(json.dumps(result, indent=2) + '\n')
temporary.replace(args.output)
print(json.dumps(result), flush=True)
