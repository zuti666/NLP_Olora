"""Seeded CUDA witness for the existing Pilot_new environment; no training."""
import argparse
import json
import os
import platform
import socket
from pathlib import Path
import torch
import torchvision
import timm
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
torch.manual_seed(0)
x = torch.randn(8, 8, device='cuda', requires_grad=True)
y = x @ x
y.square().mean().backward()
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
result = dict(host=socket.gethostname(), python=platform.python_version(),
              executable=os.sys.executable, torch=torch.__version__,
              torchvision=torchvision.__version__, timm=timm.__version__,
              yaml=yaml.__version__, cuda=torch.version.cuda,
              visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
              device=torch.cuda.get_device_name(),
              memory_total_bytes=torch.cuda.get_device_properties(0).total_memory,
              shape=list(y.shape), finite_forward_backward=True,
              matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
              cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
print('WITNESS', json.dumps(result), flush=True)
