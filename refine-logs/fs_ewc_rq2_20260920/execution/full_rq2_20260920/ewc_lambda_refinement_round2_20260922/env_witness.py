"""Seeded FP32 CUDA witness for the unchanged Pilot_new environment."""
import argparse
import json
import os
from pathlib import Path
import platform
import socket

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
import torch
import torchvision
import timm

assert torch.__version__ == '2.4.1+cu121'
assert torchvision.__version__ == '0.19.1+cu121'
assert timm.__version__ == '1.0.9'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(4)
torch.manual_seed(0)
x = torch.randn(8, 8, device='cuda', dtype=torch.float32, requires_grad=True)
y = x @ x
y.square().mean().backward()
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
payload = {'status': 'passed', 'host': socket.gethostname(), 'python': platform.python_version(),
           'torch': torch.__version__, 'torchvision': torchvision.__version__, 'timm': timm.__version__,
           'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'device': torch.cuda.get_device_name(), 'shape': list(y.shape), 'dtype': str(y.dtype),
           'finite_forward_backward': True, 'matmul_allow_tf32': False, 'cudnn_allow_tf32': False}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2) + '\n')
print('WITNESS', json.dumps(payload), flush=True)
