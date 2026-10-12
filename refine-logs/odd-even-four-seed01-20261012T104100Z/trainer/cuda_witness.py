"""Seeded FP32 CUDA dispatch witness; no training."""
import argparse
import json
import os
from pathlib import Path
import torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--seed",type=int,choices=(0,1,2),default=2)
    args=ap.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in execution namespace")
    x=torch.randn(64,64,device="cuda",requires_grad=True)
    y=(x@x.T).square().mean()
    y.backward()
    if not bool(torch.isfinite(y)) or not bool(torch.isfinite(x.grad).all()):
        raise FloatingPointError("Nonfinite CUDA kernel witness")
    data=dict(status="passed",seed=args.seed,device=torch.cuda.get_device_name(0),shape=list(x.shape),
              loss=float(y.detach()),gradient_norm=float(x.grad.norm()),
              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
              torch=torch.__version__,cuda=torch.version.cuda,
              tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
              tf32_cudnn=torch.backends.cudnn.allow_tf32)
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2)+"\n")
    print("WITNESS "+json.dumps(data),flush=True)

if __name__=="__main__":
    main()
