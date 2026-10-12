#!/usr/bin/env python3
import argparse
import ctypes
import os
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run a simple PyTorch GPU matrix multiplication benchmark.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size N for NxN @ NxN.")
    parser.add_argument("--repeat", type=int, default=20, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Matrix dtype.",
    )
    return parser.parse_args()


def dtype_from_name(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def cuda_driver_status():
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        cu_init = libcuda.cuInit(0)
        count = ctypes.c_int()
        cu_count = libcuda.cuDeviceGetCount(ctypes.byref(count))
        return f"cuInit={cu_init}, cuDeviceGetCount={cu_count}, driver_device_count={count.value}"
    except Exception as exc:
        return f"CUDA driver check failed: {type(exc).__name__}: {exc}"


def main():
    args = parse_args()

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"torch={torch.__version__}, torch_cuda={torch.version.cuda}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(cuda_driver_status())
        raise SystemExit("CUDA is not available for torch in this process.")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    print(f"visible_cuda_devices={torch.cuda.device_count()}")

    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"device={device}, name={torch.cuda.get_device_name(device)}")
    else:
        print(f"device={device}")

    torch.manual_seed(0)
    a = torch.randn((args.size, args.size), device=device, dtype=dtype)
    b = torch.randn((args.size, args.size), device=device, dtype=dtype)

    for _ in range(args.warmup):
        c = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.repeat):
        c = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed * 1000.0 / args.repeat
    flops = 2 * args.size**3 * args.repeat
    tflops = flops / elapsed / 1e12

    checksum = c.float().sum().item()
    print(f"N={args.size}, repeat={args.repeat}, dtype={args.dtype}")
    print(f"avg_time_ms={avg_ms:.3f}, approx_tflops={tflops:.3f}, checksum={checksum:.6e}")


if __name__ == "__main__":
    main()
