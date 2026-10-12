"""Bounded four-rank transport diagnostic; no model or optimizer update."""
from datetime import timedelta
import json
import os
import torch
import torch.distributed as dist

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=20))
rank = dist.get_rank()
print('NCCL_INIT', json.dumps({'rank': rank, 'local_rank': local_rank}), flush=True)
dist.barrier()
torch.cuda.synchronize()
print('INITIAL_BARRIER_PASS', rank, flush=True)
x = torch.tensor([rank + 1.0], dtype=torch.float32, device=f'cuda:{local_rank}')
dist.all_reduce(x, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
assert x.item() == 10.0 and dist.get_world_size() == 4
dist.barrier(device_ids=[local_rank])
torch.cuda.synchronize()
print('NCCL_PASS', json.dumps({'rank': rank, 'world_size': 4, 'sum': x.item(),
                             'p2p_disable': os.environ.get('NCCL_P2P_DISABLE')}), flush=True)
dist.destroy_process_group()
