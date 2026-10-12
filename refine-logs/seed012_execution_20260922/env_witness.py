import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PY='/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'
CODE='''import torch, json, yaml, timm, torchvision
torch.set_num_threads(2)
assert torch.cuda.is_available()
torch.manual_seed(0)
x=torch.randn(8,8,device="cuda",requires_grad=True)
y=x@x
y.square().mean().backward()
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
torch.cuda.synchronize()
print(json.dumps(dict(witness="PASS",shape=list(y.shape),device=torch.cuda.get_device_name(),torch=torch.__version__,timm=timm.__version__,torchvision=torchvision.__version__,yaml=yaml.__version__,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)))'''
def main():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    gpus={int(p[0]):dict(index=int(p[0]),uuid=p[1],memory_used=int(p[2]),utilization=int(p[3])) for p in ([x.strip() for x in line.split(',')]for line in lines)}
    records=[]
    for index in [2,3,4,5]:
        gpu=gpus[index]
        assert gpu['memory_used']<500 and gpu['utilization']<5,gpu
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        run=subprocess.run([PY,'-c',CODE],env=env,capture_output=True,text=True,check=True)
        result=json.loads(run.stdout.strip().splitlines()[-1]);records.append(dict(gpu=gpu,result=result));print(index,result,flush=True)
    (ROOT/'env_witness.json').write_text(json.dumps(dict(passed=True,python=PY,gpus=records),indent=2)+'\n')
if __name__=='__main__':main()
