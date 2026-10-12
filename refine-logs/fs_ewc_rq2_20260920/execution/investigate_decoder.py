import json,numpy as np,torch
from rq2_common import *
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
cfg=json.loads((ROOT/'diagnostic_config.json').read_text())
z=np.load(ROOT/'investigation/test307_pillow11.npz')
x=torch.from_numpy(z['x']).permute(0,3,1,2).contiguous().float()/255
y=torch.from_numpy(z['y'])
rows=[]
for method in ['sgd','gam']:
 for task in [0,9]:
    model=load_network(checkpoint_path(method,task,cfg),cfg['source_path'],'cuda:0')
    row={'method':method,'task':task,'metrics':evaluate(model,x,y,'cuda:0',64)}
    rows.append(row);print(json.dumps(row),flush=True)
    del model;torch.cuda.empty_cache()
atomic_json(ROOT/'investigation/decoder_gpu_witness.json',rows)
