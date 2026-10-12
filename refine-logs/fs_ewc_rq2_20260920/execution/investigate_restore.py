import json,sys,torch
from pathlib import Path
from rq2_common import *
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
cfg=json.loads((ROOT/'diagnostic_config.json').read_text())
x,y,data=prepare_data(cfg,ROOT/'restore_investigation')
sys.path.insert(0,cfg['source_path'])
from utils.data import build_transform
from utils.data_manager import pil_loader
from torchvision import transforms
tf=transforms.Compose(build_transform(False,None))
checks=[]
for i,r in enumerate(data['samples']['test'] if 'samples' in data else []):
    pass
model=load_network(checkpoint_path('sgd',0,cfg),cfg['source_path'],'cuda:0')
variants=[('native_exact',{}),('gelu_tanh',{'gelu':'tanh'}),('norm_eps1e-5',{'eps':1e-5}),('unfused',{'fused':False})]
rows=[]
for label,opts in variants:
    for mod in model.modules():
        if isinstance(mod,torch.nn.GELU): mod.approximate=opts.get('gelu','none')
        if isinstance(mod,torch.nn.LayerNorm): mod.eps=opts.get('eps',1e-6)
        if hasattr(mod,'fused_attn'):mod.fused_attn=opts.get('fused',True)
    row={'variant':label,'metrics':evaluate(model,x['test'],y['test'],'cuda:0',64)}
    rows.append(row);print(json.dumps(row),flush=True)
atomic_json(ROOT/'restore_investigation/variants.json',rows)
