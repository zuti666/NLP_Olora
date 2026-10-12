"""CPU-only AST extraction of archived/live LoRA wrappers. No trainer or GPU use."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import ast
import contextlib
import copy
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F, Parameter
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop
import torchvision

HERE=Path(__file__).resolve().parent
torch.set_num_threads(1)
manifest=json.loads((HERE/'rng_source_manifest.json').read_text())

def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()

def load_class(source, name):
    tree=ast.parse(source)
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==name)
    env={'torch':torch,'nn':nn,'F':F,'Parameter':Parameter,'os':os,'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<extracted:'+name+'>','exec'),env)
    return env[name]

class Scale(nn.Module):
    def __init__(self,v):
        super().__init__()
        self.scale=nn.Parameter(torch.tensor(float(v)))
    def forward(self,x):
        return self.scale*x

def make_model(cls,task_id,is_eval_wrapper=False):
    torch.random.default_generator.manual_seed(1001)
    d,r=6,2
    qkv=nn.Linear(d,d*3,bias=True)
    aq,bq,av,bv=nn.Linear(d,r,False),nn.Linear(r,d,False),nn.Linear(d,r,False),nn.Linear(r,d,False)
    paq,pbq,pav,pbv=nn.Linear(d,r,False),nn.Linear(r,d,False),nn.Linear(d,r,False),nn.Linear(r,d,False)
    saved_a={'saved_A_0':[paq,pav]};saved_b={'saved_B_0':[pbq,pbv]}
    scales=nn.ModuleList([Scale(.8)])
    previous=nn.ModuleList([Scale(.7) for _ in range(task_id)])
    if is_eval_wrapper:
        saved_a[f'saved_A_{task_id}']=[aq,av];saved_b[f'saved_B_{task_id}']=[bq,bv]
        return cls(task_id,qkv,saved_a,saved_b,0,r,scales,previous,str(HERE/'unused_nonexistent'),learn_alpha=False)
    return cls(qkv,aq,bq,av,bv,task_id,saved_a,saved_b,0,r,scales,previous)

head_source=manifest['auxiliary']['SimpleLinear']
head_text=Path(head_source['captured']).read_text()
assert hashlib.sha256(head_text.encode()).hexdigest()==head_source['sha256']
SimpleLinear=load_class(head_text,'SimpleLinear')

def downstream(state):
    # Each downstream example begins at precisely the post-forward CPU RNG state.
    torch.set_rng_state(state.clone())
    order=torch.cat(list(DataLoader(torch.arange(20),batch_size=5,shuffle=True,num_workers=0))).tolist()
    torch.set_rng_state(state.clone())
    head=SimpleLinear(6,7)
    torch.set_rng_state(state.clone())
    crop=RandomCrop.get_params(torch.arange(3*32*32).reshape(3,32,32),(24,24))
    flip_draw=float(torch.rand(1).item())
    # DataLoader creates the workers' base seed by this global generator draw.
    torch.set_rng_state(state.clone())
    worker_base_seed=int(torch.empty((),dtype=torch.int64).random_().item())
    return {'next_dataloader_order':order,'next_head_weight_sha256':digest(head.weight),
            'next_crop_params':crop,'next_flip_uniform':flip_draw,'next_worker_base_seed_draw':worker_base_seed}

cases=[];tensors={}
for source in manifest['sources']:
    text=Path(source['captured']).read_text()
    assert hashlib.sha256(text.encode()).hexdigest()==source['file_sha256']
    for cls_name in ['_LoRA_qkv_timm_train','_LoRA_qkv_timm_eval']:
        cls=load_class(text,cls_name)
        for task in [0,1]:
            for mode in ['train_backward','eval_backward','eval_no_grad']:
                model=make_model(cls,task,is_eval_wrapper=cls_name.endswith('_eval'))
                # This class stores a bool named .eval; use Module.train(False), as parent .eval() does.
                model.train(mode=='train_backward')
                x=torch.linspace(-.7,.8,2*3*6).reshape(2,3,6).requires_grad_(True)
                target=torch.linspace(-.2,.3,2*3*18).reshape(2,3,18)
                if mode!='eval_no_grad' and cls_name.endswith('_train'):
                    model._register_delta_hook=True
                torch.random.default_generator.manual_seed(777)
                pre=torch.get_rng_state().clone()
                with (torch.no_grad() if mode=='eval_no_grad' else contextlib.nullcontext()):
                    y=model(x)
                    if mode!='eval_no_grad' and y.requires_grad:
                        ((y-target)**2).mean().backward()
                post=torch.get_rng_state().clone()
                torch.set_rng_state(pre.clone())
                for a,b in [(6,2),(2,6),(6,2),(2,6)]:
                    nn.Linear(a,b,bias=False)
                simulated=torch.get_rng_state().clone()
                key=(source['name'],cls_name,task,mode)
                grads={name:p.grad.detach().clone() if p.grad is not None else None for name,p in model.named_parameters()}
                if x.grad is not None:grads['__input__']=x.grad.detach().clone()
                for attr in ['delta_w_q_new_grad','delta_w_v_new_grad']:
                    grad=getattr(model,attr,None)
                    if grad is not None:grads[attr]=grad.detach().clone()
                saved_grads=all(p.grad is None for a in model.saved_A.values() for m in a for p in m.parameters()) and all(p.grad is None for a in model.saved_B.values() for m in a for p in m.parameters())
                tensors[key]=(y.detach().clone(),grads,post)
                cases.append({'source':source['name'],'class':cls_name,'task_id':task,'mode':mode,
                              'cpu_rng_changed':not torch.equal(pre,post),'output_requires_grad':y.requires_grad,
                              'rng_matches_four_temp_linear_initializations':torch.equal(post,simulated),
                              'output_sha256':digest(y),'cpu_rng_before_sha256':digest(pre),'cpu_rng_after_sha256':digest(post),
                              'saved_adapter_grads_all_none':saved_grads,'downstream':downstream(post)})

comparisons=[]
for case in cases:
    key=(case['source'],case['class'],case['task_id'],case['mode'])
    refkey=('Copy_git_HEAD',case['class'],case['task_id'],case['mode'])
    y,g,s=tensors[key];yr,gr,sr=tensors[refkey]
    assert set(g)==set(gr)
    grad_error=max([float((v-gr[k]).abs().max()) for k,v in g.items() if v is not None]+[0.0])
    assert all((g[k] is None)==(gr[k] is None) for k in g)
    comparisons.append({'source':case['source'],'reference':'Copy_git_HEAD','class':case['class'],'task_id':case['task_id'],'mode':case['mode'],
                        'output_max_abs_diff':float((y-yr).abs().max()),'grad_max_abs_diff':grad_error,'cpu_rng_states_equal':torch.equal(s,sr)})

assert all(c['output_max_abs_diff']==0.0 and c['grad_max_abs_diff']==0.0 for c in comparisons)
assert all(c['saved_adapter_grads_all_none'] for c in cases)
assert not torch.cuda.is_initialized()
result={'checked_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,'torch':torch.__version__,'torchvision':torchvision.__version__,
        'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'cuda_initialized':torch.cuda.is_initialized(),
        'device':'CPU','dims':{'batch':2,'tokens':3,'dimension':6,'rank':2},
        'source_manifest':str(HERE/'rng_source_manifest.json'),'cases':cases,'comparisons':comparisons,
        'assertions':{'all_outputs_and_gradients_bit_exact_against_HEAD':True,'saved_adapter_gradients_none':True,'cuda_never_initialized':True}}
(HERE/'rng_diff.json').write_text(json.dumps(result,indent=2)+'\n')
for source in manifest['sources']:
    selected=[c for c in cases if c['source']==source['name']]
    print(source['name'],'trainclass changed',sorted(set(c['cpu_rng_changed'] for c in selected if c['class'].endswith('_train'))),
          'evalclass changed',sorted(set(c['cpu_rng_changed'] for c in selected if c['class'].endswith('_eval'))))
print('PASS',len(cases),'CPU cases; outputs and available gradients bit exact; CUDA never initialized')
