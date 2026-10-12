import json,numpy as np,torch
from rq2_common import *
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
cfg=json.loads((ROOT/'diagnostic_config.json').read_text())
from PIL import Image,features
from torchvision import datasets,transforms
print('PIL',Image.__version__,features.version_codec('jpg'),features.version_feature('libjpeg_turbo'),flush=True)
ds=datasets.ImageFolder(Path(cfg['data_root'])/'test',transform=transforms.Compose([transforms.Resize(256,interpolation=3),transforms.CenterCrop(224),transforms.ToTensor()]))
items=[ds[i] for i,(_,label) in enumerate(ds.samples) if label<20]
x=torch.stack([i[0] for i in items]);y=torch.tensor([i[1] for i in items])
rows=[]
for method in ['sgd','gam']:
 for task in [0,9]:
    model=load_network(checkpoint_path(method,task,cfg),cfg['source_path'],'cuda:0')
    row={'method':method,'task':task,'metrics':evaluate(model,x,y,'cuda:0',64)}
    rows.append(row);print(json.dumps(row),flush=True)
    del model;torch.cuda.empty_cache()
atomic_json(ROOT/'investigation/pillow113_gpu_witness.json',rows)
