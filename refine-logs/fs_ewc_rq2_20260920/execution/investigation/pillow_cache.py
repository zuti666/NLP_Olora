import json,sys,hashlib
from pathlib import Path
from PIL import Image,features
import PIL
import numpy as np
root=Path('/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution')
d=json.loads((root/'results_seed0/sgd/data_manifest.json').read_text())
xx=[];yy=[]
for rec in d['records']['test']:
 with Image.open(Path(d['data_root'])/rec['id']) as raw:
  img=raw.convert('RGB')
 width,height=img.size
 resized=(256,int(256*height/width)) if width<=height else (int(256*width/height),256)
 img=img.resize(resized,resample=Image.Resampling.BICUBIC)
 left=round((img.width-224)/2.0);top=round((img.height-224)/2.0)
 img=img.crop((left,top,left+224,top+224))
 xx.append(np.asarray(img));yy.append(rec['label'])
out=Path(sys.argv[1]);np.savez(out,x=np.stack(xx),y=np.array(yy))
print(json.dumps({'pillow':PIL.__version__,'jpeg':features.version('jpg'),'jpeg_turbo':features.version('libjpeg_turbo'),'output':str(out),'shape':[len(xx),224,224,3],'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
