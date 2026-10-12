import json,hashlib,sys
from pathlib import Path
from PIL import Image,features
import PIL
root=Path('/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution')
d=json.loads((root/'results_seed0/sgd/data_manifest.json').read_text())
images=[]
for rec in d['records']['test']:
    with Image.open(Path(d['data_root'])/rec['id']) as raw:
        img=raw.convert('RGB')
    decoded_hash=hashlib.sha256(img.tobytes()).hexdigest()
    width,height=img.size
    resized=(256,int(256*height/width)) if width<=height else (int(256*width/height),256)
    im=img.resize(resized,resample=Image.Resampling.BICUBIC)
    left=round((im.width-224)/2.0); top=round((im.height-224)/2.0)
    im=im.crop((left,top,left+224,top+224))
    images.append({'id':rec['id'],'decoded_hash':decoded_hash,'processed_hash':hashlib.sha256(im.tobytes()).hexdigest(),'shape':img.size})
res={'pillow_version':PIL.__version__,'jpeg':features.version('jpg'),'jpeg_turbo':features.version('libjpeg_turbo'),'images':images}
Path(sys.argv[1]).write_text(json.dumps(res,indent=2)+'\n')
print({k:v for k,v in res.items() if k!='images'})
