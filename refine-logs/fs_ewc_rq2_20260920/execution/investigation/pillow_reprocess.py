import json,hashlib,sys
from pathlib import Path
from PIL import Image,features
import PIL
root=Path('/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution')
d=json.loads((root/'results_seed0/sgd/data_manifest.json').read_text())
cache=root/'investigation/test307_olddecode';cache.mkdir(exist_ok=True)
result=[]
for i,rec in enumerate(d['records']['test']):
    png=cache/('%04d.png'%i)
    if sys.argv[1]=='export':
        with Image.open(Path(d['data_root'])/rec['id']) as raw:
            image=raw.convert('RGB')
        image.save(png,compress_level=1)
    else:
        with Image.open(png) as raw:
            image=raw.convert('RGB')
    decoded=hashlib.sha256(image.tobytes()).hexdigest()
    width,height=image.size
    resized=(256,int(256*height/width)) if width<=height else (int(256*width/height),256)
    image=image.resize(resized,resample=Image.Resampling.BICUBIC)
    left=round((image.width-224)/2.0); top=round((image.height-224)/2.0)
    image=image.crop((left,top,left+224,top+224))
    result.append({'id':rec['id'],'png':str(png),'decoded_hash':decoded,'processed_hash':hashlib.sha256(image.tobytes()).hexdigest()})
record={'pillow':PIL.__version__,'jpeg':features.version('jpg'),'images':result}
Path(sys.argv[2]).write_text(json.dumps(record,indent=2)+'\n')
print({'pillow':PIL.__version__,'images':len(result),'output':sys.argv[2]})
