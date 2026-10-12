"""Load the actual default timm backbone on CPU and trace its offline HF file resolution."""
import hashlib
import json
import os
import sys
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
os.environ.update(json.loads((HERE/'environment.json').read_text()))
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import timm
import huggingface_hub
import timm.models._hub as hub
torch.set_num_threads(2)
assert os.environ['HF_HUB_OFFLINE']=='1'
assert os.environ['HF_HOME']=='/data/115-2/users/liying/cache/huggingface'
trace=[]
original=hub.hf_hub_download
def tracked(*args,**kwargs):
    result=original(*args,**kwargs)
    trace.append({'args':list(args),'kwargs':kwargs,'returned_path':str(result)})
    return result
hub.hf_hub_download=tracked
model=timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=0)
assert next(model.parameters()).device.type=='cpu'
assert not torch.cuda.is_initialized()
weight_calls=[t for t in trace if t['returned_path'].endswith('model.safetensors')]
assert len(weight_calls)==1,trace
path=Path(weight_calls[0]['returned_path']);resolved=path.resolve()
digest=hashlib.sha256()
with path.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
actual=digest.hexdigest()
expected='32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b'
assert actual==expected,(actual,expected)
revision=path.parts[path.parts.index('snapshots')+1]
repo_id=model.pretrained_cfg['hf_hub_id']
again=huggingface_hub.hf_hub_download(repo_id=repo_id,filename='model.safetensors',revision=revision,local_files_only=True)
assert Path(again).resolve()==resolved
out={'checked_utc':datetime.now(timezone.utc).isoformat(),'pass':True,'python':sys.executable,
     'torch':torch.__version__,'timm':timm.__version__,'huggingface_hub':huggingface_hub.__version__,
     'model_call':{'name':'vit_base_patch16_224','pretrained':True,'num_classes':0},'pretrained_cfg':model.pretrained_cfg,
     'hf_home':os.environ['HF_HOME'],'hf_hub_offline':os.environ['HF_HUB_OFFLINE'],
     'actual_timm_hf_download_calls':trace,'repo_id':repo_id,'snapshot_revision':revision,'weight_path':str(path),
     'weight_realpath':str(resolved),'weight_bytes':path.stat().st_size,'weight_sha256':actual,
     'expected_cllora_weight_sha256':expected,'same_weight_bytes_as_cllora':True,'local_files_only_confirmation_path':again,
     'device':str(next(model.parameters()).device),'parameter_count':sum(p.numel() for p in model.parameters()),
     'cuda_initialized':False,'training_launched':False,'note':'Confirms current candidate environment resolves the same file bytes; does not certify historical March/Aug run weights.'}
(HERE/'weight_resolution.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
print('PASS loaded CPU timm backbone;',repo_id,revision,actual,'CUDA uninitialized')
