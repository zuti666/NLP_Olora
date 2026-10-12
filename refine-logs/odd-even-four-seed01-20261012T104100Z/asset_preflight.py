import hashlib, importlib.metadata as meta, json, sys
from pathlib import Path
from datetime import datetime, timezone
import torch, torchvision, timm
from torchvision.datasets import ImageFolder
H=Path(__file__).resolve().parent
spec=json.loads((H/"environment_spec.json").read_text())
def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
canonical=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(",",":")).encode()).hexdigest()
report=dict(status="running",at=datetime.now(timezone.utc).isoformat(),canonical_spec_sha256=canonical,package_rebuild=False,gpu_calls=0)
try:
 assert canonical=="4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8",canonical
 observed={"python":".".join(map(str,sys.version_info[:3])),"torch_runtime":torch.__version__,"torchvision_runtime":torchvision.__version__,"timm":timm.__version__}
 for key,val in observed.items():assert val==spec["software"][key],(key,val,spec["software"][key])
 report["software"]=observed
 mp=Path(spec["provenance"]["old_source_manifest"])
 source=Path(spec["provenance"]["source_root"])
 m=json.loads(mp.read_text());files=m["files"]
 assert digest(mp)==spec["provenance"]["source_manifest_sha256"]
 for rel,expected in files.items():assert digest(source/rel)==expected,rel
 report["source"]=dict(root=str(source),source_file_count=len(files),manifest_path=str(mp),manifest_sha256=digest(mp),all_source_hashes_exact=True)
 hashes={};counts={}
 base=Path(spec["data"]["root"])/"imagenet-a"
 for split in ("train","test"):
  data=ImageFolder(str(base/split));a=hashlib.sha256();b=hashlib.sha256()
  class_counts=[0]*200
  assert len(data.classes)==200
  for filename,label in data.samples:
   p=Path(filename);rel=str(p.relative_to(base));a.update((rel+"\t"+str(label)+"\n").encode());b.update((rel+"\t"+str(label)+"\t"+str(p.stat().st_size)+"\n").encode());class_counts[label]+=1
  hashes[split+"_path_label"]=a.hexdigest();hashes[split+"_path_label_size"]=b.hexdigest()
  assert len(data)==spec["data"]["expected_"+split+"_images"]
  tasks=[sum(class_counts[k:k+20]) for k in range(0,200,20)]
  assert tasks==spec["data"][split+"_samples_per_task"],tasks
  counts[split]=dict(images=len(data),classes=len(data.classes),task_counts=tasks)
 assert hashes==spec["data"]["asset_hashes"],hashes
 report["data"]=dict(counts=counts,asset_hashes=hashes,all_expected_hashes_exact=True)
 w=spec["weight_dirs"]["vit_b16"];p=Path(w["file"])
 assert p.stat().st_size==w["bytes"]
 assert digest(p)==w["sha256"]
 report["pretrained"]=dict(path=str(p),bytes=p.stat().st_size,sha256=w["sha256"],verified=True)
 report["status"]="passed"
except BaseException as e:
 report.update(status="failed",error=repr(e));raise
finally:
 (H/"gates/INPUT_ASSETS.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2))
