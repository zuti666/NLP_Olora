import os,sys,runpy,json
from pathlib import Path
import torch
assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8"
torch.use_deterministic_algorithms(True,warn_only=False)
assert torch.are_deterministic_algorithms_enabled()
runner=Path("/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-recovery-20261007T064813Z/trainer/four_probe_train.py")
Path(os.environ["DETERMINISTIC_WITNESS_FILE"]).write_text(json.dumps({"torch_version":torch.__version__,"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"warn_only":torch.is_deterministic_algorithms_warn_only_enabled(),"cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"runner":str(runner)},indent=2)+"\n")
sys.path.insert(0,str(runner.parent))
sys.argv=[str(runner),*sys.argv[1:]]
runpy.run_path(str(runner),run_name="__main__")
