from pathlib import Path
import sys,os,subprocess,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
OLD=Path("/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-recovery-20261007T064813Z")
sys.path.insert(0,str(OLD))
from common import load_suite,assert_host,verify_pins,query_gpus,atomic_json,proc_identity
from run_queue import acquire_idle_gpu
suite=load_suite(OLD/"suite.json");assert_host(suite);verify_pins(suite)
used=set();records=[]
for variant in ("N5","N6"):
 selected=acquire_idle_gpu(suite,query_gpus(),used)
 if selected is None:raise RuntimeError("No eligible idle GPU for bounded preflight")
 index,record,held=selected
 env=os.environ.copy();env.update(suite["environment"]);env["CUDA_VISIBLE_DEVICES"]=record["uuid"];env["DETERMINISTIC_WITNESS_FILE"]=str(H/(variant+"_deterministic_witness.json"))
 command=[suite["python"],str(H/"deterministic_entry.py"),"--source",suite["source_root"],"--config",str(H/"configs"/("deterministic_"+variant+".json")),"--output-dir",str(H/"outputs"/("deterministic_"+variant)),"--max-tasks","2","--max-epochs","1","--max-batches","1"]
 with (H/"logs"/(variant+".log")).open("xb") as log:
  p=subprocess.Popen(command,env=env,cwd=H,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(x.fileno() for x in held))
 records.append({"variant":variant,"gpu":index,"gpu_uuid":record["uuid"],"process":proc_identity(p.pid),"command":command})
 used.add(index)
 for lease in held:lease.close()
atomic_json(H/"LAUNCH.json",{"status":"bounded_2_smoke_preflight_started","formal_started":0,"jobs":records})
print(json.dumps(records))
