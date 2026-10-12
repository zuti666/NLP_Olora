"""Detached smoke -> verified handoff -> two formal jobs, with no further cohort."""
import fcntl,json,os,subprocess,sys
from pathlib import Path
from common import atomic_json,assert_host,load_suite,now,proc_identity
from run_queue import verify_pins
H=Path(__file__).resolve().parent

def main():
 s=load_suite();assert_host(s);verify_pins(s)
 guard=json.loads((H/'handoff_guard.json').read_text())
 if os.readlink('/proc/self/ns/pid')!=guard['execution_pid_namespace']:raise RuntimeError('host namespace required')
 (H/'runtime').mkdir(exist_ok=True)
 lock=open(H/'runtime/pipeline.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status={'status':'running','started_at':now(),'process':proc_identity(os.getpid())}
 env=os.environ.copy();env.update(s['environment']);env.pop('CUDA_VISIBLE_DEVICES',None)
 phases=[('validate_baselines',['validate_baselines.py']),('gpu_smokes',['run_queue.py','--smoke-only']),('verify_gpu_smokes',['check_gpu_smokes.py']),('formal_training',['run_queue.py'])]
 for phase,args in phases:
  status.update(phase=phase,updated_at=now());atomic_json(H/'runtime/pipeline_state.json',status)
  rc=subprocess.run([s['python'],'-u',str(H/args[0]),*args[1:]],cwd=H,env=env).returncode
  if rc:
   status.update(status='blocked_failure',returncode=rc,updated_at=now());atomic_json(H/'runtime/pipeline_state.json',status);return rc
 status.update(status='completed',updated_at=now());atomic_json(H/'runtime/pipeline_state.json',status);return 0
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:
  atomic_json(H/'runtime/pipeline_error.json',dict(at=now(),error=repr(e)));raise
