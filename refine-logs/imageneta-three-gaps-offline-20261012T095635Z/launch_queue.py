"""Documented detached entrypoint; source pins are verified before launching."""
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
from queue_utils import atomic_json,now,identity
from run_queue import verify
from queue_utils import read
HERE=Path(__file__).resolve().parent
def main():
    suite=read(HERE/'suite.json');verify(suite)
    # Foreground dry-run is the documented non-GPU acceptance route.
    subprocess.run([suite['python'],str(HERE/'run_queue.py'),'--dry-run'],check=True)
    runtime=HERE/'runtime';runtime.mkdir(exist_ok=True)
    with open(runtime/'controller.log','xb') as log:
        proc=subprocess.Popen([suite['python'],str(HERE/'run_queue.py')],cwd=HERE,
                              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                              env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
    record=dict(process=identity(proc.pid),launched_at=now(),command=[suite['python'],str(HERE/'run_queue.py')])
    atomic_json(runtime/'launcher.json',record);print(__import__('json').dumps(record))
if __name__=='__main__':main()
