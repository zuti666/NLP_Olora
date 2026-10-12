"""Execute one frozen experiment and require finite, complete task metrics."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    suite=json.loads((ROOT/'suite.json').read_text());job=next(j for j in suite['jobs']if j['id']==args.job)
    receipt=ROOT/'receipts'/f"{job['id']}.json"
    assert not receipt.exists(),'Never overwrite an existing attempt'
    record=dict(job,status='running',pid=os.getpid(),started_unix=time.time(),gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))
    save(receipt,record)
    try:
        preflight=json.loads((ROOT/'preflight.json').read_text());assert preflight['passed']
        assert sha(job['config'])==job['config_sha256']
        assert sha(job['source_manifest'])==job['source_manifest_sha256']
        assert sha(job['data_manifest'])==job['data_manifest_sha256']
        for item in preflight['checkpoint_files']:assert sha(item['path'])==item['sha256'],item['path']
        manifest=json.loads(Path(job['source_manifest']).read_text())
        for relative,digest in manifest['files'].items():assert sha(Path(job['source_root'])/relative)==digest,relative
        for relative,digest in json.loads((ROOT/'runtime_frozen.json').read_text())['files'].items():assert sha(ROOT/relative)==digest,relative
        sys.path.insert(0,job['source_root']);os.chdir(job['source_root'])
        import torch
        import timm
        import torchvision
        torch.set_num_threads(2)
        assert torch.cuda.is_available(),'Refuse CPU training fallback'
        assert torch.cuda.device_count()==1, 'Each child must see exactly one GPU'
        record.update(environment={k:os.environ.get(k)for k in list(suite['environment'])+['CUDA_VISIBLE_DEVICES','PYTHONHASHSEED']},
            python=sys.executable,torch=torch.__version__,timm=timm.__version__,torchvision=torchvision.__version__,
            checkpoint_files=preflight['checkpoint_files'],preflight_sha256=sha(ROOT/'preflight.json'),suite_sha256=sha(ROOT/'suite.json'),
            runtime_manifest_sha256=sha(ROOT/'runtime_frozen.json'))
        save(receipt,record)
        original_step=torch.optim.SGD.step
        def finite_step(optimizer,*a,**kw):
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite SGD gradient')
            result=original_step(optimizer,*a,**kw)
            for group in optimizer.param_groups:
                for p in group['params']:
                    if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite SGD parameter')
            return result
        torch.optim.SGD.step=finite_step
        from src.main import main as train
        assert Path(sys.modules['src.main'].__file__).resolve().is_relative_to(Path(job['source_root']).resolve())
        train(['--config',job['config'],'--override','device=[0]'])
        cfg=json.loads(Path(job['config']).read_text())
        paths=list(Path(cfg['outputs_root']).glob('**/'+cfg['prefix']+'*cl_metrics.json'))
        assert len(paths)==1,paths
        path=paths[0];raw=json.loads(path.read_text());matrix=raw['cnn']['matrices']['final']
        assert len(matrix)==job['expected_tasks'] and all(len(row)==len(matrix)for row in matrix),len(matrix)
        rows=[row[:i+1]for i,row in enumerate(matrix)]
        assert all(len(row)==i+1 and all(math.isfinite(x)and 0<=x<=100 for x in row)for i,row in enumerate(rows))
        log=path.with_name(path.name.replace('_cl_metrics.json','.log'));assert log.is_file()
        assert not re.search(r'(?:Loss|loss)[^\n]{0,80}?(?:\bnan\b|\binf\b)',log.read_text())
        record.update(status='completed',exit_code=0,finished_unix=time.time(),metric_path=str(path),metric_sha256=sha(path),
            log=str(log),log_sha256=sha(log),metrics=dict(tasks=len(rows),faa=statistics.mean(rows[-1]),
            aaa_stage=statistics.mean(statistics.mean(row)for row in rows)),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            source_manifest_sha256=sha(job['source_manifest']),device_name=torch.cuda.get_device_name())
        save(receipt,record);return 0
    except BaseException as exc:
        record.update(status='failed',exit_code=1,finished_unix=time.time(),error=repr(exc),traceback=traceback.format_exc())
        save(receipt,record);traceback.print_exc();return 1
if __name__=='__main__':raise SystemExit(main())
