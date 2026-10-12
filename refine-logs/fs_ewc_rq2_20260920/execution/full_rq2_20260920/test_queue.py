import importlib.util
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('rq2_full_queue', Path(__file__).with_name('campaign_runner.py'))
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class QueueTests(unittest.TestCase):
    def test_locked_idle_gpu_skipped_and_locks_inherited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            locks = root / 'locks'
            locks.mkdir()
            out = root / 'gpu.json'
            worker = """import fcntl,json,sys
from pathlib import Path
checks = {}
for name,path in [('gpu',sys.argv[3]),('pipeline',sys.argv[4])]:
    with open(path,'a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            checks[name+'_lock_inherited'] = False
        except BlockingIOError:
            checks[name+'_lock_inherited'] = True
Path(sys.argv[1]).write_text(json.dumps({'status':'complete','gpu':sys.argv[2],**checks}))
"""
            job = dict(id='first', gpu=0, gpu_pool=[0,6],
                       command=[sys.executable,'-c',worker,str(out),'{gpu}',str(locks/'GPU-6.lock'),str(root/'run/pipeline.lock')],
                       receipt=str(out),receipt_require={'status':'complete','gpu':'6',
                                                        'gpu_lock_inherited':True,'pipeline_lock_inherited':True})
            plan = dict(gpus=[0,6],max_parallel=1,run_dir=str(root/'run'),gpu_lock_dir=str(locks),
                        code_files=[__file__],stages=[dict(id='a',depends_on=[],jobs=[job])])
            pp = root/'plan.json'
            pp.write_text(json.dumps(plan))
            def gpu_status(gpu):
                return dict(index=gpu,uuid=f'GPU-{gpu}',memory_mib=2,utilization=0,has_compute_process=False)
            original_run_job = queue.PipelineRunner.run_job
            def guarded_run_job(runner, launch_job):
                # Fail immediately on the old behavior instead of hanging behind our held GPU0 lock.
                self.assertEqual(launch_job['gpu'],6)
                return original_run_job(runner,launch_job)
            with (locks/'GPU-0.lock').open('a') as external:
                fcntl.flock(external,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(queue,'gpu_status',side_effect=gpu_status), \
                     patch.object(queue.PipelineRunner,'run_job',new=guarded_run_job):
                    state = queue.execute_plan(plan,pp)
            self.assertEqual(state['status'],'complete')
            self.assertEqual(state['jobs']['first']['gpu'],6)
            # Completed jobs must release the parent's GPU reservation.
            with (locks/'GPU-6.lock').open('a') as released:
                fcntl.flock(released,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_resume_rejects_changed_receipt_and_frozen_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root/'result.json'
            marker = root/'source.py'
            marker.write_text('VERSION=1\n')
            job = dict(id='cpu',gpu=None,
                       command=[sys.executable,'-c',f"from pathlib import Path;Path({str(out)!r}).write_text('{{\"status\":\"complete\"}}')"],
                       receipt=str(out),receipt_require={'status':'complete'})
            plan = dict(gpus=[0],max_parallel=1,run_dir=str(root/'run'),gpu_lock_dir=str(root/'locks'),
                        code_files=[str(marker)],stages=[dict(id='a',depends_on=[],jobs=[job])])
            pp = root/'plan.json'
            pp.write_text(json.dumps(plan))
            queue.execute_plan(plan,pp)
            original = out.read_text()
            out.write_text('{"status":"complete","changed":true}')
            with self.assertRaisesRegex(RuntimeError,'Completed receipt changed'):
                queue.execute_plan(plan,pp,resume=True)
            out.write_text(original)
            marker.write_text('VERSION=2\n')
            with self.assertRaisesRegex(RuntimeError,'Code/plan changed'):
                queue.execute_plan(plan,pp,resume=True)

    def test_dynamic_idle_gpu_and_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'gpu.json'
            final = root / 'final.json'
            worker = "import json,os,sys;from pathlib import Path;Path(sys.argv[1]).write_text(json.dumps({'status':'complete','gpu':sys.argv[2],'visible':os.environ['CUDA_VISIBLE_DEVICES']}))"
            first = dict(id='first', gpu=0, gpu_pool=[0,6], command=[sys.executable,'-c',worker,str(out),'{gpu}'],
                         receipt=str(out), receipt_require={'status':'complete','gpu':'6','visible':'GPU-6'})
            second = dict(id='second', gpu=None, command=[sys.executable,'-c',worker,str(final),'cpu'],
                          receipt=str(final), receipt_require={'status':'complete','gpu':'cpu'})
            plan = dict(gpus=[0,6], max_parallel=2, run_dir=str(root/'run'), gpu_lock_dir=str(root/'locks'),
                        code_files=[__file__], stages=[dict(id='a',depends_on=[],jobs=[first]),dict(id='b',depends_on=['a'],jobs=[second])])
            pp=root/'plan.json';pp.write_text(json.dumps(plan))
            def gpu_status(gpu):
                return dict(index=gpu, uuid=f'GPU-{gpu}', memory_mib=2000 if gpu==0 else 2,
                            utilization=99 if gpu==0 else 0,has_compute_process=gpu==0)
            with patch.object(queue,'gpu_status',side_effect=gpu_status):
                state=queue.execute_plan(plan,pp)
            self.assertEqual(state['status'],'complete')
            self.assertEqual(state['jobs']['first']['gpu'],6)
            self.assertGreaterEqual(state['jobs']['second']['started_utc'],state['jobs']['first']['ended_utc'])

    def test_bad_receipt_blocks_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'bad.json';child=root/'child.json'
            a=dict(id='a',gpu=None,command=[sys.executable,'-c',f"from pathlib import Path;Path({str(out)!r}).write_text('{{}}')"],receipt=str(out),receipt_require={'status':'complete'})
            b=dict(id='b',gpu=None,command=[sys.executable,'-c',f"from pathlib import Path;Path({str(child)!r}).write_text('{{}}')"],receipt=str(child),receipt_require={'status':'complete'})
            plan=dict(gpus=[0],max_parallel=1,run_dir=str(root/'run'),gpu_lock_dir=str(root/'locks'),code_files=[__file__],
                      stages=[dict(id='a',depends_on=[],jobs=[a]),dict(id='b',depends_on=['a'],jobs=[b])])
            pp=root/'plan.json';pp.write_text(json.dumps(plan))
            with self.assertRaises(RuntimeError):queue.execute_plan(plan,pp)
            state=json.loads((root/'run/pipeline_state.json').read_text())
            self.assertEqual(state['jobs']['b']['status'],'blocked')
            self.assertFalse(child.exists())


if __name__=='__main__':unittest.main()
