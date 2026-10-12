#!/usr/bin/env python3
"""Execute the real CLI/main with a CPU mock native trainer, including evaluation.

This tests scope/restoration of SDPA policy, not a ViT/GPU training run.
"""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType
from unittest.mock import patch

sys.dont_write_bytecode=True
import component_horizon as trainer
import hvp_helpers as helper

SOURCE=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')


def run_case(backend,fail):
    before=helper.backend_state();observed=[]
    class ExpectedFailure(RuntimeError):pass
    class FakeController:
        def __init__(self,learner,cfg,cli):
            self.out=Path(cli.output_dir);self.mode=cfg['component_scale'];self.handle=io.StringIO()
            self.matrix=[[50.]];self.task_rows=[{'steps':0}];self.rows=[{}]
            self.after_task_checks=[{}];self.steps=0;self.match_count=0;self.stepper=None
            self.counters={k:0 for k in ('forbidden_work_attempts','fisher_estimation_passes','weight_merge_updates',
                'hvp_forward_passes','hvp_first_derivative_passes','hvp_second_derivative_passes','total_training_forward_passes')}
            self.checkpoints=[];self.task_identities=[{}];self.max_match_error=0.;self.started=time.time()
            self.trace=self.out/'step_trace.jsonl';self.trace.write_text('')
            for name,value in [('task_identities.json',[{}]),('task_checkpoints.json',{}),('task_matrix.json',{'matrix':self.matrix})]:
                (self.out/name).write_text(json.dumps(value))
        def install(self):observe('controller_install')
    def observe(phase):
        flags=helper.backend_state()
        if backend=='math':helper.require_math_backend()
        else:assert flags==before
        observed.append({'phase':phase,'flags':flags})
    factory=ModuleType('utils.factory');factory.get_model=lambda *_:observe('model_initialization')
    original_factory=factory.get_model
    utils=ModuleType('utils');utils.factory=factory
    random=ModuleType('utils.random_reproduce');random.json_safe=lambda x:x
    native=ModuleType('src.trainer')
    def train(cfg):
        observe('train_entry');factory.get_model('mock',cfg)
        with helper.backend_context(backend):observe('nested_training_batch')
        observe('evaluation_callback')
        if fail:raise ExpectedFailure('intentional exception after evaluation')
        observe('train_return')
    native.train=train;src=ModuleType('src');src.trainer=native
    modules={'utils':utils,'utils.factory':factory,'utils.random_reproduce':random,'src':src,'src.trainer':native}
    cwd=Path.cwd();path=list(sys.path)
    with tempfile.TemporaryDirectory(prefix='fdhvp_main_backend_') as directory:
        root=Path(directory);config=root/'config.json';output=root/'out'
        config.write_text(json.dumps({'component_rule':'ug','component_scale':'natural','seed':2,
            'component_attention_backend':backend,'component_first_response':'fd','component_second_response':'hvp' if backend=='math' else 'fd'}))
        argv=['component_horizon.py','--source',str(SOURCE),'--config',str(config),'--output-dir',str(output),
              '--max-tasks','1','--max-epochs','1','--max-batches','1']
        try:
            with patch.dict(sys.modules,modules),patch.object(trainer,'Controller',FakeController),patch.object(sys,'argv',argv):
                try:trainer.main()
                except ExpectedFailure:
                    assert fail
                else:assert not fail
            summary=json.loads((output/'final_summary.json').read_text())
            assert summary['status']==('failed' if fail else 'completed')
            assert factory.get_model is original_factory
        finally:os.chdir(cwd);sys.path[:]=path
    assert helper.backend_state()==before, 'SDPA state leaked outside main'
    assert {'model_initialization','evaluation_callback','nested_training_batch'}<={row['phase'] for row in observed}
    return {'backend':backend,'intentional_failure':fail,'main_return_status':summary['status'],
            'observed':observed,'outer_backend_restored_exact':True,'factory_restored_exact':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    report={'status':'passed','device':'cpu','scope':'real CLI main with mock native trainer and evaluation callback',
            'cases':[run_case(b,f) for b in ('native','math') for f in (False,True)],
            'trainer_sha256':trainer.audit.digest(trainer.__file__),'test_sha256':trainer.audit.digest(__file__)}
    if args.output:trainer.audit.atomic_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
