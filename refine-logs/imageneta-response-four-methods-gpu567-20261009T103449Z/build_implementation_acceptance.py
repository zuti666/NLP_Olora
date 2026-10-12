"""Read-only acceptance of the previously completed eight implementation smokes."""
import json,sys,hashlib
from pathlib import Path
N=Path(__file__).resolve().parent
P=N.parent/'imageneta-response-four-methods-plan-20261009T090226Z'
sys.path.insert(0,str(P))
from common import sha256,verify_source
from validate_job import validate_job
suite=json.loads((P/'suite.json').read_text());state=json.loads((P/'runtime/queue_state.json').read_text());pins=json.loads((P/'RUN_PINS.json').read_text())
assert state['status']=='completed'
verify_source(suite)
prior={}
for name,digest in pins['files'].items():
    f=P/name;assert sha256(f)==digest,name
    prior[str(f)]=digest
for name in ['RUN_PINS.json','runtime/queue_state.json','GPU_VALIDATION.json','IMPLEMENTATION_TESTS.json']:
    prior[str(P/name)]=sha256(P/name)
accepted=[]
for job in suite['jobs']:
    jid=job['id'];record=state['jobs'][jid];receipt_path=P/'runtime/jobs'/jid/'completed.json';receipt=json.loads(receipt_path.read_text())
    assert record['status']=='completed' and receipt['exit']['returncode']==0
    checked=validate_job(job);assert checked['summary_sha256']==record['summary_sha256']==receipt['summary_sha256']
    prior[str(receipt_path)]=sha256(receipt_path)
    out=Path(job['output_dir']);summary=out/job.get('report_file','final_summary.json');prior[str(summary)]=sha256(summary)
    if job['kind']=='training':
        trace=out/'step_trace.jsonl';prior[str(trace)]=sha256(trace)
        result=json.loads(summary.read_text())
        for d in result['checkpoint_dirs']:
            manifest=Path(d)/'manifest.json';prior[str(manifest)]=sha256(manifest)
            for name,digest in json.loads(manifest.read_text())['files_sha256'].items():
                f=Path(d)/name;assert sha256(f)==digest;prior[str(f)]=digest
    accepted.append(dict(job=jid,**checked))
identical={}
for f in (N/'trainer').rglob('*'):
    if f.is_file() and '__pycache__' not in f.parts:
        rel=f.relative_to(N);digest=sha256(f);assert digest==sha256(P/rel),str(rel);identical[str(rel)]=digest
result=dict(status='passed',prior_run_dir=str(P),accepted_smoke_jobs=8,accepted_pairing_jobs=1,prior_artifact_hashes=prior,identical_trainer_files=identical,validated_jobs=accepted,formal_training_already_run_in_new_campaign=0,meaning='same byte-identical four-method training implementation accepted in prior bounded real GPU smokes; not a new formal performance result')
(N/'gates/PREVIOUS_IMPLEMENTATION_ACCEPTED.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status='passed',smokes=8,pairing=1,trainer_files=len(identical),prior_evidence_files=len(prior))))
