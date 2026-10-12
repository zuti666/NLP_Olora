"""Create an isolated four-condition ordinary Stage2 queue; never change old run."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z')
NEW = ROOT/'ordinary_stage2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    NEW.mkdir(exist_ok=False)
    (NEW/'configs').mkdir()
    shutil.copytree(ORIGINAL/'trainer', NEW/'trainer',ignore=shutil.ignore_patterns('__pycache__'))
    for filename in ('common.py','job_worker.py','validate_job.py','collect_results.py'):
        shutil.copy2(ORIGINAL/filename,NEW/filename)
    shutil.copy2(ROOT/'EXPERIMENT_PLAN.md',NEW/'EXPERIMENT_PLAN.md')
    suite = json.loads((ORIGINAL/'suite.json').read_text())
    ids = {'S2-B01_Sec_math_natural_s2','S2-B02_Sec_math_matched_s2',
           'S2-05_Residual_natural_s2','S2-06_Residual_matched_s2'}
    selected = [j for j in suite['jobs'] if j['id'] in ids or j['id'].removeprefix('smoke_') in ids]
    assert len(selected)==8
    smoke_ids = [j['id'] for j in selected if j['smoke']]
    for job in selected:
        original_config = json.loads(Path(job['config']).read_text())
        config = dict(original_config)
        output = NEW/'outputs'/job['id']
        config['outputs_root'] = str(output/'native')
        path = NEW/'configs'/Path(job['config']).name
        path.write_text(json.dumps(config,indent=2)+'\n')
        assert {k for k in config if config[k]!=original_config[k]}=={'outputs_root'}
        command = list(job['command'])
        command[1] = str(NEW/'trainer/four_probe_train.py')
        command[command.index('--config')+1] = str(path)
        command[command.index('--output-dir')+1] = str(output)
        job.update(command=command,config=str(path),config_sha256=sha(path),output_dir=str(output),
                   depends_on=[] if job['smoke'] else smoke_ids,phase='ordinary_stage2_smoke' if job['smoke'] else 'ordinary_stage2')
    env = dict(suite['environment'])
    for key in ('TMPDIR','TMP','TEMP'):
        env[key] = '/tmp/g012-stage2-ordinary-20261003'
    suite.update(run_id='g012-stage2-ordinary-20261003',run_dir=str(NEW),jobs=selected,environment=env,
        failure_policy='halt launches; active ordinary jobs finish; no HVP jobs present',
        prerequisite_receipt=str(ROOT/'MATH_GRADIENT_ACCEPTANCE.json'),
        parent_run=str(ORIGINAL),scope='four ordinary-response conditions, not full HVP Stage2')
    (NEW/'suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    launcher = (ORIGINAL/'run_queue.py').read_text()
    launcher = launcher.replace('from validate_job import validate_job','from validate_job import validate_job\nfrom gradient_prerequisite import verify_gradient_prerequisite')
    launcher = launcher.replace('def verify_pins(suite):\n','def verify_pins(suite):\n    verify_gradient_prerequisite(suite)\n')
    (NEW/'run_queue.py').write_text(launcher)
    pins = {str(p.relative_to(NEW)):sha(p) for p in NEW.rglob('*') if p.is_file()}
    # The prerequisite validator is added by the root before sealing RUN_PINS.
    (NEW/'PROVISIONAL_PINS.json').write_text(json.dumps({'files':pins},indent=2)+'\n')
    for p in NEW.rglob('*.py'):
        ast.parse(p.read_text())
    print(json.dumps({'created':str(NEW),'formal':4,'smoke':4,'HVP_jobs':0,'not_launched':True}))


if __name__=='__main__':
    main()
