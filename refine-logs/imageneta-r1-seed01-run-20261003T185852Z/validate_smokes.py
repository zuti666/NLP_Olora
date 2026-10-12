"""Both protocols and both new seeds must pass before any formal training."""
from pathlib import Path
import subprocess
from common import read_json,sha256,atomic_json,now
H=Path(__file__).resolve().parent

def validate_smokes(suite):
    receipt=H/'runtime/SMOKES_PASSED.json'
    jobs=[j for j in suite['jobs'] if j['smoke']]
    assert len(jobs)==60
    hashes={}
    for j in jobs:
        p=H/'runtime/jobs'/j['id']/'completed.json'
        r=read_json(p)
        if not r.get('passed') or r['job_id']!=j['id']:raise RuntimeError('Unaccepted smoke '+j['id'])
        if r['summary_sha256']!=sha256(Path(j['output_dir'])/'final_summary.json'):
            raise RuntimeError('Smoke evidence changed '+j['id'])
        hashes[j['id']]=sha256(p)
    if receipt.exists():
        old=read_json(receipt)
        if old['completion_hashes']!=hashes:raise RuntimeError('Smoke acceptance evidence changed')
        return old
    reports={}
    for family in ['native','probe']:
        out=H/'runtime'/('SMOKES_'+family+'.json')
        process=subprocess.run([suite['python'],str(H/family/'validate.py'),'--suite',str(H/'suite.json'),
                        '--smokes','--output',str(out)],capture_output=True,text=True,cwd=H)
        out.with_suffix('.console.log').write_text(process.stdout+process.stderr)
        if process.returncode:raise RuntimeError('Smoke gate failed '+family+': '+process.stderr[-3000:])
        reports[family]=read_json(out)
        if not reports[family].get('passed'):raise RuntimeError('Failed '+family+' smoke pairing')
    result=dict(passed=True,at=now(),smokes=60,formal_release=True,completion_hashes=hashes,reports=reports)
    atomic_json(receipt,result)
    return result
