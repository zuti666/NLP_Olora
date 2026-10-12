"""Family-specific result validation in isolated Python processes."""
import json
from pathlib import Path
import subprocess
from common import read_json, sha256
H=Path(__file__).resolve().parent

def validate_job(job):
    suite=read_json(H/'suite.json')
    folder='native' if job['condition'].startswith('N') else 'probe'
    report=H/'runtime/jobs'/job['id']/'validation.json'
    report.parent.mkdir(parents=True,exist_ok=True)
    process=subprocess.run([suite['python'],str(H/folder/'validate.py'),'--suite',str(H/'suite.json'),
                    '--job',job['id'],'--output',str(report)],capture_output=True,text=True,cwd=H)
    report.with_suffix('.console.log').write_text(process.stdout+process.stderr)
    if process.returncode:raise RuntimeError('Validation failed '+job['id']+': '+process.stderr[-2000:])
    result=read_json(report)
    if not result.get('passed'):raise RuntimeError('Family validation failed: '+job['id']+' '+str(result))
    summary=Path(job['output_dir'])/'final_summary.json'
    if result.get('summary_sha256')!=sha256(summary):raise RuntimeError('Summary receipt hash mismatch')
    return result
