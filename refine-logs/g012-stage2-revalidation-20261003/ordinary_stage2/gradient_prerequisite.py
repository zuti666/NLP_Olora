"""Require verified full math repeatability before any ordinary Stage2 job."""
import json
from pathlib import Path
from common import sha256

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {'S2-B01_Sec_math_natural_s2','S2-B02_Sec_math_matched_s2',
           'S2-05_Residual_natural_s2','S2-06_Residual_matched_s2'}


def validate_shards():
    state = json.loads((ROOT/'gradient_runtime/queue_state.json').read_text())
    manifest = json.loads((ROOT/'gradient_manifest.json').read_text())
    if state['manifest_sha256']!=sha256(ROOT/'gradient_manifest.json'):
        raise RuntimeError('Gradient diagnostic manifest receipt changed')
    for name,digest in manifest['pins'].items():
        if sha256(name)!=digest:
            raise RuntimeError('Numerical prerequisite code/old artifact changed: '+name)
    proof, seen = {}, set()
    for task in (0,1,4,9):
        jid = 'gradient_task%d'%task
        row = state['jobs'][jid]
        if row['status']!='completed':
            raise RuntimeError('Full math gradient prerequisite is incomplete: '+jid)
        path = ROOT/'verification'/jid/'gradient_revalidation_report.json'
        report = json.loads(path.read_text())
        exit_path = ROOT/'gradient_runtime/jobs'/jid/'exit.json'
        exit_record = json.loads(exit_path.read_text())
        if exit_record['job_id']!=jid or exit_record['returncode']!=0 or row['report_sha256']!=sha256(path):
            raise RuntimeError('Gradient exit/report receipt mismatch')
        if report['status']!='passed' or not report['shard_passed'] or report['backend']!='math' or report['tasks']!=[task]:
            raise RuntimeError('A gradient shard failed or has the wrong scientific condition')
        if report['seed']!=2 or report['rho']!=.05 or report['radius']!=.25 or report['case_count']!=6:
            raise RuntimeError('Gradient shard changed protocol/budget')
        if any(report[k]!=0 for k in ('optimizer_updates','fisher_calls','weight_merge_calls','hvp_calls')):
            raise RuntimeError('Gradient prerequisite performed forbidden work')
        if not all(report[k] for k in ('registered_batches_exact','source_unchanged','trainer_unchanged',
                                      'reference_assets_unchanged','native_origins_restored')):
            raise RuntimeError('Gradient prerequisite integrity failed')
        registry_path = path.parent/'batch_registry.json'
        if sha256(registry_path)!=report['batch_registry_sha256']:
            raise RuntimeError('Registered batch receipt changed')
        registry = json.loads(registry_path.read_text())
        for registered in registry['tasks'][str(task)]['batches']:
            if sha256(registered['archive_path'])!=registered['archive_sha256']:
                raise RuntimeError('Registered input archive changed')
        for case in report['cases']:
            coordinate = (case['task'],case['point'],case['batch'])
            if coordinate in seen or case['task']!=task or case['status']!='passed' or not case['all_comparisons_bitwise_exact'] or not case['hashes_restored']:
                raise RuntimeError('A registered same-state case failed')
            seen.add(coordinate)
            if sha256(case['case_report'])!=case['case_report_sha256']:
                raise RuntimeError('Case evidence hash changed')
            detail = json.loads(Path(case['case_report']).read_text())
            if detail['backend']!='math' or detail['status']!='passed' or not detail['parameter_identities_preserved']:
                raise RuntimeError('Case state/parameter identity check failed')
            if not all(detail['final_restoration'].values()) or detail['ordinary_gradient_calls']!=20:
                raise RuntimeError('Case restoration/probe accounting failed')
            if [r['branch'] for r in detail['engine_results']]!=['P','P','D','C','R']:
                raise RuntimeError('Missing repetition/four-cell comparisons')
            for engine in detail['engine_results']:
                if engine['status']!='passed' or not all(engine['restoration'].values()):
                    raise RuntimeError('Probe state transaction failed')
                for group in ('shared_or_repeated_gradients','shared_or_repeated_points','repeat_residuals_and_direction'):
                    if not all(v['bitwise_exact'] for v in engine[group].values()):
                        raise RuntimeError('Actual gradient/direction comparisons did not pass')
        proof[str(task)] = {'report':str(path),'report_sha256':sha256(path),
                            'exit':str(exit_path),'exit_sha256':sha256(exit_path),'case_count':6}
    if seen!={(t,p,b) for t in (0,1,4,9) for p in ('entry','epoch1','end') for b in (0,1)}:
        raise RuntimeError('The prerequisite does not cover all 24 registered cases')
    return proof


def verify_gradient_prerequisite(suite):
    for job in suite['jobs']:
        if job['id'].removeprefix('smoke_') not in ALLOWED or job['direction'] not in ('A','residual_only') or job['backend']!='math':
            raise RuntimeError('This queue can only launch the four ordinary math controls')
    receipt_path = ROOT/'MATH_GRADIENT_ACCEPTANCE.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['status']!='passed' or receipt['case_count']!=24 or receipt['HVP_training_allowed'] is not False:
        raise RuntimeError('Missing complete ordinary-only gradient acceptance')
    actual = validate_shards()
    if actual!=receipt['shards']:
        raise RuntimeError('Full gradient acceptance receipt no longer matches actual evidence')
    return receipt
