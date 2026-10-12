"""Read archived evidence and recompute metrics; never invokes training or CUDA."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv, hashlib, json, math, statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE = ROOT / 'refine-logs'
OLD = 'g012-two-stage-execution-20261002T122014Z'
NEW = 'reverse-probe-reflected-update-20261003T161621Z'
OUTER = 'g2-central-location-gpu01-20261003T105925Z'
SPECS = [
    (OLD, 'S1-01_CE_natural_s2', 'local_historical_native', 'g0', 4),
    (OLD, 'S1-02_Splus_natural_s2', 'local_historical_native', 'g_plus = g0+s+e', 4),
    (OLD, 'S1-03_Sminus_natural_s2', 'local_historical_native', 'g_minus = g0-s+e', 4),
    (NEW, 'RB01_Backward_reflected_natural_s2', 'local_reflected_math', 'g0+(g0-g_minus) = g0+s-e', 2),
    (OUTER, 'G2-CD01_O_P_natural_s2', 'outer_location_math', 'g(theta+.25*N(s+e))', 4),
    (OUTER, 'G2-CD02_O_C_natural_s2', 'outer_location_math', 'g(theta+.25*N(s-e))', 4),
    (OUTER, 'G2-CD03_O_CD_natural_s2', 'outer_location_math', 'g(theta+.25*N(s))', 4),
]
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(matrix):
    n=len(matrix)
    assert n==10 and all(len(row)>=i+1 for i,row in enumerate(matrix))
    assert all(math.isfinite(float(v)) for i,row in enumerate(matrix) for v in row[:i+1])
    return dict(AAA=sum(sum(row[:i+1])/(i+1) for i,row in enumerate(matrix))/n,
        FAA=sum(matrix[-1][:n])/n, LA=sum(matrix[i][i] for i in range(n))/n,
        BWT=sum(matrix[-1][j]-matrix[j][j] for j in range(n-1))/(n-1),
        Forget=sum(max(matrix[i][j] for i in range(j,n))-matrix[-1][j] for j in range(n-1))/(n-1))

def main():
    rows=[]
    for folder,ident,cohort,formula,ce in SPECS:
        run=BASE/folder; output=run/'outputs'/ident; runtime=run/'runtime/jobs'/ident
        matrix_file=output/'task_matrix.json'; summary_file=output/'final_summary.json'; trace_file=output/'step_trace.jsonl'
        matrix=read(matrix_file); summary=read(summary_file); completed=read(runtime/'completed.json'); exit_record=read(runtime/'exit.json')
        assert exit_record['returncode']==0 and summary['status']=='completed' and summary['completed_tasks']==10
        assert completed['summary_sha256']==sha(summary_file) and completed['matrix_sha256']==sha(matrix_file)
        assert completed['trace_sha256']==sha(trace_file) and completed['updates']==2550
        result=metrics(matrix['matrix'])
        for name,key in [('AAA','AAA_task_macro'),('FAA','FAA_task_macro'),('LA','LA'),('BWT','BWT'),('Forget','forgetting')]:
            assert abs(result[name]-matrix[key])<1e-9
        suite=read(run/'suite.json'); job=next(j for j in suite['jobs'] if j['id']==ident)
        cfg=read(Path(job['config'])); assert sha(Path(job['config']))==job['config_sha256']
        assert cfg['gam_grad_rho']==.05 and cfg['lora_rank']==10 and cfg['seed']==[2]
        assert cfg['ewc_lambda']==0 and cfg['ewc_skip_task_fisher'] and not cfg['in_task_weight_merge'] and not cfg['task_endpoint_weight_merge']
        row=dict(run_id=ident,cohort=cohort,formula=formula,seed=2,rho=.05,rank=10,
            attention_backend=summary['attention_backend'],training_ce_per_update=ce,status='validated_completed',
            expected_tasks=10,completed_tasks=10,updates=2550,elapsed_seconds=summary['elapsed_seconds'],**result,
            FAA_sample_weighted=matrix['FAA_sample_weighted'],matrix_path=str(matrix_file),matrix_sha256=sha(matrix_file),
            completion_receipt=str(runtime/'completed.json'),config_path=job['config'],config_sha256=sha(Path(job['config'])))
        rows.append(row)
    diag_path=BASE/OUTER/'outputs/G2-CD_location_diagnostics/g2_location_report.json'; diag=read(diag_path)
    assert diag['status']=='passed' and diag['ordinary_gradient_calls']==108 and diag['sgd_updates']==0 and diag['entry_state_restored']
    batches=[b for state in diag['state_results'] for b in state['batches']]; assert len(batches)==18 and all(b['status']=='passed' for b in batches)
    aggregations={
       's_over_g0':[b['norms']['s']/b['norms']['g0'] for b in batches],
       'e_over_g0':[b['norms']['e']/b['norms']['g0'] for b in batches],
       'e_over_s':[b['e_over_s_plus_eps'] for b in batches],
       'cos_qP_qC':[b['q_cosines']['qP_qC'] for b in batches],
    }
    diag_out=dict(status='validated_existing_only',source=str(diag_path),sha256=sha(diag_path),state_batches=18,
        source_training_trajectories=3,task_checkpoints=[0,4,9],ordinary_ce=108,updates=0,
        statistics={k:dict(mean=statistics.mean(v),min=min(v),max=max(v)) for k,v in aggregations.items()},
        scope='18 equally weighted within-state comparisons; different trajectories are not paired gradient states')
    rb=rows[3]
    background_deltas={r['run_id']:{k:rb[k]-r[k] for k in ['AAA','FAA','LA','Forget']} for r in rows[:3]}
    original_local=BASE/'local-cd-method-20261003T082009Z/suite.json'; oldlocal=read(original_local); assert oldlocal['jobs']==[]
    at=datetime.now(ZoneInfo('Europe/Paris')); stamp=at.strftime('%Y%m%d_%H%M%S')
    ledger=dict(at=at.isoformat(),analysis_only=True,new_training_launched=False,queues_modified=False,
        selected_existing_formal_cells=dict(validated=7,expected=7,failed=0,incomplete=0),
        cohorts={'local_historical_native':3,'local_reflected_math':1,'outer_location_math':3},
        proposed_local_panel=dict(formulas_with_any_historical_complete_result=4,formulas_without_complete_result=3,
            proposed_formulas=7,unified_three_probe_panel_runs=0,queued_new_jobs=0,conditional_B_reuse='pending identity bridge; not automatically accepted'),
        rows=rows,existing_diagnostics=diag_out,
        cross_protocol_background_deltas_RB_minus_old=background_deltas,
        cross_protocol_deltas_are_causal=False,old_local_cd_suite=str(original_local),old_local_cd_jobs=0,
        single_seed_sample_SD=None,forgetting_convention='max includes final stage; old task denominator9',
        metric_precision='aggregates recomputed from archived task accuracy matrices rounded to two decimals')
    for name,value in [('RESULTS_AUDIT',ledger),('EXISTING_DIAGNOSTICS',diag_out)]:
        text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'; (HERE/f'{name}_{stamp}.json').write_text(text); (HERE/f'{name}.json').write_text(text)
    for filename in [f'EXISTING_RESULTS_{stamp}.csv','EXISTING_RESULTS.csv']:
        with (HERE/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(validated=7,expected=7,diagnostic_batches=18,new_training_launched=False,rows=[{k:r[k] for k in ['run_id','AAA','FAA','LA','Forget']} for r in rows]),ensure_ascii=False))
if __name__=='__main__': main()
