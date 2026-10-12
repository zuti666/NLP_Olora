"""Fail-closed genuine shared-parent/head/input pairing gate after both smoke commits."""
import argparse,json
from pathlib import Path
from common import atomic_json,sha256,load_suite
H=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True);a=ap.parse_args()
 suite=load_suite();jobs={j['id']:j for j in suite['jobs']};summaries=[];traces=[];hashes={};checks={}
 for b in ['S','R']:
  j=jobs[f'smoke_{b}_s2'];out=Path(j['output_dir'])
  s=json.loads((out/'final_summary.json').read_text());rows=[json.loads(x) for x in (out/'step_trace.jsonl').read_text().splitlines()]
  checks[b+'_completed_budget']=s['status']=='completed' and s['completed_steps']==4 and s['completed_epochs']==2 and s['seed']==2 and s['smoke']
  checks[b+'_switch_no_reset']=len(s['switch_receipts'])==1 and all(s['switch_receipts'][0][k] for k in ['optimizer_same','scheduler_same','parameters_exact','parameter_references_exact','rng_exact'])
  checks[b+'_native_methods']=[r['active_method'] for r in rows]==(['sam']*2+['sgd']*2 if b=='S' else ['sf_g2']*2+['sgd']*2)
  for f in ['final_summary.json','step_trace.jsonl','effective_config.json']:hashes[str(out/f)]=sha256(out/f)
  summaries.append(s);traces.append(rows)
 checks['shared_expanded_start']=summaries[0]['common_start_snapshot']==summaries[1]['common_start_snapshot'] and summaries[0]['common_start_sha256']==summaries[1]['common_start_sha256']
 for key in ['common_start_model_sha256','common_start_rng_sha256','new_fc_rows_sha256']:
  vals=[s['task_identities'][0].get(key) for s in summaries]
  checks[key]=vals[0] is not None and vals[0]==vals[1]
 checks['all_realized_augmented_inputs_paired']=len(traces[0])==len(traces[1])==4 and all(x['identity']==y['identity'] for x,y in zip(*traces))
 checks['first_base_gradient_equal']=traces[0][0]['g0_sha256']==traces[1][0]['g0_sha256']
 checks['all_summary_checks']=all(all(s['checks'].values()) for s in summaries)
 path=Path(summaries[0]['common_start_snapshot']);hashes[str(path)]=sha256(path)
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
 report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,evidence_hashes=hashes,smoke_count=2,actual_paired_updates=4)
 atomic_json(out/'pairing_report.json',report)
 if not all(checks.values()):raise AssertionError('Shared-start/switch/data pairing failed: '+repr(checks))
 print(json.dumps({'status':'passed','checks':len(checks)}))
if __name__=='__main__':main()
