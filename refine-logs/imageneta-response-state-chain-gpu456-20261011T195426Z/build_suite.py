"""Deterministic manifest/config builder: no job is launched here."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
PYTHON='/home-local/liying/.conda/envs/Pilot_new_local/bin/python'
OLD=H.parent/'imageneta-g2-minimal-fork-gpu456-20261009T223549Z'
UP=H.parent/'imageneta-upward-eight-fork-20261010T123703Z'
SOURCE='/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source'
OUT=Path('/data/115-2/users/liying/cache/tmp')/H.name/'outputs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
base=json.loads((OLD/'configs/formal_parent_sgd_s2.json').read_text())
jobs=[]
def train(jid,seed,role,branch,deps,smoke=False):
 old=OLD if seed==2 else UP
 parent=(old/'outputs'/f'formal_parent_sgd_s{seed}').resolve()
 ps=json.loads((parent/'final_summary.json').read_text())
 cfg=dict(base)
 cfg.update(seed=[seed],prefix=jid,outputs_root=str(OUT/jid/'native'),study_dev_split_path=str(H/'dev_split.json'),
    chain_role=role,chain_branch=branch,chain_smoke=smoke,chain_switch_after_epoch=1 if smoke else 25,
    study_parent_output_dir=str(parent),study_parent_checkpoint=ps['checkpoint_dirs'][0],
    study_parent_receipt=str(old/'runtime/jobs'/f'formal_parent_sgd_s{seed}'/'completed.json'),
    chain_common_start_snapshot=str(OUT/f'prepare_start_s{seed}'/'common_start.pt'),
    chain_training_ce_bin_width=.05,chain_training_ce_bin_max=5.)
 c=H/'configs'/f'{jid}.json';write(c,cfg)
 cmd=[PYTHON,str(H/'trainer/response_chain_train.py'),'--source',SOURCE,'--config',str(c),'--output-dir',str(OUT/jid)]
 if smoke:cmd+=['--max-epochs','2','--max-batches','2']
 jobs.append(dict(id=jid,kind='training' if role=='branch' else 'prepare',formal=role=='branch' and not smoke,smoke=smoke,
  seed=seed,chain_role=role,chain_branch=branch,config=str(c),config_sha256=sha(c),output_dir=str(OUT/jid),
  report_file='final_summary.json',depends_on=deps,command=cmd,expected_tasks=2,
  expected_epochs=2 if smoke else 50 if role=='branch' else 0,
  expected_steps=4 if smoke else 400 if role=='branch' else 0,
  expected_gradient_reads=(6 if branch=='S' else 8) if smoke else (600 if branch=='S' else 800) if role=='branch' else 0))
for seed in [2,0,1]:train(f'prepare_start_s{seed}',seed,'start','S',[])
for branch in ['S','R']:train(f'smoke_{branch}_s2',2,'branch',branch,['prepare_start_s2'],True)
pair='smoke_pairing_s2'
jobs.append(dict(id=pair,kind='pairing',formal=False,smoke=True,seed=2,output_dir=str(OUT/pair),report_file='pairing_report.json',
 depends_on=['smoke_S_s2','smoke_R_s2'],command=[PYTHON,str(H/'smoke_pairing.py'),'--output-dir',str(OUT/pair)]))
jid='smoke_diagnostics_s2'
jobs.append(dict(id=jid,kind='diagnostic',formal=False,smoke=True,seed=2,output_dir=str(OUT/jid),report_file='completion_receipt.json',
 depends_on=[pair],command=[PYTHON,str(H/'diagnostics/offline_diagnostics.py'),'--run-root',str(H),'--checkpoint',str(OUT/'smoke_R_s2/checkpoints/epoch_001.pt'),
 '--seed','2','--output-dir',str(OUT/jid),'--smoke']))
gates=[pair,jid]
for seed in [2,0,1]:
 for branch in ['S','R']:train(f'formal_{branch}_s{seed}',seed,'branch',branch,[f'prepare_start_s{seed}']+gates)
for seed in [2,0,1]:
 jid=f'offline_s{seed}';deps=[f'formal_{b}_s{seed}' for b in ['S','R']]+([] if seed==2 else ['offline_s2'])
 cmd=[PYTHON,str(H/'diagnostics/offline_diagnostics.py'),'--run-root',str(H),'--seed',str(seed),'--output-dir',str(OUT/jid)]
 for b in ['S','R']:
  for epoch in [1,10,25]:cmd+=['--checkpoint',str(OUT/f'formal_{b}_s{seed}'/'checkpoints'/f'epoch_{epoch:03d}.pt')]
 jobs.append(dict(id=jid,kind='diagnostic',formal=False,smoke=False,seed=seed,output_dir=str(OUT/jid),report_file='completion_receipt.json',depends_on=deps,command=cmd))
suite=json.loads((OLD/'suite.json').read_text())
suite.update(run_id=H.name,run_dir=str(H),purpose='paired_SAM_and_SF_g2_then_shared_SGD_three_layer_evidence_chain',
 protocol='response_state_chain_50ep_v1',allowed_gpus=[4,5,6],max_parallel=3,poll_seconds=10,outputs_root=str(OUT),
 source_manifest=str(H/'source_manifest.json'),source_manifest_sha256=sha(H/'source_manifest.json'),jobs=jobs,
 implementation_acceptance_file=str(H/'gates/CPU_IMPLEMENTATION_ACCEPTED.json'),
 independent_implementation_review_file=str(H/'gates/INDEPENDENT_IMPLEMENTATION_REVIEW.json'),
 fresh_doc_check=str(H/'gates/FRESH_DOC_CHECK.json'),parent_reuse_audit_file=str(H/'docs/PARENT_REUSE_AUDIT.json'))
suite['environment'].update(TMPDIR='/tmp/olora_chain_20261011T195426Z',TMP='/tmp/olora_chain_20261011T195426Z',TEMP='/tmp/olora_chain_20261011T195426Z')
write(H/'suite.json',suite)
tracker=['# 实验追踪','', '| Run | 用途 | seed | 分支/阶段 | 状态 |', '|---|---|---:|---|---|']
for j in jobs:tracker.append(f"| {j['id']} | {j['kind']} | {j['seed']} | {j.get('chain_branch','诊断')} | 已注册，等待实现及短测验收 |")
tracker+=['','队列：runtime/queue_state.json；正式结果只填验收完成的6条分支，离线诊断独立记录。']
(H/'EXPERIMENT_TRACKER_20261011T195426Z.md').write_text('\n'.join(tracker)+'\n')
(H/'EXPERIMENT_TRACKER.md').write_text('\n'.join(tracker)+'\n')
print(json.dumps({'jobs':len(jobs),'formal_new_training':sum(j['formal'] for j in jobs),'reused_sgd50_parents':3,'gpus':suite['allowed_gpus']}))
