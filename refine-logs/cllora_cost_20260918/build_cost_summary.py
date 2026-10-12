"""Cost accounting from frozen code/configs and completed receipts; no training."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
RUN = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2')
D, R, BASE = 768, 16, 85798656
MIB, GIB = 2**20, 2**30
SHARED = SPECIFIC = 6 * 2 * 2 * D * R
manifest = []
frozen = json.loads((RUN/'source_frozen.json').read_text())['files']
source_paths = [
    'models_LoRAbasedCL/cllora_aligned.py',
    'backbone/cllora_aligned.py',
    'backbone/cllora_archived_vit.py',
    'src/trainer.py',
    'models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py',
    'models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py',
    'optimer_PerturabtionType/gam.py',
]
for rel in source_paths:
    p = RUN/'code'/rel
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest == frozen[rel], rel
    manifest.append({'path':str(p), 'sha256':digest})

rows=[]
for p in sorted((RUN/'receipts').glob('cllora*.json')):
    receipt=json.loads(p.read_text())
    assert receipt['status']=='completed' and receipt['exit_code']==0
    cfg=json.loads(Path(receipt['config']).read_text())
    assert hashlib.sha256(Path(receipt['config']).read_bytes()).hexdigest()==receipt['config_sha256']
    assert hashlib.sha256(Path(receipt['log']).read_bytes()).hexdigest()==receipt['log_sha256']
    log=Path(receipt['log']).read_text()
    task_rows=re.findall(r'CL-LoRA task=(\d+) sizes=(\[[^\]]+\]) trainable=(\d+)',log)
    assert len(task_rows)==receipt['expected_tasks']
    for task,sizes,trainable in task_rows:
        size=json.loads(sizes)[-1]
        expected=6*2*D*R + SPECIFIC + 6*3 + size*D + 1
        assert int(trainable)==expected, (task,trainable,expected)
    sizes=json.loads(task_rows[-1][1]);T=len(sizes);C=sum(sizes)
    block_weights=T*6*3
    lora=SHARED*2+T*SPECIFIC
    fc=C*D*T+1
    proxy=sizes[-1]*D+1
    incremental=lora+block_weights+fc+proxy
    compact=SHARED+T*SPECIFIC+block_weights+C*D+1
    vanilla_adapter=12*2*2*D*R
    vanilla_deploy_extra=vanilla_adapter+C*(D+1)
    weights=[]
    root=Path(receipt['metric_path']).parents[2]
    for f in root.rglob('*'):
        if f.is_file() and f.suffix in {'.pt','.pth','.pkl','.safetensors','.ckpt','.bin'}:
            weights.append(str(f))
    rows.append(dict(dataset=cfg['dataset'],tasks=T,classes=C,task_sizes=sizes,
        wall_minutes=(receipt['finished_unix']-receipt['started_unix'])/60,
        peak_torch_allocated_gib=receipt['peak_cuda_bytes']/GIB,
        lora_parameters_including_old_shared=lora,
        lora_fp32_mib=lora*4/MIB,
        dense_prototype_parameters=fc,
        dense_prototype_fp32_mib=fc*4/MIB,
        incremental_state_fp32_mib=incremental*4/MIB,
        full_state_parameters=BASE+incremental,
        full_state_fp32_mib=(BASE+incremental)*4/MIB,
        hypothetical_compact_inference_extra_fp32_mib=compact*4/MIB,
        vanilla_compact_inference_extra_fp32_mib=vanilla_deploy_extra*4/MIB,
        final_trainable_parameters=int(task_rows[-1][2]),
        full_encoder_calls_per_inference_batch=T,
        transformer_block_calls_per_inference_batch=12*T,
        hypothetical_shared_prefix_block_calls=6+6*T,
        prototype_current_dataset_passes_at_final_task=T,
        checkpoint_files=weights,receipt=str(p),log=receipt['log']))

data={'checked_at':datetime.now(ZoneInfo('Europe/Madrid')).isoformat(),
      'scope':'Frozen implementation accounting and existing receipt measurements; no new GPU benchmark or training.',
      'bytes_convention':'FP32 tensor payload only, no serialization metadata, gradients or activations.',
      'baseline_scope':'Vanilla compact inference extra excludes the common ViT and any optional duplicate frozen backbone.',
      'rows':rows,'source_manifest':manifest,
      'gam_ewc_current_code':{'task_gradient_full_encoder_passes_per_batch':5,
          'explanation':'One clean gradient cache plus four GAM closure evaluations; Fisher penalty uses parameter algebra.',
          'fisher_delta_diagonal_fp32_mib':24*D*D*4/MIB,
          'task_reference_fp32_mib':24*D*D*4/MIB,
          'state_scope':'Training-only accumulated Fisher and task-start effective delta-W reference; not a task-growing inference ensemble.'}}
(OUT/'cost_summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
with (OUT/'cost_summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
    for row in rows:writer.writerow({k:json.dumps(v)if isinstance(v,list)else v for k,v in row.items()})
print(json.dumps({'datasets':len(rows),'all_trainable_counts_match_logs':True,'checkpoint_count':sum(len(r['checkpoint_files'])for r in rows),'rows':[{k:r[k]for k in ['dataset','wall_minutes','peak_torch_allocated_gib','incremental_state_fp32_mib','full_state_fp32_mib']}for r in rows]},indent=2))
