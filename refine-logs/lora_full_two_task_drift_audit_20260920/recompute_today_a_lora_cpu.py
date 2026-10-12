"""Read newly completed A continuation endpoints; no GPU or evaluation forward."""
import csv
import hashlib
import json
import math
from pathlib import Path
import torch

torch.set_num_threads(1)
OUT = Path(__file__).resolve().parent
TRAIN = OUT.parent/'fs_ewc_rq2_20260920/execution/training'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


rows, sources = [], []
for method in ('sgd', 'gam'):
    run = TRAIN/f'A_{method}_to_sgd_seed0'
    manifest = json.loads((run/'run_manifest.json').read_text())
    cfg = manifest['config']
    assert cfg['optimizer_type'] == 'sgd' and cfg['lambda_flat'] == 0
    anchor = Path(cfg['resume_from_checkpoint_dir'])/'task_00.pt'
    endpoint = run/'checkpoints/task_01.pt'
    a = torch.load(anchor, map_location='cpu', weights_only=True, mmap=True)['network_state']
    b = torch.load(endpoint, map_location='cpu', weights_only=True, mmap=True)['network_state']
    keys = [k for k in a if 'linear_a_' in k]
    assert len(keys) == 24
    sq, sq32 = 0., 0.
    for k in keys:
        bk = k.replace('linear_a_', 'linear_b_')
        diff = b[bk].double() @ b[k].double() - a[bk].double() @ a[k].double()
        sq += diff.square().sum().item()
        sq32 += float(((b[bk] @ b[k])-(a[bk] @ a[k])).square().sum())
    changed = [k for k in a if not k.startswith('fc.')
               and not any(t in k for t in ('linear_a', 'linear_b', 'w_As', 'w_Bs'))
               and not torch.equal(a[k], b[k])]
    assert not changed, changed
    old_head_equal = all(torch.equal(a[k], b[k][:a[k].shape[0]]) for k in ('fc.weight','fc.bias'))
    assert old_head_equal
    measured = {x['task']:x for x in map(json.loads,(run/'task0_local_metrics.jsonl').read_text().splitlines())}
    first, last = measured[0], measured[1]
    metrics_path = next((run/'trainer_outputs').rglob('*cl_metrics.json'))
    metrics = json.loads(metrics_path.read_text())
    matrix = metrics['cnn']['matrices']['final']
    logged_sq = metrics['mechanism']['steps']['1']['delta_norm']
    assert math.isclose(sq32, logged_sq, rel_tol=1e-5, abs_tol=1e-8)
    row = dict(anchor_method=method, continuation='SGD+EWC', seed=0,
               a11_fp32=first['test']['seen_accuracy'], a11_inherited_historical=matrix[0][0],
               a21_fp32=last['test']['seen_accuracy'], a22_trainer_rounded=matrix[1][1],
               BWT_fp32=last['test']['seen_accuracy']-first['test']['seen_accuracy'],
               radius_fp64=math.sqrt(sq), radius_fp32=math.sqrt(sq32),
               radius_training_record=last['effective_weight_displacement'],
               squared_drift=logged_sq, squared_abs_error_fp32=abs(logged_sq-sq32),
               old_test_local_loss_initial=first['test']['local_ce'],
               old_test_local_loss_final=last['test']['local_ce'],
               old_test_local_loss_delta=last['test']['local_ce']-first['test']['local_ce'],
               old_test_seen_loss_delta=last['test']['seen_ce']-first['test']['seen_ce'],
               old_head_bitwise_equal=old_head_equal, changed_nonlora_backbone_count=len(changed),
               dedicated_endpoint_Fisher_status='not measured; pipeline blocked')
    rows.append(row)
    sources.append(dict(anchor_method=method, anchor=str(anchor), endpoint=str(endpoint),
                        anchor_sha256=sha(anchor), endpoint_sha256=sha(endpoint),
                        metrics=str(metrics_path), metrics_sha256=sha(metrics_path),
                        manifest=str(run/'run_manifest.json'), manifest_sha256=sha(run/'run_manifest.json'),
                        local_measurements=str(run/'task0_local_metrics.jsonl'),
                        receipt=str(run/'COMPLETE.json'),
                        coordinate='24 effective Q/V blocks, BA1-BA0; fixed backbone checked',
                        measurement_note='Accuracy/loss read from training-time FP32 evaluation; no new forward pass. Endpoint raw Fisher not measured.'))
with (OUT/'today_imageneta_pair_recomputed.csv').open('w', newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'today_imageneta_pair_provenance.json').write_text(json.dumps(sources,indent=2)+'\n')
print(json.dumps(rows,indent=2))
