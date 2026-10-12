"""CPU-only reconstruction of the six saved ImageNet-R ratioFS LoRA pairs."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

torch.set_num_threads(1)
# Historical endpoint sidecars contain standard Linear modules, anchors tensors.
torch.serialization.add_safe_globals([torch.nn.Linear, set])
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def factors(path):
    return [(x.weight if isinstance(x, torch.nn.Linear) else x).detach()
            for x in load(path)]


rows, provenance, layer_rows = [], [], []
for run in json.loads((OUT / 'new_r_ratio_pair_assets.json').read_text()):
    anchor, end = run['anchor_sidecars'], run['endpoint_sidecars']
    a0, b0 = factors(anchor['lora_w_a']), factors(anchor['lora_w_b'])
    a1, b1 = factors(end['lora_w_a']), factors(end['lora_w_b'])
    assert run['anchor_metadata'] == run['endpoint_metadata']
    assert len(a0) == len(b0) == len(a1) == len(b1) == 24
    assert {m['channel'] for m in run['anchor_metadata']} == {'q', 'v'}
    hashes = {}
    for group in (anchor, end):
        for key, path in group.items():
            if key != 'fisher_state':
                hashes[path] = digest(path)
    verified_parent = []
    for line in (Path(run['anchor_dir']) / 'parent_artifacts.sha256').read_text().splitlines():
        expected, path = line.split(maxsplit=1)
        if path in hashes:
            assert hashes[path] == expected, f'Anchor hash changed: {path}'
            verified_parent.append(path)
    assert len(verified_parent) == 3
    sq32, sq64, anchor_sq, endpoint_sq = 0., 0., 0., 0.
    for i, (x0, y0, x1, y1) in enumerate(zip(a0, b0, a1, b1)):
        z0, z1 = y0 @ x0, y1 @ x1
        d32 = float((z1 - z0).square().sum())
        dz = y1.double() @ x1.double() - y0.double() @ x0.double()
        d64 = float(dz.square().sum())
        sq32 += d32
        sq64 += d64
        anchor_sq += float((y0.double() @ x0.double()).square().sum())
        endpoint_sq += float((y1.double() @ x1.double()).square().sum())
        layer_rows.append(dict(setting=run['setting'], seed=run['seed'],
                               **run['anchor_metadata'][i], displacement_squared_fp64=d64))
    h0, h1 = load(anchor['fc_state']), load(end['fc_state'])
    old_head_sq = sum(float((h1[k][:v.shape[0]].double() - v.double()).square().sum())
                      for k, v in h0.items())
    new_head_sq = sum(float(h1[k][v.shape[0]:].double().square().sum())
                      for k, v in h0.items())
    metrics = json.loads(Path(run['metrics']).read_text())
    mat = metrics['cnn']['matrices']['final']
    logged_sq = metrics['mechanism']['steps']['1']['delta_norm']
    assert math.isclose(sq32, logged_sq, rel_tol=1e-5, abs_tol=1e-8)
    row = dict(setting=run['setting'], seed=run['seed'],
               a11=mat[0][0], a21=mat[1][0], a22=mat[1][1], bwt=mat[1][0]-mat[0][0],
               logged_delta_norm_squared=logged_sq,
               recomputed_squared_fp32=sq32, recomputed_squared_fp64=sq64,
               recomputed_radius_fp32=math.sqrt(sq32), recomputed_radius_fp64=math.sqrt(sq64),
               squared_abs_error_fp32=abs(sq32-logged_sq),
               anchor_BA_norm=math.sqrt(anchor_sq), endpoint_BA_norm=math.sqrt(endpoint_sq),
               old_classifier_drift=math.sqrt(old_head_sq),
               new_classifier_absolute_norm_not_training_drift=math.sqrt(new_head_sq),
               parent_hashes_verified=len(verified_parent), effective_blocks=24,
               effective_coordinates=24*768*768)
    rows.append(row)
    provenance.append(dict(setting=run['setting'], seed=run['seed'],
                           files_sha256=hashes, parent_verified_paths=verified_parent,
                           metrics=run['metrics'], metrics_sha256=digest(run['metrics']),
                           log=run['log'], log_sha256=digest(run['log']),
                           caveat='Endpoint sidecars preserve factors/head, not full backbone; frozen base follows source/config. No GPU/forward/training.'))

summaries = []
for setting in ('control', 'factored_s1'):
    rr = [r for r in rows if r['setting'] == setting]
    s = dict(setting=setting, seeds='3,4,5', n=3)
    for key in ('a11', 'a21', 'a22', 'bwt', 'recomputed_radius_fp64', 'anchor_BA_norm', 'endpoint_BA_norm'):
        vals = [r[key] for r in rr]
        s[key+'_mean'], s[key+'_sd'] = statistics.mean(vals), statistics.stdev(vals)
    summaries.append(s)
for filename, records in [('new_r_lora_drift_recomputed.csv', rows),
                           ('new_r_lora_drift_summary.csv', summaries),
                           ('new_r_lora_drift_per_layer.csv', layer_rows)]:
    with (OUT / filename).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
(OUT / 'new_r_lora_drift_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
print(json.dumps(dict(rows=len(rows), summary=summaries,
                     max_squared_abs_error_fp32=max(r['squared_abs_error_fp32'] for r in rows),
                     max_old_classifier_drift=max(r['old_classifier_drift'] for r in rows)), indent=2))
