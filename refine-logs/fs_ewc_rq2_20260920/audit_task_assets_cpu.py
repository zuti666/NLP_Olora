"""Read existing CopyFlatCV assets on CPU; writes only this audit directory."""
import csv
import hashlib
import json
from pathlib import Path

import torch

torch.set_num_threads(1)
ROOT = Path('/data/140-1/users/liying/CopyFlatCV')
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'outputs_logs/sgd_gam_s1c'
T0 = BASE / 'stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50'
SEQ = BASE / 'stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50'
LOGS = ROOT / 'logs/sgd_gam_s1c/staged/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/shell'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


rows, assets, configs = [], [], {}
for method in ('sgd', 'gam'):
    metrics_path = next((SEQ / 'trainer_outputs').rglob(f's1c_{method}_from_task0_T10_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json'))
    log_path = LOGS / f'{method}_T10_seed0.log'
    data = json.loads(metrics_path.read_text())
    steps = data['task0_trajectory']['steps']
    matrix = data['cnn']['matrices']['final']
    origin_path = T0 / method / 'seed0/checkpoints/task_00.pt'
    origin = load(origin_path)['network_state']
    factor_keys = [k for k in origin if 'linear_a_' in k]
    origin_delta = {k: origin[k.replace('linear_a_', 'linear_b_')] @ origin[k] for k in factor_keys}
    config = {}
    for line in log_path.read_text().splitlines()[:96]:
        if '[trainer.py] => ' in line:
            s = line.split('[trainer.py] => ', 1)[1]
            if ': ' in s:
                k, v = s.split(': ', 1)
                config[k] = v
    configs[method] = config
    for t in range(10):
        checkpoint = origin_path if t == 0 else SEQ / method / f'T10/seed0/checkpoints/task_{t:02d}.pt'
        ckpt = load(checkpoint)
        state = ckpt['network_state']
        old_head_equal = all(torch.equal(state[k][:origin[k].shape[0]], origin[k]) for k in ('fc.weight', 'fc.bias'))
        changed_nonlora = [k for k in origin if not k.startswith('fc.') and not any(x in k for x in ('linear_a', 'linear_b', 'w_As', 'w_Bs')) and not torch.equal(origin[k], state[k])]
        displacement_sq = sum(float(((state[k.replace('linear_a_', 'linear_b_')] @ state[k]) - origin_delta[k]).square().sum()) for k in factor_keys)
        drift = displacement_sq ** 0.5
        m = steps[str(t)]
        row = {'method': method, 'seed': 0, 'task': t, 'continuation_optimizer': method, 'metrics_json': str(metrics_path), 'metrics_sha256': sha(metrics_path), 'checkpoint': str(checkpoint), 'delta_norm_json': m['delta_norm'], 'delta_norm_cpu_recomputed': drift, 'delta_norm_abs_error': abs(drift - m['delta_norm']), 'task0_head_equal_to_origin': old_head_equal, 'nonlora_backbone_changed_tensors': len(changed_nonlora), 'new_task_cnn_global_acc': matrix[t][t], 'seen_tasks_macro_cnn_acc': sum(matrix[t][:t+1])/(t+1), 'task0_fisher_normalized': m['task0_fisher_normalized'], 'task0_fisher_current_trace': m['task0_fisher_current_trace'], 'task0_fisher_anchor_cosine_sim': m['task0_fisher_anchor_cosine_sim']}
        for split in ('train', 'test'):
            row[f'{split}_samples'] = m[f'{split}_samples']
            for scope in ('local', 'full'):
                prefix = f'{split}_{scope}'
                row[f'{prefix}_loss'] = m[f'{prefix}_loss']
                row[f'{prefix}_loss_delta_recomputed'] = m[f'{prefix}_loss'] - steps['0'][f'{prefix}_loss']
                row[f'{prefix}_loss_delta_abs_error'] = abs(row[f'{prefix}_loss_delta_recomputed'] - m[f'{prefix}_loss_delta'])
                row[f'{prefix}_acc'] = m[f'{prefix}_acc']
                row[f'{prefix}_acc_drop_from_origin'] = steps['0'][f'{prefix}_acc'] - m[f'{prefix}_acc']
        rows.append(row)
        assets.append({'method': method, 'task': t, 'checkpoint': str(checkpoint), 'size_bytes': checkpoint.stat().st_size, 'manifest': {k: v for k, v in ckpt.items() if k != 'network_state'}, 'state_keys': len(state), 'scaling_or_wrapped_parameter_keys': [k for k in state if 'scaling' in k or 'wrapped_param' in k], 'effective_delta_tensors': len(factor_keys), 'effective_delta_numel': sum(x.numel() for x in origin_delta.values()), 'old_task0_head_exact_equal': old_head_equal, 'changed_nonlora_backbone_keys': changed_nonlora, 'effective_weight_drift_cpu': drift, 'sidecar_paths': {kind: str(checkpoint.parent / f'{kind}_{t}.pt') for kind in ('lora_w_a','lora_w_b','fc_state','fisher_state')}, 'sidecars_exist': all((checkpoint.parent / f'{kind}_{t}.pt').is_file() for kind in ('lora_w_a','lora_w_b','fc_state','fisher_state'))})
    fisher_path = origin_path.parent / 'fisher_state_0.pt'
    fisher = load(fisher_path)
    configs[method]['task0_fisher_sidecar_summary'] = {k: v for k,v in fisher.items() if k != 'fisher_past_delta'}
    configs[method]['task0_fisher_tensor_count'] = len(fisher['fisher_past_delta'])
    configs[method]['task0_fisher_stored_trace'] = sum(float(x.sum()) for x in fisher['fisher_past_delta'].values())
    configs[method]['log_sha256'] = sha(log_path)

with (OUT / 'task_trajectory_recomputed.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT / 'task_assets_manifest.json').write_text(json.dumps({'audit_mode': 'CPU tensor/read-only source; no forward pass or training', 'torch_version': torch.__version__, 'source_root': str(ROOT), 'selected_sequence_root': str(SEQ), 'configs_from_historical_logs': configs, 'assets': assets}, indent=2) + '\n')
print(json.dumps({'rows': len(rows), 'checkpoint_count': len(assets), 'max_drift_abs_error': max(r['delta_norm_abs_error'] for r in rows), 'all_task0_heads_equal': all(a['old_task0_head_exact_equal'] for a in assets), 'all_backbones_unchanged': all(not a['changed_nonlora_backbone_keys'] for a in assets), 'all_sidecars_exist': all(a['sidecars_exist'] for a in assets)}))
