"""Read-only CPU revalidation of nine reusable lambda-zero control endpoints."""
import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
NEW = OUT.parent
BASE = NEW.parent
torch.set_num_threads(4)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def same(value, other):
    assert value == other, (value, other)


def check_file(path, expected):
    same(sha(path), expected)


old_table = read(BASE / 'seven_row_extension_20260921/final_report/table_data.json')
definitions = {
    'vanilla': ('sgd', 'sgd', 'ewclora'),
    'shaping_only': ('gam', 'gam', 'fs_ewclora'),
    'task1_shaping_no_ewc': ('gam', 'sgd', 'task1_shaping'),
}
controls = []
parent_cache = {}
for arm, (parent_method, future_optimizer, paired_arm) in definitions.items():
    for seed in range(3):
        old_row = next(row for row in old_table['individual'] if row['arm'] == arm and row['seed'] == seed)
        if arm == 'shaping_only':
            run = BASE / f'seven_row_extension_20260921/training/shaping_only_seed{seed}'
        else:
            group = 'A_sgd' if arm == 'vanilla' else 'B_gam'
            run = BASE / f'formal_training/B3_{group}_lam0_seed{seed}'
        paired_run = NEW / f'training/{paired_arm}_seed{seed}'
        complete, manifest = read(run / 'COMPLETE.json'), read(run / 'run_manifest.json')
        paired_manifest = read(paired_run / 'run_manifest.json')
        same(complete['status'], 'complete')
        same(complete['completed_tasks'], 2)
        assert not complete.get('smoke_only', False)
        same(complete['identity_sha256'], manifest['identity_sha256'])
        same(complete['validation']['epochs_per_new_task'], 50)
        same(complete['validation']['validated_tasks'], [1])
        same(manifest['seed'], seed)
        same(manifest['anchor_method'], parent_method)
        same(manifest['source_manifest_sha256'], paired_manifest['source_manifest_sha256'])
        same(manifest['protocol'], paired_manifest['protocol'])
        same(manifest['production_precision'], paired_manifest['production_precision'])
        same(manifest['matmul_allow_tf32'], False)
        same(manifest['cudnn_allow_tf32'], False)
        cfg, new_cfg = manifest['config'], paired_manifest['config']
        same(cfg['optimizer_type'], future_optimizer)
        same(cfg['ewc_lambda'], 0)
        same(cfg['lambda_flat'], 1. if future_optimizer == 'gam' else 0.)
        for key in ('ewc_project_grad', 'ewc_dual_ascent', 'delta_w_penalty', 'weight_merge', 'in_task_weight_merge', 'endpoint_weight_merge', 'approximate_fisher_in_task'):
            same(cfg[key], False)
        same(cfg['ewc_eta'], 0.)
        differences = {k: {'control': cfg.get(k), 'new': new_cfg.get(k)} for k in sorted(cfg.keys() | new_cfg.keys()) if cfg.get(k) != new_cfg.get(k)}
        allowed_diff = {'ewc_lambda', 'ewc_normalize_fisher', 'ewc_normalize_accumulated_fisher', 'outputs_root', 'prefix', 'optimizer_tag_override', 'save_task_checkpoint_dir', 'rq2_extension_arm', 'rq2_anchor_method', 'rq2_fisher_policy', 'rq2_prepared_fisher', 'rq2_smoke_only'}
        same(set(differences) - allowed_diff, set())
        parent_file = Path(manifest['parent']['task_00.pt']['path'])
        parent_hash = manifest['parent']['task_00.pt']['sha256']
        same(parent_hash, paired_manifest['parent']['task_00.pt']['sha256'])
        same(str(parent_file), paired_manifest['parent']['task_00.pt']['path'])
        parent_metrics_file = Path(manifest['parent']['metrics']['path'])
        check_file(parent_metrics_file, manifest['parent']['metrics']['sha256'])
        parent_metrics = read(parent_metrics_file)
        same(parent_metrics['mechanism']['final']['last']['in_task_fisher_final_raw_trace'], old_row['B1_batch_proxy'])
        if parent_hash not in parent_cache:
            check_file(parent_file, parent_hash)
            parent_cache[parent_hash] = torch.load(parent_file, map_location='cpu', weights_only=True, mmap=True)['network_state']
        parent = parent_cache[parent_hash]
        for rel, item in complete['validation']['checkpoint_assets'].items():
            check_file(run / rel, item['sha256'])
            same((run / rel).stat().st_size, item['bytes'])
        metrics_file = Path(complete['validation']['metrics'])
        check_file(metrics_file, complete['validation']['metrics_sha256'])
        metrics = read(metrics_file)
        matrix = metrics['cnn']['matrices']['t01']
        same(metrics['cnn']['matrices']['final'][1], matrix[1])
        a11, a21, a22 = float(matrix[0][0]), float(matrix[1][0]), float(matrix[1][1])
        local = {r['task']: r for r in jsonl(run / 'task0_local_metrics.jsonl')}
        same(round(local[0]['test']['seen_accuracy'], 2), a11)
        same(round(local[1]['test']['seen_accuracy'], 2), a21)
        same(old_row['a11'], local[0]['test']['seen_accuracy'])
        same(old_row['a21'], a21)
        same(old_row['a22'], a22)
        mechanism = metrics['mechanism']['final']['last']
        same(mechanism['task'], 1)
        same(mechanism['lambda_ewc'], 0.)
        same(mechanism['train_fisher_loss_mean'], 0.)
        same(mechanism['ewc_penalty_value'], 0.)
        rng = jsonl(run / 'rng_receipts.jsonl')
        paired_rng = jsonl(paired_run / 'rng_receipts.jsonl')
        check_file(run / 'rng_receipts.jsonl', complete['validation']['rng_receipts_sha256'])
        same(rng, paired_rng)
        train_iter = [r for r in rng if r['event'] == 'loader_iteration' and r.get('shuffle')]
        same(len(train_iter), 51)
        same([r['iteration'] for r in train_iter], list(range(51)))
        assert all(r['batches'] == 8 and r['samples'] == r['sample_count'] for r in train_iter)
        runtime = jsonl(run / 'runtime_receipts.jsonl')
        check_file(run / 'runtime_receipts.jsonl', complete['validation']['runtime_receipts_sha256'])
        resume = [r for r in runtime if r['event'] == 'complete_resume']
        same(len(resume), 1)
        same(resume[0]['checkpoint_sha256'], parent_hash)
        same(resume[0]['verified_exact'], True)
        endpoint_file = run / 'checkpoints/task_01.pt'
        endpoint = torch.load(endpoint_file, map_location='cpu', weights_only=True, mmap=True)['network_state']
        same(tuple(parent['fc.weight'].shape), (20, 768))
        same(tuple(endpoint['fc.weight'].shape), (40, 768))
        assert torch.equal(parent['fc.weight'], endpoint['fc.weight'][:20])
        assert torch.equal(parent['fc.bias'], endpoint['fc.bias'][:20])
        frozen_count = 0
        for key, tensor in parent.items():
            assert torch.isfinite(endpoint[key]).all()
            if not key.startswith('fc.') and 'linear_a_' not in key and 'linear_b_' not in key:
                assert torch.equal(tensor, endpoint[key]), key
                frozen_count += 1
        raw_file = NEW / f'prepared_fisher/{parent_method}_seed{seed}/fisher_state_raw.pt'
        raw_receipt = read(raw_file.with_name('PREPARED.json'))
        same(paired_manifest['prepared_fisher']['parent_checkpoint_sha256'], parent_hash)
        check_file(raw_file, paired_manifest['prepared_fisher']['payload_sha256'])
        raw = torch.load(raw_file, map_location='cpu', weights_only=True, mmap=True)['fisher_past_delta']
        squared, weighted = 0., 0.
        for layer in range(12):
            for channel in ('q', 'v'):
                stem = f'backbone.lora_vit.blocks.{layer}.attn.qkv.'
                ak, bk = stem + f'linear_a_{channel}.weight', stem + f'linear_b_{channel}.weight'
                same(tuple(parent[ak].shape), (10, 768))
                same(tuple(parent[bk].shape), (768, 10))
                delta = endpoint[bk] @ endpoint[ak] - parent[bk] @ parent[ak]
                assert torch.isfinite(delta).all()
                squared += float(delta.double().square().sum())
                weighted += float((raw[stem + channel].double() * delta.double().square()).sum())
        d_qv = math.sqrt(squared)
        assert math.isclose(d_qv, old_row['d_QV_cont'], rel_tol=1e-6, abs_tol=1e-8)
        assert math.isclose(d_qv, local[1]['effective_weight_displacement'], rel_tol=1e-6, abs_tol=1e-8)
        assert math.isclose(squared, mechanism['delta_norm'], rel_tol=1e-5, abs_tol=1e-8)
        rows = {
            'arm': arm, 'seed': seed, 'parent_method': parent_method, 'future_optimizer': future_optimizer,
            'run_dir': str(run), 'status': 'complete_verified', 'ewc_lambda': 0.,
            'a11': a11, 'a11_exact': old_row['a11'], 'a21': a21, 'a22': a22,
            'BWT': a21-a11, 'average_accuracy': (a21+a22)/2,
            'B1_batch_proxy': old_row['B1_batch_proxy'], 'd_QV_cont': d_qv,
            'B1_metrics_source': str(parent_metrics_file), 'B1_metrics_sha256': sha(parent_metrics_file),
            'raw_fisher_weighted_drift_squared_vs_common_parent_rawF': weighted,
            'raw_fisher_path': str(raw_file), 'raw_fisher_sha256': sha(raw_file),
            'metrics_source': str(metrics_file), 'metrics_sha256': sha(metrics_file),
            'parent_checkpoint': str(parent_file), 'parent_checkpoint_sha256': parent_hash,
            'endpoint_checkpoint': str(endpoint_file), 'endpoint_checkpoint_sha256': complete['validation']['checkpoint_assets']['checkpoints/task_01.pt']['sha256'],
            'manifest_path': str(run / 'run_manifest.json'), 'manifest_sha256': sha(run / 'run_manifest.json'),
            'complete_path': str(run / 'COMPLETE.json'), 'complete_sha256': sha(run / 'COMPLETE.json'),
            'paired_ewc_arm': paired_arm, 'paired_ewc_run_dir': str(paired_run),
            'configuration_differences_from_paired_20000_arm': differences,
            'checks': {'epochs': 50, 'all_51_loader_receipts_exact_match_including_first_augmented_batches': True,
                       'all_permutation_indices_match': True, 'new_head_initialization_exact_match': True,
                       'frozen_backbone_tensors_verified': frozen_count, 'old_head_exactly_unchanged': True,
                       'complete_parent_restore_receipt_verified': True, 'active_ewc_loss_zero': True,
                       'checkpoint_metrics_and_rng_hashes_verified': True, 'dQV_recomputed_CPU': True,
                       'weighted_drift_recomputed_with_paired_unormalized_fisher_CPU': True,
                       'GPU_used': False},
        }
        controls.append(rows)
        print(arm, seed, a11, a21, a22, d_qv, weighted, flush=True)

aggregate = []
for arm in definitions:
    rows = [r for r in controls if r['arm'] == arm]
    fields = ('a11', 'B1_batch_proxy', 'a21', 'a22', 'BWT', 'average_accuracy', 'd_QV_cont', 'raw_fisher_weighted_drift_squared_vs_common_parent_rawF')
    aggregate.append({'arm': arm, 'statistics': {key: {'mean': statistics.mean(r[key] for r in rows), 'sample_sd': statistics.stdev(r[key] for r in rows), 'n': 3} for key in fields}})

result = {'status': 'passed', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'Nine completed lambda-zero controls compatible with user-fixed common lambda 20000 campaign',
          'validated_controls': 9, 'controls': controls, 'aggregates': aggregate,
          'a11_and_BWT_policy': 'Native saved CIL matrices rounded to two decimals per seed across all rows; a11_exact retained separately.',
          'fisher_normalization_at_lambda_zero': 'Historical controls retain true normalization flags, but native penalty is multiplied by lambda=0, mean recorded EWC loss is zero, dual/projection/extra penalty are false. Flags only change inactive stored Fisher or diagnostic quantities and do not alter these two-task trained weights.',
          'pairing_limit': 'Every input index, head initialization and first augmented batch in each of 50 training plus final fit-Fisher passes are verified; later augmented batches were not independently hashed.',
          'causal_limit': 'Valid matched EWC-vs-zero comparisons exist for Vanilla/EWCLoRA, Shaping-only/FS-EWCLoRA, and Task1-shaping-no-EWC/Task1-shaping. No SGD-to-GAM lambda-zero arm is present, so Task2-shaping does not have that direct no-EWC control.',
          'raw_Fisher_weighted_drift_definition': 'sum over 24 Q/V effective BA matrices of prepared Task0 raw batch-gradient Fisher times squared endpoint-minus-parent BA; FP64 reduction; same raw Fisher as matched EWC arm.'}
(OUT / 'CONTROL_AUDIT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
lines = ['# Lambda-zero controls audit', '', 'All 9 controls passed fresh CPU verification of saved assets, parent identity, 50 epochs, input/head pairing, frozen backbone/old head, and effective Q/V displacement.', '', '| arm | seed | a11 | a21 | a22 | BWT | A2 | dQV | raw-F weighted squared drift |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in controls:
    lines.append(f"| {r['arm']} | {r['seed']} | {r['a11']:.2f} | {r['a21']:.2f} | {r['a22']:.2f} | {r['BWT']:.2f} | {r['average_accuracy']:.3f} | {r['d_QV_cont']:.8f} | {r['raw_fisher_weighted_drift_squared_vs_common_parent_rawF']:.10g} |")
for key in ('a11_and_BWT_policy', 'fisher_normalization_at_lambda_zero', 'pairing_limit', 'causal_limit', 'raw_Fisher_weighted_drift_definition'):
    lines.extend(['', result[key]])
lines.extend(['', 'Historical files were only read. No model inference, training, or GPU was used. All per-file hashes and configuration differences are in CONTROL_AUDIT.json.'])
(OUT / 'CONTROL_AUDIT.md').write_text('\n'.join(lines) + '\n')
print('PASSED nine controls', flush=True)
