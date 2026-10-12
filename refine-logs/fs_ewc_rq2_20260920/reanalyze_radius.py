#!/usr/bin/env python3
"""CPU-only, standard-library reanalysis; never imports training code or torch.

G(r) = mean(max(loss_delta, 0)); Q95 uses linear interpolation of the
positive-part per-direction damage. Prefix tolerance is a grid statistic:
the last radius before the FIRST threshold violation, not last crossing.
Direction samples are not training seeds or paired rays across radii.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path('/data/140-1/users/liying/CopyFlatCV')
TAG = 'r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50'


def quantile(values, q):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def dump_csv(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source_dir = args.repo / 'logs/sgd_gam_s1c/staged/stage2_r-2' / TAG / 'fqa_partial'
    metrics_dir = args.repo / 'outputs_logs/sgd_gam_s1c/stage2_r-2' / TAG
    epsilons = [.001, .005, .01, .02, .05]
    all_rows, all_thresholds, sources = [], [], []
    for method in ['sgd', 'gam']:
        matches = list(metrics_dir.rglob(f's1c_{method}_r_fqa_from_task0_s0_vit_base_patch16_224_cl_metrics.json'))
        assert len(matches) == 1, (method, matches)
        metrics_path = matches[0]
        metrics = json.loads(metrics_path.read_text())['mechanism']['steps']['0']
        for split in ['train', 'test']:
            path = source_dir / f's1c_{method}_r_fqa_from_task0_s0_task00_endpoint_{split}_fqa_partial.json'
            raw_bytes = path.read_bytes()
            records = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
            assert [r['radius'] for r in records] == [i / 4 for i in range(81)]
            rows = []
            for line_num, record in enumerate(records, 1):
                losses = record['per_u']['loss_delta']
                assert len(losses) == record['K'] == 100
                assert all(math.isfinite(value) for value in losses)
                positive = [max(0., value) for value in losses]
                acc = record['per_u']['perturbed_acc']
                prefix = f'endpoint_{split}'
                row = dict(method=method, split=split, training_seed=0,
                           radius=record['radius'], K=len(losses),
                           delta_loss_mean=mean(losses), G=mean(positive),
                           Q95=quantile(positive, .95),
                           damage_std=pstdev(positive), sampled_damage_max=max(positive),
                           positive_fraction=mean([float(v > 0) for v in losses]),
                           base_loss=metrics[f'{prefix}_fqa_base_loss'],
                           base_accuracy=record['mean']['base_acc'],
                           accuracy_mean=mean(acc),
                           fixed_anchor_normalized_fisher_quad=record['mean'][f'{prefix}_fisher_quad'],
                           source_jsonl=str(path), source_line=line_num)
                rows.append(row)
            assert rows[0]['G'] == rows[0]['Q95'] == 0.
            all_rows.extend(rows)
            for metric in ['G', 'Q95']:
                for epsilon in epsilons:
                    passing_radius = None
                    first_failure = None
                    for row in rows:
                        if row[metric] > epsilon:
                            first_failure = row['radius']
                            break
                        passing_radius = row['radius']
                    all_thresholds.append(dict(method=method, split=split, training_seed=0,
                        metric=metric, epsilon=epsilon, prefix_r_epsilon=passing_radius,
                        first_failing_grid_radius=first_failure,
                        right_censored=first_failure is None,
                        max_observed_radius=rows[-1]['radius'], grid_step=.25,
                        interpretation='empirical_grid_statistic_no_continuous_ball_guarantee'))
            sources.append(dict(method=method, split=split, path=str(path),
                sha256=hashlib.sha256(raw_bytes).hexdigest(), rows=len(records),
                metrics_path=str(metrics_path),
                metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                zero_radius_max_abs=max(abs(v) for v in records[0]['per_u']['loss_delta']),
                normalized_anchor_fisher_trace=metrics[f'{prefix}_fisher_total_mass'],
                coordinate_dim=metrics[f'{prefix}_param_count']))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(args.output_dir / 'radius_damage_seed0.csv', all_rows)
    dump_csv(args.output_dir / 'radius_tolerance_seed0.csv', all_thresholds)
    manifest = dict(
        sources=sources, epsilons=epsilons,
        threshold_provenance='User-supplied experiment-plan analysis thresholds; no historical preregistration claimed.',
        G='mean(max(delta_loss, 0))', Q95='linear_interpolated_quantile(max(delta_loss,0), 0.95)',
        prefix='last observed radius before first threshold violation; no interpolation',
        caveats=['seed0 only', 'each radius samples fresh directions',
                 'no saved direction IDs/hashes: cross-method pairing unverified',
                 'train uses one cached draw of random training augmentation',
                 'Fisher quadratic is based on trace-normalized fixed anchor Fisher',
                 'right-censored >=20 is not evidence for radii above 20'])
    (args.output_dir / 'radius_reanalysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'radius_rows': len(all_rows), 'threshold_rows': len(all_thresholds),
                      'output_dir': str(args.output_dir)}, indent=2))


if __name__ == '__main__':
    main()
