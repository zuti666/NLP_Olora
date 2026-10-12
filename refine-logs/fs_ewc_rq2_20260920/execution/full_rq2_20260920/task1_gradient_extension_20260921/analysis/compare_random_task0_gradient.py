"""Read-only source analysis; writes only this new experiment's comparison files."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    random = BASE / 'radius50_extension_20260921'
    fixed = BASE / 'fixed_gradient_extension_20260921'
    sources, ledger = [], []
    for root, name, count in ((random, 'radius50_complete.json', 12),
                              (fixed, 'fixed_gradient_complete.json', 6)):
        path = root / 'report' / name
        receipt = json.loads(path.read_text())
        assert receipt['status'] == 'complete'
        assert receipt['expected_jobs'] == receipt['validated_jobs'] == count
        assert receipt['automatic_numeric_and_layout_checks'] == 'passed'
        for item in receipt['output_files']:
            assert sha(item['path']) == item['sha256'], item['path']
        ledger.append(dict(experiment=root.name, expected=count, validated=count, failed=0, missing=0))
        sources.append(dict(path=str(path), sha256=sha(path)))
    frames = []
    files = [(random / 'figures/radius50/dense_training_per_seed.csv', 'random100',
              {'G_train': 'loss_change', 'train_accuracy_percent': 'accuracy_percent'}),
             (random / 'figures/radius50/fisher_training_per_seed.csv', 'random16',
              {'raw_fisher_trace': 'fisher_trace'}),
             (fixed / 'figures/all_per_seed.csv', 'task0_positive_gradient',
              {'D_train': 'loss_change', 'train_accuracy_percent': 'accuracy_percent',
               'raw_fisher_trace': 'fisher_trace'})]
    for path, experiment, metrics in files:
        frame = pd.read_csv(path)
        assert not frame.duplicated(['seed', 'method', 'radius']).any()
        for original, metric in metrics.items():
            assert np.isfinite(frame[original]).all()
            part = frame[['seed', 'method', 'radius', original]].rename(columns={original: 'value'})
            part['metric'], part['experiment'] = metric, experiment
            frames.append(part)
        sources.append(dict(path=str(path), sha256=sha(path)))
    long = pd.concat(frames, ignore_index=True)
    rows = []
    for (experiment, metric, radius), group in long.groupby(['experiment', 'metric', 'radius']):
        pivot = group.pivot(index='seed', columns='method', values='value').sort_index()
        assert pivot.index.tolist() == [0, 1, 2] and set(pivot.columns) == {'sgd', 'gam'}
        delta = pivot.gam - pivot.sgd
        rows.append(dict(experiment=experiment, metric=metric, radius=radius, paired_n=3,
                         EWCLoRA_mean=pivot.sgd.mean(), FS_EWCLoRA_mean=pivot.gam.mean(),
                         paired_change_mean=delta.mean(), paired_change_sample_sd=delta.std(ddof=1),
                         gam_lower_seeds=int((delta < 0).sum()),
                         delta_seed0=delta.loc[0], delta_seed1=delta.loc[1], delta_seed2=delta.loc[2]))
    summary = pd.DataFrame(rows)
    long.to_csv(HERE / 'random_vs_task0_positive_per_seed.csv', index=False)
    summary.to_csv(HERE / 'random_vs_task0_positive_paired.csv', index=False)
    selected = summary[summary.radius.isin([.001, .1, .25, .5, 1, 2, 10, 20, 50])]
    selected.to_csv(HERE / 'selected_radius_comparison.csv', index=False)
    # Exact decomposition on the fixed positive ray: D(r)=r*||g0||+remainder.
    # The remainder is not reported as a point Hessian estimate.
    norms = []
    fixed_receipt = json.loads((fixed / 'report/fixed_gradient_complete.json').read_text())
    fixed_sources = {item['path']: item['sha256'] for item in fixed_receipt['source_files']}
    for seed in range(3):
        for method in ('sgd', 'gam'):
            path = fixed / f'results/seed{seed}/{method}/fixed_gradient/gradient_manifest.json'
            assert sha(path) == fixed_sources[str(path)]
            metadata = json.loads(path.read_text())
            norms.append(dict(seed=seed, method=method, initial_directional_slope=metadata['raw_grad_norm']))
            sources.append(dict(path=str(path), sha256=sha(path)))
    remainder = pd.read_csv(fixed / 'figures/all_per_seed.csv').merge(
        pd.DataFrame(norms), on=['seed', 'method'], validate='many_to_one')
    remainder['first_order_term'] = remainder.radius * remainder.initial_directional_slope
    remainder['nonlinear_remainder'] = remainder.D_train - remainder.first_order_term
    remainder[['seed', 'method', 'radius', 'D_train', 'initial_directional_slope',
               'first_order_term', 'nonlinear_remainder']].to_csv(
                   HERE / 'task0_positive_nonlinear_remainder.csv', index=False)
    fixed_loss = long[(long.experiment == 'task0_positive_gradient') & (long.metric == 'loss_change')]
    record = dict(status='passed', generated_utc=datetime.now(timezone.utc).isoformat(),
                  ledger=ledger, source_files=sources, fixed_signed_loss_min=float(fixed_loss.value.min()),
                  paired_change_definition='FS-EWCLoRA minus EWCLoRA, same training seed',
                  random_loss_definition='mean over 100 positive per-direction CE increments',
                  fixed_loss_definition='signed CE increment on own positive Task0-gradient direction',
                  same_effective_coordinate_dimension=14155776,
                  table=list(selected.replace({np.nan: None}).to_dict(orient='records')))
    (HERE / 'COMPARISON_VALIDATION.json').write_text(json.dumps(record, indent=2) + '\n')
    print(selected[['experiment', 'metric', 'radius', 'EWCLoRA_mean', 'FS_EWCLoRA_mean']].to_string(index=False))
    print('All fixed-direction signed train damage nonnegative:', bool((fixed_loss.value >= 0).all()))


if __name__ == '__main__':
    main()
