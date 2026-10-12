"""Add completed Task1 results to the prior paired comparison without new inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    receipt = json.loads((ROOT / 'report/task1_gradient_complete.json').read_text())
    assert receipt['status'] == 'complete' and receipt['validated_jobs'] == 6
    assert receipt['gradient_task'] == 1 and receipt['direction_sign'] == -1
    data_path = ROOT / 'figures/all_per_seed.csv'
    entries = {item['path']: item['sha256'] for item in receipt['output_files']}
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == entries[str(data_path)]
    data = pd.read_csv(data_path)
    assert len(data) == 558 and not data.duplicated(['seed', 'method', 'radius']).any()
    metrics = {'D_train': 'loss_change', 'train_accuracy_percent': 'accuracy_percent',
               'raw_fisher_trace': 'fisher_trace'}
    added = []
    for original, metric in metrics.items():
        assert np.isfinite(data[original]).all()
        for radius, group in data.groupby('radius'):
            pivot = group.pivot(index='seed', columns='method', values=original).sort_index()
            assert pivot.index.tolist() == [0, 1, 2]
            delta = pivot.gam - pivot.sgd
            added.append(dict(experiment='task1_negative_gradient', metric=metric, radius=radius,
                paired_n=3, EWCLoRA_mean=pivot.sgd.mean(), FS_EWCLoRA_mean=pivot.gam.mean(),
                paired_change_mean=delta.mean(), paired_change_sample_sd=delta.std(ddof=1),
                gam_lower_seeds=int((delta < 0).sum()), delta_seed0=delta.loc[0],
                delta_seed1=delta.loc[1], delta_seed2=delta.loc[2]))
    previous = pd.read_csv(HERE / 'random_vs_task0_positive_paired.csv')
    combined = pd.concat([previous, pd.DataFrame(added)], ignore_index=True)
    combined.to_csv(HERE / 'three_directions_paired.csv', index=False)
    selected = combined[combined.radius.isin([.1, .25, .5, 1, 2, 10, 20, 50])]
    selected.to_csv(HERE / 'three_directions_selected.csv', index=False)
    print(selected[selected.experiment == 'task1_negative_gradient'][
        ['metric', 'radius', 'EWCLoRA_mean', 'FS_EWCLoRA_mean', 'gam_lower_seeds']].to_string(index=False))


if __name__ == '__main__':
    main()
