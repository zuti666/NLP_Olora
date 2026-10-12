"""Summarize completed native-protocol mechanisms; never pool other protocols."""
from pathlib import Path
import json
import statistics
from datetime import datetime, timezone

from common import atomic_json, atomic_text, sha256

HERE = Path(__file__).resolve().parent
OLD = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-reference-hyperparams-20260920')


def main():
    current = json.loads((HERE / 'results.json').read_text())
    previous = json.loads((OLD / 'results.json').read_text())
    records = []
    for origin, payload in [('reused_native_sweep', previous), ('new_mechanisms', current)]:
        for row in payload['jobs']:
            row = dict(row, origin=origin)
            if row['status'] == 'completed':
                path = Path(row['metrics_path'])
                assert sha256(path) == row['metrics_sha256'], path
                raw = json.loads(path.read_text())
                steps = raw.get('mechanism', {}).get('steps', {})
                summaries = {}
                for key in ('grad_fisher_to_clean_norm_ratio_mean', 'grad_cos_clean_gam_mean',
                            'grad_flat_to_clean_norm_ratio_mean', 'in_task_weight_merge_shift_l2',
                            'in_task_path_weight_effective_n', 'fisher_past_trace_mass'):
                    values = [float(s[key]) for k, s in steps.items() if int(k) > 0 and s.get(key) is not None]
                    if values:
                        summaries[key] = {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
                row['mechanism_task1_to9'] = summaries
                row['fisher_sources'] = sorted({str(s.get('current_fisher_source')) for s in steps.values()})
            records.append(row)
    lookup = {r['id']: r for r in records}
    contrasts = []
    plan = json.loads((HERE / 'REVIEWER_MECHANISM_PLAN.json').read_text())
    # Read the defining config overrides; do not depend on informal job labels.
    on, off = {2: lookup.get('s3_center_s2')}, {}
    for job in plan['jobs']:
        row = lookup[job['id']]
        override = {k: v for k, v in job['overrides'].items() if not k.startswith('s3_')}
        if not override:
            on[job['seed']] = row
        if override == {'in_task_weight_merge_start_task': 10}:
            off[job['seed']] = row
    paired = []
    for seed in (1, 2, 3):
        a, b = on.get(seed), off.get(seed)
        if a and b and a['status'] == b['status'] == 'completed':
            paired.append({'seed': seed, 'on_id': a['id'], 'off_id': b['id'],
                           'delta_faa': a['faa_task_macro'] - b['faa_task_macro'],
                           'delta_aaa': a['aaa_task_macro'] - b['aaa_task_macro']})
    pair_summary = {'completed_pairs': len(paired), 'required_pairs': 3, 'rows': paired}
    if len(paired) == 3:
        for key in ('delta_faa', 'delta_aaa'):
            values = [r[key] for r in paired]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            margin = 4.302652729911275 * sd / (3 ** .5)
            pair_summary[key] = {'mean': mean, 'sample_sd': sd, 'paired_t95_df2': [mean-margin, mean+margin]}
        pair_summary['interpretation'] = 'Three paired seeds; report signs, mean and sample SD. No automatic significance claim.'
    counts = {state: sum(r['status'] == state for r in current['jobs'])
              for state in sorted({r['status'] for r in current['jobs']})}
    document = {'updated_at': datetime.now(timezone.utc).isoformat(), 'new_job_counts': counts,
                'weight_fusion_paired': pair_summary, 'records': records,
                'scope': 'Native frozen S3 protocol only; single-seed design controls are exploratory. Integration smokes excluded.'}
    atomic_json(HERE / 'mechanism_summary.json', document)
    lines = ['# S3 mechanism results', '', document['scope'], '',
             'New jobs: ' + json.dumps(counts), '',
             f"Weight fusion paired seeds complete: {len(paired)}/3.", '',
             '| Job | Origin | Seed | Status | FAA macro | AAA stage macro |',
             '| --- | --- | --- | --- | --- | --- |']
    for r in records:
        lines.append('| ' + ' | '.join(str(r.get(k, '')) for k in
                     ('id', 'origin', 'seed', 'status', 'faa_task_macro', 'aaa_task_macro')) + ' |')
    lines += ['', 'Weight fusion comparison (A on minus computation/RNG-matched A off):', '',
              '```json', json.dumps(pair_summary, indent=2), '```', '',
              'FAA and AAA are recomputed from the full task accuracy matrix. Separate sample-Fisher studies remain in EVIDENCE_REUSE.md and are not pooled. Boundary archives support factor/ΔW algebra only; no paired endpoint/retained Fisher cosine has been measured by this observer.']
    atomic_text(HERE / 'MECHANISM_RESULTS.md', '\n'.join(lines) + '\n')
    print(json.dumps({'new_job_counts': counts, 'paired_weight_seeds': len(paired)}))


if __name__ == '__main__':
    main()
