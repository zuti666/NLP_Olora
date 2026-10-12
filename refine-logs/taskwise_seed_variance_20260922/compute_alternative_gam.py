"""Independently recompute the separate RNG-controlled GAM seed groups."""
import math
import statistics

from compute_variance import HERE, read, sha, save_json, save_csv, stats


def main():
    source = read(HERE / 'gam_alternative_cohorts.json')
    raw, summaries, stage_rows = [], [], []
    for group in source['groups']:
        configs, metrics, curves = [], [], []
        for r in group['runs']:
            for name, digest_key in [('source', 'source_sha256'), ('log', 'log_sha256'),
                                     ('config', 'config_sha256'), ('receipt', 'receipt_sha256')]:
                assert sha(r[name]) == r[digest_key]
            configs.append(read(r['config']))
            matrix = read(r['source'])['cnn']['matrices']['final']
            assert len(matrix) == r['expected_tasks'] == 10
            aa = [statistics.mean(row[:i+1]) for i, row in enumerate(matrix)]
            assert all(math.isfinite(v) and 0 <= v <= 100 for v in aa)
            assert all(abs(a-b) < 1e-10 for a, b in zip(aa, r['aa']))
            faa, aaa = aa[-1], statistics.mean(aa)
            assert abs(faa-r['faa']) < 1e-10 and abs(aaa-r['aaa_stage']) < 1e-10
            raw.append({'dataset': r['dataset'], 'method': 'GAM-EWCLoRA',
                        'cohort': r['cohort'], 'seed': r['seed'], 'faa': faa, 'aaa_stage': aaa,
                        'used_in_current_table': False})
            metrics.append((faa, aaa))
            curves.append(aa)
        diffs = {key for key in set.union(*(set(c) for c in configs))
                 if configs[0].get(key) != configs[1].get(key)}
        assert diffs <= {'seed', 'loader_seed', 'prefix'}, diffs
        st_faa, st_aaa = stats([v[0] for v in metrics]), stats([v[1] for v in metrics])
        for calculated, key in [(st_faa, 'faa'), (st_aaa, 'aaa_stage')]:
            assert abs(calculated['sample_variance']-group[key]['sample_variance']) < 1e-10
        summaries.append({'dataset': group['dataset'], 'method': 'GAM-EWCLoRA',
                          'cohort': 'separate RNG-controlled campaign; not current table recipe',
                          'seeds': '1,2', 'n': 2,
                          'faa_mean': st_faa['mean'], 'faa_sample_sd': st_faa['sample_sd'],
                          'faa_sample_variance_pp2': st_faa['sample_variance'],
                          'aaa_mean': st_aaa['mean'], 'aaa_sample_sd': st_aaa['sample_sd'],
                          'aaa_sample_variance_pp2': st_aaa['sample_variance']})
        for i in range(10):
            s = stats([curve[i] for curve in curves])
            stage_rows.append({'dataset': group['dataset'], 'method': 'GAM-EWCLoRA',
                               'cohort': 'separate RNG-controlled campaign', 'task': i+1,
                               'n': 2, 'aa_mean': s['mean'], 'aa_sample_sd': s['sample_sd'],
                               'aa_sample_variance_pp2': s['sample_variance']})
    save_csv(HERE / 'alternative_gam_raw_metrics.csv', raw)
    save_csv(HERE / 'alternative_gam_summary.csv', summaries)
    save_csv(HERE / 'alternative_gam_stage_variance.csv', stage_rows)
    save_json(HERE / 'alternative_gam_recalculation.json', {
        'status': 'independent source/config/log/receipt hashes and numeric checks passed',
        'ddof': 1, 'current_table_included': False, 'raw': raw, 'summary': summaries})
    print(summaries)


if __name__ == '__main__':
    main()
