#!/usr/bin/env python3
"""Independent CPU reader; writes only versioned audit files beside this script.

Usage: PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python SCRIPT
       --require-complete --output-prefix final_20261012
No experiment modules, torch, model execution, checkpoint loading, or GPU calls.
"""
import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT.parent / 'imageneta-response-state-chain-gpu456-20261011T195426Z'
GRIDS = {'original': [.001, .003, .01],
         'exploratory': [1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01]}
COUNTS = {'curvature': 36, 'matching': 24, 'evaluation': 30}
DATASETS = {'train_old': 1141, 'train_new': 906, 'dev_old': 118,
            'dev_new': 91, 'test_old': 307, 'test_new': 254}


def near(a, b, atol=1e-10, rtol=1e-8):
    assert abs(a-b) <= atol+rtol*max(abs(a), abs(b)), (a, b)


def exact_restore(d):
    assert d and all(v is True for v in d.values()), d


def adjacent(rows):
    valid = [r.get('degenerate') is False and r.get('relative_error') is not None
             and math.isfinite(r['relative_error']) and r['relative_error'] <= .08 for r in rows]
    return any(a and b for a, b in zip(valid, valid[1:]))


def comparisons(value, checks):
    if isinstance(value, dict):
        if all(k in value for k in ('a_norm', 'b_norm', 'relative_error', 'cosine')):
            if value['cosine'] is not None:
                a, b, c = value['a_norm'], value['b_norm'], value['cosine']
                assert a >= 0 and b > 0 and -1 <= c <= 1
                expected = math.sqrt(max(0, a*a+b*b-2*a*b*c))/(b+1e-12)
                near(value['relative_error'], expected, atol=1e-7)
                checks['relative_error_denominator_identity'] += 1
            else:
                assert value.get('degenerate') is True and value['relative_error'] is None
        for v in value.values(): comparisons(v, checks)
    elif isinstance(value, list):
        for v in value: comparisons(v, checks)


def gate(g):
    assert g['fixed_base'] is True and g['second_hvp_vector_detached'] is True
    first = g['first_order']; c = first['comparison']
    ok = (first['repeated_gradient_exact'] is True and c['degenerate'] is False
          and c['relative_error'] <= 2e-4 and c['cosine'] >= .99999)
    assert first['accepted'] == ok
    passes = {name: [] for name in GRIDS}
    for product in ('H_g0', 'H_detached_H_g0'):
        block = g['finite_difference'].get(product)
        for name, distances in GRIDS.items():
            passed = False
            if block is not None:
                rows = block[name]
                assert [r['distance'] for r in rows] == distances
                passed = adjacent(rows)
                assert block[name+'_passed'] == passed
            passes[name].append(passed)
    original = bool(ok and all(passes['original']))
    extended = bool(ok and all(passes['exploratory']))
    assert g['original_gate_accepted'] == original
    assert g['extended_gate_accepted'] == g['accepted'] == extended
    return original, extended


def effective_part(part, checks):
    assert len(part['layers']) == 24
    assert {x['layer_index'] for x in part['layers']} == set(range(24))
    near(part['effective_norm']**2, part['lora_effective_norm']**2+part['fc_norm']**2)
    near(part['lora_effective_norm']**2, sum(x['effective_delta_norm']**2 for x in part['layers']))
    for x in part['layers']:
        assert x['delta_Q_identity_error'] <= 1e-10
        assert x['effective_identity_error'] <= 1e-10
        near(x['pi_A'], x['T_A_norm']/(x['T_A_norm']+x['T_B_norm']+1e-12))
        assert x['delta_Q_norm'] <= x['delta_Q_first_order_norm']+x['delta_Q_second_order_norm']+1e-10
    checks['effective_and_Q_identity_layers'] += 24


def eval_arrays(row, read_bytes, checks):
    path = Path(row['logits_file'])
    assert hashlib.sha256(read_bytes(path)).hexdigest() == row['logits_sha256']
    with np.load(path, allow_pickle=False) as arrays:
        for name, count in DATASETS.items():
            pre = name+'__'; z = arrays[pre+'logits'].astype(np.float64); y = arrays[pre+'labels']
            assert z.shape == (count, 40) and y.shape == (count,) and np.isfinite(z).all()
            lo = 0 if name.endswith('old') else 20
            assert ((y >= lo) & (y < lo+20)).all()
            pred = z.argmax(1); cond = z[:, lo:lo+20].argmax(1)+lo
            cross = (pred < lo) | (pred >= lo+20); within = (pred != y) & ~cross
            d = row['datasets'][name]
            assert d['samples'] == count
            for key, v in [('prediction_joint40', pred), ('prediction_conditional20', cond),
                           ('cross_group_error', cross), ('within_group_error', within)]:
                assert np.array_equal(arrays[pre+key], v)
            for mode, p in [('joint40', pred), ('conditional20', cond)]:
                correct = int((p == y).sum())
                assert d[mode]['correct'] == correct and d[mode]['errors'] == count-correct
                near(d[mode]['accuracy'], 100*correct/count)
            assert d['cross_group_error_count'] == int(cross.sum())
            assert d['within_group_error_count'] == int(within.sum())
            assert d['conditional_correct_lost_to_other_group_count'] == int(((cond == y) & cross).sum())
            assert d['conditional_error_and_cross_group_count'] == int(((cond != y) & cross).sum())
            def lse(x):
                m = x.max(1); return m+np.log(np.exp(x-m[:, None]).sum(1))
            true = z[np.arange(count), y]
            joint = lse(z)-true; conditional = lse(z[:, lo:lo+20])-true
            penalty = joint-conditional
            for key, actual in [('joint_ce', joint), ('conditional_ce', conditional),
                                ('ce_competition_penalty', penalty)]:
                assert np.max(np.abs(arrays[pre+key]-actual)) <= 3e-6
            near(d['joint40']['ce'], float(joint.mean()), atol=1e-5)
            near(d['conditional20']['ce'], float(conditional.mean()), atol=1e-5)
            near(d['ce_competition_penalty'], float(penalty.mean()), atol=3e-6)
            checks['evaluated_datasets_from_saved_logits'] += 1
        for key in arrays.files:
            if np.issubdtype(arrays[key].dtype, np.floating): assert np.isfinite(arrays[key]).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--output-prefix', default='audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = parser.parse_args()
    assert '/' not in args.output_prefix and '\\' not in args.output_prefix
    hashes = {}; failures = []; checks = collections.Counter(); observed = collections.Counter()
    receipts = []; curvature = []; matching = []; evaluation = []; originals = []
    def read_bytes(path):
        path = Path(path); raw = path.read_bytes(); hashes[str(path)] = hashlib.sha256(raw).hexdigest(); return raw
    def read(path): return json.loads(read_bytes(path))
    suite = read(ROOT/'suite.json'); queue = read(ROOT/'runtime/queue_state.json')
    manifest = read(ROOT/'implementation_manifest.json')['files']
    read_bytes(Path(__file__))
    for script in ('collect_results.py', 'curvature_diagnostics.py', 'effective_match.py', 'checkpoint_eval.py'):
        read_bytes(ROOT/script)
        if script != 'collect_results.py': assert hashes[str(ROOT/script)] == manifest[str(ROOT/script)]
    for rel in ('diagnostics/offline_core.py', 'trainer/effective_step.py', 'trainer/response_chain_train.py', 'trainer/state_transaction.py'):
        path = OLD/rel; read_bytes(path); assert hashes[str(path)] == manifest[str(path)]
    for kind, expected_count in COUNTS.items():
        jobs = [j for j in suite['jobs'] if j['kind'] == kind and not j['pilot']]
        assert sorted(j['seed'] for j in jobs) == [0, 1, 2]
        assert sum(j['expected_items'] for j in jobs) == expected_count
    assert len([j for j in suite['jobs'] if j['pilot']]) == 3
    for job in suite['jobs']:
        path = Path(job['output_dir'])/job['report_file']
        if not path.exists(): continue
        report = read(path); rows = report['results']; keys = set()
        assert report['seed'] == job['seed'] and report['pilot'] == job['pilot']
        assert report['submitted_training_updates'] == 0
        prefix = 'pilot' if job['pilot'] else 'formal'; kind = job['kind']
        if job['pilot']:
            assert job['seed'] == 2
            allowed = ({(2, 'S', 1, 0)} if kind == 'curvature' else
                       {(2, 'S', 10, 0)} if kind == 'matching' else
                       {(2, b, 1, None) for b in ('S', 'R')})
        else:
            epochs = (0, 1, 10, 25, 50) if kind == 'evaluation' else (1, 10, 25) if kind == 'curvature' else (10, 25)
            allowed = {(job['seed'], b, e, batch) for b in ('S', 'R') for e in epochs
                       for batch in ((None,) if kind == 'evaluation' else (0, 1))}
        for row in rows:
            try:
                key = (row['seed'], row['branch'], row['epoch'], row.get('batch_index'))
                assert key not in keys; keys.add(key)
                assert key in allowed, (key, allowed)
                assert row['seed'] == job['seed'] and row['branch'] in ('S', 'R')
                assert row.get('submitted_training_updates', 0) == 0
                assert manifest[str(Path(row['checkpoint']).resolve())] == row['checkpoint_sha256']
                comparisons(row, checks)
                observed[prefix+'_'+kind] += 1
                if kind == 'curvature':
                    d = row['diagnostic']; exact_restore(d['restoration'])
                    op, ep = gate(d['fp32']) if 'fp32' in d else (False, False)
                    accepted64 = gate(d['fp64'])[1] if 'fp64' in d else False
                    assert job['pilot'] or 'fp64' not in d
                    assert d['candidate_comparisons_permitted'] == (ep or accepted64)
                    assert len(d['scales']) == (2 if ep or accepted64 else 0)
                    if 'fp64_restoration' in d: exact_restore(d['fp64_restoration'])
                    observed[prefix+'_original_fp32_gate'] += op
                    observed[prefix+'_exploratory_fp32_gate'] += ep
                    observed[prefix+'_exploratory_fp64_gate'] += accepted64
                    for scale in d['scales']:
                        if scale['relative_error'] is not None:
                            near(scale['a_norm'], scale['delta_native_norm'])
                            near(scale['b_norm'], scale['delta_quadratic_norm'])
                        curvature.append(dict(stratum=prefix, seed=row['seed'], branch=row['branch'], epoch=row['epoch'],
                                              batch_index=row['batch_index'], **scale))
                    if len(d['scales']) == 2 and all(x['delta_quadratic_norm'] is not None for x in d['scales']):
                        near(d['scales'][1]['delta_quadratic_norm'], .5*d['scales'][0]['delta_quadratic_norm'])
                elif kind == 'matching':
                    d = row['match']; exact_restore(row['restoration']); exact_restore(d['restoration'])
                    assert row['HVP_calls'] == d['submitted_training_updates'] == 0
                    observed[prefix+'_accepted_matches'] += d['status'] == 'accepted'
                    if d['status'] == 'accepted':
                        for part in ('sam', 'g2_native', 'g2_matched'): effective_part(d[part], checks)
                        near(d['sam']['effective_norm'], d['g2_matched']['effective_norm'])
                        near(d['norm_match_relative_error'], abs(d['g2_matched']['effective_norm']-d['sam']['effective_norm'])/d['sam']['effective_norm'])
                        near(d['scalar_solver']['target'], d['sam']['effective_norm'])
                        near(d['scalar_solver']['predicted_norm'], d['sam']['effective_norm'])
                        near(d['matched_lr_ratio'], d['scalar_solver']['scale']/row['probe_lr'])
                        for label, cand in [('native', 'g2_native'), ('matched', 'g2_matched')]:
                            for part, part_norm in [('all', 'effective_norm'), ('lora_only', 'lora_effective_norm'), ('fc_only', 'fc_norm')]:
                                near(d['direction_parts'][part][label]['a_norm'], d[cand][part_norm])
                                near(d['direction_parts'][part][label]['b_norm'], d['sam'][part_norm])
                        matching.append(dict(stratum=prefix, row=row))
                else:
                    exact_restore(row['restoration']); eval_arrays(row, read_bytes, checks)
                    near(row['FAA'], .5*(row['datasets']['test_old']['joint40']['accuracy']+row['datasets']['test_new']['joint40']['accuracy']))
                    evaluation.append(dict(stratum=prefix, row=row))
                pattern = ('checkpoint%02d_batch%02d.json' % (row['checkpoint_index'], row['batch_index'])
                           if kind == 'curvature' else 'branch%s_epoch%03d%s.json' %
                           (row['branch'], row['epoch'], '_batch%d' % row['batch_index'] if kind == 'matching' else ''))
                assert read(path.parent/pattern) == row
            except (AssertionError, KeyError, ValueError, OSError) as error:
                failures.append({'job': job['id'], 'cell': str(key), 'error': str(error)})
        receipt_path = path.parent/'completion_receipt.json'; execution_complete = False
        if receipt_path.exists():
            receipt = read(receipt_path)
            exit_path = ROOT/'runtime/jobs'/job['id']/'exit.json'
            exit_record = read(exit_path) if exit_path.exists() else {}
            execution_complete = (receipt['results_sha256'] == hashes[str(path)] and
                                  len(rows) == job['expected_items'] and
                                  keys == allowed and
                                  queue['jobs'][job['id']]['status'] == 'completed' and
                                  exit_record.get('returncode') == 0)
        receipts.append(dict(job=job['id'], pilot=job['pilot'], observed=len(rows), expected=job['expected_items'],
                             report_status=report['status'], execution_complete=execution_complete))
    shared = {}
    for seed in range(3):
        pair = {x['row']['branch']: x['row'] for x in evaluation if x['stratum'] == 'formal'
                and x['row']['seed'] == seed and x['row']['epoch'] == 0}
        if set(pair) != {'S', 'R'}: continue
        a, b = pair['S'], pair['R']
        assert a['full_model_parameter_sha256'] == b['full_model_parameter_sha256']
        assert a['dev_split_sha256'] == b['dev_split_sha256']
        with np.load(a['logits_file'], allow_pickle=False) as aa, np.load(b['logits_file'], allow_pickle=False) as bb:
            assert set(aa.files) == set(bb.files)
            assert all(aa[k].dtype == bb[k].dtype and np.array_equal(aa[k], bb[k]) for k in aa.files)
            shared[seed] = dict(verified=True, arrays_compared=len(aa.files))
    paired = []
    for epoch in (25, 50):
        for dataset in ('test_old', 'test_new'):
            for mode in ('conditional20', 'joint40'):
                for metric in ('accuracy', 'ce'):
                    values = []; seeds = []
                    for seed in range(3):
                        pair = {x['row']['branch']: x['row'] for x in evaluation if x['stratum'] == 'formal'
                                and x['row']['seed'] == seed and x['row']['epoch'] == epoch}
                        if set(pair) != {'S', 'R'}: continue
                        values.append(pair['R']['datasets'][dataset][mode][metric]-pair['S']['datasets'][dataset][mode][metric]); seeds.append(seed)
                    paired.append(dict(epoch=epoch, dataset=dataset, metric=mode+'.'+metric, seeds=seeds, values=values,
                                       mean=statistics.mean(values) if values else None,
                                       sample_sd=statistics.stdev(values) if len(values)>1 else None))
    seed_curvature = []; curvature_summary = []
    groups = collections.defaultdict(list)
    for r in curvature:
        groups[(r['stratum'], r['branch'], r['epoch'], r['scale'], r['curvature_precision'])].append(r)
    for key, cells in sorted(groups.items()):
        for metric in ('relative_error', 'cosine', 'delta_native_norm', 'delta_quadratic_norm'):
            vals = []; seeds = []; counts = []
            for seed in range(3):
                actual = [r[metric] for r in cells if r['seed'] == seed and r[metric] is not None]
                if not actual: continue
                mean = statistics.mean(actual); vals.append(mean); seeds.append(seed); counts.append(len(actual))
                seed_curvature.append(dict(group=list(key), seed=seed, metric=metric, cells=len(actual), mean=mean))
            curvature_summary.append(dict(group=list(key), metric=metric, seeds=seeds, cells_per_seed=counts, values=vals,
                                           mean=statistics.mean(vals) if vals else None,
                                           sample_sd=statistics.stdev(vals) if len(vals)>1 else None))
    for path in sorted((OLD/'outputs').glob('offline*/diagnostic_results.json')):
        r = read(path)
        for row in r.get('results', []):
            g = row['correction']['numerical_gate']; originals.append(dict(seed=row['seed'], branch=row['branch'], epoch=row['epoch'], batch=row['batch_index'], accepted=g.get('accepted', False), status=g['status']))
    phase_summary = []
    for branch in ('S', 'R'):
        for scale in (1., .5):
            for metric in ('relative_error', 'cosine', 'delta_native_norm', 'delta_quadratic_norm'):
                values = []; seeds = []; counts = []
                for seed in range(3):
                    actual = [r[metric] for r in curvature if r['stratum'] == 'formal' and r['seed'] == seed
                              and r['branch'] == branch and r['scale'] == scale and r[metric] is not None]
                    if not actual: continue
                    values.append(statistics.mean(actual)); seeds.append(seed); counts.append(len(actual))
                phase_summary.append(dict(stratum='formal', phase='probe_1_25', branch=branch, scale=scale,
                                          metric=metric, seeds=seeds, cells_per_seed=counts, values=values,
                                          mean=statistics.mean(values) if values else None,
                                          sample_sd=statistics.stdev(values) if len(values)>1 else None,
                                          estimand='mean over all fixed e1/e10/e25 batches within each seed, then seed mean/sample SD'))
    complete = (all(observed['formal_'+k] == v for k, v in COUNTS.items()) and
                sum(r['execution_complete'] and not r['pilot'] for r in receipts) == 9)
    collector_path = ROOT/'results/offline_analysis/OFFLINE_ANALYSIS.json'
    collector = read(collector_path) if collector_path.exists() else {}
    # Summaries may lag a live raw snapshot; never mark lag as scientific failure.
    for stored in collector.get('curvature_correction_summary', []):
        key = [stored[k] for k in ('stratum', 'branch', 'epoch', 'scale', 'curvature_precision')]
        own = next((x for x in curvature_summary if x['group'] == key and x['metric'] == stored['metric']), None)
        if own and own['cells_per_seed'] == stored['cells_per_seed'] and own['seeds'] == stored['seeds']:
            try:
                if own['mean'] is not None: near(own['mean'], stored['mean'])
                if own['sample_sd'] is not None: near(own['sample_sd'], stored['sample_sd'])
                checks['collector_seed_curvature_summaries'] += 1
            except AssertionError as error: failures.append(dict(summary='curvature', error=str(error)))
    for own in paired:
        stored = next((x for x in collector.get('paired_evaluation_summary', []) if all(x[k] == own[k] for k in ('epoch', 'dataset', 'metric'))), None)
        if stored and stored['seeds'] == own['seeds']:
            try:
                near(own['mean'], stored['mean']); near(own['sample_sd'], stored['sample_sd'])
                assert own['values'] == stored['values']; checks['collector_paired_endpoint_summaries'] += 1
            except AssertionError as error: failures.append(dict(summary='paired_endpoint', error=str(error)))
    csv_path = ROOT/'results/offline_analysis/CURVATURE_CORRECTION_SUMMARY.csv'
    if csv_path.exists():
        stored_rows = list(csv.DictReader(read_bytes(csv_path).decode().splitlines()))
        for stored in stored_rows:
            key = [stored['stratum'], stored['branch'], int(stored['epoch']), float(stored['scale']), stored['curvature_precision']]
            own = next((x for x in curvature_summary if x['group'] == key and x['metric'] == stored['metric']), None)
            if own and own['cells_per_seed'] == json.loads(stored['cells_per_seed']) and own['seeds'] == json.loads(stored['seeds']):
                try:
                    if own['mean'] is not None: near(own['mean'], float(stored['mean']))
                    if own['sample_sd'] is not None: near(own['sample_sd'], float(stored['sample_sd']))
                    assert own['values'] == json.loads(stored['values'])
                    checks['collector_curvature_summary_CSV'] += 1
                except AssertionError as error: failures.append(dict(summary='curvature_CSV', error=str(error)))
    artifacts_path = ROOT/'results/offline_analysis/ARTIFACTS.json'
    if artifacts_path.exists():
        artifacts = read(artifacts_path)
        for name, expected_hash in artifacts['files'].items():
            try:
                assert hashlib.sha256(read_bytes(Path(name))).hexdigest() == expected_hash
                checks['collector_artifact_SHA256'] += 1
            except (AssertionError, OSError) as error: failures.append(dict(artifact=name, error=str(error)))
    report = dict(protocol='independent_cpu_final_integrity_v1', time_utc=datetime.now(timezone.utc).isoformat(),
                  status='failed' if failures else 'complete' if complete else 'partial',
                  cpu_only=True, submitted_training_updates=0, actual_formal_scope_complete=complete,
                  planned_formal=COUNTS, observed=dict(observed), execution_receipts=receipts, failures=failures,
                  checks=dict(checks), shared_e0=shared, paired_fixed_endpoints=paired,
                  curvature_within_seed_means=seed_curvature, curvature_seed_summary=curvature_summary,
                  curvature_branch_phase_summary=phase_summary,
                  historical_original_gates_unchanged=originals, source_sha256=hashes,
                  interpretation={'relative_error':'||a-b||/(||b||+1e-12); reference_norm affects degeneracy only. FD b=exact HVP; correction b=Delta_quad; matched effective b=SAM.',
                                  'curvature_phase':'All formal curvature e1/e10/e25 cells are in the pre-switch phase; cell means are within seed and branch/epoch/scale/precision before across-seed sample SD.',
                                  'matching_geometry':'Ideal finite FP64 BA and all FC geometry from represented native FP32 parameters/gradients; not an actually rounded FP32 optimizer commit.',
                                  'fixed_vs_fit':'Fixed e25/e50 paired values above are distinct estimands from the first-common-CE-bin fit matching in state_analysis.',
                                  'scientific_vs_execution':'Execution receipt and exit 0 never imply HVP convergence, correction agreement, or a causal mechanism.'})
    output = Path(__file__).parent/args.output_prefix
    for suffix in ('.json', '.md'):
        assert not output.with_suffix(suffix).exists(), 'Use a new output prefix to preserve previous audit'
    output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    text = '# Independent offline integrity audit\n\nStatus: '+report['status']+'; this is the actual snapshot, not an inferred completion.\n\n'
    text += 'Formal observed/planned: '+', '.join('%s %d/%d' % (k, observed['formal_'+k], v) for k, v in COUNTS.items())+'.\n\n'
    text += 'Failures: '+str(len(failures))+'. Full checks, raw paired values, original gates, receipts and input hashes are in the sibling JSON.\n'
    output.with_suffix('.md').write_text(text)
    print(json.dumps({'status': report['status'], 'complete': complete, 'observed': dict(observed), 'failures': failures, 'checks': dict(checks), 'audit': str(output.with_suffix('.json'))}))
    return 1 if failures else 2 if args.require_complete and not complete else 0


if __name__ == '__main__':
    raise SystemExit(main())
