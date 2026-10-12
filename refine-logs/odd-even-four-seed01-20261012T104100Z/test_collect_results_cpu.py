"""No GPU, no training: completion gating and three-seed statistics."""
import copy
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import collect_results as cr


def synthetic_matrix(shift=0):
    m=[[90-3*i-j+shift for j in range(i+1)] for i in range(10)]
    result=dict(matrix=m,AAA_task_macro=sum(sum(row)/len(row) for row in m)/10,
        FAA_task_macro=sum(m[-1])/10,LA=sum(m[i][i] for i in range(10))/10,
        BWT=sum(m[-1][j]-m[j][j] for j in range(9))/9,
        forgetting=sum(max(m[i][j] for i in range(j,10))-m[-1][j] for j in range(9))/9)
    return result


def synthetic_rows():
    return [dict(direction=direction,seed=seed,status='validated_complete',
                 **{m:10*index+seed for m in cr.METRICS})
            for index,(_,direction) in enumerate(cr.CONDITIONS) for seed in cr.SEEDS]


class CollectorTests(unittest.TestCase):
    def test_recompute_complete_matrix(self):
        values=cr.metric_values(synthetic_matrix())
        self.assertEqual(values['FAA'],58.5)
        self.assertEqual(values['forgetting'],15)

    def test_reject_partial_matrix(self):
        data=synthetic_matrix();data['matrix'].pop()
        with self.assertRaises(ValueError):cr.metric_values(data)

    def test_reject_nonfinite_accuracy(self):
        data=synthetic_matrix();data['matrix'][0][0]=float('nan')
        with self.assertRaises(ValueError):cr.metric_values(data)

    def test_reject_inconsistent_summary_metric(self):
        data=synthetic_matrix();data['AAA_task_macro']+=.1
        with self.assertRaises(ValueError):cr.metric_values(data)

    def test_incomplete_means_withheld(self):
        rows=[r for r in synthetic_rows() if r['seed']==2]
        groups,pairs=cr.aggregate(rows)
        self.assertTrue(all(g['n_complete']==1 and g['expected_n']==3 for g in groups))
        self.assertTrue(all(g['AAA_mean'] is None and g['AAA_sample_sd'] is None for g in groups))
        self.assertTrue(all(p['paired_mean'] is None and p['wins'] is None for p in pairs))
        self.assertEqual(groups[0]['AAA_seed2'],2)
        self.assertIsNone(groups[0]['AAA_seed0'])

    def test_complete_mean_sample_sd(self):
        groups,_=cr.aggregate(synthetic_rows())
        self.assertEqual(groups[0]['AAA_mean'],1)
        self.assertEqual(groups[0]['AAA_sample_sd'],1)
        self.assertTrue(all(g['n_complete']==3 and g['eligible_for_three_seed_comparison'] for g in groups))

    def test_matched_seed_pairing_and_wins(self):
        _,pairs=cr.aggregate(synthetic_rows())
        aaa=next(p for p in pairs if p['direction']=='central' and p['baseline']=='g1_plus' and p['metric']=='AAA')
        self.assertEqual([aaa[f'delta_seed{s}'] for s in cr.SEEDS],[20,20,20])
        self.assertEqual((aaa['paired_mean'],aaa['paired_sample_sd'],aaa['wins']),(20,0,3))
        forget=next(p for p in pairs if p['direction']=='central' and p['baseline']=='g0' and p['metric']=='forgetting')
        self.assertEqual(forget['wins'],3)

    def test_pair_not_complete_when_missing_one_side(self):
        rows=[r for r in synthetic_rows() if (r['direction'],r['seed'])!=('g0',1)]
        _,pairs=cr.aggregate(rows)
        row=next(p for p in pairs if p['direction']=='central' and p['baseline']=='g0' and p['metric']=='AAA')
        self.assertEqual(row['n_complete'],2)
        self.assertIsNone(row['paired_mean'])
        self.assertIsNone(row['delta_seed1'])

    def test_reject_duplicate_seed(self):
        rows=synthetic_rows()
        with self.assertRaises(ValueError):cr.aggregate(rows+[rows[0]])

    def test_unverified_rows_excluded(self):
        rows=synthetic_rows()
        for row in rows:
            if row['seed']==1:row['status']='completed_unverified'
        groups,_=cr.aggregate(rows)
        self.assertTrue(all(g['n_complete']==2 and g['AAA_mean'] is None for g in groups))

    def test_parent_original_receipts_valid(self):
        audit=cr.read(cr.H/'PARENT_REUSE_AUDIT.json')
        rows=cr.parent_rows(audit)
        self.assertEqual(len(rows),4)
        self.assertAlmostEqual(rows[0]['AAA'],63.10206547619047)
        self.assertEqual({r['seed'] for r in rows},{2})

    def test_parent_binding_tamper_rejected(self):
        audit=copy.deepcopy(cr.read(cr.H/'PARENT_REUSE_AUDIT.json'))
        first=next(iter(audit['bound_files']))
        audit['bound_files'][first]='0'*64
        with self.assertRaises(ValueError):cr.parent_rows(audit)

    def test_parent_receipt_mismatch_rejected(self):
        audit=cr.read(cr.H/'PARENT_REUSE_AUDIT.json');job=audit['jobs'][0]
        actual_read=cr.read
        def tampered(path):
            result=actual_read(path)
            if Path(path).name=='completed.json':result['matrix_sha256']='0'*64
            return result
        with patch.object(cr,'read',tampered):
            with self.assertRaises(ValueError):cr.verify_receipt(cr.PARENT,job,2)

    def test_nonzero_exit_rejected(self):
        audit=cr.read(cr.H/'PARENT_REUSE_AUDIT.json');job=audit['jobs'][0]
        actual_read=cr.read
        def tampered(path):
            result=actual_read(path)
            if Path(path).name=='exit.json':result['returncode']=1
            return result
        with patch.object(cr,'read',tampered):
            with self.assertRaises(ValueError):cr.verify_receipt(cr.PARENT,job,2)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CollectorTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    cr.atomic_json(cr.H/'COLLECTOR_CPU_GATE.json',dict(status='passed' if result.wasSuccessful() else 'failed',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        cpu_only=True,gpu_calls=0,training_runs=0,generated_at=cr.now(),
        python_executable=sys.executable,python_version=sys.version,
        collector_sha256=cr.sha256(cr.H/'collect_results.py'),
        tests_sha256=cr.sha256(Path(__file__)),
        parent_audit_sha256=cr.sha256(cr.H/'PARENT_REUSE_AUDIT.json')))
    raise SystemExit(0 if result.wasSuccessful() else 1)
