"""CPU-only admission, phase-budget and dependency checks; no CUDA imports."""
import copy,json,tempfile,unittest
from pathlib import Path
from common import HERE as H,load_suite
from run_queue import verify_output_path,verify_implementation
from validate_job import active,READS
class QueueTests(unittest.TestCase):
    def setUp(self):self.suite=json.loads((H/'suite.json').read_text())
    def test_registered_success_dependencies_and_counts(self):
        jobs=self.suite['jobs'];smokes=[j for j in jobs if j['kind']=='training' and j['smoke']];formals=[j for j in jobs if j['kind']=='training' and not j['smoke']];pair=[j for j in jobs if j['kind']=='pairing']
        self.assertEqual((len(smokes),len(formals),len(pair)),(6,18,1));self.assertEqual(self.suite['allowed_gpus'],[4,5,6]);self.assertEqual(self.suite['max_parallel'],3)
        required={j['id'] for j in smokes}|{pair[0]['id']}
        for j in formals:self.assertEqual(set(j['depends_on']),required)
        self.assertEqual(set(pair[0]['depends_on']),{j['id'] for j in smokes})
        self.assertEqual({(j['condition'],j['seed']) for j in formals},{(c,s) for c in ('sgd','sam','gam_residual','sf_g2','sf_g2_then_sgd','sgd_then_sf_g2') for s in (0,1,2)})
    def test_method_boundary_and_native_probe_counts(self):
        for j in self.suite['jobs']:
            if j['kind']!='training':continue
            actual=sum(j['expected_batches_per_epoch'][t]*READS[active(j['condition'],t,e)] for t in (0,1) for e in range(1,21))
            self.assertEqual(actual,j['expected_gradient_reads']);self.assertTrue(all(active(j['condition'],1,e)=='sgd' for e in range(1,21)))
            self.assertEqual(j['expected_steps'],40 if j['smoke'] else 340)
        self.assertEqual(active('sf_g2_then_sgd',0,10),'sf_g2');self.assertEqual(active('sf_g2_then_sgd',0,11),'sgd')
        self.assertEqual(active('sgd_then_sf_g2',0,10),'sgd');self.assertEqual(active('sgd_then_sf_g2',0,11),'sf_g2')
    def test_reject_unauthorized_gpu_and_parallelism(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'suite.json'
            for gpu in (0,1,2,3):
                invalid=copy.deepcopy(self.suite);invalid['allowed_gpus']=[gpu];invalid['max_parallel']=1;p.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):load_suite(p)
            invalid=copy.deepcopy(self.suite);invalid['max_parallel']=4;p.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):load_suite(p)
    def test_outputs_and_cli_cannot_escape_registered_campaign(self):
        for j in self.suite['jobs']:verify_output_path(self.suite,j)
        invalid=copy.deepcopy(self.suite['jobs'][-1]);invalid['output_dir']='/tmp/other_campaign'
        with self.assertRaises(RuntimeError):verify_output_path(self.suite,invalid)
        invalid=copy.deepcopy(self.suite['jobs'][0]);invalid['command'][invalid['command'].index('--output-dir')+1]='/tmp/different_cli_output'
        with self.assertRaises(RuntimeError):verify_output_path(self.suite,invalid)
    def test_actual_launch_requires_independent_and_fresh_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            suite=copy.deepcopy(self.suite)
            for name in ('implementation_acceptance_file','independent_implementation_review_file','fresh_doc_check'):
                p=Path(d)/(name+'.json');p.write_text(json.dumps(dict(status='passed',checks=dict(done=True),evidence_hashes={})));suite[name]=str(p)
            self.assertTrue(verify_implementation(suite))
            p=Path(suite['fresh_doc_check']);p.write_text(json.dumps(dict(status='failed',checks=dict(done=False),evidence_hashes={})))
            with self.assertRaises(RuntimeError):verify_implementation(suite)
if __name__=='__main__':unittest.main()
