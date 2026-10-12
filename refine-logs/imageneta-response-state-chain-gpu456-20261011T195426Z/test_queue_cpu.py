import unittest,copy,tempfile,json
from pathlib import Path
from common import load_suite
from aggregate_helpers import fit_match,mean_sd
H=Path(__file__).resolve().parent
class ScientificAdmission(unittest.TestCase):
 def bad(self,s):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'suite.json';p.write_text(json.dumps(s))
   with self.assertRaises(ValueError):load_suite(p)
 def test_external_gpu_forbidden(self):
  for i in [0,1,2,3,7]:
   s=copy.deepcopy(load_suite());s['allowed_gpus']=[4,5,i];self.bad(s)
 def test_manifest_exact_six_branch_budgets(self):
  s=load_suite();formal=[j for j in s['jobs'] if j['formal']]
  self.assertEqual(len(formal),6);self.assertEqual(sum(j['expected_steps'] for j in formal),2400);self.assertEqual(sum(j['expected_gradient_reads'] for j in formal),4200)
  self.assertEqual({(j['seed'],j['chain_branch']) for j in formal},{(i,b) for i in range(3) for b in ['S','R']})
  for j in formal:self.assertIn('smoke_diagnostics_s2',j['depends_on']);self.assertIn('smoke_pairing_s2',j['depends_on'])
 def test_dependencies_cycle_rejected(self):
  s=copy.deepcopy(load_suite());s['jobs'][0]['depends_on']=[s['jobs'][-1]['id']];self.bad(s)
 def test_root_cuda_binding_cannot_override(self):
  s=copy.deepcopy(load_suite());s['environment']['CUDA_VISIBLE_DEVICES']='0';self.bad(s)
 def test_first_fit_bin_never_selects_best_test(self):
  rows=[]
  for b,values in [('S',[(1,.128,30),(2,.126,99)]),('R',[(1,.129,40),(2,.127,100)])]:
   for epoch,ce,acc in values:rows.append(dict(seed=2,branch=b,epoch=epoch,clean_training_ce=ce,old_accuracy=acc,new_accuracy=20,new_conditional20_ce=.3))
  match=fit_match(rows);self.assertEqual(len(match),1);self.assertEqual(match[0]['old_accuracy_R_minus_S'],10);self.assertEqual(match[0]['S_first_epoch'],1);self.assertEqual(match[0]['R_first_epoch'],1)
  self.assertEqual(fit_match(rows[:2]),[])
 def test_sample_sd_and_paired_unit(self):
  s=mean_sd([1,2,3]);self.assertEqual(s,{'n':3,'mean':2,'sample_sd':1})
if __name__=='__main__':unittest.main()
