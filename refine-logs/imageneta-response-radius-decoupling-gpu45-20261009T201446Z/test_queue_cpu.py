"""Negative allowlist and real campaign manifest acceptance checks, CPU only."""
import copy,json,tempfile,unittest
from pathlib import Path
from common import HERE,load_suite,sha256,verify_config,verify_source
from collect_results import external_seed2_ready

class QueueTests(unittest.TestCase):
    def test_real_manifest_and_budget(self):
        s=load_suite();self.assertEqual(s['allowed_gpus'],[4,5]);self.assertEqual(s['max_parallel'],2)
        sm=[j for j in s['jobs'] if j['kind']=='training' and j['smoke']];fo=[j for j in s['jobs'] if j['kind']=='training' and not j['smoke']]
        self.assertEqual(len(sm),4);self.assertEqual(len(fo),3)
        self.assertEqual({j['radius_variant'] for j in fo},{'fixed_inner','fixed_outer','fixed_both'})
        for j in fo:
            self.assertEqual(j['depends_on'],['radius_smoke_pairing']);self.assertEqual(j['expected_steps'],2550)
            self.assertFalse(any(a in j['command'] for a in ('--max-tasks','--max-epochs','--max-batches')))
        gate=next(j for j in s['jobs'] if j['kind']=='pairing');self.assertEqual(set(gate['depends_on']),{j['id'] for j in sm})
    def test_unauthorized_gpu_and_cycles_rejected(self):
        s=load_suite()
        for mutation in ([0],[1],[6],[7],[4,5,6],[4,4]):
            r=copy.deepcopy(s);r['allowed_gpus']=mutation
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'bad.json';p.write_text(json.dumps(r))
                with self.assertRaises(ValueError):load_suite(p)
        r=copy.deepcopy(s);r['jobs'][0]['depends_on']=[r['jobs'][-1]['id']]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text(json.dumps(r))
            with self.assertRaises(ValueError):load_suite(p)
    def test_configs_outputs_unique(self):
        outputs=[]
        for j in load_suite()['jobs']:
            if j['kind']!='training':continue
            _,c,o=verify_config(j);self.assertEqual(c['history_weight'],0.);self.assertEqual(c['response_method'],'current_g2')
            self.assertFalse(any(o==p or o.is_relative_to(p) or p.is_relative_to(o) for p in outputs));outputs.append(o)
    def test_source_and_frozen_reuse(self):
        self.assertEqual(len(verify_source(load_suite())['source_files']),190);self.assertTrue(external_seed2_ready())

if __name__=='__main__':unittest.main()
