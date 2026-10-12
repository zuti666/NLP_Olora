"""CPU acceptance of exact archived-state loading and read-only diagnostics."""
import copy
import json
import unittest
import torch

from mechanism_diagnostics import audit_same_state, overlay_observation, trainable_sha
from response_hvp import State, rng_get, rng_equal
from test_direction_cpu import make_case


class MechanismAcceptance(unittest.TestCase):
    def test_overlay_schema_identity_exact_values_and_atomic_rejection(self):
        model,params,_,_,_,_ = make_case()
        identities = {n:id(p) for n,p in model.named_parameters()}
        frozen = model.frozen.clone()
        payload = {"task":4,"point":"epoch1","config":{},"rng":rng_get(),
                   "trainable_state":{n:p.detach().clone()+.2 for n,p in model.named_parameters() if p.requires_grad}}
        result = overlay_observation(model,payload,4,"epoch1")
        self.assertTrue(result["names_shapes_dtypes_exact"])
        self.assertTrue(all(id(p)==identities[n] for n,p in model.named_parameters()))
        self.assertTrue(torch.equal(model.frozen,frozen))
        expected_hash = trainable_sha(model)
        for malformed in ("name","shape","dtype","task","point","nonfinite"):
            bad = copy.deepcopy(payload)
            key = next(iter(bad["trainable_state"]))
            if malformed=="name":bad["trainable_state"]["unexpected"] = torch.ones(1)
            elif malformed=="shape":bad["trainable_state"][key] = torch.ones(1,dtype=torch.float64)
            elif malformed=="dtype":bad["trainable_state"][key] = bad["trainable_state"][key].float()
            elif malformed=="task":bad["task"] = 0
            elif malformed=="point":bad["point"] = "end"
            else:bad["trainable_state"][key].view(-1)[0] = float("nan")
            with self.assertRaises(ValueError):overlay_observation(model,bad,4,"epoch1")
            self.assertEqual(trainable_sha(model),expected_hash)

    def test_four_cell_exact_common_inputs_state_rng_and_zero_updates(self):
        model,_,geom,_,_,closure = make_case()
        model.train()
        live = State(model)
        before = trainable_sha(model)
        report = audit_same_state(model,None,None,closure,geom,backend="native")
        live.check()
        self.assertEqual(before,trainable_sha(model))
        self.assertEqual(report["status"],"passed")
        self.assertEqual(report["base_ordinary_gradient_calls"],16)
        self.assertEqual(report["optimizer_updates"],0)
        self.assertEqual(report["fisher_calls"],0)
        self.assertEqual(report["weight_merge_calls"],0)
        self.assertEqual(set(report["branches"]),{"P","D","C","R"})
        self.assertTrue(all(all(v.values()) for v in report["base_gradient_identities"].values()))
        self.assertLess(report["cross_cell_contrasts"]["P_vs_C"]["O_g2"]["relative_error"],1e-12)
        self.assertGreater(report["cross_cell_contrasts"]["P_vs_C"]["A"]["relative_error"],1e-3)
        json.dumps(report,allow_nan=False)

    def test_response_reading_common_math_endpoints_and_zero_updates(self):
        model,_,geom,_,_,closure = make_case()
        live = State(model)
        report = audit_same_state(model,None,None,closure,geom,backend="math",responses=True)
        live.check()
        self.assertEqual(report["status"],"passed")
        self.assertEqual(report["base_ordinary_gradient_calls"],4)
        self.assertEqual(report["additional_ordinary_reference_gradient_calls"],1)
        self.assertEqual(report["hvp_calls"],2)
        self.assertEqual(report["optimizer_updates"],0)
        self.assertEqual(set(report["responses"]),{"Sec","Hstart","Hmid"})
        for label in ("Hstart","Hmid"):
            result = report["responses"][label]
            self.assertLess(result["response_vs_Sec"]["relative_error"],1e-12)
            self.assertAlmostEqual(result["response_vs_Sec"]["cosine"],1.,places=12)
            self.assertAlmostEqual(result["response_vs_Sec"]["norm_ratio"],1.,places=12)
        json.dumps(report,allow_nan=False)

    def test_failure_is_transactional(self):
        model,_,geom,_,_,closure = make_case()
        live = State(model)
        count = 0
        def failure(x,y):
            nonlocal count
            count += 1
            if count==7:
                model.delta_w_q_new_grad = torch.ones(3)
                raise RuntimeError("intentional diagnostic failure")
            return closure(x,y)
        with self.assertRaisesRegex(RuntimeError,"intentional"):
            audit_same_state(model,None,None,failure,geom)
        live.check()


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
