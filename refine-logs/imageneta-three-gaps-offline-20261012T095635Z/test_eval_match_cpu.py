#!/usr/bin/env python3
"""Nontrivial CPU invariants for conditional evaluation and finite BA matching."""
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent/'imageneta-response-state-chain-gpu456-20261011T195426Z'
sys.path.insert(0,str(CAMPAIGN/'trainer'))
sys.path.insert(0,str(CAMPAIGN/'diagnostics'))
import torch
from checkpoint_eval import group_metrics,assert_historical_agreement
from effective_match import vector_part_comparisons
from effective_step import EffectiveStep,solve_positive_scale,StepPolynomial
from offline_core import finite_lora_parts,comparison,matched_effective_diagnostic


class EvaluationMatchingTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(513)

    def test_old_joint_errors_partition_and_conditional_competition(self):
        # correct; wrong same group; conditional correct beaten by new;
        # conditional incorrect AND beaten by new. These are distinct cases.
        logits=torch.full((4,40),-10.)
        labels=torch.tensor([0,1,2,3])
        logits[0,0]=3
        logits[1,0]=3;logits[1,1]=2
        logits[2,2]=3;logits[2,20]=4
        logits[3,0]=3;logits[3,3]=2;logits[3,21]=4
        metrics,arrays=group_metrics(logits,labels,'old')
        self.assertEqual(metrics['joint40']['errors'],3)
        self.assertEqual(metrics['within_group_error_count'],1)
        self.assertEqual(metrics['cross_group_error_count'],2)
        self.assertEqual(metrics['conditional_correct_lost_to_other_group_count'],1)
        self.assertEqual(metrics['conditional_error_and_cross_group_count'],1)
        self.assertEqual(metrics['conditional20']['accuracy'],50.)
        self.assertAlmostEqual(metrics['joint40']['ce'],metrics['conditional20']['ce']+metrics['ce_competition_penalty'],places=6)
        self.assertEqual(arrays['logits'].shape,(4,40))

    def test_new_group_uses_global_original_labels_and_conditional_slice(self):
        logits=torch.full((3,40),-8.)
        labels=torch.tensor([20,21,22])
        logits[0,20]=2
        logits[1,21]=2;logits[1,0]=4
        logits[2,23]=3;logits[2,22]=2
        metrics,arrays=group_metrics(logits,labels,'new')
        self.assertEqual(metrics['cross_group_error_count'],1)
        self.assertEqual(metrics['within_group_error_count'],1)
        self.assertEqual(arrays['prediction_conditional20'].tolist(),[20,21,23])
        self.assertLess(metrics['ce_partition_max_abs_error'],3e-6)
        self.assertTrue((arrays['true_class_probability_conditional20']>=arrays['true_class_probability_joint40']).all())

    def test_historical_agreement_rejects_wrong_classification_scope(self):
        reference=dict(accuracy=80.,ce=.5,samples=254)
        self.assertTrue(assert_historical_agreement(reference,reference,'identical')['accepted'])
        with self.assertRaises(AssertionError):
            assert_historical_agreement(dict(accuracy=81.,ce=.5,samples=254),reference,'wrongscope')
        with self.assertRaises(AssertionError):
            assert_historical_agreement(dict(accuracy=80.,ce=.501,samples=254),reference,'wrongloss')

    def geometry(self):
        a=torch.nn.Parameter(torch.randn(2,5,dtype=torch.float64))
        b=torch.nn.Parameter(torch.randn(4,2,dtype=torch.float64))
        fc=torch.nn.Parameter(torch.randn(3,4,dtype=torch.float64))
        bias=torch.nn.Parameter(torch.randn(3,dtype=torch.float64))
        return EffectiveStep([(a,b)],[fc,bias])

    def test_finite_BA_plus_full_head_match_root_and_identities(self):
        geometry=self.geometry()
        sam=[torch.randn_like(p) for p in geometry.params]
        sf=[torch.randn_like(p) for p in geometry.params]
        sam_parts,ds=finite_lora_parts(geometry,sam,.02)
        root=solve_positive_scale(geometry.polynomial(sf),sam_parts['effective_norm'],initial_scale=.02)
        sf_parts,dm=finite_lora_parts(geometry,sf,root['scale'])
        self.assertLess(abs(sf_parts['effective_norm']/sam_parts['effective_norm']-1),1e-9)
        self.assertLess(sf_parts['layers'][0]['effective_identity_error'],1e-14)
        self.assertLess(sf_parts['layers'][0]['delta_Q_identity_error'],1e-14)
        self.assertAlmostEqual(sf_parts['effective_norm']**2,sf_parts['lora_effective_norm']**2+sf_parts['fc_norm']**2,places=12)
        self.assertTrue(root['smallest_positive_root'])

    def test_unavailable_root_and_zero_direction_do_not_get_fallback(self):
        with self.assertRaises(ValueError):
            solve_positive_scale(StepPolynomial(0.,0.,0.),1.)
        with self.assertRaises(ValueError):
            solve_positive_scale(StepPolynomial(1.,0.,0.),20.,max_scale=10.)
        # Nonmonotonic effective norm |s-s²| has two small roots; first is chosen.
        root=solve_positive_scale(StepPolynomial(1.,-2.,1.),.1,max_scale=2.)
        self.assertAlmostEqual(root['scale'],(1-(1-.4)**.5)/2,places=9)

    def test_direction_components_are_distinct_and_include_bias(self):
        sam=(torch.tensor([[1.,0.]],dtype=torch.float64),torch.tensor([1.,0.],dtype=torch.float64),torch.tensor([1.],dtype=torch.float64))
        native=(torch.tensor([[0.,1.]],dtype=torch.float64),torch.tensor([1.,0.],dtype=torch.float64),torch.tensor([1.],dtype=torch.float64))
        record=vector_part_comparisons(native,native,sam,1,comparison,3**.5)
        self.assertAlmostEqual(record['lora_only']['native']['cosine'],0.)
        self.assertAlmostEqual(record['fc_only']['native']['cosine'],1.)
        self.assertAlmostEqual(record['all']['native']['cosine'],2/3)

    def test_existing_match_runs_without_HVP_and_keeps_parameters(self):
        geometry=self.geometry();saved=tuple(p.detach().clone() for p in geometry.params)
        candidate=dict(status='accepted',g1=tuple(torch.randn_like(p) for p in geometry.params),
            g2=tuple(torch.randn_like(p) for p in geometry.params),inner_radius=.04)
        class NoHVP:
            def __init__(self):self.geometry=geometry;self.calls=0
            def native_candidates(self,*args):self.calls+=1;return candidate
            def close(self):return dict(parameters_exact=all(torch.equal(p,v) for p,v in zip(geometry.params,saved)))
            def hvp(self,*args):raise AssertionError('This method must never be called')
        engine=NoHVP()
        result=matched_effective_diagnostic(engine,torch.zeros(128,1),torch.zeros(128,dtype=torch.long),.02,solve_positive_scale)
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(result['submitted_training_updates'],0)
        self.assertTrue(result['restoration']['parameters_exact'])
        self.assertEqual(engine.calls,1)
        self.assertLess(result['norm_match_relative_error'],1e-9)


if __name__=='__main__':unittest.main(verbosity=2)
