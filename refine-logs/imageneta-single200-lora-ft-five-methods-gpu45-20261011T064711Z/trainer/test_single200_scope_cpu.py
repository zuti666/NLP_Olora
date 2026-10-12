"""Scope tests catch accidental LoRA retention/freezing and RNG reinitialization."""
import argparse
import copy
import json
from pathlib import Path
import unittest
import torch
from scope_helper import FullFTClassifier,full_ft_from_lora_origin,parameter_hash,scope_receipt
from train_single200 import resolve_config


class TinyViT(torch.nn.Module):
    def __init__(self):
        super().__init__();self.proj=torch.nn.Linear(3,768)
    def forward(self,x):return self.proj(x)


class SimpleHead(torch.nn.Linear):
    def forward(self,x):return dict(logits=super().forward(x))


class ScopeTests(unittest.TestCase):
    def test_plain_ft_copies_pretrained_and_classifier_without_rng_draws(self):
        origin=torch.nn.Module();origin.backbone=torch.nn.Module()
        origin.backbone.base_vit=TinyViT();origin.fc=SimpleHead(768,200)
        for p in origin.backbone.parameters():p.requires_grad_(False)
        old=torch.get_rng_state().clone();model=full_ft_from_lora_origin(origin)
        self.assertTrue(torch.equal(old,torch.get_rng_state()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.backbone.parameters(),origin.backbone.base_vit.parameters())))
        self.assertTrue(torch.equal(model.fc.weight,origin.fc.weight))
        self.assertNotEqual(model.fc.weight.data_ptr(),origin.fc.weight.data_ptr())
        receipt=scope_receipt(model,'full_ft')
        self.assertEqual(receipt['frozen_parameter_count'],0)
        self.assertTrue(receipt['no_lora_modules'])

    def test_ft_rejects_any_frozen_tensor(self):
        model=FullFTClassifier(TinyViT(),SimpleHead(768,200));model.backbone.proj.weight.requires_grad_(False)
        with self.assertRaises(AssertionError):scope_receipt(model,'full_ft')

    def test_full200_head_required(self):
        model=FullFTClassifier(TinyViT(),SimpleHead(768,20))
        with self.assertRaises(AssertionError):scope_receipt(model,'full_ft')

    def test_config_explicit_scope_lr_and_forbidden_modules(self):
        root=Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'configs/smoke_full_ft_g2_s2.json').read_text())
        cli=argparse.Namespace(max_epochs=1,max_batches=2)
        self.assertEqual(resolve_config(cfg,cli)['ordinary_gradient_calls'],3)
        bad=copy.deepcopy(cfg);bad['lr']=.04
        with self.assertRaises(ValueError):resolve_config(bad,cli)
        bad=copy.deepcopy(cfg);bad['in_task_weight_merge']=True
        with self.assertRaises(ValueError):resolve_config(bad,cli)
        bad=copy.deepcopy(cfg);bad['class_shuffle']=True
        with self.assertRaises(ValueError):resolve_config(bad,cli)

    def test_model_hash_tracks_weight_changes(self):
        model=FullFTClassifier(TinyViT(),SimpleHead(768,200));old=parameter_hash(model)
        with torch.no_grad():model.fc.bias.add_(1.)
        self.assertNotEqual(old,parameter_hash(model))


if __name__=='__main__':unittest.main()
