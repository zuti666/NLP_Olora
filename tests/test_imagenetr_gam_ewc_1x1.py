"""CPU scientific checks for native four-probe GAM with additive clean EWC.

Run with an environment containing CopyFlatCV dependencies, for example:
  CUDA_VISIBLE_DEVICES='' /data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python \
      -m unittest discover -s tests -p test_imagenetr_gam_ewc_1x1.py -v

These tests use small deterministic models, the production additive_step, and
the real upstream GAM. No pretrained weights, data download, or GPU is needed.
"""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("COPYFLATCV_SOURCE", "/data/140-1/users/liying/CopyFlatCV"))
sys.path.insert(0, str(SOURCE))
_SPEC = importlib.util.spec_from_file_location(
    "imagenetr_native_gam_test_learner", ROOT / "scripts/imagenetr_gam_ewc_1x1/learner.py"
)
production = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(production)

from backbone.lora import LoRA_ViT_timm
from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from timm.models.vision_transformer import VisionTransformer
from utils.inc_net import IncrementalNet


class TinyClassifier(nn.Module):
    def __init__(self, dropout=0.0, fail_on=None):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(
            [[0.31, -0.22, 0.14], [-0.17, 0.28, 0.39], [0.12, 0.41, -0.25]],
            dtype=torch.float64,
        ))
        self.dropout = nn.Dropout(dropout)
        self.probe_weights = []
        self.probe_inputs = []
        self.fail_on = fail_on

    def forward(self, inputs):
        values = self.dropout(inputs)
        self.probe_weights.append(self.weight.detach().clone())
        self.probe_inputs.append(values.detach().clone())
        if self.fail_on == len(self.probe_weights):
            raise RuntimeError("deliberate probe failure")
        return {"logits": F.linear(values, self.weight)}


class TinySourceLoRANet(nn.Module):
    """Real source Q/V wrapper and classifier lifecycle, on a small ViT."""
    update_fc = IncrementalNet.update_fc
    generate_fc = IncrementalNet.generate_fc
    save_fc = IncrementalNet.save_fc
    load_fc = IncrementalNet.load_fc

    def __init__(self):
        super().__init__()
        vit = VisionTransformer(img_size=8, patch_size=4, in_chans=3,
                                num_classes=0, embed_dim=12, depth=1,
                                num_heads=3, mlp_ratio=2, drop_rate=0,
                                attn_drop_rate=0, drop_path_rate=0)
        self.backbone = LoRA_ViT_timm(vit, r=2, increment=2, max_prev_tasks=2)
        self.feature_dim = 12
        self.fc = None

    def forward(self, images):
        return self.fc(self.backbone(images))

    def effective_updates(self):
        return [b.weight @ a.weight for a, b in zip(self.backbone.w_As, self.backbone.w_Bs)]


def task_loss(logits, targets, offset):
    local_logits = logits[:, offset:]
    local_targets = targets - offset
    return F.cross_entropy(local_logits, local_targets), local_logits, local_targets


def native_gam(model, lr=0.037, rho=0.06, norm_rho=0.11, gamma=0.13):
    params = list(model.parameters())
    base = torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0)
    args = SimpleNamespace(grad_beta_1=1.0, grad_beta_2=1.0, grad_beta_3=1.0,
                           grad_gamma=gamma, use_sgv=False)
    return production.GAM(params, base_optimizer=base, model=model,
                          grad_rho=rho, grad_norm_rho=norm_rho,
                          adaptive=False, args=args)


def independent_gam_reference(weight, inputs, targets, rho, norm_rho, gamma, eps=1e-12):
    """Four CE gradients and the native GAM vector formula, independent of step."""
    def gradient(value):
        value = value.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(F.linear(inputs, value), targets)
        return torch.autograd.grad(loss, value)[0].detach()

    g0 = gradient(weight)
    theta1 = weight + rho * g0 / (torch.linalg.vector_norm(g0) + eps)
    g1 = gradient(theta1)
    difference = g1 - g0
    theta2 = weight + norm_rho * difference / (torch.linalg.vector_norm(difference) + eps)
    g2 = gradient(theta2)
    theta3 = theta2 + rho * g2 / (torch.linalg.vector_norm(g2) + eps)
    g3 = gradient(theta3)
    minus, plus = g0 + g2, g1 + g3
    nm, np = torch.linalg.vector_norm(minus), torch.linalg.vector_norm(plus)
    cosine = (minus * plus).sum() / (nm * np + eps)
    result = plus - gamma * (minus - cosine * nm * plus / (np + eps))
    return [weight, theta1, theta2, theta3], result, g0


class NativeGAMEWCSanity(unittest.TestCase):
    def setUp(self):
        self.cuda_patch = patch.object(torch.cuda, "is_available", return_value=False)
        self.cuda_patch.start()
        self.addCleanup(self.cuda_patch.stop)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.x = torch.tensor([[0.7, -1.2, 0.4], [1.1, 0.3, -0.8],
                               [-0.2, 0.9, 1.3], [-1.0, -0.5, 0.6]], dtype=torch.float64)
        self.y = torch.tensor([0, 2, 1, 2])

    def assertTensorClose(self, actual, expected, atol=2e-12, rtol=2e-11):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_native_four_ce_probes_match_independent_formula(self):
        model = TinyClassifier()
        initial = model.weight.detach().clone()
        expected_points, direction, _ = independent_gam_reference(
            initial, self.x, self.y, rho=0.06, norm_rho=0.11, gamma=0.13)
        opt = native_gam(model)
        logits, reported_loss, stats = production.additive_step(
            model, opt, self.x, self.y, 0, lambda: initial.new_zeros(()), task_loss)
        self.assertEqual(stats['ce_backward_count'], 4)
        self.assertEqual(len(model.probe_weights), 4)
        for actual, expected in zip(model.probe_weights, expected_points):
            self.assertTensorClose(actual, expected)
        self.assertTensorClose(model.weight, initial - 0.037 * direction)
        self.assertTensorClose(logits, F.linear(self.x, initial))
        self.assertAlmostEqual(reported_loss,
                               F.cross_entropy(F.linear(self.x, initial), self.y).item(), places=13)
        self.assertIsNone(opt._direction_projector)

    def test_ewc_changes_only_final_update_not_any_gam_probe(self):
        unprotected, protected = TinyClassifier(), TinyClassifier()
        initial = protected.weight.detach().clone()
        reference = initial - torch.linspace(0.03, 0.17, initial.numel(), dtype=initial.dtype).reshape_as(initial)
        fisher = torch.linspace(0.4, 1.6, initial.numel(), dtype=initial.dtype).reshape_as(initial)
        lam, observed_penalty_points = 2.7, []

        def penalty():
            observed_penalty_points.append(protected.weight.detach().clone())
            return 0.5 * lam * (fisher * (protected.weight - reference).square()).sum()

        production.additive_step(unprotected, native_gam(unprotected), self.x, self.y, 0,
                                 lambda: initial.new_zeros(()), task_loss)
        _, _, stats = production.additive_step(protected, native_gam(protected), self.x, self.y, 0,
                                               penalty, task_loss)
        self.assertEqual(len(observed_penalty_points), 1)
        self.assertTrue(torch.equal(observed_penalty_points[0], initial))
        self.assertEqual(len(protected.probe_weights), 4)
        for a, b in zip(unprotected.probe_weights, protected.probe_weights):
            self.assertTensorClose(a, b)
        clean_ewc_gradient = lam * fisher * (initial - reference)
        self.assertTensorClose(protected.weight - unprotected.weight, -0.037 * clean_ewc_gradient)
        self.assertAlmostEqual(stats['ewc_penalty'],
                               (0.5 * lam * (fisher * (initial-reference).square()).sum()).item(), places=13)

    def test_sgd_matches_direct_clean_ce_plus_ewc_with_local_head(self):
        model = TinyClassifier()
        initial = model.weight.detach().clone()
        reference = initial - 0.09
        expected_weight = initial.clone().requires_grad_(True)
        targets = torch.tensor([1, 2, 1, 2])
        total = F.cross_entropy(F.linear(self.x, expected_weight)[:, 1:], targets-1)
        total = total + 0.5 * 1.8 * (expected_weight-reference).square().sum()
        expected_gradient = torch.autograd.grad(total, expected_weight)[0]
        opt = torch.optim.SGD(model.parameters(), lr=0.041, momentum=0, weight_decay=0)
        _, reported_loss, stats = production.additive_step(
            model, opt, self.x, targets, 1,
            lambda: 0.5 * 1.8 * (model.weight-reference).square().sum(), task_loss)
        self.assertEqual(stats['ce_backward_count'], 1)
        self.assertEqual(len(model.probe_weights), 1)
        self.assertTensorClose(model.weight, initial-0.041*expected_gradient)
        self.assertAlmostEqual(reported_loss, total.item(), places=13)

    def test_probe_rng_replayed_and_global_rng_advances_once(self):
        model = TinyClassifier(dropout=0.35)
        initial = model.weight.detach().clone()
        torch.manual_seed(421)
        start_rng = torch.get_rng_state().clone()
        clean = model(self.x)['logits'].detach()
        expected_post_clean = torch.get_rng_state().clone()
        model.probe_weights.clear()
        model.probe_inputs.clear()
        torch.set_rng_state(start_rng)
        logits, _, _ = production.additive_step(
            model, native_gam(model), self.x, self.y, 0,
            lambda: initial.new_zeros(()), task_loss)
        self.assertEqual(len(model.probe_inputs), 4)
        for probe_inputs in model.probe_inputs[1:]:
            self.assertTrue(torch.equal(probe_inputs, model.probe_inputs[0]))
        self.assertTensorClose(logits, clean)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_post_clean))

    def test_exception_restores_clean_parameters_rng_and_callback(self):
        model = TinyClassifier(dropout=0.35, fail_on=3)
        initial = model.weight.detach().clone()
        torch.manual_seed(721)
        start_rng = torch.get_rng_state().clone()
        model(self.x)
        expected_post_clean = torch.get_rng_state().clone()
        model.probe_weights.clear()
        model.probe_inputs.clear()
        torch.set_rng_state(start_rng)
        opt = native_gam(model)
        with self.assertRaisesRegex(RuntimeError, 'deliberate probe failure'):
            production.additive_step(model, opt, self.x, self.y, 0,
                                     lambda: initial.new_zeros(()), task_loss)
        self.assertTrue(torch.equal(model.weight, initial))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_post_clean))
        self.assertIsNone(opt._direction_projector)

    def test_zero_radii_preserve_native_two_gradient_scale(self):
        # Native beta1=beta3=1 sums two clean gradients; it does not average.
        model = TinyClassifier()
        initial = model.weight.detach().clone()
        ref = initial.clone().requires_grad_(True)
        gradient = torch.autograd.grad(F.cross_entropy(F.linear(self.x, ref), self.y), ref)[0]
        _, _, stats = production.additive_step(
            model, native_gam(model, rho=0, norm_rho=0, gamma=0), self.x, self.y, 0,
            lambda: initial.new_zeros(()), task_loss)
        self.assertEqual(stats['ce_backward_count'], 4)
        self.assertTensorClose(model.weight, initial - 2 * 0.037 * gradient)

    def test_effective_weight_ewc_chain_rule_for_lora_factors(self):
        # Exercise the production inherited penalty against analytic BA derivatives.
        model = nn.Module()
        model.a = nn.Parameter(torch.tensor([[0.3, -0.5, 0.2], [0.1, 0.4, -0.2]], dtype=torch.float64))
        model.b = nn.Parameter(torch.tensor([[0.2, -0.1], [0.6, 0.3]], dtype=torch.float64))
        reference = torch.zeros(2, 3, dtype=torch.float64)
        fisher = torch.tensor([[0.2, 0.4, 0.7], [0.9, 0.3, 0.5]], dtype=torch.float64)
        learner = object.__new__(production.Learner)
        learner._fisher_past_delta = {'q': fisher}
        learner._delta_reference = {'q': reference}
        learner._ewc_lambda = 3.4
        learner._ewc_eta = 0.0
        learner._device = torch.device('cpu')
        learner._iter_delta_terms = lambda _: iter([('q', model.b @ model.a)])
        penalty = learner._past_fisher_penalty(model)
        ga, gb = torch.autograd.grad(penalty, [model.a, model.b])
        weighted_delta = 3.4 * fisher * (model.b.detach() @ model.a.detach()-reference)
        self.assertTensorClose(ga, model.b.detach().T @ weighted_delta)
        self.assertTensorClose(gb, weighted_delta @ model.a.detach().T)

    def test_source_lora_persists_old_head_and_base_freeze_fullstate_restore(self):
        torch.manual_seed(101)
        model = TinySourceLoRANet()
        model.update_fc(2)
        self.assertTrue(all(torch.count_nonzero(v) == 0 for v in model.effective_updates()))
        self.assertTrue(all(torch.count_nonzero(a.weight) > 0 for a in model.backbone.w_As))
        factor_ids = {id(p) for part in model.backbone.w_As + model.backbone.w_Bs
                      for p in part.parameters()}
        frozen = {name: p.detach().clone() for name, p in model.backbone.named_parameters()
                  if id(p) not in factor_ids}
        self.assertTrue(all(not p.requires_grad for name, p in model.backbone.named_parameters()
                            if name in frozen))
        images = torch.randn(4, 3, 8, 8)
        opt = native_gam(model, lr=0.025)
        production.additive_step(model, opt, images, torch.tensor([0, 1, 1, 0]), 0,
                                 lambda: images.new_zeros(()), task_loss)
        anchor = [v.detach().clone() for v in model.effective_updates()]
        self.assertTrue(any(torch.count_nonzero(v) > 0 for v in anchor))
        factor_values = [(a.weight.detach().clone(), b.weight.detach().clone())
                         for a, b in zip(model.backbone.w_As, model.backbone.w_Bs)]
        old_head = {key: value.detach().clone() for key, value in model.fc.state_dict().items()}
        before_save_logits = model(images)['logits'].detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            # Upstream sidecar save mutates the outer Python task_id, not Q/V factors.
            model.backbone.save_lora_parameters(directory + '/', 0)
            self.assertEqual(model.backbone.task_id, 1)
            self.assertEqual(model.backbone.lora_vit.blocks[0].attn.qkv.task_id, 0)
            self.assertTrue(torch.equal(model(images)['logits'], before_save_logits))
            model.update_fc(4)
            for (a, b), (expected_a, expected_b) in zip(
                    zip(model.backbone.w_As, model.backbone.w_Bs), factor_values):
                self.assertTrue(torch.equal(a.weight, expected_a))
                self.assertTrue(torch.equal(b.weight, expected_b))
            for key, value in old_head.items():
                self.assertTrue(torch.equal(model.fc.state_dict()[key][:2], value))

            def effective_penalty():
                return 0.5 * 0.4 * sum((v-ref).square().sum()
                                      for v, ref in zip(model.effective_updates(), anchor))

            opt = native_gam(model, lr=0.025)
            production.additive_step(model, opt, images, torch.tensor([2, 3, 3, 2]), 2,
                                     effective_penalty, task_loss)
            for key, value in old_head.items():
                self.assertTrue(torch.equal(model.fc.state_dict()[key][:2], value))
            for name, value in model.backbone.named_parameters():
                if name in frozen:
                    self.assertTrue(torch.equal(value, frozen[name]), name)
            self.assertTrue(any(not torch.equal(v, ref)
                                for v, ref in zip(model.effective_updates(), anchor)))

            # Exercise the actual upstream full network/Fisher checkpoint writer.
            learner = object.__new__(production.Learner)
            learner._network = model
            learner._device = torch.device('cpu')
            learner._known_classes = learner._total_classes = 4
            learner.args = {'dataset': 'imagenetr', 'model_name': 'test_tiny_source_lora',
                            'seed': [3], 'lora_rank': 2, 'init_cls': 2, 'increment': 2}
            learner._fisher_past_delta = {f'block0.{c}': torch.ones_like(v.detach())
                                          for c, v in zip(('q', 'v'), model.effective_updates())}
            learner._log = lambda *args: None
            LoraBaseLearner.save_task_checkpoint(learner, directory, 1)
            payload = torch.load(Path(directory)/'task_01.pt', map_location='cpu', weights_only=True)
            self.assertTrue((Path(directory)/'fisher_state_1.pt').is_file())
            self.assertEqual(set(payload['network_state']), set(model.state_dict()))

            torch.manual_seed(999)  # Deliberately different frozen base before full restore.
            restored = TinySourceLoRANet()
            restored.update_fc(4)
            restored.backbone.task_id = 1
            block = restored.backbone.lora_vit.blocks[0].attn.qkv
            block.task_id = 1
            block.saved_A = {'saved_A_0': restored.backbone.w_As}
            block.saved_B = {'saved_B_0': restored.backbone.w_Bs}
            LoraBaseLearner._load_current_lora_factors(learner, restored.backbone, directory, 1)
            self.assertEqual(restored.backbone.task_id, 0)
            self.assertEqual(block.task_id, 0)
            self.assertEqual(block.saved_A, {})
            self.assertEqual(block.saved_B, {})
            # Same ordering as production: native lifecycle restore, then ALL weights.
            restored.load_state_dict(payload['network_state'], strict=True)
            for key, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
            self.assertTrue(torch.equal(restored(images)['logits'], model(images)['logits']))


if __name__ == '__main__':
    unittest.main()
