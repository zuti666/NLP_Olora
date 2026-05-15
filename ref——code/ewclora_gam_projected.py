import logging
import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm

from methods.base import BaseLearner
from methods.ewclora import FisherComputer
from models.net_ewclora import Net
from optimer_PerturabtionType.gam import GAM
from utils.toolkit import check_params_consistency, print_trainable_params, tensor2numpy


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_max_batches(value, default=None):
    if value is None:
        return default
    if isinstance(value, str) and value.lower() in {"none", "all", "full"}:
        return None
    value = int(value)
    return None if value <= 0 else value


class EWCLoRAGAMProjected(BaseLearner):
    """Standalone GAM + EWCLoRA projected-gradient variant.

    This is the extracted version of ``method_type="gam_ewclora_projected"``
    from ``ewclora_verify.py``.

    Per-batch update:
        1. Build the clean EWC gradient g_ewc from the accumulated Delta-W Fisher.
        2. Run GAM on current-task CE only, producing its decomposed gradient g_gam.
        3. Project g_gam away from g_ewc and then add g_ewc back:

           g_final = g_gam - <g_gam, g_ewc> / (||g_ewc||^2 + eps) * g_ewc + g_ewc

    Fisher storage follows ``ewclora_gam_todo.py``: it is computed in Delta-W
    space for each LoRA k/v block by registering hooks on B_new @ A_new.
    """

    def __init__(self, args):
        super().__init__(args)

        self.topk = 1
        self.method_type = "gam_ewclora_projected"
        self.use_gam = True
        self.use_ewc = True
        self.network = Net(args)

        self.gamma = float(args.get("ewc_gamma", args.get("gamma", 1.0)))
        self.ewc_weight = float(args.get("ewc_lambda", args.get("lambda", 0.0)))
        self.ewc_max_batches = _parse_max_batches(
            args.get("ewc_max_batches", None), default=None
        )
        self.fisher_project_eps = float(args.get("fisher_project_eps", 1e-12))

        self.mechanism_eval = _as_bool(args.get("mechanism_eval", False))
        self.mechanism_eps = float(args.get("mechanism_eps", 1e-12))
        self._mechanism_sums = {}
        self._mechanism_count = 0
        self._last_mechanism_metrics = {}
        self._last_fisher_info = ""

        self.omega_W = []
        self.count_updates = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def incremental_train(self, data_manager):
        self.build_train_loader(data_manager)
        self.build_test_loader(data_manager)
        logging.info(
            "Task {} learning on class {}-{}".format(
                self.cur_task, self.known_classes, self.total_classes
            )
        )
        self._train(self.train_loader)

    def after_task(self):
        super().after_task()

        logging.info("=== Update Projected EWCLoRA Delta-W Fisher ===")
        self.count_updates += 1
        task_start, _ = self._task_range(self.cur_task)
        fisher_W = self._compute_delta_fisher(
            self.network,
            self.train_loader,
            task_id=self.cur_task,
            increment=self.increment,
            task_start=task_start,
            max_batches=self.ewc_max_batches,
        )
        self._last_fisher_info = self._summarize_fisher_info(fisher_W)
        logging.info("[Projected Fisher][Task %d] %s", self.cur_task, self._last_fisher_info)

        omega_W_bk = self.omega_W[:]
        self.omega_W = []
        new_a_params = filter(lambda p: getattr(p, "_is_new_a", False), self.network.parameters())
        new_b_params = filter(lambda p: getattr(p, "_is_new_b", False), self.network.parameters())
        for idx, (_p_a, _p_b) in enumerate(zip(new_a_params, new_b_params)):
            if len(omega_W_bk) != 0:
                self.omega_W.append(self.gamma * omega_W_bk[idx] + fisher_W[idx])
            else:
                self.omega_W.append(fisher_W[idx])

        if self.mechanism_eval:
            self._last_mechanism_metrics = {
                "task": int(self.cur_task),
                "method_type": self.method_type,
                "lambda_ewc": float(self.ewc_weight),
                **self._summarize_mechanism_stats(),
                **self._compute_delta_drift_metrics(),
            }
            logging.info("[Projected][Mechanism] %s", self._last_mechanism_metrics)

        self.network.accumulate_and_reset_lora()

    def get_mechanism_metrics(self):
        return dict(self._last_mechanism_metrics)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self, train_loader):
        self.network.to(self.device)
        self.freeze_network()
        print_trainable_params(self.network)
        self._reset_mechanism_stats()

        encoder_params = {
            "params": self.network.image_encoder.parameters(),
            "lr": self.lrate,
            "weight_decay": self.weight_decay,
        }
        cls_params = {
            "params": [p for p in self.network.classifier_pool.parameters() if p.requires_grad],
            "lr": self.fc_lrate if self.fc_lrate is not None else self.lrate,
            "weight_decay": self.weight_decay,
        }

        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

        optimizer, scheduler = self._build_gam_optimizer([encoder_params, cls_params])
        check_params_consistency(self.network, optimizer)

        self._train_function(train_loader, optimizer, scheduler)

        if len(self.multiple_gpus) > 1:
            self.network = self.network.module

    def _build_gam_optimizer(self, param_groups):
        filtered_groups = []
        for group in param_groups:
            params = [p for p in group["params"] if p.requires_grad]
            if params:
                new_group = group.copy()
                new_group["params"] = params
                filtered_groups.append(new_group)
        if not filtered_groups:
            raise ValueError("No trainable parameters found!")

        base_name = str(self.args.get("gam_base_optimizer", "adam")).lower()
        if base_name == "sgd":
            base_optimizer = optim.SGD(
                filtered_groups,
                lr=self.lrate,
                momentum=float(self.args.get("momentum", self.momentum)),
                weight_decay=self.weight_decay,
            )
        elif base_name == "adam":
            base_optimizer = optim.Adam(
                filtered_groups,
                lr=self.lrate,
                weight_decay=self.weight_decay,
                betas=(
                    float(self.args.get("adam_beta1", 0.9)),
                    float(self.args.get("adam_beta2", 0.999)),
                ),
            )
        elif base_name == "adamw":
            base_optimizer = optim.AdamW(
                filtered_groups,
                lr=self.lrate,
                weight_decay=self.weight_decay,
                betas=(
                    float(self.args.get("adam_beta1", 0.9)),
                    float(self.args.get("adam_beta2", 0.999)),
                ),
            )
        else:
            raise ValueError(f"Unknown GAM base optimizer: {base_name}")

        gam_args = SimpleNamespace(
            grad_beta_1=float(self.args.get("gam_grad_beta_1", self.args.get("grad_beta_1", 1.0))),
            grad_beta_2=float(self.args.get("gam_grad_beta_2", self.args.get("grad_beta_2", 1.0))),
            grad_beta_3=float(self.args.get("gam_grad_beta_3", self.args.get("grad_beta_3", 1.0))),
            grad_gamma=float(self.args.get("gam_grad_gamma", self.args.get("grad_gamma", 0.03))),
        )
        optimizer = GAM(
            filtered_groups,
            base_optimizer,
            model=self.network,
            grad_rho=float(self.args.get("gam_grad_rho", 0.02)),
            grad_norm_rho=float(self.args.get("gam_grad_norm_rho", 0.2)),
            adaptive=_as_bool(self.args.get("gam_adaptive", False)),
            perturb_eps=float(self.args.get("gam_perturb_eps", 1e-12)),
            args=gam_args,
            grad_reduce=str(self.args.get("gam_grad_reduce", "mean")),
        )

        if self.scheduler == "constant":
            scheduler = None
        elif self.scheduler == "cosine":
            min_lr_ratio = float(self.args.get("cosine_min_lr_ratio", 0.0))
            if min_lr_ratio > 0:
                min_lr_ratio = min(min_lr_ratio, 1.0)

                def lr_lambda(epoch):
                    cosine = 0.5 * (1.0 + math.cos(math.pi * epoch / self.epochs))
                    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

                scheduler = optim.lr_scheduler.LambdaLR(base_optimizer, lr_lambda=lr_lambda)
            else:
                eta_min = float(self.args.get("cosine_eta_min", 0.0))
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    base_optimizer, T_max=self.epochs, eta_min=eta_min
                )
        elif self.scheduler == "steplr":
            scheduler = optim.lr_scheduler.MultiStepLR(
                base_optimizer, milestones=self.milestone, gamma=self.lrate_decay
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

        return optimizer, scheduler

    def _train_function(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.0
            ce_losses = 0.0
            ewc_losses = 0.0
            ewc_count = 0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self.known_classes
                if inputs.size(0) == 0:
                    continue

                logits, loss_value, log_dict = self._step_projected(optimizer, inputs, targets)
                losses += loss_value

                ce_losses += float(log_dict.get("ce", 0.0))
                ewc_losses += float(log_dict.get("ewc", 0.0))
                ewc_count += 1

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = (
                0.0
                if total == 0
                else np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            )
            avg_loss = losses / max(1, len(train_loader))
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self.cur_task, epoch + 1, self.epochs, avg_loss, train_acc
            )
            if ewc_count > 0:
                avg_ce = ce_losses / ewc_count
                avg_ewc = ewc_losses / ewc_count
                info += ", CE {:.3f}, EWC {:.3e}, EWC/CE {:.3e}".format(
                    avg_ce, avg_ewc, avg_ewc / (avg_ce + 1e-12)
                )

            prog_bar.set_description(info)

            if (epoch + 1) % 5 == 0 and hasattr(self, "test_loader"):
                test_acc = self._quick_test_acc()
                logging.info("[Periodic Eval] %s, Test_accy %.2f", info, test_acc)
                self.network.train()

        logging.info(info)

    @torch.no_grad()
    def _quick_test_acc(self):
        self.network.eval()
        net = self._net()
        correct, total = 0, 0
        for _, inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = net.interface(inputs)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
        return round(correct * 100 / total, 2) if total > 0 else 0.0

    def freeze_network(self):
        target_suffix = f".{self.cur_task}"
        unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            "lora_new_A_k",
            "lora_new_A_v",
            "lora_new_B_k",
            "lora_new_B_v",
        ]
        for name, param in self.network.named_parameters():
            param.requires_grad_(any(key in name for key in unfrozen_keys))

    # ------------------------------------------------------------------
    # Projected GAM + EWC step
    # ------------------------------------------------------------------

    def _compute_task_loss(self, inputs, targets):
        outputs = self.network(inputs, use_new=True)
        ce_loss = F.cross_entropy(outputs["logits"], targets)
        return outputs["logits"], ce_loss

    def _step_projected(self, optimizer, inputs, targets):
        params = self._flatten_optimizer_params(optimizer)
        ewc_grad_cache = self._compute_ewc_grad_cache(params)
        log_cache = {"ce": 0.0, "ewc": 0.0}

        optimizer.set_direction_projector(
            self._build_projected_ewc_projector(ewc_grad_cache)
        )

        def closure():
            optimizer.zero_grad()
            logits, ce_loss = self._compute_task_loss(inputs, targets)
            ce_loss.backward()
            log_cache["ce"] = float(ce_loss.detach().item())
            return logits, ce_loss.detach()

        try:
            logits, loss_value = optimizer.step(closure=closure)
        finally:
            optimizer.set_direction_projector(None)

        ewc_loss_value = float(self._compute_ewc_loss().detach().item())
        log_cache["ewc"] = ewc_loss_value
        self._record_mechanism_stats(log_cache, increment=True)

        return logits.detach(), float(loss_value) + ewc_loss_value, log_cache

    @staticmethod
    def _flatten_optimizer_params(optimizer):
        params = []
        seen = set()
        for group in optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                pid = id(param)
                if pid in seen:
                    continue
                seen.add(pid)
                params.append(param)
        return params

    def _compute_ewc_grad_cache(self, params):
        if self.count_updates == 0 or not self.omega_W:
            return {}

        with torch.enable_grad():
            ewc_loss = self._compute_ewc_loss()

        grads = torch.autograd.grad(
            ewc_loss,
            params,
            allow_unused=True,
            retain_graph=False,
        )
        return {
            id(param): grad.detach().clone()
            for param, grad in zip(params, grads)
            if grad is not None
        }

    def _build_projected_ewc_projector(self, ewc_grad_cache):
        eps = float(self.fisher_project_eps)

        def projector(param_groups):
            gam_norm_sq = 0.0
            ewc_norm_sq = 0.0
            proj_norm_sq = 0.0
            final_norm_sq = 0.0
            gam_ewc_dot = 0.0

            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad or param.grad is None:
                        continue

                    g_gam = param.grad.detach().clone()
                    g_ewc = ewc_grad_cache.get(id(param))

                    if g_ewc is None:
                        g_final = g_gam
                        g_proj = g_gam
                    else:
                        denom = torch.sum(g_ewc * g_ewc) + eps
                        coeff = torch.sum(g_gam * g_ewc) / denom
                        g_proj = g_gam - coeff * g_ewc
                        g_final = g_proj + g_ewc

                    if self.mechanism_eval:
                        gam_norm_sq += float(g_gam.pow(2).sum().item())
                        proj_norm_sq += float(g_proj.pow(2).sum().item())
                        final_norm_sq += float(g_final.pow(2).sum().item())
                        if g_ewc is not None:
                            ewc_norm_sq += float(g_ewc.pow(2).sum().item())
                            gam_ewc_dot += float((g_gam * g_ewc).sum().item())

                    param.grad.data.copy_(g_final)

            if self.mechanism_eval:
                self._record_mechanism_stats(
                    {
                        "gam_grad_norm": gam_norm_sq ** 0.5,
                        "ewc_grad_norm": ewc_norm_sq ** 0.5,
                        "projected_grad_norm": proj_norm_sq ** 0.5,
                        "final_grad_norm": final_norm_sq ** 0.5,
                        "cos_gam_ewc": self._cosine_from_sums(
                            gam_ewc_dot, gam_norm_sq, ewc_norm_sq
                        ),
                    },
                    increment=False,
                )

        return projector

    # ------------------------------------------------------------------
    # Delta-W Fisher and EWC
    # ------------------------------------------------------------------

    def _net(self):
        return self.network.module if isinstance(self.network, nn.DataParallel) else self.network

    @staticmethod
    def _is_delta_module(module):
        return all(
            hasattr(module, attr)
            for attr in (
                "lora_new_A_k",
                "lora_new_B_k",
                "lora_new_A_v",
                "lora_new_B_v",
            )
        )

    def _iter_delta_modules(self, model):
        for module in model.modules():
            if self._is_delta_module(module):
                yield module

    def _iter_delta_terms(self, model):
        for module in self._iter_delta_modules(model):
            yield module.lora_new_B_k.weight @ module.lora_new_A_k.weight
            yield module.lora_new_B_v.weight @ module.lora_new_A_v.weight

    def _compute_ewc_loss(self):
        if self.count_updates == 0 or not self.omega_W:
            return torch.zeros((), device=self.device)

        model = self._net()
        penalty = None
        for idx, delta_now in enumerate(self._iter_delta_terms(model)):
            if idx >= len(self.omega_W):
                break
            fisher = self.omega_W[idx].to(device=delta_now.device, dtype=delta_now.dtype)
            term = (fisher * delta_now.pow(2)).sum()
            penalty = term if penalty is None else penalty + term

        if penalty is None:
            return torch.zeros((), device=self.device)
        return 0.5 * float(self.ewc_weight) * penalty

    def _compute_delta_fisher(self, model, loader, task_id, increment, task_start=None, max_batches=None):
        fisher = FisherComputer(
            task_id,
            model,
            loader,
            increment,
            F.cross_entropy,
            self.device,
            task_start=task_start,
        )
        return fisher.compute(max_batches=max_batches)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _reset_mechanism_stats(self):
        self._mechanism_sums = {}
        self._mechanism_count = 0
        self._last_mechanism_metrics = {}

    def _record_mechanism_stats(self, stats, increment=False):
        if not self.mechanism_eval:
            return
        for key, value in stats.items():
            self._mechanism_sums[key] = self._mechanism_sums.get(key, 0.0) + float(value)
        if increment:
            self._mechanism_count += 1

    def _summarize_mechanism_stats(self):
        if self._mechanism_count <= 0:
            return {}
        denom = float(self._mechanism_count)
        return {
            f"train_{key}_mean": value / denom
            for key, value in sorted(self._mechanism_sums.items())
        }

    def _compute_delta_drift_metrics(self):
        if self.count_updates == 0 or not self.omega_W:
            return {
                "delta_norm": 0.0,
                "fisher_weighted_drift": 0.0,
                "normalized_fisher_drift": 0.0,
                "ewc_penalty_value": 0.0,
            }

        eps = float(self.mechanism_eps)
        delta_norm = 0.0
        fisher_weighted = 0.0
        with torch.no_grad():
            for idx, delta_now in enumerate(self._iter_delta_terms(self._net())):
                if idx >= len(self.omega_W):
                    break
                fisher = self.omega_W[idx].to(device=delta_now.device, dtype=delta_now.dtype)
                delta_sq = delta_now.pow(2)
                delta_norm += float(delta_sq.sum().item())
                fisher_weighted += float((fisher * delta_sq).sum().item())

        return {
            "delta_norm": float(delta_norm),
            "fisher_weighted_drift": float(fisher_weighted),
            "normalized_fisher_drift": float(fisher_weighted / (delta_norm + eps)),
            "ewc_penalty_value": float(0.5 * float(self.ewc_weight) * fisher_weighted),
        }

    def _summarize_fisher_info(self, fisher_W):
        if fisher_W is None or len(fisher_W) == 0:
            return "Fisher blocks 0"

        fw_means = [fw.detach().float().mean().item() for fw in fisher_W]
        fw_maxs = [fw.detach().float().max().item() for fw in fisher_W]
        fw_sums = [fw.detach().float().sum().item() for fw in fisher_W]

        return "Fisher blocks {}, mean {:.3e}, max {:.3e}, sum {:.3e}".format(
            len(fisher_W),
            float(np.mean(fw_means)),
            float(np.max(fw_maxs)),
            float(np.sum(fw_sums)),
        )

    def _cosine_from_sums(self, dot, norm_a_sq, norm_b_sq):
        denom = (max(norm_a_sq, 0.0) ** 0.5) * (max(norm_b_sq, 0.0) ** 0.5)
        if denom <= self.mechanism_eps:
            return 0.0
        return float(dot / (denom + self.mechanism_eps))


class EWCLoRA(EWCLoRAGAMProjected):
    """Backward-compatible alias when this module is selected directly."""
