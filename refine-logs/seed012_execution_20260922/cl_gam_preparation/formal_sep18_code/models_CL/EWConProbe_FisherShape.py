"""Full-model EWC + FisherShape optimizer (FS-EWC method).

This is the learner for the Fisher-Shaped EWC design
(``newmethod_design_wiki/FisherShapedEWC_method_wiki.md``).  It inherits
``EWCon_Probe.Learner`` to get the full EWC infrastructure (Fisher
estimation/accumulation, checkpoint buffers, mechanism/FQA diagnostics)
and replaces the per-batch optimization with the two-gradient
Fisher-trace shaping step:

    g_0  = grad L_CE(theta; B)                  # pass 1 (clean point)
    theta_+ = theta + s_0 g_0,  s_0 = rho/(||g_0|| + xi)
    g_1  = grad L_CE(theta_+; B)                # pass 2 (same minibatch)
    d_fs = g_0 + alpha * (g_1 - g_0)/s_0        # ~= g_0 + alpha * H_0 g_0
    g_final = d_fs + lambda * F_{t-1} (theta - theta*)   # additive EWC

Design constraints (matching the paper's Algorithm 1):
  * The EWC gradient is excluded from both the probing perturbation and
    the finite-difference construction of r_F; it is computed analytically
    from the stored diagonal Fisher and checkpoint (no extra backward) and
    injected through the optimizer's direction projector.
  * Fisher shaping is active from task 0 on: shaping the first task's
    Fisher is exactly what makes it protectable later.  On task 0 the EWC
    cache is empty and the projector is a no-op.

Two forward-backward passes per step (SAM cost, half of GAM).
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from models_CL.EWCon_Probe import Learner as _EWConProbe
from optimer_PerturabtionType.fishershape import FisherShape
from utils.mechanism_eval import as_bool
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(_EWConProbe):
    """Full-model EWC + two-gradient Fisher-trace shaping (FS-EWC)."""

    def __init__(self, args):
        super().__init__(args)

        self._fs_rho = float(args.get("fs_rho", args.get("gam_grad_rho", 0.05)))
        self._fs_alpha = float(args.get("fs_alpha", 0.2))
        self._fs_alpha_mode = str(args.get("fs_alpha_mode", "ratio")).lower()
        self._fs_xi = float(args.get("fs_xi", args.get("fs_perturb_eps", 1e-12)))
        self._fs_sam_equivalent = as_bool(args.get("fs_sam_equivalent", False))
        self._fs_adaptive = as_bool(args.get("fs_adaptive", False))
        self._fs_grad_reduce = str(args.get("fs_grad_reduce", "mean"))
        self._fs_preserve_probe_rng = as_bool(
            args.get("fs_preserve_probe_rng", False)
        )

        # This learner defaults to its own optimizer; other types fall back
        # to the stock EWCon_Probe training path (useful for baselines).
        if "optimizer_type" not in args:
            self._optimizer_type = "fishershape"
        if self._optimizer_type != "fishershape":
            logger.warning(
                "[FS-EWC] optimizer_type=%s (not 'fishershape'); "
                "falling back to the stock EWCon_Probe training path.",
                self._optimizer_type,
            )

        logger.info(
            "[FS-EWC] rho=%.4g alpha=%.4g alpha_mode=%s xi=%.3g sam_equivalent=%s "
            "adaptive=%s preserve_probe_rng=%s lambda=%.4g use_ewc=%s",
            self._fs_rho,
            self._fs_alpha,
            self._fs_alpha_mode,
            self._fs_xi,
            self._fs_sam_equivalent,
            self._fs_adaptive,
            self._fs_preserve_probe_rng,
            self._ewc_lambda,
            self._use_ewc,
        )

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------
    def _build_optimizer(self, params, stage):
        if self._optimizer_type != "fishershape":
            return super()._build_optimizer(params, stage)

        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
        base_name = str(
            self.args.get(
                "fs_base_optimizer",
                self.args.get("gam_base_optimizer", self.args.get("optimizer", "sgd")),
            )
        ).lower()

        if base_name == "adam":
            base_opt = optim.Adam(
                params,
                lr=lr,
                betas=(
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
                weight_decay=weight_decay,
            )
        elif base_name == "adamw":
            base_opt = optim.AdamW(
                params,
                lr=lr,
                betas=(
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
                weight_decay=weight_decay,
            )
        else:
            base_opt = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

        return FisherShape(
            params,
            base_optimizer=base_opt,
            model=self._network,
            rho=self._fs_rho,
            alpha=self._fs_alpha,
            alpha_mode=self._fs_alpha_mode,
            sam_equivalent=self._fs_sam_equivalent,
            adaptive=self._fs_adaptive,
            perturb_eps=self._fs_xi,
            grad_reduce=self._fs_grad_reduce,
            preserve_probe_rng=self._fs_preserve_probe_rng,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _train(self, train_loader, test_loader):
        if self._optimizer_type != "fishershape":
            super()._train(train_loader, test_loader)
            return

        model = self._unwrap_network()
        model.to(self._device)
        self._reset_mechanism_stats()

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage=stage)

        if stage == "init":
            epochs = int(self.args.get("init_epoch", 1))
            scheduler = self.build_scheduler(
                optimizer.base_optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("init_milestones", []),
                gamma=float(self.args.get("init_lrate_decay", self.args.get("init_lr_decay", 1.0))),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )
        else:
            epochs = int(self.args.get("epochs", 1))
            scheduler = self.build_scheduler(
                optimizer.base_optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("milestones", []),
                gamma=float(self.args.get("lrate_decay", 1.0)),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )

        prog_bar = tqdm(range(epochs))
        info = ""
        for epoch in prog_bar:
            model.train()
            total_loss = 0.0
            ce_losses = 0.0
            ewc_losses = 0.0
            loss_count = 0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits, ce_val, ewc_val, step_stats = self._step_fishershape(
                    optimizer, model, inputs, targets, stage
                )

                total_loss += ce_val + ewc_val
                ce_losses += ce_val
                ewc_losses += ewc_val
                loss_count += 1

                if self._mechanism_eval:
                    stats = {
                        "ce": ce_val,
                        "ewc": ewc_val,
                        "ewc_ce_ratio": ewc_val / (ce_val + self._mechanism_eps),
                    }
                    stats.update(step_stats)
                    if self._should_measure_batch_param_drift():
                        stats.update(self._compute_param_drift_metrics())
                    self._record_mechanism_stats(stats, increment=True)

                eval_logits, eval_targets = self._eval_view_from_logits(logits, targets, stage)
                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_loss = total_loss / max(1, len(train_loader))
            avg_ce = ce_losses / max(1, loss_count)
            avg_ewc = ewc_losses / max(1, loss_count)
            ewc_ce_ratio = float(avg_ewc / (avg_ce + self._mechanism_eps))
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} [fs] => "
                f"Loss {avg_loss:.3f}, Train_accy {train_acc:.2f}, "
                f"CE {avg_ce:.3f}, EWC {avg_ewc:.3e}"
            )
            last = getattr(optimizer, "last_stats", None)
            if last:
                info += (
                    f", g0 {last.get('fs_g0_norm', 0.0):.3e}"
                    f", rF {last.get('fs_rf_norm', 0.0):.3e}"
                )

            all_seen_test_acc = None
            cur_test_acc = None
            if epoch % 5 == 4:
                all_seen_test_acc = self._compute_accuracy(model, test_loader)
                cur_test_acc = self._compute_current_task_accuracy(model, test_loader, stage)
                info += f", Test_accy {all_seen_test_acc:.2f}, CurTask_Test {cur_test_acc:.2f}"

            if self._epoch_dynamics_eval:
                drift = self._compute_param_drift_metrics()
                epoch_metrics = {
                    "task": int(self._cur_task),
                    "epoch": int(epoch + 1),
                    "epochs": int(epochs),
                    "stage": stage,
                    "lambda_ewc": float(self._ewc_lambda if self._use_ewc else 0.0),
                    "use_ewc": bool(self._use_ewc),
                    "loss_total": float(avg_loss),
                    "loss_ce": float(avg_ce),
                    "loss_ewc": float(avg_ewc),
                    "ewc_ce_ratio": float(ewc_ce_ratio),
                    "train_acc_current": float(train_acc),
                    "loss_total_finite": bool(np.isfinite(avg_loss)),
                    "loss_ce_finite": bool(np.isfinite(avg_ce)),
                    "loss_ewc_finite": bool(np.isfinite(avg_ewc)),
                    "fs_rho": float(self._fs_rho),
                    "fs_alpha": float(self._fs_alpha),
                    "fs_sam_equivalent": bool(self._fs_sam_equivalent),
                    **{k: float(v) for k, v in (last or {}).items()},
                    **drift,
                }
                epoch_metrics["delta_norm_sq"] = float(epoch_metrics.get("delta_norm", 0.0))
                epoch_metrics["param_delta_norm_sq"] = float(
                    epoch_metrics.get("param_delta_norm", epoch_metrics["delta_norm_sq"])
                )
                if all_seen_test_acc is not None:
                    epoch_metrics["test_acc_seen"] = float(all_seen_test_acc)
                if cur_test_acc is not None:
                    epoch_metrics["test_acc_current"] = float(cur_test_acc)
                elif self._epoch_dynamics_current_test:
                    epoch_metrics["test_acc_current"] = float(
                        self._compute_current_task_accuracy(model, test_loader, stage)
                    )
                self._last_epoch_dynamics.append(epoch_metrics)
                logger.info("[FS-EWC][EpochDynamics] %s", epoch_metrics)

            prog_bar.set_description(info)
        if epochs > 0:
            logging.info(info)

    # ------------------------------------------------------------------
    # One FS-EWC batch step
    # ------------------------------------------------------------------
    def _step_fishershape(
        self,
        optimizer: FisherShape,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        stage: str,
    ):
        """Run one FisherShape step with additive analytic EWC gradient.

        Returns (clean_logits, ce_value, ewc_value, step_stats).
        """
        params = self._flatten_optimizer_params(optimizer)
        ewc_grad_cache = self._build_ewc_grad_cache(params)
        # Record CE and EWC at the same clean, pre-update parameter point.
        # The analytic EWC gradient cache above is also constructed there.
        if self._use_ewc and self._fisher is not None and self._checkpoint is not None:
            with torch.no_grad():
                ewc_val = float(self._ewc_penalty(model).detach().item())
        else:
            ewc_val = 0.0
        proj_stats: Dict[str, float] = {}
        optimizer.set_direction_projector(
            self._build_additive_ewc_projector(ewc_grad_cache, proj_stats)
        )

        def closure():
            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs["logits"]
            loss = self._task_loss_from_logits(logits, targets, stage)
            loss.backward()
            return outputs, loss.detach()

        try:
            # FisherShape returns the FIRST (clean-point) outputs/loss.
            outputs, loss_value = optimizer.step(closure=closure)
        finally:
            optimizer.set_direction_projector(None)

        ce_val = float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)

        step_stats = dict(getattr(optimizer, "last_stats", {}) or {})
        step_stats.update(proj_stats)

        logits = (outputs["logits"] if isinstance(outputs, dict) else outputs).detach()
        return logits, ce_val, ewc_val, step_stats

    # ------------------------------------------------------------------
    # Analytic EWC gradient cache (no extra backward)
    # ------------------------------------------------------------------
    def _build_ewc_grad_cache(self, params: List[torch.Tensor]) -> Dict[int, torch.Tensor]:
        """grad L_EWC = lambda * F_i * (theta_i - theta_i*), from stored buffers."""
        if not self._use_ewc or self._fisher is None or self._checkpoint is None:
            return {}

        model = self._unwrap_network()
        param_id_set = {id(p) for p in params}
        cache: Dict[int, torch.Tensor] = {}

        for name, p in self._iter_trainable(model):
            if id(p) not in param_id_set:
                continue
            ref = self._checkpoint.get(name)
            fisher = self._fisher.get(name)
            if ref is None or fisher is None:
                continue
            with torch.no_grad():
                ref_d = ref.to(device=p.device, dtype=p.dtype)
                fisher_d = fisher.to(device=p.device, dtype=p.dtype)
                # Gradient of 0.5 * lambda * sum F_i (theta_i - theta_i*)^2
                cache[id(p)] = self._ewc_lambda * fisher_d * (p.detach() - ref_d)

        return cache

    # ------------------------------------------------------------------
    # Additive projector: g_final = d_fs + g_ewc
    # ------------------------------------------------------------------
    def _build_additive_ewc_projector(
        self,
        ewc_grad_cache: Dict[int, torch.Tensor],
        log_cache: Dict[str, float],
    ):
        collect = self._mechanism_eval
        eps = float(self._mechanism_eps)

        def projector(param_groups):
            fs_norm_sq = 0.0
            ewc_norm_sq = 0.0
            fs_ewc_dot = 0.0
            for group in param_groups:
                for p in group["params"]:
                    if not p.requires_grad or p.grad is None:
                        continue
                    if collect:
                        fs_norm_sq += float(p.grad.detach().pow(2).sum().item())
                    g_ewc = ewc_grad_cache.get(id(p))
                    if g_ewc is None:
                        continue
                    g_ewc = g_ewc.to(device=p.grad.device, dtype=p.grad.dtype)
                    if collect:
                        ewc_norm_sq += float(g_ewc.pow(2).sum().item())
                        fs_ewc_dot += float((p.grad.detach() * g_ewc).sum().item())
                    p.grad.data.add_(g_ewc)

            if collect:
                denom = (fs_norm_sq ** 0.5) * (ewc_norm_sq ** 0.5) + eps
                log_cache["fs_dir_norm"] = fs_norm_sq ** 0.5
                log_cache["ewc_grad_norm"] = ewc_norm_sq ** 0.5
                log_cache["cos_fs_ewc"] = fs_ewc_dot / denom

        return projector

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_optimizer_params(optimizer) -> List[torch.Tensor]:
        params, seen = [], set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params
