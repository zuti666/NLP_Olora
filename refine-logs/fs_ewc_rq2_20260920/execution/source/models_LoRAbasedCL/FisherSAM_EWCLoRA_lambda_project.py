"""SeqLoRA EWCLoRA with Fisher-SAM, adaptive EWC lambda, and gradient projection.

This learner keeps the SeqLoRA lifecycle and delta-W Fisher/EWC geometry from
``ewclora_youyue_fitArchitechture.py`` while replacing the per-batch optimizer
step with the same additive Fisher-SAM/EWC schedule used by
``models_CL/FisherSAM-EWC-lambda.py``:

    1. compute the current-task CE gradient at theta;
    2. build the diagonal Fisher-SAM perturbation from that CE gradient only;
    3. compute the current-task CE gradient at theta + eps;
    4. restore theta;
    5. compute the past-task delta-W Fisher/EWC gradient at theta;
    6. optionally project the FSAM gradient away from the EWC outward direction;
    7. apply the base optimizer step with g_task_projected + g_ewc;
    8. optionally update lambda from the configured drift constraints.

The Fisher buffer is still estimated in effective LoRA delta-W space through the
hook-based logic inherited from the SeqLoRA EWCLoRA implementation.  In-task
raw Fisher snapshots can be cached during training and mixed with the final
task Fisher before online EWC accumulation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models_LoRAbasedCL.ewclora_youyue_fitArchitechture import Learner as _SeqEWCLoRA
from optimer_PerturabtionType.optimer_fishersam_diagonal import DiagonalFisherSAM
from utils.mechanism_eval import as_bool

logger = logging.getLogger(__name__)


class Learner(_SeqEWCLoRA):
    """SeqLoRA + delta-W EWC + Fisher-SAM current-task optimizer."""

    @staticmethod
    def _optional_positive_float(value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in {"", "none", "null", "false", "off"}:
                return None
            value = stripped
        parsed = float(value)
        return parsed if parsed > 0.0 else None

    def __init__(self, args):
        super().__init__(args)

        self._configured_optimizer_type = str(getattr(self, "_optimizer_type", "sgd")).lower()
        self._optimizer_type = "fishersam"
        self._base_optimizer_type = "fishersam"
        self._optimizer_type_by_task = None

        self._lambda_ewc = float(self._ewc_lambda)
        self._fisher_eta = float(getattr(self, "_ewc_eta", args.get("ewc_eta", args.get("eta", 1e-12))))

        self._fsam_gamma = float(
            args.get(
                "fsam_gamma",
                args.get("fishersam_gamma", args.get("sam_rho", args.get("rho", 0.1))),
            )
        )
        self._fsam_eta = float(args.get("fsam_eta", args.get("fisher_eta", 1.0)))
        self._fsam_damping = float(args.get("fsam_damping", args.get("fisher_sam_damping", 1e-12)))
        self._fsam_inverse_mode = str(args.get("fsam_inverse_mode", "paper")).lower()
        self._fsam_adaptive = as_bool(args.get("fsam_adaptive", args.get("sam_adaptive", False)))
        self._fsam_base_optimizer = self._resolve_fsam_base_optimizer(args)

        self._dual_ascent = as_bool(args.get("ewc_dual_ascent", args.get("fsam_ewc_dual_ascent", True)))
        dual_budget_keys = ("ewc_dual_delta", "gamma_a_sq", "fsam_ewc_gamma_a_sq")
        dual_budget_is_explicit = any(key in args for key in dual_budget_keys)
        dual_budget_default = float(args.get("ewc_dual_raw_budget_default", 1e-3))
        self._dual_budget = float(  # σ: D_F ≤ σ budget
            args.get(
                "ewc_dual_delta",
                args.get("gamma_a_sq", args.get("fsam_ewc_gamma_a_sq", dual_budget_default)),
            )
        )
        if not dual_budget_is_explicit:
            logger.warning(
                "[FisherSAM-EWCLoRA-lambda] Using raw-Fisher default ewc_dual_delta=%.4g. "
                "Delta-W Fisher drift is scale-dependent; tune ewc_dual_delta explicitly for final runs.",
                self._dual_budget,
            )
        self._dual_relative = as_bool(args.get("ewc_dual_relative", True))
        self._dual_lambda_min = float(args.get("ewc_dual_lambda_min", 1.0))
        self._dual_lambda_max = float(args.get("ewc_dual_lambda_max", 1e6))
        self._dual_eps = float(args.get("ewc_dual_eps", 1e-12))
        self._dual_use_eta = as_bool(args.get("ewc_dual_use_eta", True))
        self._dual_update_by_epoch = as_bool(args.get("ewc_dual_update_by_epoch", True))
        default_dual_interval = 5 if self._dual_update_by_epoch else 100
        self._dual_update_interval = max(
            1,
            int(args.get("ewc_dual_update_interval", default_dual_interval)),
        )
        self._dual_eta_up = float(args.get("ewc_dual_eta_up", 0.05))    # η↑
        self._dual_eta_down = float(args.get("ewc_dual_eta_down", 0.05))  # η↓
        self._dual_rho = float(                                            # ρ hysteresis: decrease when D_F < ρσ
            args.get("ewc_dual_rho", args.get("ewc_dual_low_ratio", 0.5))
        )
        self._dual_step_count = 0
        self._dual_euclidean_budget = self._optional_positive_float(
            args.get(
                "ewc_dual_euclidean_delta",
                args.get("ewc_dual_l2_delta", args.get("ewc_dual_r_e_sq", None)),
            )
        )
        self._dual_spectral_budget = self._optional_positive_float(
            args.get(
                "ewc_dual_spectral_delta",
                args.get("ewc_dual_fmax_l2_delta", args.get("ewc_dual_r_m_sq", None)),
            )
        )
        self._dual_use_joint_constraints = as_bool(
            args.get(
                "ewc_dual_use_joint_constraints",
                self._dual_euclidean_budget is not None or self._dual_spectral_budget is not None,
            )
        )

        # Fisher-aware numerical stability cap:
        # keep lr * lambda * F_max <= c for the EWC gradient scale.
        self._safe_lambda_cap = as_bool(args.get("ewc_safe_lambda_cap", True))
        self._safe_lambda_c = float(args.get("ewc_safe_lambda_c", 0.5))
        self._safe_lambda_eps = float(args.get("ewc_safe_lambda_eps", 1e-12))
        self._safe_lambda_override_min = as_bool(args.get("ewc_safe_lambda_override_min", True))
        self._last_base_lr = float(
            args.get(
                "lrate",
                args.get(
                    "lr",
                    args.get(
                        "init_lr",
                        args.get("full_lr", args.get("learning_rate", 0.0)),
                    ),
                ),
            )
        )
        self._last_safe_lambda_max = float(self._dual_lambda_max)

        # --- commented-out engineering guardrails (removed from core) ---
        # self._dual_alpha = float(args.get("ewc_dual_alpha", ...))
        # self._dual_loss_window / _dual_loss_patience  (CE-loss windowing — orthogonal to D_F constraint)
        # self._dual_loss_buffer / _dual_ewc_ratio_buffer / _dual_prev_loss_window
        # self._dual_use_ewc_ratio_gate / _dual_ewc_ratio_threshold  (EWC/FSAM ratio gate)

        self._approximate_fisher_in_task = as_bool(
            args.get(
                "approximte_fisher_inTask",
                args.get(
                    "approximate_fisher_inTask",
                    args.get("approximate_fisher_in_task", True),
                ),
            )
        )
        self._in_task_fisher_interval = int(
            args.get(
                "in_task_fisher_interval",
                args.get(
                    "approximate_fisher_interval",
                    args.get("approximte_fisher_interval", 10),
                ),
            )
        )
        self._in_task_fisher_omega = min(
            1.0,
            max(
                0.0,
                float(
                    args.get(
                        "in_task_fisher_omega",
                        args.get(
                            "path_fisher_omega",
                            args.get(
                                "approximate_fisher_omega",
                                args.get("approximte_fisher_omega", 0.5),
                            ),
                        ),
                    )
                ),
            ),
        )
        self._in_task_fisher_snapshots: List[
            Tuple[int, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = []
        self._in_task_fisher_last_metrics: Dict[str, float] = {}

        self._mechanism_eval = as_bool(args.get("mechanism_eval", False))
        self._mechanism_eps = float(args.get("mechanism_eps", 1e-12))
        self._mechanism_top_fraction = float(args.get("mechanism_top_fraction", 0.1))
        topk = args.get("mechanism_topk", None)
        self._mechanism_topk = None if topk is None else int(topk)
        self._mechanism_sums: Dict[str, float] = {}
        self._mechanism_count = 0
        self._last_mechanism_metrics: Dict[str, float] = {}

        self._ewc_project_grad = as_bool(args.get("ewc_project_grad", True))
        self._ewc_project_eps = float(args.get("ewc_project_eps", args.get("fisher_project_eps", 1e-12)))
        self._ewc_project_rho = float(args.get("ewc_project_rho", args.get("fisher_project_rho", self._dual_rho)))
        self._ewc_project_boundary = as_bool(args.get("ewc_project_boundary", True))
        self._ewc_project_outward_only = as_bool(args.get("ewc_project_outward_only", True))

        if self._is_main_process:
            logger.info(
                "[FisherSAM-EWCLoRA-lambda] gamma=%.4g eta=%.4g inverse=%s base=%s "
                "dual=%s budget(σ)=%.4g relative=%s eta_up(η↑)=%.4g eta_down(η↓)=%.4g rho(ρ)=%.4g "
                "update_interval=%d by_epoch=%s "
                "joint_constraints=%s euclidean_budget=%.4g spectral_budget=%.4g "
                "lambda_bounds=[%.4g, %.4g] lambda_init=%.4g "
                "safe_cap=%s safe_c=%.4g safe_eps=%.4g override_min=%s "
                "project_grad=%s project_rho=%.4g outward_only=%s boundary_gate=%s "
                "approximte_fisher_inTask=%s interval=%d omega=%.3g",
                self._fsam_gamma,
                self._fsam_eta,
                self._fsam_inverse_mode,
                self._fsam_base_optimizer,
                self._dual_ascent,
                self._dual_budget,
                self._dual_relative,
                self._dual_eta_up,
                self._dual_eta_down,
                self._dual_rho,
                self._dual_update_interval,
                self._dual_update_by_epoch,
                self._dual_use_joint_constraints,
                0.0 if self._dual_euclidean_budget is None else self._dual_euclidean_budget,
                0.0 if self._dual_spectral_budget is None else self._dual_spectral_budget,
                self._dual_lambda_min,
                self._dual_lambda_max,
                self._ewc_lambda,
                self._safe_lambda_cap,
                self._safe_lambda_c,
                self._safe_lambda_eps,
                self._safe_lambda_override_min,
                self._ewc_project_grad,
                self._ewc_project_rho,
                self._ewc_project_outward_only,
                self._ewc_project_boundary,
                self._approximate_fisher_in_task,
                self._in_task_fisher_interval,
                self._in_task_fisher_omega,
            )
        if (
            self._dual_use_joint_constraints
            and (self._dual_euclidean_budget is not None or self._dual_spectral_budget is not None)
            and self._fisher_eta <= 0.0
        ):
            logger.warning(
                "[FisherSAM-EWCLoRA-lambda] Euclidean/spectral lambda constraints are enabled "
                "with ewc_eta<=0. A single Fisher-EWC lambda cannot control drift in low-Fisher "
                "or zero-Fisher directions unless an eta floor is used."
            )

    def _default_cosine_eta_min(self, lr: float) -> float:
        ratio = self.args.get("cosine_min_lr_ratio", self.args.get("min_lr_ratio", None))
        if ratio is not None:
            ratio = min(max(float(ratio), 0.0), 1.0)
            return float(lr) * ratio

        absolute = self.args.get("cosine_eta_min", self.args.get("min_lr", None))
        if absolute is not None:
            return float(absolute)

        return 0.01 * float(lr)

    def _resolve_fsam_base_optimizer(self, args) -> str:
        base = str(
            args.get(
                "fsam_base_optimizer",
                args.get("fishersam_base_optimizer", args.get("optimizer", self._configured_optimizer_type)),
            )
        ).lower()
        if base in {"fishersam", "sam", "gam", "cflat", "rwp", "arwp"}:
            return "sgd"
        if base not in {"sgd", "adam", "adamw"}:
            logger.warning(
                "[FisherSAM-EWCLoRA-lambda] Unsupported fsam_base_optimizer=%s; using SGD.",
                base,
            )
            return "sgd"
        return base

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------
    def _build_optimizer(self, params, stage):
        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
        self._last_base_lr = float(lr)
        base_name = self._fsam_base_optimizer

        if base_name == "adam":
            base_cls = torch.optim.Adam
            opt_kwargs = {
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": (
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
            }
        elif base_name == "adamw":
            base_cls = torch.optim.AdamW
            opt_kwargs = {
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": (
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
            }
        else:
            base_cls = torch.optim.SGD
            opt_kwargs = {
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
            }

        return DiagonalFisherSAM(
            params,
            base_cls,
            gamma=self._fsam_gamma,
            eta=self._fsam_eta,
            damping=self._fsam_damping,
            adaptive=self._fsam_adaptive,
            inverse_mode=self._fsam_inverse_mode,
            **opt_kwargs,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _run_regularized_train(
        self,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        *,
        epochs: int,
        class_offset: int,
    ) -> None:
        model = self._unwrap_network()
        self._snapshot_task_reference()
        self._reset_dual_controller_state()
        self._in_task_fisher_snapshots = []
        self._in_task_fisher_last_metrics = {}
        self._diag_first_batch_done = False  # reset per-task first-batch diagnostic flag
        self._apply_joint_lambda_clip(optimizer, reason="task_start")

        # ── Task-start diagnostic ──────────────────────────────────────────
        if self._is_main_process:
            has_f = bool(self._fisher_past_delta)
            has_r = bool(self._delta_reference)
            f_info = ""
            if has_f:
                f_vals = list(self._fisher_past_delta.values())
                f_max = float(max(v.max().item() for v in f_vals))
                f_sum = float(sum(v.sum().item() for v in f_vals))
                f_numel = sum(int(v.numel()) for v in f_vals)
                f_mean = f_sum / max(f_numel, 1)
                f_info = (f"F_tensors={len(f_vals)} numel={f_numel} "
                          f"F_max={f_max:.3e} F_mean={f_mean:.3e} F_sum={f_sum:.3e}")
            r_info = f"ref_tensors={len(self._delta_reference)}" if has_r else "ref=EMPTY"
            self._log(
                f"[FisherSAM-EWCLoRA-lambda][TaskStart] task={self._cur_task} "
                f"lambda={self._ewc_lambda:.3e} "
                f"has_fisher={has_f} {f_info} | "
                f"has_ref={has_r} {r_info}"
            )

        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        info = ""

        for epoch in prog_bar:
            self._network.train()
            total_loss = 0.0
            ce_losses = 0.0
            fsam_losses = 0.0
            ewc_losses = 0.0
            loss_count = 0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                logits, loss_value, log_dict = self._fsam_ewc_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=inputs,
                    targets=targets,
                    class_offset=class_offset,
                )

                total_loss += float(loss_value)
                ce_losses += float(log_dict.get("ce_clean", 0.0))
                fsam_losses += float(log_dict.get("fsam", 0.0))
                ewc_losses += float(log_dict.get("ewc", 0.0))
                loss_count += 1

                if self._mechanism_eval:
                    self._record_mechanism_stats(log_dict)

                with torch.no_grad():
                    if class_offset > 0:
                        eval_logits = logits[:, class_offset:]
                        eval_targets = targets - class_offset
                    else:
                        eval_logits = logits
                        eval_targets = targets
                    preds = torch.argmax(eval_logits, dim=1)
                    correct += int(preds.eq(eval_targets).sum().item())
                    total += int(eval_targets.numel())

            if scheduler is not None:
                scheduler.step()

            denom = max(1, loss_count)
            epoch_ewc_ratio = float(ewc_losses / (fsam_losses + self._mechanism_eps))
            if self._dual_update_by_epoch:
                epoch_dual_stats = self._maybe_update_dual_lambda(
                    model,
                    current_loss=ce_losses / denom,
                    current_ewc_ratio=epoch_ewc_ratio,
                    optimizer=optimizer,
                )
                if self._mechanism_eval and epoch_dual_stats.get("dual_updated", 0.0):
                    self._record_mechanism_stats(epoch_dual_stats)

            train_acc = 0.0 if total == 0 else round(100.0 * correct / total, 2)
            drift_display = (
                float(self._compute_fisher_drift_value(model).detach().item())
                if self._fisher_past_delta and self._delta_reference
                else 0.0
            )
            info = (
                f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                f"Loss {total_loss / denom:.3f}, Train_accy {train_acc:.2f}, "
                f"CE {ce_losses / denom:.3f}, FSAM {fsam_losses / denom:.3f}, "
                f"EWC {ewc_losses / denom:.3e}, D_F {drift_display:.3e}, "
                f"EWC/FSAM {epoch_ewc_ratio:.3e}, lambda {float(self._ewc_lambda):.3g}"
            )
            if ((epoch % 5 == 4) or epoch == epochs - 1) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            if self._should_capture_in_task_fisher(epoch + 1):
                snapshot_count = self._capture_in_task_fisher_snapshot(
                    train_loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    epoch_idx=epoch + 1,
                    class_offset=class_offset,
                )
                if snapshot_count > 0:
                    info += f", Fisher_snap {snapshot_count}"

            if self._is_main_process:
                prog_bar.set_description(info)

        if self._is_main_process and info:
            self._log(info)

    def _should_capture_in_task_fisher(self, epoch_idx: int) -> bool:
        if not self._approximate_fisher_in_task:
            return False
        if self._in_task_fisher_interval <= 0:
            return False
        if self._in_task_fisher_omega <= 0.0:
            return False
        return int(epoch_idx) % int(self._in_task_fisher_interval) == 0

    def _capture_lora_weight_snapshot(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        markers = (
            "linear_a",
            "linear_b",
            "lora_a",
            "lora_b",
            "w_as",
            "w_bs",
            "scaling_factor",
            "wrapped_param",
        )
        snapshot: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            lower_name = name.lower()
            if any(marker in lower_name for marker in markers):
                snapshot[name] = param.detach().cpu().clone()
        return snapshot

    def _capture_in_task_fisher_snapshot(
        self,
        train_loader,
        model: nn.Module,
        optimizer: DiagonalFisherSAM,
        epoch_idx: int,
        class_offset: int,
    ) -> int:
        was_training = bool(model.training)
        logger.info(
            "[FisherSAM-EWCLoRA-lambda][InTaskFisher] Computing raw delta-W "
            "snapshot task=%d epoch=%d",
            self._cur_task,
            epoch_idx,
        )
        try:
            fisher = super()._compute_delta_fisher(train_loader, model, class_offset)
        finally:
            optimizer.zero_grad()
            model.train(was_training)

        fisher_snapshot = {
            key: value.detach().cpu().clone()
            for key, value in fisher.items()
        }
        lora_snapshot = self._capture_lora_weight_snapshot(model)
        self._in_task_fisher_snapshots.append((int(epoch_idx), fisher_snapshot, lora_snapshot))
        logger.info(
            "[FisherSAM-EWCLoRA-lambda][InTaskFisher] Saved snapshot "
            "task=%d epoch=%d count=%d fisher_trace=%.6g lora_tensors=%d",
            self._cur_task,
            epoch_idx,
            len(self._in_task_fisher_snapshots),
            self._fisher_trace(fisher_snapshot),
            len(lora_snapshot),
        )
        return len(self._in_task_fisher_snapshots)

    # ------------------------------------------------------------------
    # Fisher-SAM + EWC batch step
    # ------------------------------------------------------------------
    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        del scheduler
        logits, loss_value, log_dict = self._fsam_ewc_step(
            model=self._unwrap_network(),
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            class_offset=class_offset,
        )
        if self._mechanism_eval:
            self._record_mechanism_stats(log_dict)
        return logits, loss_value

    def _iter_optimizer_params(self, optimizer):
        base_opt = getattr(optimizer, "base_optimizer", optimizer)
        for group in getattr(base_opt, "param_groups", []):
            for param in group.get("params", []):
                yield param

    @torch.no_grad()
    def _capture_grad_cache(self, optimizer) -> Dict[int, torch.Tensor]:
        grads: Dict[int, torch.Tensor] = {}
        for param in self._iter_optimizer_params(optimizer):
            if param.grad is not None:
                grads[id(param)] = param.grad.detach().clone()
        return grads

    @staticmethod
    def _cosine_from_sums(dot: float, norm_a_sq: float, norm_b_sq: float, eps: float) -> float:
        denom = (max(norm_a_sq, 0.0) ** 0.5) * (max(norm_b_sq, 0.0) ** 0.5)
        return float(dot / (denom + eps))

    @torch.no_grad()
    def _write_projected_fsam_ewc_grads(
        self,
        optimizer,
        fsam_grad_cache: Dict[int, torch.Tensor],
        ewc_grad_cache: Dict[int, torch.Tensor],
        drift_value: float,
    ) -> Dict[str, float]:
        eps = float(self._ewc_project_eps)
        task_norm_sq = 0.0
        ewc_norm_sq = 0.0
        task_ewc_dot = 0.0
        param_count = 0

        for param in self._iter_optimizer_params(optimizer):
            task_grad = fsam_grad_cache.get(id(param))
            ewc_grad = ewc_grad_cache.get(id(param))
            if task_grad is None and ewc_grad is None:
                continue

            param_count += 1
            if task_grad is not None:
                task_f = task_grad.detach().float()
                task_norm_sq += float((task_f * task_f).sum().item())
            if ewc_grad is not None:
                ewc_f = ewc_grad.detach().float()
                ewc_norm_sq += float((ewc_f * ewc_f).sum().item())
            if task_grad is not None and ewc_grad is not None:
                task_ewc_dot += float((task_grad.detach().float() * ewc_grad.detach().float()).sum().item())

        threshold = float(self._ewc_project_rho * self._dual_budget)
        boundary_active = (not self._ewc_project_boundary) or (float(drift_value) > threshold)
        outward = task_ewc_dot < 0.0
        apply_projection = (
            self._ewc_project_grad
            and boundary_active
            and ewc_norm_sq > eps
            and ((not self._ewc_project_outward_only) or outward)
        )
        coeff = float(task_ewc_dot / (ewc_norm_sq + eps)) if apply_projection else 0.0

        for param in self._iter_optimizer_params(optimizer):
            task_grad = fsam_grad_cache.get(id(param))
            ewc_grad = ewc_grad_cache.get(id(param))
            if task_grad is None and ewc_grad is None:
                param.grad = None
                continue

            if task_grad is None:
                task_term = torch.zeros_like(ewc_grad, device=param.device, dtype=param.dtype)
            else:
                task_term = task_grad.to(device=param.device, dtype=param.dtype)

            if ewc_grad is None:
                ewc_term = torch.zeros_like(task_term, device=param.device, dtype=param.dtype)
            else:
                ewc_term = ewc_grad.to(device=param.device, dtype=param.dtype)

            if apply_projection:
                task_term = task_term - coeff * ewc_term
            final_grad = task_term + ewc_term

            if param.grad is None:
                param.grad = final_grad.detach().clone()
            else:
                param.grad.data.copy_(final_grad)

        return {
            "project_enabled": float(self._ewc_project_grad),
            "project_applied": float(apply_projection),
            "project_boundary_active": float(boundary_active),
            "project_outward": float(outward),
            "project_drift": float(drift_value),
            "project_threshold": float(threshold),
            "project_dot": float(task_ewc_dot),
            "project_task_norm": float(task_norm_sq ** 0.5),
            "project_ewc_norm": float(ewc_norm_sq ** 0.5),
            "project_cosine": self._cosine_from_sums(task_ewc_dot, task_norm_sq, ewc_norm_sq, eps),
            "project_coeff": float(coeff),
            "project_param_count": float(param_count),
        }

    def _fsam_ewc_step(
        self,
        model: nn.Module,
        optimizer: DiagonalFisherSAM,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        class_offset: int,
    ) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        optimizer.zero_grad()
        perturbed = False
        safe_stats: Dict[str, float] = {}
        project_stats: Dict[str, float] = {}

        try:
            clean_outputs = self._network(inputs)
            clean_logits = clean_outputs["logits"]
            clean_loss, _, _ = self._task_loss(clean_logits, targets, class_offset)
            clean_loss.backward()

            optimizer.first_step(zero_grad=True)
            perturbed = True

            pert_outputs = self._network(inputs)
            pert_logits = pert_outputs["logits"]
            fsam_loss, _, _ = self._task_loss(pert_logits, targets, class_offset)
            fsam_loss.backward()
            fsam_grad_cache = self._capture_grad_cache(optimizer)

            optimizer.restore()
            perturbed = False

            safe_stats = self._apply_joint_lambda_clip(
                optimizer,
                reason="before_ewc_backward",
            )
            optimizer.zero_grad()
            ewc_loss = self._past_fisher_penalty(model)
            if torch.is_tensor(ewc_loss) and ewc_loss.requires_grad:
                ewc_loss.backward()
            ewc_grad_cache = self._capture_grad_cache(optimizer)

            drift_value = (
                float(self._compute_fisher_drift_value(model).detach().item())
                if self._fisher_past_delta and self._delta_reference
                else 0.0
            )
            project_stats = self._write_projected_fsam_ewc_grads(
                optimizer,
                fsam_grad_cache=fsam_grad_cache,
                ewc_grad_cache=ewc_grad_cache,
                drift_value=drift_value,
            )

            # ── First-batch diagnostic: verify EWC scale and gradient health ──
            if not getattr(self, '_diag_first_batch_done', True) and self._is_main_process:
                self._diag_first_batch_done = True
                ewc_diag = float(ewc_loss.detach().item()) if torch.is_tensor(ewc_loss) else float(ewc_loss)
                total_norm_sq = 0.0
                nan_params: list = []
                for pname, p in model.named_parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            nan_params.append(pname)
                        total_norm_sq += float(p.grad.detach().float().norm(2).pow(2).item())
                grad_norm = total_norm_sq ** 0.5
                self._log(
                    f"[FisherSAM-EWCLoRA-lambda][FirstBatch] task={self._cur_task} "
                    f"ewc_penalty={ewc_diag:.3e} grad_norm={grad_norm:.3e} "
                    f"nan_grad_params={nan_params[:5] if nan_params else 'none'}"
                )

            optimizer.base_optimizer.step()
        finally:
            if perturbed:
                optimizer.restore()
            optimizer.zero_grad()

        fsam_value = float(fsam_loss.detach().item())
        ewc_value = float(ewc_loss.detach().item() if torch.is_tensor(ewc_loss) else ewc_loss)
        clean_value = float(clean_loss.detach().item())
        fsam_ewc_ratio = float(ewc_value / (fsam_value + self._mechanism_eps))
        if self._dual_update_by_epoch:
            dual_stats = self._default_dual_stats(ewc_ratio=fsam_ewc_ratio)
        else:
            dual_stats = self._maybe_update_dual_lambda(
                model,
                current_loss=clean_value,
                current_ewc_ratio=fsam_ewc_ratio,
                optimizer=optimizer,
            )
        log_dict = {
            "ce_clean": clean_value,
            "fsam": fsam_value,
            "ewc": ewc_value,
            "lambda_ewc": float(self._ewc_lambda),
            "fsam_ewc_ratio": fsam_ewc_ratio,
            **dual_stats,
            **safe_stats,
            **project_stats,
        }
        return pert_logits.detach(), fsam_value + ewc_value, log_dict

    # ------------------------------------------------------------------
    # In-task raw Fisher path averaging
    # ------------------------------------------------------------------
    def _fisher_trace(self, fisher: Optional[Dict[str, torch.Tensor]]) -> float:
        if not fisher:
            return 0.0
        total = 0.0
        for value in fisher.values():
            total += float(value.detach().float().clamp_min(0).sum().item())
        return float(total)

    def _match_delta_tensor(self, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        ref = reference.detach().cpu()
        value_cpu = value.detach().cpu().to(dtype=ref.dtype)
        if tuple(value_cpu.shape) == tuple(ref.shape):
            return value_cpu

        matched = torch.zeros_like(ref)
        slices = tuple(
            slice(0, min(int(value_cpu.shape[dim]), int(ref.shape[dim])))
            for dim in range(min(value_cpu.ndim, ref.ndim))
        )
        if len(slices) == value_cpu.ndim == ref.ndim:
            matched[slices] = value_cpu[slices]
        return matched

    def _mean_in_task_fisher(self, model: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        if not self._in_task_fisher_snapshots:
            return None

        mean_fisher: Dict[str, torch.Tensor] = {}
        for key, delta_now in self._iter_delta_terms(model):
            acc = None
            count = 0
            for _, fisher_snapshot, _ in self._in_task_fisher_snapshots:
                value = fisher_snapshot.get(key)
                if value is None:
                    continue
                value = self._match_delta_tensor(value, delta_now)
                acc = value.clone() if acc is None else acc + value
                count += 1
            if acc is not None and count > 0:
                mean_fisher[key] = acc / float(count)
        return mean_fisher or None

    def _mix_in_task_path_fisher(
        self,
        task_fisher: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        snapshot_epochs = [epoch for epoch, _, _ in self._in_task_fisher_snapshots]
        path_fisher = self._mean_in_task_fisher(model)
        final_trace = self._fisher_trace(task_fisher)
        path_trace = self._fisher_trace(path_fisher)
        lora_snapshot_tensors = sum(
            len(lora_snapshot)
            for _, _, lora_snapshot in self._in_task_fisher_snapshots
        )

        use_path = (
            self._approximate_fisher_in_task
            and self._in_task_fisher_omega > 0.0
            and path_fisher is not None
        )
        if not use_path:
            self._in_task_fisher_last_metrics = {
                "approximte_fisher_inTask": bool(self._approximate_fisher_in_task),
                "in_task_fisher_used_path": False,
                "in_task_fisher_interval": int(self._in_task_fisher_interval),
                "in_task_fisher_omega": float(self._in_task_fisher_omega),
                "in_task_fisher_snapshots": int(len(snapshot_epochs)),
                "in_task_fisher_first_epoch": int(snapshot_epochs[0]) if snapshot_epochs else 0,
                "in_task_fisher_last_epoch": int(snapshot_epochs[-1]) if snapshot_epochs else 0,
                "in_task_fisher_final_trace": float(final_trace),
                "in_task_fisher_path_trace": float(path_trace),
                "in_task_fisher_mixed_trace": float(final_trace),
                "in_task_lora_weight_snapshots": int(len(snapshot_epochs)),
                "in_task_lora_snapshot_tensors": int(lora_snapshot_tensors),
            }
            return task_fisher

        omega = float(self._in_task_fisher_omega)
        mixed: Dict[str, torch.Tensor] = {}
        for key, delta_now in self._iter_delta_terms(model):
            final_value = task_fisher.get(key)
            path_value = path_fisher.get(key)
            if final_value is None and path_value is None:
                continue
            if final_value is None:
                mixed[key] = self._match_delta_tensor(path_value, delta_now).clone()
            elif path_value is None:
                mixed[key] = self._match_delta_tensor(final_value, delta_now).clone()
            else:
                final_t = self._match_delta_tensor(final_value, delta_now)
                path_t = self._match_delta_tensor(path_value, delta_now)
                mixed[key] = (1.0 - omega) * final_t + omega * path_t

        mixed_trace = self._fisher_trace(mixed)
        self._in_task_fisher_last_metrics = {
            "approximte_fisher_inTask": bool(self._approximate_fisher_in_task),
            "in_task_fisher_used_path": True,
            "in_task_fisher_interval": int(self._in_task_fisher_interval),
            "in_task_fisher_omega": float(omega),
            "in_task_fisher_snapshots": int(len(snapshot_epochs)),
            "in_task_fisher_first_epoch": int(snapshot_epochs[0]) if snapshot_epochs else 0,
            "in_task_fisher_last_epoch": int(snapshot_epochs[-1]) if snapshot_epochs else 0,
            "in_task_fisher_final_trace": float(final_trace),
            "in_task_fisher_path_trace": float(path_trace),
            "in_task_fisher_mixed_trace": float(mixed_trace),
            "in_task_lora_weight_snapshots": int(len(snapshot_epochs)),
            "in_task_lora_snapshot_tensors": int(lora_snapshot_tensors),
        }
        logger.info(
            "[FisherSAM-EWCLoRA-lambda][InTaskFisher] Mixed final/path raw Fisher "
            "task=%d snapshots=%d omega=%.3g final_trace=%.6g path_trace=%.6g "
            "mixed_trace=%.6g lora_snapshot_tensors=%d",
            self._cur_task,
            len(snapshot_epochs),
            omega,
            final_trace,
            path_trace,
            mixed_trace,
            lora_snapshot_tensors,
        )
        return mixed

    # ------------------------------------------------------------------
    # Dual ascent and drift metrics
    # ------------------------------------------------------------------
    def _sync_lambda(self, value: float) -> None:
        self._ewc_lambda = float(value)
        self._lambda_ewc = float(value)

    def _get_current_base_lr(self, optimizer=None) -> float:
        """Return the live base optimizer lr, falling back to the last built lr."""
        if optimizer is not None:
            base_opt = getattr(optimizer, "base_optimizer", optimizer)
            if hasattr(base_opt, "param_groups") and len(base_opt.param_groups) > 0:
                lr_values = []
                for group in base_opt.param_groups:
                    lr = float(group.get("lr", 0.0))
                    if lr > 0.0:
                        lr_values.append(lr)
                if lr_values:
                    return max(lr_values)

        return float(getattr(self, "_last_base_lr", 0.0))

    @torch.no_grad()
    def _current_effective_fisher_max(self) -> float:
        """Return max diagonal Fisher in the same effective form used by EWC."""
        if not self._fisher_past_delta:
            return 0.0

        f_max = 0.0
        for fisher in self._fisher_past_delta.values():
            if fisher is None or fisher.numel() == 0:
                continue
            value = float(fisher.detach().float().clamp_min(0).max().item())
            f_max = max(f_max, value)

        if self._dual_use_eta and self._fisher_eta != 0.0:
            f_max += float(self._fisher_eta)

        return float(f_max)

    def _compute_safe_lambda_bounds(self, optimizer=None) -> Tuple[float, float, Dict[str, float]]:
        """Compute effective lambda bounds under lr * lambda * F_max <= c."""
        static_min = float(self._dual_lambda_min)
        static_max = float(self._dual_lambda_max)

        if not self._safe_lambda_cap:
            return static_min, static_max, {
                "safe_lambda_enabled": 0.0,
                "safe_lambda_min_eff": static_min,
                "safe_lambda_max_eff": static_max,
                "safe_lambda_raw_max": static_max,
                "safe_lambda_fmax": 0.0,
                "safe_lambda_lr": 0.0,
                "safe_lambda_c": float(self._safe_lambda_c),
                "safe_lambda_min_overridden": 0.0,
            }

        lr = self._get_current_base_lr(optimizer)
        f_max = self._current_effective_fisher_max()
        if lr <= 0.0 or f_max <= 0.0:
            raw_safe_max = static_max
        else:
            raw_safe_max = float(self._safe_lambda_c) / (
                lr * (f_max + float(self._safe_lambda_eps))
            )

        capped_max = min(static_max, float(raw_safe_max))
        min_overridden = 0.0
        if capped_max < static_min:
            if self._safe_lambda_override_min:
                effective_min = capped_max
                effective_max = capped_max
                min_overridden = 1.0
            else:
                effective_min = static_min
                effective_max = static_min
        else:
            effective_min = static_min
            effective_max = capped_max

        self._last_safe_lambda_max = float(effective_max)

        return float(effective_min), float(effective_max), {
            "safe_lambda_enabled": 1.0,
            "safe_lambda_min_eff": float(effective_min),
            "safe_lambda_max_eff": float(effective_max),
            "safe_lambda_raw_max": float(raw_safe_max),
            "safe_lambda_fmax": float(f_max),
            "safe_lambda_lr": float(lr),
            "safe_lambda_c": float(self._safe_lambda_c),
            "safe_lambda_min_overridden": float(min_overridden),
        }

    def _clip_lambda_by_joint_bounds(
        self,
        lambda_value: float,
        optimizer=None,
        reason: str = "",
    ) -> Tuple[float, Dict[str, float]]:
        """Clip lambda by both static dual bounds and the Fisher stability cap."""
        effective_min, effective_max, stats = self._compute_safe_lambda_bounds(optimizer)
        old_value = float(lambda_value)
        clipped_value = min(effective_max, max(effective_min, old_value))

        stats.update({
            "safe_lambda_before_clip": old_value,
            "safe_lambda_after_clip": float(clipped_value),
            "safe_lambda_clipped": float(abs(clipped_value - old_value) > 0.0),
        })

        if abs(clipped_value - old_value) > 0.0 and self._is_main_process:
            self._log(
                f"[FisherSAM-EWCLoRA-lambda][JointLambdaClip] "
                f"reason={reason} lambda {old_value:.6g} -> {clipped_value:.6g} "
                f"bounds=[{effective_min:.6g}, {effective_max:.6g}] "
                f"raw_safe_max={stats['safe_lambda_raw_max']:.6g} "
                f"F_max={stats['safe_lambda_fmax']:.6g} "
                f"lr={stats['safe_lambda_lr']:.6g} "
                f"c={stats['safe_lambda_c']:.6g}"
            )

        return float(clipped_value), stats

    def _apply_joint_lambda_clip(self, optimizer=None, reason: str = "") -> Dict[str, float]:
        """Apply joint lambda clipping to the current EWC lambda."""
        old_lambda = float(self._ewc_lambda)
        new_lambda, stats = self._clip_lambda_by_joint_bounds(
            old_lambda,
            optimizer=optimizer,
            reason=reason,
        )
        self._sync_lambda(new_lambda)

        stats.update({
            "lambda_before_joint_clip": old_lambda,
            "lambda_after_joint_clip": float(new_lambda),
        })
        return stats

    @torch.no_grad()
    def _compute_fisher_drift_value(self, model: nn.Module) -> torch.Tensor:
        if not self._fisher_past_delta or not self._delta_reference:
            return torch.zeros((), device=self._device)

        drift = torch.zeros((), device=self._device)
        for key, delta_now in self._iter_delta_terms(model):
            fisher = self._fisher_past_delta.get(key)
            delta_ref = self._delta_reference.get(key)
            if fisher is None or delta_ref is None:
                continue

            fisher_t = fisher.to(device=delta_now.device, dtype=delta_now.dtype).clamp_min(0)
            if self._dual_use_eta and self._fisher_eta != 0.0:
                fisher_t = fisher_t + self._fisher_eta
            delta_ref_t = delta_ref.to(device=delta_now.device, dtype=delta_now.dtype)
            drift = drift + 0.5 * (fisher_t * (delta_now.detach() - delta_ref_t).pow(2)).sum()
        return drift

    @torch.no_grad()
    def _compute_joint_drift_values(self, model: nn.Module) -> Dict[str, float]:
        """Compute Fisher, Euclidean, and F_max-scaled Euclidean drift.

        The Fisher term follows the existing controller convention:
            D_F = 0.5 * sum(F_eff * delta^2)

        Euclidean and spectral terms follow the requested constraints:
            D_E = ||delta||_2^2
            D_M = F_max * ||delta||_2^2
        """
        if not self._fisher_past_delta or not self._delta_reference:
            return {
                "fisher": 0.0,
                "euclidean": 0.0,
                "spectral": 0.0,
                "fmax": 0.0,
            }

        fisher_drift = 0.0
        euclidean_drift = 0.0
        f_max = 0.0

        for key, delta_now in self._iter_delta_terms(model):
            fisher = self._fisher_past_delta.get(key)
            delta_ref = self._delta_reference.get(key)
            if fisher is None or delta_ref is None:
                continue

            fisher_t = fisher.to(device=delta_now.device, dtype=delta_now.dtype).clamp_min(0)
            if self._dual_use_eta and self._fisher_eta != 0.0:
                fisher_t = fisher_t + self._fisher_eta
            delta_ref_t = delta_ref.to(device=delta_now.device, dtype=delta_now.dtype)
            update_sq = (delta_now.detach() - delta_ref_t).pow(2)

            fisher_drift += float((0.5 * fisher_t * update_sq).sum().item())
            euclidean_drift += float(update_sq.sum().item())
            if fisher_t.numel() > 0:
                f_max = max(f_max, float(fisher_t.max().item()))

        return {
            "fisher": float(fisher_drift),
            "euclidean": float(euclidean_drift),
            "spectral": float(f_max * euclidean_drift),
            "fmax": float(f_max),
        }

    def _default_dual_constraint_stats(self) -> Dict[str, float]:
        return {
            "dual_fisher_drift": 0.0,
            "dual_fisher_budget": float(self._dual_budget),
            "dual_fisher_rho_budget": float(self._dual_rho * self._dual_budget),
            "dual_fisher_ratio": 0.0,
            "dual_fisher_violation": 0.0,
            "dual_euclidean_drift": 0.0,
            "dual_euclidean_budget": float(self._dual_euclidean_budget or 0.0),
            "dual_euclidean_rho_budget": float(self._dual_rho * (self._dual_euclidean_budget or 0.0)),
            "dual_euclidean_ratio": 0.0,
            "dual_euclidean_violation": 0.0,
            "dual_euclidean_active": float(
                self._dual_use_joint_constraints and self._dual_euclidean_budget is not None
            ),
            "dual_spectral_drift": 0.0,
            "dual_spectral_budget": float(self._dual_spectral_budget or 0.0),
            "dual_spectral_rho_budget": float(self._dual_rho * (self._dual_spectral_budget or 0.0)),
            "dual_spectral_ratio": 0.0,
            "dual_spectral_violation": 0.0,
            "dual_spectral_active": float(
                self._dual_use_joint_constraints and self._dual_spectral_budget is not None
            ),
            "dual_spectral_fmax": 0.0,
            "dual_max_ratio": 0.0,
            "dual_critical_constraint": 0.0,
        }

    def _compute_dual_constraint_stats(self, model: nn.Module) -> Dict[str, float]:
        values = self._compute_joint_drift_values(model)

        fisher_budget = float(self._dual_budget)
        fisher_ratio = float(values["fisher"] / (fisher_budget + self._dual_eps))
        constraints = [(fisher_ratio, 1.0)]

        euclidean_active = bool(
            self._dual_use_joint_constraints and self._dual_euclidean_budget is not None
        )
        euclidean_budget = float(self._dual_euclidean_budget or 0.0)
        euclidean_ratio = (
            float(values["euclidean"] / (euclidean_budget + self._dual_eps))
            if euclidean_active
            else 0.0
        )
        if euclidean_active:
            constraints.append((euclidean_ratio, 2.0))

        spectral_active = bool(
            self._dual_use_joint_constraints and self._dual_spectral_budget is not None
        )
        spectral_budget = float(self._dual_spectral_budget or 0.0)
        spectral_ratio = (
            float(values["spectral"] / (spectral_budget + self._dual_eps))
            if spectral_active
            else 0.0
        )
        if spectral_active:
            constraints.append((spectral_ratio, 3.0))

        max_ratio, critical_constraint = max(constraints, key=lambda item: item[0])

        return {
            "dual_fisher_drift": float(values["fisher"]),
            "dual_fisher_budget": fisher_budget,
            "dual_fisher_rho_budget": float(self._dual_rho * fisher_budget),
            "dual_fisher_ratio": fisher_ratio,
            "dual_fisher_violation": float(fisher_ratio - 1.0),
            "dual_euclidean_drift": float(values["euclidean"]),
            "dual_euclidean_budget": euclidean_budget,
            "dual_euclidean_rho_budget": float(self._dual_rho * euclidean_budget),
            "dual_euclidean_ratio": euclidean_ratio,
            "dual_euclidean_violation": float(euclidean_ratio - 1.0) if euclidean_active else 0.0,
            "dual_euclidean_active": float(euclidean_active),
            "dual_spectral_drift": float(values["spectral"]),
            "dual_spectral_budget": spectral_budget,
            "dual_spectral_rho_budget": float(self._dual_rho * spectral_budget),
            "dual_spectral_ratio": spectral_ratio,
            "dual_spectral_violation": float(spectral_ratio - 1.0) if spectral_active else 0.0,
            "dual_spectral_active": float(spectral_active),
            "dual_spectral_fmax": float(values["fmax"]),
            "dual_max_ratio": float(max_ratio),
            "dual_critical_constraint": float(critical_constraint),
        }

    def _reset_dual_controller_state(self) -> None:
        self._dual_step_count = 0
        # commented-out: loss buffer tracking (engineering guardrail, not core dual ascent)
        # self._dual_loss_buffer = []
        # self._dual_ewc_ratio_buffer = []
        # self._dual_prev_loss_window = None

    # commented-out: ratio gate was an indirect proxy, not part of D_F ≤ σ formulation
    # def _dual_ratio_gate_value(self, ewc_ratio: float) -> float:
    #     if not self._dual_use_ewc_ratio_gate:
    #         return 1.0
    #     return float(float(ewc_ratio) > self._dual_ewc_ratio_threshold)

    def _default_dual_stats(self, ewc_ratio: float = 0.0, optimizer=None) -> Dict[str, float]:
        del ewc_ratio
        budget = float(self._dual_budget)
        _, _, safe_stats = self._compute_safe_lambda_bounds(optimizer)
        return {
            "dual_drift": 0.0,
            "dual_budget": budget,
            "dual_rho_budget": float(self._dual_rho * budget),
            "dual_violation": 0.0,
            "lambda_before": float(self._ewc_lambda),
            "lambda_after": float(self._ewc_lambda),
            "dual_updated": 0.0,
            "dual_mode": 0.0,
            **self._default_dual_constraint_stats(),
            **safe_stats,
        }

    def _maybe_update_dual_lambda(
        self,
        model: nn.Module,
        current_loss: float,
        current_ewc_ratio: float = 0.0,
        optimizer: Optional[DiagonalFisherSAM] = None,
    ) -> Dict[str, float]:
        del current_loss, current_ewc_ratio  # dual update uses configured drift constraints only
        default_stats = self._default_dual_stats(optimizer=optimizer)
        if not self._dual_ascent or not self._fisher_past_delta or not self._delta_reference:
            return default_stats

        self._dual_step_count += 1
        if self._dual_step_count % self._dual_update_interval != 0:
            return default_stats

        return self._update_dual_lambda(model, optimizer=optimizer)

        # --- commented-out engineering guardrails ---
        # self._dual_loss_buffer.append(float(current_loss))
        # self._dual_ewc_ratio_buffer.append(float(current_ewc_ratio))
        # if len(self._dual_loss_buffer) < self._dual_loss_window:
        #     return default_stats
        # curr_window = float(np.mean(self._dual_loss_buffer[-self._dual_loss_window:]))
        # curr_ewc_ratio = float(np.mean(self._dual_ewc_ratio_buffer[-self._dual_loss_window:]))
        # ... loss_improve window comparison ...
        # loss_improve = (prev_window - curr_window) / (abs(prev_window) + self._dual_eps)

    def _update_dual_lambda(
        self,
        model: nn.Module,
        optimizer: Optional[DiagonalFisherSAM] = None,
    ) -> Dict[str, float]:
        """Version B dual ascent plus Fisher-aware stability clipping.

        First propose lambda from the worst normalized active drift constraint,
        then clip it to the stability range constrained by
        lr * lambda * F_max <= c.
        """
        if not self._dual_ascent or not self._fisher_past_delta or not self._delta_reference:
            return self._default_dual_stats(optimizer=optimizer)

        constraint_stats = self._compute_dual_constraint_stats(model)
        drift_value = float(constraint_stats["dual_fisher_drift"])
        budget = float(self._dual_budget)   # σ
        rho_budget = self._dual_rho * budget  # ρσ
        max_ratio = float(constraint_stats["dual_max_ratio"])

        old_lambda = float(self._ewc_lambda)
        base = old_lambda if old_lambda > 0.0 else max(float(self._dual_lambda_min), self._dual_eps)
        lambda_proposal = old_lambda
        updated = 0.0
        mode = 0.0

        if max_ratio > 1.0:
            # Any active constraint is over budget: respond to the worst normalized violation.
            rel = max_ratio - 1.0
            lambda_proposal = base * float(np.exp(np.clip(self._dual_eta_up * rel, -50.0, 50.0)))
            updated = 1.0
            mode = 1.0
        elif max_ratio < self._dual_rho:
            # All active constraints are well under budget: relax EWC.
            rel = 1.0 - max_ratio
            lambda_proposal = base * float(np.exp(np.clip(-self._dual_eta_down * rel, -50.0, 50.0)))
            updated = 1.0
            mode = -1.0
        # else: dead-band [ρ, 1] on the worst normalized constraint.

        new_lambda, safe_stats = self._clip_lambda_by_joint_bounds(
            lambda_proposal,
            optimizer=optimizer,
            reason="dual_update",
        )
        self._sync_lambda(new_lambda)

        violation = drift_value / (budget + self._dual_eps) - 1.0
        return {
            "dual_drift": drift_value,
            "dual_budget": budget,
            "dual_rho_budget": rho_budget,
            "dual_violation": float(max_ratio - 1.0),
            "dual_fisher_legacy_violation": violation,
            "lambda_before": old_lambda,
            "lambda_proposal": float(lambda_proposal),
            "lambda_after": float(new_lambda),
            "dual_updated": updated,
            "dual_mode": mode,
            **constraint_stats,
            **safe_stats,
        }

    def _compute_delta_drift_metrics(self) -> Dict[str, float]:
        if not self._fisher_past_delta or not self._delta_reference:
            return {
                "delta_drift_available": 0,
                "delta_tensors": 0,
                "delta_norm": 0.0,
                "fisher_weighted_drift": 0.0,
                "fisher_eta_weighted_drift": 0.0,
                "normalized_fisher_drift": 0.0,
                "normalized_fisher_eta_drift": 0.0,
                "high_fisher_delta_energy_ratio": 0.0,
                "high_fisher_weighted_energy_ratio": 0.0,
                "ewc_penalty_value": 0.0,
            }

        model = self._unwrap_network()
        eps = float(self._mechanism_eps)
        delta_norm = 0.0
        fisher_weighted = 0.0
        fisher_eta_weighted = 0.0
        high_delta_energy = 0.0
        high_weighted_energy = 0.0
        fisher_sum = 0.0
        fisher_max = 0.0
        fisher_numel = 0
        tensor_count = 0

        with torch.no_grad():
            for key, delta_now in self._iter_delta_terms(model):
                fisher = self._fisher_past_delta.get(key)
                delta_ref = self._delta_reference.get(key)
                if fisher is None or delta_ref is None:
                    continue

                fisher_t = fisher.to(device=delta_now.device, dtype=delta_now.dtype).clamp_min(0)
                delta_ref_t = delta_ref.to(device=delta_now.device, dtype=delta_now.dtype)
                update_sq = (delta_now.detach() - delta_ref_t).pow(2)
                fisher_update = fisher_t * update_sq

                delta_norm += float(update_sq.sum().item())
                fisher_weighted += float(fisher_update.sum().item())
                fisher_eta_weighted += float(((fisher_t + self._fisher_eta) * update_sq).sum().item())
                fisher_sum += float(fisher_t.sum().item())
                fisher_max = max(fisher_max, float(fisher_t.max().item()) if fisher_t.numel() else 0.0)
                fisher_numel += int(fisher_t.numel())
                tensor_count += 1

                flat_fisher = fisher_t.reshape(-1)
                flat_update_sq = update_sq.reshape(-1)
                if flat_fisher.numel() > 0:
                    if self._mechanism_topk is not None:
                        k = min(max(int(self._mechanism_topk), 1), int(flat_fisher.numel()))
                    else:
                        frac = min(max(float(self._mechanism_top_fraction), 0.0), 1.0)
                        k = min(max(int(round(float(flat_fisher.numel()) * frac)), 1), int(flat_fisher.numel()))
                    top_idx = torch.topk(flat_fisher, k=k, largest=True, sorted=False).indices
                    high_delta_energy += float(flat_update_sq[top_idx].sum().item())
                    high_weighted_energy += float((flat_fisher[top_idx] * flat_update_sq[top_idx]).sum().item())

        return {
            "delta_drift_available": 1,
            "delta_tensors": int(tensor_count),
            "delta_norm": float(delta_norm),
            "fisher_weighted_drift": float(fisher_weighted),
            "fisher_eta_weighted_drift": float(fisher_eta_weighted),
            "normalized_fisher_drift": float(fisher_weighted / (delta_norm + eps)),
            "normalized_fisher_eta_drift": float(fisher_eta_weighted / (delta_norm + eps)),
            "high_fisher_delta_energy_ratio": float(high_delta_energy / (delta_norm + eps)),
            "high_fisher_weighted_energy_ratio": float(high_weighted_energy / (fisher_weighted + eps)),
            "fisher_mean": float(fisher_sum / max(fisher_numel, 1)),
            "fisher_max": float(fisher_max),
            "ewc_penalty_value": float(0.5 * self._ewc_lambda * fisher_eta_weighted),
        }

    # ------------------------------------------------------------------
    # Mechanism summaries
    # ------------------------------------------------------------------
    def _reset_mechanism_stats(self) -> None:
        self._mechanism_sums = {}
        self._mechanism_count = 0
        self._last_mechanism_metrics = {}

    def _snapshot_task_reference(self) -> None:
        super()._snapshot_task_reference()
        if self._mechanism_eval:
            self._reset_mechanism_stats()

    def _record_mechanism_stats(self, stats: Dict[str, float]) -> None:
        for key, value in stats.items():
            self._mechanism_sums[key] = self._mechanism_sums.get(key, 0.0) + float(value)
        self._mechanism_count += 1

    def _summarize_mechanism_stats(self) -> Dict[str, float]:
        if self._mechanism_count <= 0:
            return {}
        denom = float(self._mechanism_count)
        return {
            f"train_{key}_mean": value / denom
            for key, value in sorted(self._mechanism_sums.items())
        }

    def after_task(self):
        drift_before = (
            self._compute_delta_drift_metrics()
            if (self._mechanism_eval or self._dual_ascent)
            else {}
        )
        base_model = self._unwrap_network()
        new_fisher = super()._compute_delta_fisher(
            loader=self.train_loader,
            model=base_model,
            class_offset=int(self._known_classes),
        )
        new_fisher = self._mix_in_task_path_fisher(new_fisher, base_model)
        self._fisher_past_delta = super()._accumulate_fisher(self._fisher_past_delta, new_fisher)

        if self._is_main_process:
            # ── Fisher-stats diagnostic ──────────────────────────────────────
            f_vals = list(new_fisher.values())
            if f_vals:
                nf_max = float(max(v.max().item() for v in f_vals))
                nf_sum = float(sum(v.sum().item() for v in f_vals))
                nf_numel = sum(int(v.numel()) for v in f_vals)
                nf_mean = nf_sum / max(nf_numel, 1)
                self._log(
                    f"[FisherSAM-EWCLoRA-lambda][FisherStats] task={self._cur_task} "
                    f"new_fisher tensors={len(f_vals)} numel={nf_numel} "
                    f"max={nf_max:.3e} mean={nf_mean:.3e} sum={nf_sum:.3e}"
                )
            accum_vals = list(self._fisher_past_delta.values())
            if accum_vals:
                af_max = float(max(v.max().item() for v in accum_vals))
                af_sum = float(sum(v.sum().item() for v in accum_vals))
                af_numel = sum(int(v.numel()) for v in accum_vals)
                af_mean = af_sum / max(af_numel, 1)
                self._log(
                    f"[FisherSAM-EWCLoRA-lambda][FisherStats] task={self._cur_task} "
                    f"accum_fisher tensors={len(accum_vals)} numel={af_numel} "
                    f"max={af_max:.3e} mean={af_mean:.3e} sum={af_sum:.3e}"
                )
            self._log(
                "[FisherSAM-EWCLoRA-lambda] Updated raw delta-W Fisher for "
                f"{len(new_fisher)} delta tensors after task {self._cur_task}."
            )

        self._delta_reference = {}

        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._in_task_fisher_snapshots = []

        if self._mechanism_eval:
            self._last_mechanism_metrics = {
                "task": int(self._cur_task),
                "method_type": "fishersam_ewclora_lambda_project",
                "lambda_ewc": float(self._ewc_lambda),
                "fsam_gamma": float(self._fsam_gamma),
                "fsam_eta": float(self._fsam_eta),
                "fsam_base_optimizer": self._fsam_base_optimizer,
                "ewc_dual_ascent": bool(self._dual_ascent),
                "ewc_dual_delta": float(self._dual_budget),   # σ
                "ewc_dual_relative": bool(self._dual_relative),
                "ewc_dual_update_by_epoch": bool(self._dual_update_by_epoch),
                "ewc_dual_update_interval": int(self._dual_update_interval),
                "ewc_dual_eta_up": float(self._dual_eta_up),  # η↑
                "ewc_dual_lambda_min": float(self._dual_lambda_min),
                "ewc_dual_lambda_max": float(self._dual_lambda_max),
                "ewc_dual_use_joint_constraints": bool(self._dual_use_joint_constraints),
                "ewc_dual_euclidean_delta": float(self._dual_euclidean_budget or 0.0),
                "ewc_dual_spectral_delta": float(self._dual_spectral_budget or 0.0),
                "ewc_safe_lambda_cap": bool(self._safe_lambda_cap),
                "ewc_safe_lambda_c": float(self._safe_lambda_c),
                "ewc_safe_lambda_eps": float(self._safe_lambda_eps),
                "ewc_safe_lambda_override_min": bool(self._safe_lambda_override_min),
                "ewc_safe_lambda_last_max": float(self._last_safe_lambda_max),
                "ewc_project_grad": bool(self._ewc_project_grad),
                "ewc_project_eps": float(self._ewc_project_eps),
                "ewc_project_rho": float(self._ewc_project_rho),
                "ewc_project_boundary": bool(self._ewc_project_boundary),
                "ewc_project_outward_only": bool(self._ewc_project_outward_only),
                # commented-out: removed guardrail params
                # "ewc_dual_alpha": ..., "ewc_dual_eta_down": ...,
                # "ewc_dual_low_ratio": ..., "ewc_dual_loss_window": ...,
                # "ewc_dual_loss_patience": ..., "ewc_dual_use_ewc_ratio_gate": ...,
                # "ewc_dual_ewc_ratio_threshold": ...,
                **self._in_task_fisher_last_metrics,
                **self._summarize_mechanism_stats(),
                **drift_before,
            }
            if self._is_main_process:
                self._log(f"[FisherSAM-EWCLoRA-lambda][Mechanism] {self._last_mechanism_metrics}")

    def get_mechanism_metrics(self) -> Dict[str, float]:
        return dict(self._last_mechanism_metrics)


FisherSAMEWCLoRALambdaProject = Learner
FisherSAMEWCLoRALambda = Learner
EWCLoRAProject = Learner
EWCLoRA = Learner
