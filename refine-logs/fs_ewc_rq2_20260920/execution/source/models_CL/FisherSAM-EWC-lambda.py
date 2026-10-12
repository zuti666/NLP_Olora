"""Fisher-SAM + EWC with adaptive dual lambda.

This learner reuses the full-model EWCon_Probe infrastructure for incremental
training, Fisher estimation, mechanism metrics, and optional FQA diagnostics.
It can also cache current-task Fisher snapshots during training and mix their
path average with the final task Fisher before online EWC accumulation.

The per-batch update intentionally keeps the Fisher-SAM inner maximization
separate from the old-task EWC drift penalty:

    1. compute current-task CE gradient at theta;
    2. build the diagonal Fisher-SAM perturbation from that CE gradient only;
    3. compute current-task CE gradient at theta + eps;
    4. restore theta;
    5. add the EWC gradient at theta;
    6. apply the base optimizer step;
    7. optionally update lambda with a sparse asymmetric controller from the
       inherited raw-Fisher drift.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models_CL.EWCon_Probe import Learner as _EWConProbe
from optimer_PerturabtionType.optimer_fishersam_diagonal import DiagonalFisherSAM
from utils.mechanism_eval import as_bool
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(_EWConProbe):
    """Full-model Fisher-SAM current-task optimizer + EWC drift controller."""

    def __init__(self, args):
        super().__init__(args)

        self._fsam_gamma = float(
            args.get("fsam_gamma", args.get("fishersam_gamma", args.get("sam_rho", args.get("rho", 0.1))))
        )
        self._fsam_eta = float(args.get("fsam_eta", args.get("fisher_eta", 1.0)))
        self._fsam_damping = float(args.get("fsam_damping", args.get("fisher_sam_damping", 1e-12)))
        self._fsam_inverse_mode = str(args.get("fsam_inverse_mode", "paper")).lower()
        self._fsam_adaptive = as_bool(args.get("fsam_adaptive", args.get("sam_adaptive", False)))
        self._fsam_base_optimizer = str(
            args.get("fsam_base_optimizer", args.get("gam_base_optimizer", args.get("optimizer", "sgd")))
        ).lower()

        # Raw Fisher is the default inherited posterior precision. The dual
        # controller operates on the raw drift
        # D_F = 0.5 * (theta - theta_ref)^T F_cum (theta - theta_ref).
        # Trace normalization remains available as an explicit ablation.
        self._normalize_fisher = as_bool(args.get("ewc_normalize_fisher", False))
        self._fisher_norm_eps = float(args.get("ewc_fisher_norm_eps", 1e-12))
        self._renormalize_accumulated_fisher = as_bool(
            args.get("ewc_renormalize_accumulated_fisher", False)
        )
        if self._normalize_fisher:
            logger.warning(
                "[FisherSAM-EWC-lambda] ewc_normalize_fisher=True. "
                "This deviates from the default raw-Fisher posterior-precision setting."
            )

        # Dual controller for lambda. Lambda is a constraint feedback variable,
        # so it is updated sparsely and asymmetrically instead of after every
        # optimizer step.
        self._dual_ascent = as_bool(args.get("ewc_dual_ascent", args.get("fsam_ewc_dual_ascent", True)))
        self._dual_alpha = float(args.get("ewc_dual_alpha", args.get("eta_lambda", 1.0)))
        dual_budget_keys = ("ewc_dual_delta", "gamma_a_sq", "fsam_ewc_gamma_a_sq")
        dual_budget_is_explicit = any(key in args for key in dual_budget_keys)
        dual_budget_default = float(args.get("ewc_dual_raw_budget_default", 1e-3))
        self._dual_budget = float(
            args.get(
                "ewc_dual_delta",
                args.get("gamma_a_sq", args.get("fsam_ewc_gamma_a_sq", dual_budget_default)),
            )
        )
        if not dual_budget_is_explicit and not self._normalize_fisher:
            logger.warning(
                "[FisherSAM-EWC-lambda] Using raw-Fisher default ewc_dual_delta=%.4g. "
                "Raw Fisher drift is scale-dependent; tune ewc_dual_delta explicitly for final runs.",
                self._dual_budget,
            )
        self._dual_relative = as_bool(args.get("ewc_dual_relative", True))
        self._dual_lambda_min = float(args.get("ewc_dual_lambda_min", 1.0))
        self._dual_lambda_max = float(args.get("ewc_dual_lambda_max", 1e6))
        self._dual_eps = float(args.get("ewc_dual_eps", 1e-12))
        self._dual_trace_normalize_drift = as_bool(args.get("ewc_dual_trace_normalize_drift", False))
        self._dual_update_by_epoch = as_bool(args.get("ewc_dual_update_by_epoch", True))
        default_dual_interval = 5 if self._dual_update_by_epoch else 100
        default_dual_loss_window = 1 if self._dual_update_by_epoch else 50
        self._dual_update_interval = max(
            1,
            int(args.get("ewc_dual_update_interval", default_dual_interval)),
        )
        self._dual_eta_up = float(args.get("ewc_dual_eta_up", 0.05))
        self._dual_eta_down = float(args.get("ewc_dual_eta_down", 0.005))
        self._dual_low_ratio = float(args.get("ewc_dual_low_ratio", 0.5))
        self._dual_loss_window = max(
            1,
            int(args.get("ewc_dual_loss_window", default_dual_loss_window)),
        )
        self._dual_loss_patience = float(args.get("ewc_dual_loss_patience", 1e-3))
        self._dual_step_count = 0
        self._dual_loss_buffer: List[float] = []
        self._dual_prev_loss_window: Optional[float] = None

        # In-task Fisher approximation:
        #   F_bar = (1 - omega) * F_final + omega * mean_j F_epoch_j.
        # These snapshots are only used after the task ends; current-task EWC
        # still uses the previous tasks' accumulated Fisher.
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
        self._in_task_fisher_snapshots: List[Tuple[int, Dict[str, torch.Tensor]]] = []
        self._in_task_fisher_last_metrics: Dict[str, float] = {}
        self._in_task_fisher_skip_warned = False

        logger.info(
            "[FisherSAM-EWC-lambda] gamma=%.4g eta=%.4g inverse=%s base=%s "
            "normalize_fisher=%s dual=%s budget=%.4g relative=%s "
            "eta_up=%.4g eta_down=%.4g update_interval=%d by_epoch=%s "
            "low_ratio=%.4g loss_window=%d loss_patience=%.4g lambda_bounds=[%.4g, %.4g] "
            "lambda_init=%.4g approximte_fisher_inTask=%s interval=%d omega=%.3g",
            self._fsam_gamma,
            self._fsam_eta,
            self._fsam_inverse_mode,
            self._fsam_base_optimizer,
            self._normalize_fisher,
            self._dual_ascent,
            self._dual_budget,
            self._dual_relative,
            self._dual_eta_up,
            self._dual_eta_down,
            self._dual_update_interval,
            self._dual_update_by_epoch,
            self._dual_low_ratio,
            self._dual_loss_window,
            self._dual_loss_patience,
            self._dual_lambda_min,
            self._dual_lambda_max,
            self._ewc_lambda,
            self._approximate_fisher_in_task,
            self._in_task_fisher_interval,
            self._in_task_fisher_omega,
        )

    def after_task(self):
        super().after_task()
        if self._last_mechanism_metrics:
            self._last_mechanism_metrics.update({
                "method_type": "fishersam_ewc_lambda",
                "lambda_ewc": float(self._ewc_lambda if self._use_ewc else 0.0),
                "fsam_gamma": float(self._fsam_gamma),
                "fsam_eta": float(self._fsam_eta),
                "ewc_dual_ascent": bool(self._dual_ascent),
                "ewc_dual_alpha": float(self._dual_alpha),
                "ewc_dual_delta": float(self._dual_budget),
                "ewc_dual_relative": bool(self._dual_relative),
                "ewc_dual_update_by_epoch": bool(self._dual_update_by_epoch),
                "ewc_dual_update_interval": int(self._dual_update_interval),
                "ewc_dual_eta_up": float(self._dual_eta_up),
                "ewc_dual_eta_down": float(self._dual_eta_down),
                "ewc_dual_low_ratio": float(self._dual_low_ratio),
                "ewc_dual_loss_window": int(self._dual_loss_window),
                "ewc_dual_loss_patience": float(self._dual_loss_patience),
                "ewc_dual_lambda_min": float(self._dual_lambda_min),
                "ewc_dual_lambda_max": float(self._dual_lambda_max),
                "ewc_normalize_fisher": bool(self._normalize_fisher),
                **self._in_task_fisher_last_metrics,
            })
            logging.info(
                "[FisherSAM-EWC-lambda][Mechanism] %s",
                self._last_mechanism_metrics,
            )

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------
    def _build_fsam_optimizer(self, params, stage: str):
        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
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
    def _train(self, train_loader, test_loader):
        model = self._unwrap_network()
        model.to(self._device)
        self._reset_mechanism_stats()
        self._in_task_fisher_snapshots = []
        self._in_task_fisher_last_metrics = {}
        self._reset_dual_controller_state()

        stage = "init" if self._cur_task == 0 else "update"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_fsam_optimizer(params, stage=stage)

        if stage == "init":
            epochs = int(self.args.get("init_epoch", 1))
            scheduler = self.build_scheduler(
                optimizer,
                policy=self.args.get("scheduler", "constant"),
                milestones=self.args.get("init_milestones", []),
                gamma=float(self.args.get("init_lrate_decay", self.args.get("init_lr_decay", 1.0))),
                T_max=epochs,
                eta_min=self.args.get("min_lr", 0.0),
            )
        else:
            epochs = int(self.args.get("epochs", 1))
            scheduler = self.build_scheduler(
                optimizer,
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
            fsam_losses = 0.0
            ewc_losses = 0.0
            loss_count = 0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits, loss_value, log_dict = self._fsam_ewc_step(
                    model, optimizer, inputs, targets, stage
                )

                eval_logits, eval_targets = self._eval_view_from_logits(logits, targets, stage)
                total_loss += float(loss_value)
                ce_losses += float(log_dict.get("ce_clean", 0.0))
                fsam_losses += float(log_dict.get("fsam", 0.0))
                ewc_losses += float(log_dict.get("ewc", 0.0))
                loss_count += 1

                if self._mechanism_eval:
                    stats = {
                        "ce": float(log_dict.get("ce_clean", 0.0)),
                        "fsam": float(log_dict.get("fsam", 0.0)),
                        "ewc": float(log_dict.get("ewc", 0.0)),
                        "lambda_ewc_step": float(self._ewc_lambda if self._use_ewc else 0.0),
                        "fsam_ewc_ratio": float(
                            log_dict.get("ewc", 0.0) / (log_dict.get("fsam", 0.0) + self._mechanism_eps)
                        ),
                        "dual_drift": float(log_dict.get("dual_drift", 0.0)),
                        "dual_violation": float(log_dict.get("dual_violation", 0.0)),
                        "dual_loss_improve": float(log_dict.get("dual_loss_improve", 0.0)),
                        "dual_updated": float(log_dict.get("dual_updated", 0.0)),
                        "dual_mode": float(log_dict.get("dual_mode", 0.0)),
                    }
                    stats.update(self._compute_param_drift_metrics())
                    self._record_mechanism_stats(stats, increment=True)

                with torch.no_grad():
                    _, preds = torch.max(eval_logits, dim=1)
                    correct += preds.eq(eval_targets).cpu().sum()
                    total += len(eval_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2) if total > 0 else 0.0
            denom = max(1, loss_count)
            epoch_dual_stats = {}
            if self._dual_update_by_epoch:
                epoch_dual_stats = self._maybe_update_dual_lambda(
                    model,
                    current_loss=ce_losses / denom,
                )
                if self._mechanism_eval and epoch_dual_stats.get("dual_updated", 0.0):
                    self._record_mechanism_stats(epoch_dual_stats, increment=True)
            if self._fisher is not None and self._checkpoint is not None:
                drift_display = float(self._compute_fisher_drift_value(model).detach().item())
            else:
                drift_display = 0.0
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => "
                f"Loss {total_loss/denom:.3f}, Train_accy {train_acc:.2f}, "
                f"CE {ce_losses/denom:.3f}, FSAM {fsam_losses/denom:.3f}, "
                f"EWC {ewc_losses/denom:.3e}, D_F {drift_display:.3e}, "
                f"lambda {float(self._ewc_lambda):.3g}"
            )
            if epoch % 5 == 4:
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            if self._should_capture_in_task_fisher(epoch + 1):
                snapshot_count = self._capture_in_task_fisher_snapshot(
                    train_loader,
                    model,
                    optimizer,
                    epoch + 1,
                )
                if snapshot_count > 0:
                    info += f", Fisher_snap {snapshot_count}"
            prog_bar.set_description(info)

        if info:
            logging.info(info)

    def _should_capture_in_task_fisher(self, epoch_idx: int) -> bool:
        if not self._approximate_fisher_in_task:
            return False
        if not self._compute_fisher_enabled:
            return False
        if self._in_task_fisher_interval <= 0:
            return False
        if self._in_task_fisher_omega <= 0.0:
            return False
        return int(epoch_idx) % int(self._in_task_fisher_interval) == 0

    def _capture_in_task_fisher_snapshot(
        self,
        train_loader,
        model: nn.Module,
        optimizer: DiagonalFisherSAM,
        epoch_idx: int,
    ) -> int:
        if self._fisher_mode in {"ncm_empirical", "ncm_batch_empirical"}:
            if not self._in_task_fisher_skip_warned:
                logger.warning(
                    "[FisherSAM-EWC-lambda][InTaskFisher] Skipping in-task "
                    "snapshots for fisher_mode=%s because NCM Fisher needs "
                    "class means computed after training.",
                    self._fisher_mode,
                )
                self._in_task_fisher_skip_warned = True
            return len(self._in_task_fisher_snapshots)

        task_start, task_end = int(self._known_classes), int(self._total_classes)
        was_training = bool(model.training)
        logger.info(
            "[FisherSAM-EWC-lambda][InTaskFisher] Computing snapshot "
            "task=%d epoch=%d mode=%s",
            self._cur_task,
            epoch_idx,
            self._fisher_mode,
        )
        try:
            fisher = self._compute_fisher(train_loader, model, task_start, task_end)
        finally:
            optimizer.zero_grad()
            model.train(was_training)

        snapshot = {
            name: value.detach().cpu().clone()
            for name, value in fisher.items()
        }
        self._in_task_fisher_snapshots.append((int(epoch_idx), snapshot))
        logger.info(
            "[FisherSAM-EWC-lambda][InTaskFisher] Saved snapshot "
            "task=%d epoch=%d count=%d trace=%.6g",
            self._cur_task,
            epoch_idx,
            len(self._in_task_fisher_snapshots),
            self._fisher_trace(snapshot),
        )
        return len(self._in_task_fisher_snapshots)

    def _fsam_ewc_step(
        self,
        model: nn.Module,
        optimizer: DiagonalFisherSAM,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        stage: str,
    ) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        optimizer.zero_grad()

        clean_outputs = model(inputs)
        clean_logits = clean_outputs["logits"]
        clean_loss = self._task_loss_from_logits(clean_logits, targets, stage)
        clean_loss.backward()

        optimizer.first_step(zero_grad=True)

        pert_outputs = model(inputs)
        pert_logits = pert_outputs["logits"]
        fsam_loss = self._task_loss_from_logits(pert_logits, targets, stage)
        fsam_loss.backward()

        # Keep the FSAM gradient, but restore theta before adding the EWC gradient.
        optimizer.restore()

        ewc_loss = self._ewc_penalty(model)
        if torch.is_tensor(ewc_loss) and ewc_loss.requires_grad:
            ewc_loss.backward()

        optimizer.base_optimizer.step()
        optimizer.zero_grad()

        fsam_value = float(fsam_loss.detach().item())
        ewc_value = float(ewc_loss.detach().item() if torch.is_tensor(ewc_loss) else ewc_loss)
        clean_value = float(clean_loss.detach().item())
        if self._dual_update_by_epoch:
            dual_stats = self._default_dual_stats()
        else:
            dual_stats = self._maybe_update_dual_lambda(
                model,
                current_loss=clean_value,
            )
        log_dict = {
            "ce_clean": clean_value,
            "fsam": fsam_value,
            "ewc": ewc_value,
            "lambda_ewc": float(self._ewc_lambda),
            **dual_stats,
        }
        return pert_logits.detach(), fsam_value + ewc_value, log_dict

    # ------------------------------------------------------------------
    # Fisher normalization and dual ascent
    # ------------------------------------------------------------------
    def _fisher_trace(self, fisher: Optional[Dict[str, torch.Tensor]]) -> float:
        if not fisher:
            return 0.0
        total = 0.0
        for value in fisher.values():
            total += float(value.detach().float().clamp_min(0).sum().item())
        return float(total)

    def _mean_in_task_fisher(self, model: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        if not self._in_task_fisher_snapshots:
            return None

        mean_fisher: Dict[str, torch.Tensor] = {}
        for name, p in self._iter_trainable(model):
            acc = None
            count = 0
            for _, snapshot in self._in_task_fisher_snapshots:
                value = snapshot.get(name)
                if value is None:
                    continue
                value = self._match_tensor(value, p).detach().cpu()
                acc = value.clone() if acc is None else acc + value
                count += 1
            if acc is not None and count > 0:
                mean_fisher[name] = acc / float(count)
        return mean_fisher or None

    def _mix_in_task_path_fisher(
        self,
        task_fisher: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        snapshot_epochs = [epoch for epoch, _ in self._in_task_fisher_snapshots]
        path_fisher = self._mean_in_task_fisher(model)
        final_trace = self._fisher_trace(task_fisher)
        path_trace = self._fisher_trace(path_fisher)

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
            }
            return task_fisher

        omega = float(self._in_task_fisher_omega)
        mixed: Dict[str, torch.Tensor] = {}
        for name, p in self._iter_trainable(model):
            final_value = task_fisher.get(name)
            path_value = path_fisher.get(name)
            if final_value is None and path_value is None:
                continue
            if final_value is None:
                mixed[name] = self._match_tensor(path_value, p).detach().cpu().clone()
            elif path_value is None:
                mixed[name] = self._match_tensor(final_value, p).detach().cpu().clone()
            else:
                final_t = self._match_tensor(final_value, p).detach().cpu()
                path_t = self._match_tensor(path_value, p).detach().cpu()
                mixed[name] = (1.0 - omega) * final_t + omega * path_t

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
        }
        logger.info(
            "[FisherSAM-EWC-lambda][InTaskFisher] Mixed final/path Fisher "
            "task=%d snapshots=%d omega=%.3g final_trace=%.6g "
            "path_trace=%.6g mixed_trace=%.6g",
            self._cur_task,
            len(snapshot_epochs),
            omega,
            final_trace,
            path_trace,
            mixed_trace,
        )
        return mixed

    def _trace_normalize_fisher(self, fisher: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not fisher:
            return fisher
        total_trace = 0.0
        for value in fisher.values():
            total_trace += float(value.detach().float().clamp_min(0).sum().item())
        denom = total_trace + self._fisher_norm_eps
        if denom <= 0.0:
            return fisher
        return {name: value / denom for name, value in fisher.items()}

    def _accumulate_fisher(self, task_fisher: Dict[str, torch.Tensor], model: nn.Module):
        task_fisher = self._mix_in_task_path_fisher(task_fisher, model)
        if self._normalize_fisher:
            task_fisher = self._trace_normalize_fisher(task_fisher)
            self._last_fisher_estimation_metrics["ewc_fisher_trace_normalized"] = True
        super()._accumulate_fisher(task_fisher, model)
        if self._normalize_fisher and self._renormalize_accumulated_fisher and self._fisher:
            self._fisher = self._trace_normalize_fisher(self._fisher)
        self._in_task_fisher_snapshots = []

    @torch.no_grad()
    def _compute_fisher_drift_value(self, model: nn.Module) -> torch.Tensor:
        if self._fisher is None or self._checkpoint is None:
            return torch.zeros((), device=self._device)

        drift = torch.zeros((), device=self._device)
        trace = 0.0
        for name, p in self._iter_trainable(model):
            ref = self._checkpoint.get(name)
            fisher = self._fisher.get(name)
            if ref is None or fisher is None:
                continue
            fisher_d = fisher.to(device=p.device, dtype=p.dtype).clamp_min(0)
            ref_d = ref.to(device=p.device, dtype=p.dtype)
            drift = drift + (fisher_d * (p.detach() - ref_d).pow(2)).sum()
            if self._dual_trace_normalize_drift:
                trace += float(fisher_d.detach().sum().item())

        if self._dual_trace_normalize_drift:
            drift = drift / (trace + self._dual_eps)
        return drift

    def _reset_dual_controller_state(self) -> None:
        self._dual_step_count = 0
        self._dual_loss_buffer = []
        self._dual_prev_loss_window = None

    def _default_dual_stats(self) -> Dict[str, float]:
        budget = float(self._dual_budget)
        return {
            "dual_drift": 0.0,
            "dual_budget": budget,
            "dual_delta_low": float(self._dual_low_ratio * budget),
            "dual_violation": 0.0,
            "dual_loss_improve": 0.0,
            "lambda_before": float(self._ewc_lambda),
            "lambda_after": float(self._ewc_lambda),
            "dual_updated": 0.0,
            "dual_mode": 0.0,
        }

    def _maybe_update_dual_lambda(
        self,
        model: nn.Module,
        current_loss: float,
    ) -> Dict[str, float]:
        default_stats = self._default_dual_stats()
        if (
            not self._dual_ascent
            or not self._use_ewc
            or self._fisher is None
            or self._checkpoint is None
        ):
            return default_stats

        self._dual_step_count += 1
        self._dual_loss_buffer.append(float(current_loss))
        if len(self._dual_loss_buffer) < self._dual_loss_window:
            return default_stats
        if self._dual_step_count % self._dual_update_interval != 0:
            return default_stats

        curr_window = float(np.mean(self._dual_loss_buffer[-self._dual_loss_window:]))
        if self._dual_prev_loss_window is None:
            if len(self._dual_loss_buffer) < 2 * self._dual_loss_window:
                self._dual_prev_loss_window = curr_window
                return default_stats
            prev_window = float(
                np.mean(self._dual_loss_buffer[-2 * self._dual_loss_window:-self._dual_loss_window])
            )
        else:
            prev_window = float(self._dual_prev_loss_window)

        loss_improve = (prev_window - curr_window) / (abs(prev_window) + self._dual_eps)
        self._dual_prev_loss_window = curr_window

        return self._update_dual_lambda(model, loss_improve=loss_improve)

    def _update_dual_lambda(
        self,
        model: nn.Module,
        loss_improve: float = float("inf"),
    ) -> Dict[str, float]:
        default_stats = self._default_dual_stats()
        if (
            not self._dual_ascent
            or not self._use_ewc
            or self._fisher is None
            or self._checkpoint is None
        ):
            return default_stats

        drift = self._compute_fisher_drift_value(model)
        drift_value = float(drift.detach().item())
        budget = float(self._dual_budget)
        delta_low = float(self._dual_low_ratio * budget)

        if self._dual_relative:
            violation = drift_value / (budget + self._dual_eps) - 1.0
        else:
            violation = drift_value - budget

        old_lambda = float(self._ewc_lambda)
        new_lambda = old_lambda
        updated = 0.0
        mode = 0.0

        if drift_value > budget:
            rel_high = (drift_value - budget) / (budget + self._dual_eps)
            exponent = float(np.clip(self._dual_eta_up * rel_high, -50.0, 50.0))
            base_lambda = old_lambda
            if base_lambda <= 0.0:
                base_lambda = max(float(self._dual_lambda_min), self._dual_eps)
            new_lambda = base_lambda * float(np.exp(exponent))
            updated = 1.0
            mode = 1.0
        # Only relax EWC when current-task loss is nearly plateaued. If the loss
        # is rising (R < 0), hold lambda because the cause is ambiguous.
        elif drift_value < delta_low and 0.0 <= float(loss_improve) < self._dual_loss_patience:
            new_lambda = old_lambda * float(np.exp(-self._dual_eta_down))
            updated = 1.0
            mode = -1.0

        new_lambda = min(self._dual_lambda_max, max(self._dual_lambda_min, new_lambda))
        self._ewc_lambda = float(new_lambda)

        return {
            "dual_drift": drift_value,
            "dual_budget": budget,
            "dual_delta_low": delta_low,
            "dual_violation": float(violation),
            "dual_loss_improve": float(loss_improve),
            "lambda_before": old_lambda,
            "lambda_after": float(new_lambda),
            "dual_updated": updated,
            "dual_mode": mode,
        }


FisherSAMEWCLambda = Learner
