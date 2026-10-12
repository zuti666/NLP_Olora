"""SeqLoRA EWCLoRA with KFAC Fisher-SAM and adaptive EWC lambda.

This learner reuses the raw delta-W EWC geometry, adaptive lambda controller,
and in-task Fisher path averaging from ``FisherSAM_EWCLoRA_lambda.py``.  Both
the current-task Fisher-SAM perturbation and the EWC Fisher estimate use the
KFAC implementation in ``optimer_fishersam_kfac.py`` with
``kfac_target="lora_delta_w"`` so the KFAC blocks are built on the effective
LoRA update Delta W = B @ A for q/v branches.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

from models_LoRAbasedCL.FisherSAM_EWCLoRA_lambda import Learner as _LoRAFishersamEWCLambda
from optimer_PerturabtionType.optimer_fishersam_kfac import KFACFisherSAM
from utils.mechanism_eval import as_bool

logger = logging.getLogger(__name__)


class Learner(_LoRAFishersamEWCLambda):
    """SeqLoRA + delta-W EWC + KFAC Fisher-SAM current-task optimizer."""

    def __init__(self, args):
        super().__init__(args)

        self._optimizer_type = "fishersam_kfac"
        self._base_optimizer_type = "fishersam_kfac"
        self._kfac_stat_decay = float(args.get("kfac_stat_decay", args.get("fsam_kfac_stat_decay", 0.95)))
        self._kfac_tcov = int(args.get("kfac_TCov", args.get("kfac_tcov", args.get("fsam_kfac_TCov", 1))))
        self._kfac_tinv = int(args.get("kfac_TInv", args.get("kfac_tinv", args.get("fsam_kfac_TInv", 10))))
        self._kfac_token_averaged = as_bool(
            args.get("kfac_token_averaged", args.get("fsam_kfac_token_averaged", False))
        )
        self._kfac_grad_scale = str(args.get("kfac_grad_scale", "mean_x_bs")).lower()
        self._kfac_seq_agg = str(args.get("kfac_seq_agg", "token")).lower()
        self._kfac_target = "lora_delta_w"
        self._kfac_fallback = str(args.get("kfac_fallback", args.get("fsam_kfac_fallback", "diag"))).lower()
        self._kfac_fallback_inverse_mode = str(
            args.get("kfac_fallback_inverse_mode", self._fsam_inverse_mode)
        ).lower()
        max_kfac_dim = args.get("kfac_max_dim", args.get("fsam_kfac_max_dim", None))
        self._kfac_max_dim = None if max_kfac_dim in (None, "", "none", "None") else int(max_kfac_dim)
        self._kfac_verbose = as_bool(args.get("kfac_verbose", args.get("fsam_kfac_verbose", False)))
        self._last_kfac_ewc_fisher_metrics: Dict[str, float] = {}

        if self._is_main_process:
            logger.info(
                "[FisherSAM-EWCLoRA-lambda-KFAC] gamma=%.4g eta=%.4g damping=%.4g base=%s "
                "kfac_stat_decay=%.4g TCov=%d TInv=%d target=%s fallback=%s "
                "grad_scale=%s seq_agg=%s token_averaged=%s lambda_init=%.4g",
                self._fsam_gamma,
                self._fsam_eta,
                self._fsam_damping,
                self._fsam_base_optimizer,
                self._kfac_stat_decay,
                self._kfac_tcov,
                self._kfac_tinv,
                self._kfac_target,
                self._kfac_fallback,
                self._kfac_grad_scale,
                self._kfac_seq_agg,
                self._kfac_token_averaged,
                self._ewc_lambda,
            )

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------
    def _build_optimizer(self, params, stage):
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

        model = self._unwrap_network()
        base_optimizer = base_cls(params, **opt_kwargs)
        return KFACFisherSAM(
            model,
            base_optimizer,
            gamma=self._fsam_gamma,
            eta=self._fsam_eta,
            damping=self._fsam_damping,
            stat_decay=self._kfac_stat_decay,
            TCov=self._kfac_tcov,
            TInv=self._kfac_tinv,
            token_averaged=self._kfac_token_averaged,
            kfac_grad_scale=self._kfac_grad_scale,
            kfac_seq_agg=self._kfac_seq_agg,
            kfac_target="lora_delta_w",
            fallback=self._kfac_fallback,
            fallback_inverse_mode=self._kfac_fallback_inverse_mode,
            adaptive=self._fsam_adaptive,
            max_kfac_dim=self._kfac_max_dim,
            verbose=self._kfac_verbose,
        )

    # ------------------------------------------------------------------
    # KFAC Fisher-SAM + EWC batch step
    # ------------------------------------------------------------------
    def _fsam_ewc_step(
        self,
        model: nn.Module,
        optimizer: KFACFisherSAM,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        class_offset: int,
    ) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        optimizer.zero_grad()
        perturbed = False

        optimizer._lora_current_a.clear()
        optimizer._lora_current_grad_dw.clear()
        optimizer.acc_stats = True
        try:
            clean_outputs = model(inputs)
            clean_logits = clean_outputs["logits"]
            clean_loss, _, _ = self._task_loss(clean_logits, targets, class_offset)
            clean_loss.backward()
        finally:
            optimizer.acc_stats = False

        optimizer.last_clean_loss = clean_loss.detach()
        optimizer.first_step(zero_grad=True)
        perturbed = True

        try:
            pert_outputs = model(inputs)
            pert_logits = pert_outputs["logits"]
            fsam_loss, _, _ = self._task_loss(pert_logits, targets, class_offset)
            fsam_loss.backward()
            optimizer.last_fsam_loss = fsam_loss.detach()

            optimizer.restore()
            perturbed = False

            ewc_loss = self._past_fisher_penalty(model)
            if torch.is_tensor(ewc_loss) and ewc_loss.requires_grad:
                ewc_loss.backward()

            optimizer.base_optimizer.step()
            optimizer.steps += 1
        finally:
            if perturbed:
                optimizer.restore()
            optimizer.acc_stats = False
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
            )
        log_dict = {
            "ce_clean": clean_value,
            "fsam": fsam_value,
            "ewc": ewc_value,
            "lambda_ewc": float(self._ewc_lambda),
            "fsam_ewc_ratio": fsam_ewc_ratio,
            **dual_stats,
        }
        return pert_logits.detach(), fsam_value + ewc_value, log_dict

    # ------------------------------------------------------------------
    # KFAC EWC Fisher in LoRA delta-W space
    # ------------------------------------------------------------------
    def _build_kfac_ewc_fisher_estimator(self, model: nn.Module) -> KFACFisherSAM:
        params = [p for p in model.parameters() if p.requires_grad]
        base_optimizer = torch.optim.SGD(params, lr=0.0)
        return KFACFisherSAM(
            model,
            base_optimizer,
            gamma=self._fsam_gamma,
            eta=self._fsam_eta,
            damping=self._fsam_damping,
            stat_decay=0.0,
            TCov=1,
            TInv=self._kfac_tinv,
            token_averaged=self._kfac_token_averaged,
            kfac_grad_scale=self._kfac_grad_scale,
            kfac_seq_agg=self._kfac_seq_agg,
            kfac_target="lora_delta_w",
            fallback="none",
            fallback_inverse_mode=self._kfac_fallback_inverse_mode,
            adaptive=self._fsam_adaptive,
            max_kfac_dim=self._kfac_max_dim,
            verbose=False,
        )

    @staticmethod
    def _kfac_delta_w_diag(aa: torch.Tensor, gg: torch.Tensor) -> torch.Tensor:
        diag_a = torch.diagonal(aa.detach().float(), 0).clamp_min(0)
        diag_g = torch.diagonal(gg.detach().float(), 0).clamp_min(0)
        return diag_g.reshape(-1, 1) * diag_a.reshape(1, -1)

    def _compute_delta_fisher(self, loader, model: nn.Module, class_offset: int) -> Dict[str, torch.Tensor]:
        was_training = bool(model.training)
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        total = 0
        estimator = None

        try:
            estimator = self._build_kfac_ewc_fisher_estimator(model)
            module_names = {module: name for name, module in self._iter_delta_modules(model)}

            for batch_idx, batch in enumerate(loader):
                if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                    break

                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                model.zero_grad(set_to_none=True)
                estimator.zero_grad()
                estimator.lora_m_aa.clear()
                estimator.lora_m_gg.clear()
                estimator._lora_current_a.clear()
                estimator._lora_current_grad_dw.clear()

                estimator.acc_stats = True
                try:
                    with torch.enable_grad():
                        logits = model(inputs)["logits"]
                        loss, _, _ = self._task_loss(logits, targets, class_offset)
                        loss.backward()
                finally:
                    estimator.acc_stats = False

                batch_size = int(inputs.size(0))
                total += batch_size
                for key in estimator.lora_specs:
                    module, branch = key
                    module_name = module_names.get(module)
                    if module_name is None:
                        continue
                    aa = estimator.lora_m_aa.get(key)
                    gg = estimator.lora_m_gg.get(key)
                    if aa is None or gg is None:
                        continue

                    fisher_key = f"{module_name}.{branch}"
                    diag = self._kfac_delta_w_diag(aa, gg).cpu() * float(batch_size)
                    fisher[fisher_key] = diag if fisher_key not in fisher else fisher[fisher_key] + diag
        finally:
            if estimator is not None:
                estimator.acc_stats = False
                estimator.zero_grad()
                estimator.close()
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        if total > 0:
            for key in fisher:
                fisher[key] = fisher[key] / float(total)

        trace = self._fisher_trace(fisher)
        numel = sum(int(value.numel()) for value in fisher.values())
        max_value = max((float(value.max().item()) for value in fisher.values()), default=0.0)
        self._last_kfac_ewc_fisher_metrics = {
            "ewc_fisher_estimator": "kfac_lora_delta_w_diag",
            "ewc_fisher_kfac_tensors": int(len(fisher)),
            "ewc_fisher_kfac_numel": int(numel),
            "ewc_fisher_kfac_trace": float(trace),
            "ewc_fisher_kfac_max": float(max_value),
            "ewc_fisher_kfac_samples": int(total),
        }
        if self._is_main_process:
            self._log(
                "[FisherSAM-EWCLoRA-lambda-KFAC][EWCFisher] "
                f"task={self._cur_task} tensors={len(fisher)} samples={total} "
                f"trace={trace:.6g} max={max_value:.6g}"
            )
        return fisher

    def _capture_in_task_fisher_snapshot(
        self,
        train_loader,
        model: nn.Module,
        optimizer: KFACFisherSAM,
        epoch_idx: int,
        class_offset: int,
    ) -> int:
        was_training = bool(model.training)
        logger.info(
            "[FisherSAM-EWCLoRA-lambda-KFAC][InTaskFisher] Computing KFAC "
            "delta-W snapshot task=%d epoch=%d",
            self._cur_task,
            epoch_idx,
        )
        try:
            fisher = self._compute_delta_fisher(train_loader, model, class_offset)
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
            "[FisherSAM-EWCLoRA-lambda-KFAC][InTaskFisher] Saved KFAC snapshot "
            "task=%d epoch=%d count=%d fisher_trace=%.6g lora_tensors=%d",
            self._cur_task,
            epoch_idx,
            len(self._in_task_fisher_snapshots),
            self._fisher_trace(fisher_snapshot),
            len(lora_snapshot),
        )
        return len(self._in_task_fisher_snapshots)

    def after_task(self):
        drift_before = (
            self._compute_delta_drift_metrics()
            if (self._mechanism_eval or self._dual_ascent)
            else {}
        )
        base_model = self._unwrap_network()
        new_fisher = self._compute_delta_fisher(
            loader=self.train_loader,
            model=base_model,
            class_offset=int(self._known_classes),
        )
        new_fisher = self._mix_in_task_path_fisher(new_fisher, base_model)
        self._fisher_past_delta = self._accumulate_fisher(self._fisher_past_delta, new_fisher)

        if self._is_main_process:
            f_vals = list(new_fisher.values())
            if f_vals:
                nf_max = float(max(v.max().item() for v in f_vals))
                nf_sum = float(sum(v.sum().item() for v in f_vals))
                nf_numel = sum(int(v.numel()) for v in f_vals)
                nf_mean = nf_sum / max(nf_numel, 1)
                self._log(
                    f"[FisherSAM-EWCLoRA-lambda-KFAC][FisherStats] task={self._cur_task} "
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
                    f"[FisherSAM-EWCLoRA-lambda-KFAC][FisherStats] task={self._cur_task} "
                    f"accum_fisher tensors={len(accum_vals)} numel={af_numel} "
                    f"max={af_max:.3e} mean={af_mean:.3e} sum={af_sum:.3e}"
                )
            self._log(
                "[FisherSAM-EWCLoRA-lambda-KFAC] Updated KFAC delta-W Fisher for "
                f"{len(new_fisher)} delta tensors after task {self._cur_task}."
            )

        self._delta_reference = {}
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._in_task_fisher_snapshots = []

        if self._mechanism_eval and self._last_mechanism_metrics:
            self._last_mechanism_metrics.update({
                "method_type": "fishersam_ewclora_lambda_kfac",
                "fsam_kfac_stat_decay": float(self._kfac_stat_decay),
                "fsam_kfac_TCov": int(self._kfac_tcov),
                "fsam_kfac_TInv": int(self._kfac_tinv),
                "fsam_kfac_target": self._kfac_target,
                "fsam_kfac_fallback": self._kfac_fallback,
                "fsam_kfac_grad_scale": self._kfac_grad_scale,
                "fsam_kfac_seq_agg": self._kfac_seq_agg,
                "fsam_kfac_token_averaged": bool(self._kfac_token_averaged),
                **self._last_kfac_ewc_fisher_metrics,
            })
        elif self._mechanism_eval:
            self._last_mechanism_metrics = {
                "task": int(self._cur_task),
                "method_type": "fishersam_ewclora_lambda_kfac",
                "lambda_ewc": float(self._ewc_lambda),
                "fsam_gamma": float(self._fsam_gamma),
                "fsam_eta": float(self._fsam_eta),
                "fsam_base_optimizer": self._fsam_base_optimizer,
                "fsam_kfac_stat_decay": float(self._kfac_stat_decay),
                "fsam_kfac_TCov": int(self._kfac_tcov),
                "fsam_kfac_TInv": int(self._kfac_tinv),
                "fsam_kfac_target": self._kfac_target,
                "fsam_kfac_fallback": self._kfac_fallback,
                "fsam_kfac_grad_scale": self._kfac_grad_scale,
                "fsam_kfac_seq_agg": self._kfac_seq_agg,
                "fsam_kfac_token_averaged": bool(self._kfac_token_averaged),
                "ewc_dual_ascent": bool(self._dual_ascent),
                "ewc_dual_delta": float(self._dual_budget),
                "ewc_dual_relative": bool(self._dual_relative),
                "ewc_dual_update_by_epoch": bool(self._dual_update_by_epoch),
                "ewc_dual_update_interval": int(self._dual_update_interval),
                "ewc_dual_eta_up": float(self._dual_eta_up),
                "ewc_dual_lambda_min": float(self._dual_lambda_min),
                "ewc_dual_lambda_max": float(self._dual_lambda_max),
                **self._in_task_fisher_last_metrics,
                **self._last_kfac_ewc_fisher_metrics,
                **self._summarize_mechanism_stats(),
                **drift_before,
            }
        if self._mechanism_eval and self._is_main_process:
            self._log(f"[FisherSAM-EWCLoRA-lambda-KFAC][Mechanism] {self._last_mechanism_metrics}")


FisherSAMEWCLoRALambdaKFAC = Learner
FisherSAMEWCLoRALambdaKFACFisherSAM = Learner
FisherSAMEWCLoRALambda = Learner
EWCLoRA = Learner
