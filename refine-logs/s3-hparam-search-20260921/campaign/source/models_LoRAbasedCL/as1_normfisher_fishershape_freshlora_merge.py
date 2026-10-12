"""Fresh-adapter, merge-after-task EWC-LoRA with factor-space FisherShape.

The task lifecycle follows the EWC-LoRA equations rather than keeping a
single LoRA adapter across tasks:

1. task ``t`` starts from a fresh adapter whose effective update is exactly
   zero (A is Kaiming initialized and B is zero initialized);
2. past-task EWC directly penalizes ``delta_W_t = B_t @ A_t`` -- there is no
   ``delta_W_t - delta_W_{t-1}`` factor-space anchor;
3. after optimization, the task Fisher is estimated at
   ``W_{t-1} + delta_W_t`` while the trained factors are still active;
4. only then is ``delta_W_t`` added to the frozen q/v base weights and the
   adapter reinitialized for the next task.

Thus the persistent model state is the merged base weight plus the accumulated
diagonal Fisher.  Historical LoRA A/B factors are neither needed by the
forward pass nor written as per-task LoRA checkpoints by this learner.

FisherShape still acts in trainable factor/head parameter space.  The EWC
quadratic and Fisher remain in effective full-dimensional ``delta_W`` space;
autograd pulls the EWC gradient back to A/B only after the two CE-only
FisherShape probes have been composed.  This gives the controlled directions

    alpha=0                 : g_0 + g_EWC
    sam_equivalent=True     : g_1 + g_EWC
    alpha_mode="ratio"      : g_0 + c ||g_0|| unit(g_1-g_0) + g_EWC

without allowing EWC to affect the probing point or finite difference.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch import optim

from models_LoRAbasedCL.as1_normfisher_gam import Learner as _AS1NormFisherGAM
from optimer_PerturabtionType.fishershape import FisherShape
from utils.mechanism_eval import as_bool


logger = logging.getLogger(__name__)


class Learner(_AS1NormFisherGAM):
    """AS1 delta-W EWC with a fresh, weight-merged LoRA on every task."""

    _OPTIMIZER_ALIASES = {
        "fishershape",
        "fishershape_factor",
        "factor_fishershape",
        "fs_factor",
    }

    def __init__(self, args):
        # Make the dedicated learner usable without an explicit optimizer key.
        if "optimizer_type" not in args:
            args["optimizer_type"] = "fishershape"
        super().__init__(args)

        self._fs_rho = float(args.get("fs_rho", args.get("gam_grad_rho", 0.05)))
        self._fs_alpha = float(args.get("fs_alpha", 0.2))
        self._fs_alpha_mode = str(args.get("fs_alpha_mode", "ratio")).lower()
        self._fs_xi = float(args.get("fs_xi", args.get("fs_perturb_eps", 1e-12)))
        self._fs_sam_equivalent = as_bool(args.get("fs_sam_equivalent", False))
        self._fs_adaptive = as_bool(args.get("fs_adaptive", False))
        self._fs_grad_reduce = str(args.get("fs_grad_reduce", "mean"))
        fisher_damping_raw = float(
            args.get(
                "ewc_fisher_damping_beta",
                args.get("ewc_fisher_damping_alpha", args.get("fisher_damping_alpha", 0.0)),
            )
        )
        self._ewc_fisher_damping_beta = max(0.0, min(1.0, fisher_damping_raw))
        self._last_task_fisher_damping_metrics: Dict[str, float] = {}
        self._last_premerge_drift_metrics: Dict[str, float] = {}
        self._last_merge_metrics: Dict[str, float] = {}
        # Algorithm 1 (Appendix A.2.1): Fisher must be estimated at the
        # trained effective weight W_{t-1} + delta_W_t before delta_W_t is
        # integrated into the base weight and the adapter is reset.
        self._fresh_merge_pending = False

        if self._fs_alpha_mode not in {"fixed", "ratio"}:
            raise ValueError(
                f'fs_alpha_mode must be "fixed" or "ratio", got {self._fs_alpha_mode!r}'
            )
        if self._optimizer_type not in self._OPTIMIZER_ALIASES:
            raise ValueError(
                "as1_normfisher_fishershape requires optimizer_type in "
                f"{sorted(self._OPTIMIZER_ALIASES)}, got {self._optimizer_type!r}. "
                "Use fs_alpha/fs_sam_equivalent for the controlled "
                "SGD/SAM/FS comparison."
            )
        if self._ewc_fisher_damping_beta > 0.0 and not self._normalize_fisher:
            logger.warning(
                "[AS1-FS-LoRA] ewc_fisher_damping_beta=%.4g is intended for "
                "trace-normalized per-task delta-W Fisher, but "
                "ewc_normalize_fisher=False.",
                self._ewc_fisher_damping_beta,
            )
        if self._is_main_process:
            logger.info(
                "[AS1-FS-LoRA] lifecycle=fresh_adapter_merge_to_base "
                "ewc_anchor=zero_delta space=factor rho=%.4g alpha=%.4g "
                "alpha_mode=%s xi=%.3g sam_equivalent=%s adaptive=%s "
                "lambda=%.4g gamma=%.4g normalize_fisher=%s "
                "normalize_accumulated=%s fisher_damping_beta=%.4g",
                self._fs_rho,
                self._fs_alpha,
                self._fs_alpha_mode,
                self._fs_xi,
                self._fs_sam_equivalent,
                self._fs_adaptive,
                self._lambda_ewc,
                self._ewc_gamma,
                self._normalize_fisher,
                self._normalize_accumulated_fisher,
                self._ewc_fisher_damping_beta,
            )

    # ------------------------------------------------------------------
    # Paper-faithful fresh-LoRA EWC geometry
    # ------------------------------------------------------------------
    def _snapshot_task_reference(self) -> None:
        """Start each task at delta_W=0; no previous-factor anchor is stored."""
        self._delta_reference = {}
        self._last_premerge_drift_metrics = {}
        if self._mechanism_eval:
            self._reset_mechanism_stats()

    def _past_fisher_penalty(self, model: nn.Module) -> torch.Tensor:
        """Penalize the fresh task update itself: 0.5*lambda*F*(B@A)^2."""
        if not self._fisher_past_delta:
            return torch.zeros((), device=self._device)

        penalty = None
        for key, delta_now in self._iter_delta_terms(model):
            fisher = self._fisher_past_delta.get(key)
            if fisher is None:
                continue
            fisher_t = fisher.to(device=delta_now.device, dtype=delta_now.dtype)
            if self._fisher_eta != 0.0:
                fisher_t = fisher_t + float(self._fisher_eta)
            term = (fisher_t * delta_now.pow(2)).sum()
            penalty = term if penalty is None else penalty + term

        if penalty is None:
            return torch.zeros((), device=self._device)
        return 0.5 * float(self._lambda_ewc) * penalty

    def _direct_delta_drift_metrics(self) -> Dict[str, float]:
        """Measure the current fresh adapter relative to its zero start."""
        if not self._fisher_past_delta:
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

        base_model = self._unwrap_network()
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
            for key, delta_now in self._iter_delta_terms(base_model):
                fisher = self._fisher_past_delta.get(key)
                if fisher is None:
                    continue
                fisher_t = fisher.to(device=delta_now.device, dtype=delta_now.dtype)
                update_sq = delta_now.pow(2)
                fisher_update = fisher_t * update_sq

                delta_norm += float(update_sq.sum().item())
                fisher_weighted += float(fisher_update.sum().item())
                fisher_eta_weighted += float(
                    ((fisher_t + float(self._fisher_eta)) * update_sq).sum().item()
                )
                fisher_sum += float(fisher_t.sum().item())
                if fisher_t.numel():
                    fisher_max = max(fisher_max, float(fisher_t.max().item()))
                fisher_numel += int(fisher_t.numel())
                tensor_count += 1

                flat_fisher = fisher_t.reshape(-1)
                flat_update_sq = update_sq.reshape(-1)
                if flat_fisher.numel():
                    if self._mechanism_topk is not None:
                        k = min(max(int(self._mechanism_topk), 1), int(flat_fisher.numel()))
                    else:
                        fraction = min(max(float(self._mechanism_top_fraction), 0.0), 1.0)
                        k = min(
                            max(int(round(float(flat_fisher.numel()) * fraction)), 1),
                            int(flat_fisher.numel()),
                        )
                    top_idx = torch.topk(
                        flat_fisher, k=k, largest=True, sorted=False
                    ).indices
                    high_delta_energy += float(flat_update_sq[top_idx].sum().item())
                    high_weighted_energy += float(
                        (flat_fisher[top_idx] * flat_update_sq[top_idx]).sum().item()
                    )

        return {
            "delta_drift_available": int(tensor_count > 0),
            "delta_tensors": int(tensor_count),
            "delta_norm": float(delta_norm),
            "fisher_weighted_drift": float(fisher_weighted),
            "fisher_eta_weighted_drift": float(fisher_eta_weighted),
            "normalized_fisher_drift": float(fisher_weighted / (delta_norm + eps)),
            "normalized_fisher_eta_drift": float(
                fisher_eta_weighted / (delta_norm + eps)
            ),
            "high_fisher_delta_energy_ratio": float(
                high_delta_energy / (delta_norm + eps)
            ),
            "high_fisher_weighted_energy_ratio": float(
                high_weighted_energy / (fisher_weighted + eps)
            ),
            "fisher_mean": float(fisher_sum / max(fisher_numel, 1)),
            "fisher_max": float(fisher_max),
            "ewc_penalty_value": float(
                0.5 * float(self._lambda_ewc) * fisher_eta_weighted
            ),
        }

    def _compute_delta_drift_metrics(self) -> Dict[str, float]:
        # The adapter has already been merged/reset when after_task() requests
        # metrics.  Return the pre-merge snapshot in that case.
        if self._last_premerge_drift_metrics:
            return dict(self._last_premerge_drift_metrics)
        return self._direct_delta_drift_metrics()

    @staticmethod
    def _current_scale(module: nn.Module, reference: torch.Tensor) -> torch.Tensor:
        """Return the scalar multiplying the current LoRA branch."""
        scaling = getattr(module, "scaling_factor", None)
        if scaling is None or len(scaling) == 0:
            return reference.new_ones(())
        scale_module = scaling[0]
        scale_param = getattr(scale_module, "param", None)
        if torch.is_tensor(scale_param):
            return scale_param.detach().to(device=reference.device, dtype=reference.dtype).reshape(())
        return reference.new_ones(())

    @staticmethod
    def _reset_current_lora(module: nn.Module) -> None:
        """Reinitialize A and zero B so the next task starts at delta_W=0."""
        nn.init.kaiming_uniform_(module.linear_a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(module.linear_b_q.weight)
        nn.init.kaiming_uniform_(module.linear_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(module.linear_b_v.weight)

        scaling = getattr(module, "scaling_factor", None)
        if scaling is not None and len(scaling) > 0:
            scale_param = getattr(scaling[0], "param", None)
            if torch.is_tensor(scale_param):
                scale_param.fill_(1.0)

    def _merge_current_lora_into_base(self) -> Dict[str, float]:
        """Apply W <- W + B@A to q/v weights, then discard the factors."""
        base_model = self._unwrap_network()
        modules = list(self._iter_delta_modules(base_model))
        if not modules:
            raise RuntimeError(
                "as1_normfisher_fishershape_freshlora_merge found no mergeable LoRA q/v modules"
            )

        # Preserve task-drift diagnostics before B is reset to zero.
        self._last_premerge_drift_metrics = self._direct_delta_drift_metrics()

        merged_norm_sq = 0.0
        merged_numel = 0
        with torch.no_grad():
            for name, module in modules:
                qkv = getattr(module, "qkv", None)
                weight = getattr(qkv, "weight", None)
                dim = int(getattr(module, "dim", 0))
                if weight is None or weight.ndim != 2 or dim <= 0 or weight.shape[0] != 3 * dim:
                    raise TypeError(
                        f"LoRA module {name!r} is not a mergeable ViT qkv layer"
                    )

                delta_q = module.linear_b_q.weight @ module.linear_a_q.weight
                delta_v = module.linear_b_v.weight @ module.linear_a_v.weight
                scale = self._current_scale(module, delta_q)
                delta_q = scale * delta_q
                delta_v = scale * delta_v

                weight[:dim, :].add_(delta_q.to(device=weight.device, dtype=weight.dtype))
                weight[-dim:, :].add_(delta_v.to(device=weight.device, dtype=weight.dtype))

                merged_norm_sq += float(delta_q.pow(2).sum().item())
                merged_norm_sq += float(delta_v.pow(2).sum().item())
                merged_numel += int(delta_q.numel() + delta_v.numel())
                self._reset_current_lora(module)

        residual_max = 0.0
        with torch.no_grad():
            for _, delta in self._iter_delta_terms(base_model):
                if delta.numel():
                    residual_max = max(residual_max, float(delta.abs().max().item()))
        if residual_max != 0.0:
            raise RuntimeError(
                f"fresh LoRA reset did not produce delta_W=0 (max={residual_max:.3e})"
            )

        metrics = {
            "merge_task": int(self._cur_task),
            "merge_lora_modules": int(len(modules)),
            "merge_delta_tensors": int(2 * len(modules)),
            "merge_delta_numel": int(merged_numel),
            "merge_delta_norm": float(merged_norm_sq ** 0.5),
            "fresh_delta_max_abs": float(residual_max),
        }
        self._last_merge_metrics = metrics
        if self._is_main_process:
            logger.info(
                "[AS1-FS-LoRA][Merge] task=%d modules=%d delta_norm=%.6e; "
                "fresh adapter delta_max=%.1e",
                int(self._cur_task),
                int(len(modules)),
                float(metrics["merge_delta_norm"]),
                float(residual_max),
            )
        return metrics

    def _save_merged_final_model_if_requested(self) -> None:
        """Replace the trainer's pre-after_task save with merged final state."""
        if not self._is_main_process or not as_bool(self.args.get("save_final_model", False)):
            return
        nb_tasks = int(self.args.get("nb_tasks", self.args.get("total_sessions", 0)))
        if nb_tasks > 0 and int(self._cur_task) != nb_tasks - 1:
            return

        network = self._unwrap_network()
        checkpoint_dir = os.path.abspath(
            os.path.expanduser(str(self.args.get("filepath", "./")))
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = os.path.join(checkpoint_dir, "final_model.pt")
        final_ckpt = {
            "method": self.args.get("model_name", "model_name"),
            "dataset": self.args.get("dataset", "dataset"),
            "init_cls": self.args.get("init_cls", "init_cls"),
            "increment": self.args.get("increment", self.args.get("init_cls", "init_cls")),
            "seed": self.args.get("seed", "seed"),
            "tasks": int(self._cur_task),
            "model_state_dict": network.state_dict(),
        }
        torch.save(final_ckpt, final_path)
        logger.info(
            "[AS1-FS-LoRA][FinalSave] Replaced pre-merge checkpoint with merged model at %s",
            final_path,
        )

    def _train(self, train_loader, test_loader) -> None:
        """Train one fresh adapter and leave it intact for task-end Fisher."""
        network = self._unwrap_network()
        stage = "init" if self._cur_task == 0 else "update"

        if self._cur_task == 0 and not self._lora_initialized:
            network.backbone = self.build_lora_backbone()
            network.backbone.to(self._device)
            self._lora_initialized = True

        self._network = network
        self._prepare_network()
        params = self._split_param_groups(stage=stage)
        optimizer = self._build_optimizer(params, stage=stage)
        lr = optimizer.param_groups[0]["lr"]
        scheduler = self.build_scheduler(
            optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("milestones", []),
            gamma=float(self.args.get("lrate_decay", 1.0)),
            T_max=self.args.get("epochs", None),
            eta_min=self._default_cosine_eta_min(lr),
        )

        if self._cur_task == 0:
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        # Do not merge here.  Appendix Algorithm 1 requires the task Fisher
        # to be estimated at the trained LoRA solution A*, B* first.  The
        # merge/reset is performed in after_task(), only after the parent
        # lifecycle has estimated and accumulated that Fisher.
        self._fresh_merge_pending = True

    @staticmethod
    def _fisher_dict_numel(fisher: Dict[str, torch.Tensor]) -> int:
        return int(sum(int(v.numel()) for v in fisher.values()))

    @staticmethod
    def _fisher_dict_max(fisher: Dict[str, torch.Tensor]) -> float:
        vals = [v.detach().float().max().item() for v in fisher.values() if v.numel()]
        return float(max(vals)) if vals else 0.0

    def _damp_task_fisher(
        self,
        fisher: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        beta = float(getattr(self, "_ewc_fisher_damping_beta", 0.0))
        self._last_task_fisher_damping_metrics = {
            "ewc_fisher_damping_beta": beta,
            "task_fisher_damping_applied": False,
        }
        if beta <= 0.0 or not fisher:
            return fisher

        total_numel = self._fisher_dict_numel(fisher)
        if total_numel <= 0:
            return fisher

        trace_before = self._fisher_trace(fisher)
        max_before = self._fisher_dict_max(fisher)
        iso = beta / float(total_numel)
        damped = {key: (1.0 - beta) * value + iso for key, value in fisher.items()}
        trace_after = self._fisher_trace(damped)
        max_after = self._fisher_dict_max(damped)
        self._last_task_fisher_damping_metrics = {
            "ewc_fisher_damping_beta": beta,
            "task_fisher_damping_applied": True,
            "task_fisher_damping_numel": float(total_numel),
            "task_fisher_damping_iso": float(iso),
            "task_fisher_trace_before_damping": float(trace_before),
            "task_fisher_trace_after_damping": float(trace_after),
            "task_fisher_max_before_damping": float(max_before),
            "task_fisher_max_after_damping": float(max_after),
        }
        if self._is_main_process:
            logger.info(
                "[AS1-FS-LoRA] Task delta-W Fisher damped beta=%.4g numel=%d "
                "trace %.4e -> %.4e max %.4e -> %.4e iso=%.4e",
                beta,
                total_numel,
                trace_before,
                trace_after,
                max_before,
                max_after,
                iso,
            )
        return damped

    def _compute_delta_fisher(self, loader, model, class_offset: int) -> Dict[str, torch.Tensor]:
        task_fisher = super()._compute_delta_fisher(loader, model, class_offset)
        return self._damp_task_fisher(task_fisher)

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------
    def _build_optimizer(self, params, stage):
        # ``params`` is normally a list (possibly containing FC/LoRA parameter
        # group dicts). Materialize generators so both optimizers see the same
        # objects.
        if not isinstance(params, (list, tuple)):
            params = list(params)

        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
        base_name = str(
            self.args.get(
                "fs_base_optimizer",
                self.args.get("gam_base_optimizer", self.args.get("optimizer", "sgd")),
            )
        ).lower()

        if base_name == "adam":
            base_optimizer = optim.Adam(
                params,
                lr=lr,
                betas=(
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
                weight_decay=weight_decay,
            )
        elif base_name == "adamw":
            base_optimizer = optim.AdamW(
                params,
                lr=lr,
                betas=(
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
                weight_decay=weight_decay,
            )
        else:
            base_optimizer = optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        return FisherShape(
            params,
            base_optimizer=base_optimizer,
            model=self._network,
            rho=self._fs_rho,
            alpha=self._fs_alpha,
            alpha_mode=self._fs_alpha_mode,
            sam_equivalent=self._fs_sam_equivalent,
            adaptive=self._fs_adaptive,
            perturb_eps=self._fs_xi,
            grad_reduce=self._fs_grad_reduce,
        )

    # ------------------------------------------------------------------
    # Add delta-W EWC pullback after the factor-space FS direction
    # ------------------------------------------------------------------
    def _build_additive_fisher_projector(
        self,
        fisher_grad_cache: Dict[int, torch.Tensor],
        step_stats: Dict[str, float],
    ):
        collect = bool(self._mechanism_grad_stats)
        eps = float(self._mechanism_eps)

        def projector(param_groups):
            fs_norm_sq = 0.0
            fisher_norm_sq = 0.0
            combined_norm_sq = 0.0
            fs_fisher_dot = 0.0

            with torch.no_grad():
                for group in param_groups:
                    for param in group["params"]:
                        if not param.requires_grad:
                            continue

                        fs_grad = None if param.grad is None else param.grad.detach().clone()
                        fisher_grad = fisher_grad_cache.get(id(param))
                        if fisher_grad is not None:
                            fisher_grad = fisher_grad.to(
                                device=param.device,
                                dtype=param.dtype,
                            )

                        if fs_grad is None and fisher_grad is None:
                            continue
                        if fs_grad is None:
                            combined = fisher_grad.clone()
                        elif fisher_grad is None:
                            combined = fs_grad
                        else:
                            combined = fs_grad + fisher_grad

                        if param.grad is None:
                            param.grad = combined.clone()
                        else:
                            param.grad.copy_(combined)

                        if collect:
                            if fs_grad is not None:
                                fs_norm_sq += float(fs_grad.pow(2).sum().item())
                            if fisher_grad is not None:
                                fisher_norm_sq += float(fisher_grad.pow(2).sum().item())
                            if fs_grad is not None and fisher_grad is not None:
                                fs_fisher_dot += float((fs_grad * fisher_grad).sum().item())
                            combined_norm_sq += float(combined.pow(2).sum().item())

            if collect:
                fs_norm = fs_norm_sq ** 0.5
                fisher_norm = fisher_norm_sq ** 0.5
                step_stats.update(
                    {
                        "fs_dir_norm": float(fs_norm),
                        "fisher_norm": float(fisher_norm),
                        "combined_norm": float(combined_norm_sq ** 0.5),
                        "fisher_to_fs_norm_ratio": float(fisher_norm / (fs_norm + eps)),
                        "cos_fs_fisher": self._cosine_from_sums(
                            fs_fisher_dot,
                            fs_norm_sq,
                            fisher_norm_sq,
                            eps,
                        ),
                    }
                )

        return projector

    # ------------------------------------------------------------------
    # One controlled FS/SAM/SGD batch step
    # ------------------------------------------------------------------
    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        del scheduler
        if not isinstance(optimizer, FisherShape):
            raise TypeError(
                "as1_normfisher_fishershape requires a FisherShape optimizer, "
                f"got {type(optimizer).__name__}"
            )

        params: List[torch.nn.Parameter] = self._flatten_optimizer_params(optimizer)

        # This is the existing AS1 delta-W penalty. autograd.grad performs the
        # exact chain-rule pullback from delta_W=B@A to the A/B factors. It is
        # evaluated at the clean point and never enters either CE closure.
        fisher_loss_value, fisher_grad_cache = self._compute_fisher_grad_cache(params)
        projector_stats: Dict[str, float] = {}
        optimizer.set_direction_projector(
            self._build_additive_fisher_projector(fisher_grad_cache, projector_stats)
        )

        def closure():
            optimizer.zero_grad()
            outputs = self._network(inputs)
            task_loss, _, _ = self._task_loss(
                outputs["logits"],
                targets,
                class_offset,
            )
            loss_value = task_loss.detach()
            task_loss.backward()
            return outputs, loss_value

        try:
            # FisherShape returns the first, clean-point outputs and CE value.
            outputs, clean_loss = optimizer.step(closure=closure)
        finally:
            optimizer.set_direction_projector(None)

        clean_loss_value = float(
            clean_loss.item() if torch.is_tensor(clean_loss) else clean_loss
        )
        self._record_mechanism_loss_stats(clean_loss_value, fisher_loss_value)

        if self._mechanism_grad_stats:
            fs_stats = dict(getattr(optimizer, "last_stats", {}) or {})
            fs_stats.update(projector_stats)
            self._accumulate_mechanism_grad_stats(fs_stats)

        total_loss_value = clean_loss_value + float(fisher_loss_value)
        return outputs["logits"].detach(), float(total_loss_value)

    def after_task(self):
        # Parent after_task() computes the current-task delta-W Fisher while
        # the trained adapter is still active, then updates F_cum.  This is
        # Step 3 of Appendix Algorithm 1.
        super().after_task()

        # Step 4: only after Fisher estimation/accumulation, integrate the
        # trained low-rank update into the base q/v weights and reset A/B for
        # the next task.  DDP replicas have identical trained factors after
        # synchronized updates, so every rank applies the same merge.
        if getattr(self, "_fresh_merge_pending", True):
            self._maybe_barrier()
            self._network = self._unwrap_network()
            self._merge_current_lora_into_base()
            self._fresh_merge_pending = False
            self._maybe_barrier()

            # Historical A/B files are deliberately not saved: after merge,
            # the qkv base weights contain the learned task update.
            if self._is_main_process and hasattr(self._network, "save_fc"):
                self._network.save_fc(self.args.get("filepath", "./"), self._cur_task)
            self._save_merged_final_model_if_requested()

        if self._last_mechanism_metrics:
            self._last_mechanism_metrics.update(
                {
                    "method_type": "as1_normfisher_fishershape_freshlora_merge",
                    "lora_lifecycle": "fresh_adapter_merge_to_base",
                    "ewc_anchor": "zero_delta",
                    "stores_historical_lora_factors": False,
                    "fs_space": "factor",
                    "fs_rho": float(self._fs_rho),
                    "fs_alpha": float(self._fs_alpha),
                    "fs_alpha_mode": str(self._fs_alpha_mode),
                    "fs_sam_equivalent": bool(self._fs_sam_equivalent),
                    **self._last_task_fisher_damping_metrics,
                    **self._last_merge_metrics,
                }
            )


class EWCLoRA(Learner):
    """Backward-compatible alias."""
