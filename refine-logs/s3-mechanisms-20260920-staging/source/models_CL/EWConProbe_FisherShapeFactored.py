"""Full-model EWC + phase-adaptive Fisher-Shape (FS) optimizer with projected gradient.

Fisher-Shape counterpart of ``models_CL/EWConProbe_GAMFactored.py``.  Same
full-parameter 2x2 factorial structure over Phase A / Phase B tasks, but the
"shaping" optimizer is ``FisherShape`` (SAM-cost, two-gradient Fisher-trace
shaping step; see ``optimer_PerturabtionType/fishershape.py`` and
``models_CL/EWConProbe_FisherShape.py``) instead of GAM.

  fs_task_a=F, fs_task_b=F  -> Control    (base_opt on A,  base_opt + EWC penalty on B+)
  fs_task_a=T, fs_task_b=F  -> Setting 1  (FS on A,        base_opt + EWC penalty on B+)
  fs_task_a=F, fs_task_b=T  -> Setting 2  (base_opt on A,  FS + EWC projected on B+)
  fs_task_a=T, fs_task_b=T  -> Full       (FS on A,        FS + EWC projected on B+)
  use_ewc=False              -> FS-only (no EWC in either phase)

Key difference between the two EWC application methods (identical formulas
to the GAM factored variant, only the "shaping" gradient source changes):
  Penalty path   (fs_task_b=False): loss = CE + (lambda/2) * sum_i F_i (theta_i-theta_i*)^2
  Projected path (fs_task_b=True):  g_final = g_fs - <g_fs,g_ewc>/||g_ewc||^2 * g_ewc + g_ewc
    where g_fs = FisherShape's shaped direction d_fs = g0 + alpha*(g1-g0)/s0
    and g_ewc = lambda * F_{t-1} * (theta - theta*) (analytical EWC gradient, no extra loss term)
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models_CL.EWConProbe_FisherShape import Learner as _EWConProbeFisherShape
from utils.mechanism_eval import as_bool
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(_EWConProbeFisherShape):
    """Full-model EWC + phase-adaptive Fisher-Shape (projected-gradient variant)."""

    def __init__(self, args):
        super().__init__(args)
        self._fs_task_a = as_bool(args.get("fs_task_a", True))
        self._fs_task_b = as_bool(args.get("fs_task_b", True))
        self._phase_a_tasks = int(args.get("phase_a_tasks", 1))
        self._fisher_project_eps = float(args.get("fisher_project_eps", 1e-12))
        if self._phase_a_tasks < 1:
            raise ValueError("phase_a_tasks must be >= 1")

    # ------------------------------------------------------------------
    # Training dispatch
    # ------------------------------------------------------------------

    def _train(self, train_loader, test_loader):
        in_phase_a = self._cur_task < self._phase_a_tasks
        phase = "A" if in_phase_a else "B"
        use_fs_this_task = (in_phase_a and self._fs_task_a) or \
                           (not in_phase_a and self._fs_task_b)
        # Projected gradient only makes sense on Phase B where Fisher exists
        use_projected = (not in_phase_a) and self._fs_task_b and self._use_ewc

        logger.info(
            "[FSFactored] task=%d phase=%s optimizer=%s ewc_mode=%s "
            "(phase_a_tasks=%d fs_a=%s fs_b=%s use_ewc=%s)",
            self._cur_task, phase,
            "FisherShape" if use_fs_this_task else "base_opt",
            "projected" if use_projected else "penalty",
            self._phase_a_tasks,
            self._fs_task_a, self._fs_task_b, self._use_ewc,
        )

        if use_projected:
            # FisherShape with EWC projected gradient (EWC handled at gradient level, not in loss)
            self._train_projected(train_loader, test_loader)
        else:
            # Standard EWConProbe_FisherShape (or base EWCon_Probe) training path,
            # EWC as penalty in loss when active.
            saved_opt = self._optimizer_type
            self._optimizer_type = "fishershape" if use_fs_this_task else str(
                self.args.get("fs_base_optimizer", self.args.get("gam_base_optimizer", "sgd"))
            ).lower()
            try:
                super()._train(train_loader, test_loader)
            finally:
                self._optimizer_type = saved_opt

    # ------------------------------------------------------------------
    # Projected-gradient training path (Phase B + FisherShape)
    # ------------------------------------------------------------------

    def _train_projected(self, train_loader, test_loader):
        """
        Training loop for Phase B with FisherShape + EWC projected gradient.

        EWC is NOT added to the loss.  Instead the per-step shaped direction
        is projected:
            g_final = g_fs - (<g_fs, g_ewc> / ||g_ewc||^2) * g_ewc + g_ewc
        This ensures the update is not orthogonal to the EWC constraint direction.
        """
        model = self._unwrap_network()
        model.to(self._device)
        self._reset_mechanism_stats()

        saved_opt = self._optimizer_type
        self._optimizer_type = "fishershape"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage="update")
        epochs = int(self.args.get("epochs", 1))
        scheduler = self.build_scheduler(
            optimizer.base_optimizer,
            policy=self.args.get("scheduler", "constant"),
            milestones=self.args.get("milestones", []),
            gamma=float(self.args.get("lrate_decay", 1.0)),
            T_max=epochs,
            eta_min=self.args.get("min_lr", 0.0),
        )
        self._optimizer_type = saved_opt

        task_start = int(self._known_classes)
        task_end = int(self._total_classes)

        prog_bar = tqdm(range(epochs))
        info = ""
        for epoch in prog_bar:
            model.train()
            total_loss = 0.0
            ce_accum = 0.0
            ewc_accum = 0.0
            loss_count = 0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # Pre-filter to current task classes
                inputs_t, local_targets = self._current_task_inputs(
                    inputs, targets, task_start, task_end
                )
                if inputs_t is None:
                    continue

                logits, ce_val, ewc_approx, step_stats = self._step_projected(
                    optimizer, model, inputs_t, local_targets, task_start, task_end
                )

                total_loss += ce_val + ewc_approx
                ce_accum += ce_val
                ewc_accum += ewc_approx
                loss_count += 1

                if self._mechanism_eval:
                    stats = {
                        "ce": ce_val,
                        "ewc": ewc_approx,
                        "ewc_ce_ratio": ewc_approx / (ce_val + self._mechanism_eps),
                    }
                    stats.update(step_stats)
                    stats.update(self._compute_param_drift_metrics())
                    self._record_mechanism_stats(stats, increment=True)

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(local_targets).cpu().sum()
                    total += len(local_targets)

            scheduler.step()

            n = max(1, loss_count)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2) if total > 0 else 0.0
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} [fs_proj] => "
                f"Loss {total_loss/n:.3f}, Train_accy {train_acc:.2f}, "
                f"CE {ce_accum/n:.3f}, EWC {ewc_accum/n:.3e}"
            )
            if epoch % 5 == 4:
                test_acc = self._compute_accuracy(model, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            prog_bar.set_description(info)

        if info:
            logger.info(info)

    def _step_projected(
        self,
        optimizer,
        model: nn.Module,
        inputs: torch.Tensor,
        local_targets: torch.Tensor,
        task_start: int,
        task_end: int,
    ):
        """
        One projected-FisherShape step.

        Returns (task_logits, ce_value, ewc_approx_value, mechanism_stats).
        """
        flat_params = self._flatten_optimizer_params(optimizer)
        ewc_grad_cache = self._build_ewc_grad_cache(flat_params)

        projection_stats: Dict[str, float] = {}
        optimizer.set_direction_projector(
            self._build_projected_ewc_projector(ewc_grad_cache, projection_stats)
        )

        logits_out: List[torch.Tensor] = []
        ce_cache: Dict[str, float] = {"ce": 0.0}

        def closure():
            optimizer.zero_grad()
            logits = model(inputs)["logits"][:, task_start:task_end]
            loss = F.cross_entropy(logits, local_targets)
            ce_cache["ce"] = float(loss.detach().item())
            loss.backward()
            logits_out.clear()
            logits_out.append(logits.detach())
            return logits, loss.detach()

        try:
            _, loss_val = optimizer.step(closure=closure)
        finally:
            optimizer.set_direction_projector(None)

        # EWC penalty value (for logging / metric tracking only — not added to loss)
        with torch.no_grad():
            ewc_approx = float(self._ewc_penalty(model).detach().item())

        logits_det = logits_out[0] if logits_out else model(inputs)["logits"][:, task_start:task_end].detach()
        step_stats = dict(getattr(optimizer, "last_stats", {}) or {})
        step_stats.update(projection_stats)
        return logits_det, float(ce_cache["ce"]), ewc_approx, step_stats

    # ------------------------------------------------------------------
    # Projected gradient operator
    # ------------------------------------------------------------------

    def _build_projected_ewc_projector(
        self,
        ewc_grad_cache: Dict[int, torch.Tensor],
        log_cache: Dict[str, float] | None = None,
    ):
        """
        Returns a projector callable for FisherShape's set_direction_projector.

        For each parameter:
            g_final = g_fs - (<g_fs, g_ewc> / ||g_ewc||^2) * g_ewc + g_ewc

        This removes the component of g_fs parallel to g_ewc and then adds
        g_ewc back.  With eps -> 0, each parameter block therefore satisfies
        <g_final, g_ewc> = ||g_ewc||^2.
        """
        eps = self._fisher_project_eps

        def projector(param_groups):
            fs_norm_sq = 0.0
            ewc_norm_sq = 0.0
            fs_ewc_dot = 0.0
            removed_norm_sq = 0.0
            final_norm_sq = 0.0
            for group in param_groups:
                for p in group["params"]:
                    if not p.requires_grad or p.grad is None:
                        continue
                    g_fs = p.grad.detach()
                    if log_cache is not None:
                        fs_block_norm_sq = float(g_fs.pow(2).sum().item())
                        fs_norm_sq += fs_block_norm_sq
                    g_ewc = ewc_grad_cache.get(id(p))
                    if g_ewc is None:
                        if log_cache is not None:
                            final_norm_sq += fs_block_norm_sq
                        continue
                    g_ewc = g_ewc.to(device=g_fs.device, dtype=g_fs.dtype)
                    denom = (g_ewc * g_ewc).sum() + eps
                    coeff = (g_fs * g_ewc).sum() / denom
                    removed = coeff * g_ewc
                    g_final = g_fs - removed + g_ewc

                    if log_cache is not None:
                        ewc_norm_sq += float(g_ewc.pow(2).sum().item())
                        fs_ewc_dot += float((g_fs * g_ewc).sum().item())
                        removed_norm_sq += float(removed.pow(2).sum().item())
                        final_norm_sq += float(g_final.pow(2).sum().item())

                    p.grad.data.copy_(g_final)

            if log_cache is not None:
                fs_norm = fs_norm_sq ** 0.5
                ewc_norm = ewc_norm_sq ** 0.5
                removed_norm = removed_norm_sq ** 0.5
                log_cache["fs_dir_norm"] = fs_norm
                log_cache["ewc_grad_norm"] = ewc_norm
                log_cache["cos_fs_ewc"] = fs_ewc_dot / (
                    fs_norm * ewc_norm + eps
                )
                log_cache["projection_removed_norm"] = removed_norm
                log_cache["projection_removal_ratio"] = removed_norm / (
                    fs_norm + eps
                )
                log_cache["projected_final_dir_norm"] = final_norm_sq ** 0.5

        return projector
