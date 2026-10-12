"""Full-model EWC + phase-adaptive GAM with projected gradient.

This is the full-parameter (non-LoRA) equivalent of
``models_LoRAbasedCL/ewclora_gam_factored.py``.

It inherits ``EWCon_Probe.Learner`` to get:
  - ``IncrementalNet`` backbone (same framework as the rest of the codebase)
  - FQA + flatness evaluation (Hessian, sharpness, multi-Fisher)
  - Online diagonal Fisher accumulation and mechanism metrics

And adds a 2×2 factorial optimizer structure over Phase A / Phase B tasks:

  gam_task_a=F, gam_task_b=F  → Control     (SGD on A,   SGD + EWC penalty on B+)
  gam_task_a=T, gam_task_b=F  → Setting 1   (GAM on A,   SGD + EWC penalty on B+)
  gam_task_a=F, gam_task_b=T  → Setting 2   (SGD on A,   GAM + EWC projected on B+)
  gam_task_a=T, gam_task_b=T  → Full        (GAM on A,   GAM + EWC projected on B+)
  use_ewc=False                → GAM-only    (no EWC in either phase)

Key difference between the two EWC application methods:
  Penalty path  (gam_task_b=False):  loss = CE + (λ/2) Σ F_i (θ_i−θ_i*)²
  Projected path (gam_task_b=True):  g_final = g_gam − <g_gam,g_ewc>/‖g_ewc‖² · g_ewc + g_ewc
    where g_ewc = λ·F⊙(θ−θ*) (analytical EWC gradient, no extra loss term)
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models_CL.EWCon_Probe import Learner as _EWConProbe
from utils.mechanism_eval import as_bool
from utils.toolkit import tensor2numpy

logger = logging.getLogger(__name__)


class Learner(_EWConProbe):
    """Full-model EWC + phase-adaptive GAM (projected-gradient variant)."""

    def __init__(self, args):
        super().__init__(args)
        self._gam_task_a = as_bool(args.get("gam_task_a", True))
        self._gam_task_b = as_bool(args.get("gam_task_b", True))
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
        use_gam_this_task = (in_phase_a and self._gam_task_a) or \
                            (not in_phase_a and self._gam_task_b)
        # Projected gradient only makes sense on Phase B where Fisher exists
        use_projected = (not in_phase_a) and self._gam_task_b and self._use_ewc

        logger.info(
            "[Factored] task=%d phase=%s optimizer=%s ewc_mode=%s "
            "(phase_a_tasks=%d gam_a=%s gam_b=%s use_ewc=%s)",
            self._cur_task, phase,
            "GAM" if use_gam_this_task else "SGD/Adam",
            "projected" if use_projected else "penalty",
            self._phase_a_tasks,
            self._gam_task_a, self._gam_task_b, self._use_ewc,
        )

        if use_projected:
            # GAM with EWC projected gradient (EWC handled at gradient level, not in loss)
            self._train_projected(train_loader, test_loader)
        else:
            # Standard EWCon_Probe training (GAM or SGD, EWC as penalty in loss)
            saved_opt = self._optimizer_type
            if not use_gam_this_task:
                # Force non-GAM optimizer for this phase
                self._optimizer_type = str(
                    self.args.get("gam_base_optimizer", "sgd")
                ).lower()
            try:
                super()._train(train_loader, test_loader)
            finally:
                self._optimizer_type = saved_opt

    # ------------------------------------------------------------------
    # Projected-gradient training path (Phase B + GAM)
    # ------------------------------------------------------------------

    def _train_projected(self, train_loader, test_loader):
        """
        Training loop for Phase B with GAM + EWC projected gradient.

        EWC is NOT added to the loss.  Instead the per-step gradient is projected:
            g_final = g_gam − (<g_gam, g_ewc> / ‖g_ewc‖²) · g_ewc + g_ewc
        This ensures the update is not orthogonal to the EWC constraint direction.
        """
        model = self._unwrap_network()
        model.to(self._device)
        self._reset_mechanism_stats()

        # Build GAM optimizer (force optimizer_type=gam for _build_optimizer)
        saved_opt = self._optimizer_type
        self._optimizer_type = "gam"
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params, stage="update")
        epochs = int(self.args.get("epochs", 1))
        scheduler = self.build_scheduler(
            optimizer,
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

                logits, ce_val, ewc_approx = self._step_projected(
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
                    stats.update(self._compute_param_drift_metrics())
                    self._record_mechanism_stats(stats, increment=True)

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(local_targets).cpu().sum()
                    total += len(local_targets)

            scheduler.step()

            n = max(1, len(train_loader))
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2) if total > 0 else 0.0
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} [proj] => "
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
        One projected-GAM step.

        Returns (task_logits, ce_value, ewc_approx_value).
        """
        flat_params = self._flatten_optimizer_params(optimizer)
        ewc_grad_cache = self._build_ewc_grad_cache(flat_params)

        optimizer.set_direction_projector(
            self._build_projected_ewc_projector(ewc_grad_cache)
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
        return logits_det, float(ce_cache["ce"]), ewc_approx

    # ------------------------------------------------------------------
    # EWC gradient cache (analytical, no autograd)
    # ------------------------------------------------------------------

    def _build_ewc_grad_cache(self, params: List[torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Analytical EWC gradient for each optimizer parameter.

        ∂L_EWC/∂θ_i = λ · F_i · (θ_i − θ_i*)

        Avoids a second backward pass by computing the gradient directly from
        the stored diagonal Fisher and checkpoint.
        """
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
                # Gradient of 0.5 * λ * Σ F_i (θ_i - θ_i*)² w.r.t. θ_i
                cache[id(p)] = self._ewc_lambda * fisher_d * (p.detach() - ref_d)

        return cache

    # ------------------------------------------------------------------
    # Projected gradient operator
    # ------------------------------------------------------------------

    def _build_projected_ewc_projector(self, ewc_grad_cache: Dict[int, torch.Tensor]):
        """
        Returns a projector callable for GAM's set_direction_projector.

        For each parameter:
            g_final = g_gam − (<g_gam, g_ewc> / ‖g_ewc‖²) · g_ewc + g_ewc

        This removes the component of g_gam that opposes g_ewc and then
        adds g_ewc back, ensuring the update moves along the EWC direction.
        """
        eps = self._fisher_project_eps

        def projector(param_groups):
            for group in param_groups:
                for p in group["params"]:
                    if not p.requires_grad or p.grad is None:
                        continue
                    g_ewc = ewc_grad_cache.get(id(p))
                    if g_ewc is None:
                        continue
                    g_gam = p.grad.detach()
                    g_ewc = g_ewc.to(device=g_gam.device, dtype=g_gam.dtype)
                    denom = (g_ewc * g_ewc).sum() + eps
                    coeff = (g_gam * g_ewc).sum() / denom
                    p.grad.data.copy_(g_gam - coeff * g_ewc + g_ewc)

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
