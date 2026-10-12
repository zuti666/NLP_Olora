"""NCL on LoRA parameters with KFAC Fisher — faithful minimal implementation.

Reference: class NCL in ref/laplace (2).py (Titsias et al., "Functional Regularisation
for Continual Learning with Gaussian Processes", NeurIPS 2019 / laplace variant).

Core NCL update rule (in full-weight-matrix space, faithful to laplace.py):

    cost(params) = CE(params)                               # task loss only
    g_CE         = grad(cost)(params)
    g_reg        = F['L'] @ (W - W0) @ F['R']              # = G⊗C applied to ΔW_deviation
    g_combined   = g_CE + λ · g_reg
    new_params   = W - lr · (projs['L'] @ g_combined @ projs['R'])
                                                            # projs = F^{-1}
               = W - lr · (F^{-1} g_CE + λ · (W - W0))  # F cancels in reg term

So the effective update is:
    Δw = F^{-1} g_CE   +   λ · (W - W0)
         ↑ Fisher-preconditioned task grad   ↑ plain L2 anchor in ΔW space

For LoRA  ΔW = B·A  with KFAC  F_ΔW ≈ G ⊗ C  (chain-rule to A/B space):

    F_A = (B^T G B) ⊗ C   →   F_A^{-1} g_A = (B^T G B + γI)^{-1} g_A (C + γI)^{-1}
    F_B = G ⊗ (A^T C A)   →   F_B^{-1} g_B = (G + γI)^{-1} g_B (A^T C A + γI)^{-1}

Anchor terms (gradient of ½‖ΔW - ΔW_ref‖² w.r.t. A and B):
    anchor_A = λ · B^T @ (BA - B0A0)
    anchor_B = λ · (BA - B0A0) @ A^T

KFAC factors G and C are computed after each task and accumulated across tasks
using the additive nearest Kronecker-factor approximation (ref/fisher/kfsum.py,
``additive_nearest_kf``), which is also used for the damping initialisation in
``NCL.compute_projs`` of the reference.

Differences from GAM_EWCLoRA_NCL-KFAC.py:
  - No GAM perturbation — plain SGD (or Adam).
  - No trust-region.
  - No mechanism_eval infrastructure.
  - Minimised to the NCL paper's algorithm, nothing added.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from models_LoRAbasedCL.ewclora_youyue_fitArchitechture import Learner as _Base

logger = logging.getLogger(__name__)

# (A-weight attr,  B-weight attr,  branch key suffix)
_BRANCHES = (
    ("linear_a_q", "linear_b_q", "q"),
    ("linear_a_v", "linear_b_v", "v"),
)


class Learner(_Base):
    """NCL + LoRA KFAC Fisher.  No GAM, no EWC loss — faithful to laplace.py NCL.

    Hyper-parameters
    ----------------
    ncl_lambda          float, default 1.0  — anchor weight λ
    kfac_damping        float, default 1e-3 — γ added to G+γI and C+γI before inversion
                                              (plays the role of projection_alpha + reginv
                                               regulariser in the reference)
    kfac_grad_scale     str,   default "mean_x_bs"
    kfac_seq_agg        str,   default "token"
    kfac_max_batches    int,   default None (full training set)
    """

    def __init__(self, args):
        super().__init__(args)

        # Faithful: do NOT inherit ewc_lambda — NCL anchor scale is O(1).
        self._ncl_lambda: float = float(args.get("ncl_lambda", 1.0))

        # Damping γ: corresponds to projection_alpha + reginv eps in the reference.
        self._kfac_damping: float = float(
            args.get("kfac_damping", args.get("ncl_kfac_damping", 1e-3))
        )

        # Options forwarded to compute_lora_kfac_fisher
        self._kfac_grad_scale: str = str(args.get("kfac_grad_scale", "mean_x_bs")).lower()
        self._kfac_seq_agg: str = str(args.get("kfac_seq_agg", "token")).lower()
        _mb = args.get("kfac_max_batches", None)
        self._kfac_max_batches: Optional[int] = (
            None
            if (_mb is None or str(_mb).lower() in {"none", "null", ""})
            else int(_mb)
        )

        # Accumulated KFAC factors from all past tasks: {branch_key: (G, C)} on CPU
        self._kfac_past: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Pre-computed task-static inverses: {branch_key: (inv_G, inv_C)} on CPU
        # Only inv_BTGB and inv_ATCA (rank×rank) are freshly computed each batch.
        self._kfac_static_inv: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Disable EWC / in-task-Fisher machinery from base class.
        self._approximate_fisher_in_task = False
        self._in_task_fisher_interval = 0
        self._in_task_fisher_omega = 0.0
        self._in_task_fisher_snapshots = []
        self._in_task_fisher_last_metrics = {}

        if self._is_main_process:
            logger.info(
                "[NCL-KFAC] ncl_lambda=%.4g  kfac_damping=%.4g  "
                "kfac_grad_scale=%s  kfac_seq_agg=%s  kfac_max_batches=%s",
                self._ncl_lambda,
                self._kfac_damping,
                self._kfac_grad_scale,
                self._kfac_seq_agg,
                self._kfac_max_batches,
            )

    # ------------------------------------------------------------------
    # KFAC helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _additive_nearest_kf(
        G1: torch.Tensor, C1: torch.Tensor,
        G2: torch.Tensor, C2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge (G1⊗C1) + (G2⊗C2) into the nearest single KF.

        Faithful to ref/fisher/kfsum.py ``additive_nearest_kf``:
            π = sqrt(tr(G1)·tr(C2) / (tr(G2)·tr(C1)))
            G = G1 + π·G2,   C = C1 + C2/π
        """
        tr_G1 = max(float(G1.trace()), 1e-30)
        tr_C1 = max(float(C1.trace()), 1e-30)
        tr_G2 = max(float(G2.trace()), 1e-30)
        tr_C2 = max(float(C2.trace()), 1e-30)
        pi = max(((tr_G1 * tr_C2) / (tr_G2 * tr_C1)) ** 0.5, 1e-8)
        return G1 + pi * G2, C1 + (1.0 / pi) * C2

    @staticmethod
    def _damp_inv(M: torch.Tensor, gamma: float) -> torch.Tensor:
        """Return (M + γI)^{-1} for symmetric PSD matrix M via eigendecomposition.

        Never uses torch.linalg.inv — BTGB eigenvalues can be ≈1e-6 (tr_G/768),
        and numerical asymmetry from fp accumulation makes bare LU factorization fail.
        Eigenvalue floor = gamma ensures max inverse = 1/gamma.
        """
        M32 = M.detach().float()
        if not torch.isfinite(M32).all():
            n = M32.shape[0]
            return (1.0 / float(gamma)) * torch.eye(n, device=M32.device, dtype=torch.float32)
        M_sym = 0.5 * (M32 + M32.T)   # symmetrize before eigh
        eigvals, eigvecs = torch.linalg.eigh(M_sym)
        damped = (eigvals + float(gamma)).clamp_min(float(gamma))
        return eigvecs @ torch.diag(damped.pow(-1.0)) @ eigvecs.T

    def _accumulate_kfac(
        self,
        new: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Merge new KFAC factors into _kfac_past (additive nearest KF per key)."""
        if not self._kfac_past:
            self._kfac_past = {k: (G.clone(), C.clone()) for k, (G, C) in new.items()}
            return
        merged: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for key in set(self._kfac_past) | set(new):
            if key not in self._kfac_past:
                G2, C2 = new[key]
                merged[key] = (G2.clone(), C2.clone())
            elif key not in new:
                G1, C1 = self._kfac_past[key]
                merged[key] = (G1.clone(), C1.clone())
            else:
                G1, C1 = self._kfac_past[key]
                G2, C2 = new[key]
                merged[key] = self._additive_nearest_kf(G1, C1, G2, C2)
        self._kfac_past = merged

    def _refresh_static_inverses(self) -> None:
        """Pre-compute (G+γI)^{-1} and (C+γI)^{-1} once per task.

        Faithful to NCL.compute_projs: adds damping then inverts.
        Only the large out_dim×out_dim (inv_G) and in_dim×in_dim (inv_C)
        matrices are pre-computed here; the small rank×rank matrices
        inv_BTGB and inv_ATCA are computed fresh each batch.
        """
        self._kfac_static_inv = {}
        gamma = self._kfac_damping
        for key, (G_cpu, C_cpu) in self._kfac_past.items():
            G32 = 0.5 * (G_cpu + G_cpu.T).float()
            C32 = 0.5 * (C_cpu + C_cpu.T).float()
            inv_G = self._damp_inv(G32, gamma)
            inv_C = self._damp_inv(C32, gamma)
            self._kfac_static_inv[key] = (inv_G.cpu(), inv_C.cpu())

    # ------------------------------------------------------------------
    # KFAC Fisher computation — delegates to compute_lora_kfac_fisher
    # ------------------------------------------------------------------

    def _compute_kfac_delta_fisher(
        self,
        loader,
        model: nn.Module,
        class_offset: int,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute {branch_key: (G, C)} on CPU float32 for the current task.

        Delegates to scripts/compute_all_fisher_stage1.py which correctly
        slices q/v gradients and applies LoRA scaling.

        Key translation: compute_lora_kfac_fisher uses "{mod}.delta_w_{branch}";
        we translate to "{mod}.{branch}" to match _delta_reference convention.
        """
        import os as _os
        import sys as _sys

        _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from scripts.compute_all_fisher_stage1 import compute_lora_kfac_fisher

        lora_kfac_factors, *_ = compute_lora_kfac_fisher(
            mode="lora_kfac_empirical",
            net=model,
            loader=loader,
            device=self._device,
            task_start=int(class_offset),
            task_end=int(self._total_classes),
            max_batches=self._kfac_max_batches,
            kfac_grad_scale=self._kfac_grad_scale,
            kfac_seq_agg=self._kfac_seq_agg,
        )

        result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for delta_name, factors in lora_kfac_factors.items():
            key: Optional[str] = None
            for branch in ("q", "v"):
                suffix = f".delta_w_{branch}"
                if delta_name.endswith(suffix):
                    key = delta_name[: -len(suffix)] + f".{branch}"
                    break
                if delta_name == f"delta_w_{branch}":
                    key = branch
                    break
            if key is None:
                logger.warning("[NCL-KFAC] Unexpected key %r; skipping.", delta_name)
                continue
            result[key] = (factors["G"].float().cpu(), factors["C"].float().cpu())
        return result

    # ------------------------------------------------------------------
    # NCL preconditioner — modifies param.grad in-place
    # ------------------------------------------------------------------

    def _apply_ncl_preconditioner(self, model: nn.Module) -> None:
        """Replace param.grad with the NCL update direction (in-place).

        Implements the combined update faithful to laplace.py NCL:

            new_grad_A = (B^T G B + γI)^{-1} @ g_CE_A @ (C + γI)^{-1}
                         + λ · B^T @ (BA - B0A0)

            new_grad_B = (G + γI)^{-1} @ g_CE_B @ (A^T C A + γI)^{-1}
                         + λ · (BA - B0A0) @ A^T

        Equivalent to applying F^{-1} to (g_CE + λ·F·ΔW_deviation):
            F^{-1}(g + λ·F·dp) = F^{-1}g + λ·dp  [F cancels with F^{-1}]

        Non-LoRA parameters are left unchanged.
        """
        if not self._kfac_past or not self._delta_reference:
            return
        if not self._kfac_static_inv:
            self._refresh_static_inverses()

        gamma = self._kfac_damping
        lam = self._ncl_lambda

        for mod_name, module in self._iter_delta_modules(model):
            for a_attr, b_attr, branch in _BRANCHES:
                a_mod = getattr(module, a_attr, None)
                b_mod = getattr(module, b_attr, None)
                if a_mod is None or b_mod is None:
                    continue
                A: torch.Tensor = a_mod.weight   # (rank, in_dim)
                B: torch.Tensor = b_mod.weight   # (out_dim, rank)

                key = f"{mod_name}.{branch}"
                kfac_entry = self._kfac_past.get(key)
                static_inv = self._kfac_static_inv.get(key)
                d_ref = self._delta_reference.get(key)
                if kfac_entry is None or static_inv is None or d_ref is None:
                    continue

                G_cpu, C_cpu = kfac_entry
                inv_G_cpu, inv_C_cpu = static_inv
                dev, dt = A.device, A.dtype

                # Symmetrize G and C (fp accumulation can introduce minor asymmetry).
                G = 0.5 * (G_cpu + G_cpu.T).float().to(device=dev)
                C = 0.5 * (C_cpu + C_cpu.T).float().to(device=dev)
                inv_G = inv_G_cpu.float().to(device=dev)
                inv_C = inv_C_cpu.float().to(device=dev)

                with torch.no_grad():
                    A_d = A.detach().float()
                    B_d = B.detach().float()
                    d_ref_f32 = d_ref.float().to(device=dev)
                    d_delta = B_d @ A_d - d_ref_f32

                    # ── A parameter ────────────────────────────────────
                    if A.requires_grad and A.grad is not None:
                        g_A = A.grad.data.float()
                        BTGB = B_d.T @ G @ B_d
                        BTGB = 0.5 * (BTGB + BTGB.T)          # symmetrize rank×rank
                        inv_BTGB = self._damp_inv(BTGB, gamma)
                        prec_A = inv_BTGB @ g_A @ inv_C
                        anchor_A = lam * (B_d.T @ d_delta)
                        A.grad.data.copy_((prec_A + anchor_A).to(dt))

                    # ── B parameter ────────────────────────────────────
                    if B.requires_grad and B.grad is not None:
                        g_B = B.grad.data.float()
                        ATCA = A_d @ C @ A_d.T
                        ATCA = 0.5 * (ATCA + ATCA.T)           # symmetrize rank×rank
                        inv_ATCA = self._damp_inv(ATCA, gamma)
                        prec_B = inv_G @ g_B @ inv_ATCA
                        anchor_B = lam * (d_delta @ A_d.T)
                        B.grad.data.copy_((prec_B + anchor_B).to(dt))

    # ------------------------------------------------------------------
    # Training loop — plain SGD / Adam, NCL preconditioner applied to grad
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

        if self._is_main_process:
            self._log(
                f"[NCL-KFAC][TaskStart] task={self._cur_task}  "
                f"ncl_lambda={self._ncl_lambda:.3g}  kfac_damping={self._kfac_damping:.3g}  "
                f"has_kfac={bool(self._kfac_past)}  branches={len(self._kfac_past)}"
            )

        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        info = ""

        for epoch in prog_bar:
            model.train()
            total_loss = 0.0
            n_batches = 0
            correct = 0
            n_total = 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                optimizer.zero_grad()
                outputs = self._network(inputs)
                ce_loss, _, _ = self._task_loss(outputs["logits"], targets, class_offset)
                ce_loss.backward()

                # NCL preconditioner: F^{-1} g_CE + λ·(ΔW - ΔW_ref)
                # Faithful to laplace.py NCL: F^{-1}(g + λ·F·dp) = F^{-1}g + λ·dp
                self._apply_ncl_preconditioner(model)

                optimizer.step()

                total_loss += float(ce_loss.detach().item())
                n_batches += 1

                with torch.no_grad():
                    logits = outputs["logits"].detach()
                    if class_offset > 0:
                        preds = torch.argmax(logits[:, class_offset:], dim=1)
                        labels = targets - class_offset
                    else:
                        preds = torch.argmax(logits, dim=1)
                        labels = targets
                    correct += int(preds.eq(labels).sum().item())
                    n_total += int(labels.numel())

            if scheduler is not None:
                scheduler.step()

            denom = max(1, n_batches)
            train_acc = 0.0 if n_total == 0 else round(100.0 * correct / n_total, 2)
            info = (
                f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                f"Loss {total_loss / denom:.3f}, Train_accy {train_acc:.2f}"
            )
            if ((epoch % 5 == 4) or epoch == epochs - 1) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
            if self._is_main_process:
                prog_bar.set_description(info)

        if self._is_main_process and info:
            self._log(info)

    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        del scheduler
        optimizer.zero_grad()
        outputs = self._network(inputs)
        ce_loss, _, _ = self._task_loss(outputs["logits"], targets, class_offset)
        ce_loss.backward()
        self._apply_ncl_preconditioner(self._unwrap_network())
        optimizer.step()
        return outputs["logits"].detach(), float(ce_loss.detach().item())

    # ------------------------------------------------------------------
    # Disable in-task Fisher (not used by NCL)
    # ------------------------------------------------------------------

    def _should_capture_in_task_fisher(self, epoch_idx: int) -> bool:
        return False

    def _fisher_trace(self, fisher) -> float:
        if not fisher:
            return 0.0
        return float(sum(
            float(v.detach().float().clamp_min(0).sum().item())
            for v in fisher.values()
        ))

    # ------------------------------------------------------------------
    # after_task: compute KFAC factors, accumulate, refresh inverses
    # ------------------------------------------------------------------

    def after_task(self) -> None:
        """Faithful to laplace.py KFAC.update_priors:
        compute Fisher for completed task, merge with accumulated prior,
        then pre-compute preconditioner projections.
        """
        model = self._unwrap_network()

        new_kfac = self._compute_kfac_delta_fisher(
            loader=self.train_loader,
            model=model,
            class_offset=int(self._known_classes),
        )

        # Accumulate: additive_nearest_kf (faithful to KFAC.update_priors / opt_nearest_kf)
        self._accumulate_kfac(new_kfac)

        # Pre-compute task-static preconditioner matrices (faithful to compute_projs)
        self._refresh_static_inverses()

        if self._is_main_process and new_kfac:
            k0 = next(iter(new_kfac))
            G0, C0 = new_kfac[k0]
            self._log(
                f"[NCL-KFAC][after_task] task={self._cur_task}  "
                f"branches={len(new_kfac)}  "
                f"tr_G={float(G0.trace()):.3e}  tr_C={float(C0.trace()):.3e}"
            )
            if self._kfac_past:
                k0_acc = next(iter(self._kfac_past))
                G_acc, C_acc = self._kfac_past[k0_acc]
                self._log(
                    f"[NCL-KFAC][after_task] accum_branches={len(self._kfac_past)}  "
                    f"accum_tr_G={float(G_acc.trace()):.3e}  "
                    f"accum_tr_C={float(C_acc.trace()):.3e}"
                )

        self._delta_reference = {}
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


NCLKFACLoRA = Learner
EWCLoRA = Learner
