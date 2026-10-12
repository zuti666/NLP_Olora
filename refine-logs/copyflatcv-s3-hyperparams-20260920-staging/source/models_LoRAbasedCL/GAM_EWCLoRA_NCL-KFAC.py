"""SeqLoRA EWCLoRA with GAM + NCL-style update (LoRA KFAC Fisher, no EWC lambda loss).

Replaces the diagonal Fisher used in GAM_EWCLoRA_NCL.py with a full Kronecker-factored
approximation in LoRA delta-W space, following the NCL update rule from laplace.py
(see ref/laplace (2).py, class NCL).

NCL in LoRA KFAC form (this file):
    Fisher for ΔW = B @ A is approximated as F_ΔW ≈ G ⊗ C where:
        G  (out_dim × out_dim) = E[g_out g_out^T]   output-gradient covariance
        C  (in_dim  × in_dim)  = E[a    a^T   ]     input-activation covariance

    Chain-rule Gauss-Newton KFAC mapped to A/B parameter space:
        F_A ≈ (B^T G B) ⊗ C                         shape: (rank×rank) ⊗ (in×in)
        F_B ≈ G ⊗ (A^T C A)                         shape: (out×out) ⊗ (rank×rank)

    NCL update per parameter:
        v_A = (B^T G B + γI)^{-1} @ g_gam_A @ (C + γI)^{-1}
              + λ · B^T @ (Δ_now - Δ_ref)
        v_B = (G + γI)^{-1} @ g_gam_B @ (A^T C A + γI)^{-1}
              + λ · (Δ_now - Δ_ref) @ A^T

    where γ = sqrt(fisher_eta) is the KFAC damping coefficient.

KFAC factor accumulation across tasks:
    Uses the additive nearest Kronecker-factor approximation (from ref/fisher/kfsum.py
    ``additive_nearest_kf``):
        π  = sqrt(tr(G_old) · tr(C_new)) / sqrt(tr(G_new) · tr(C_old))
        G  ← G_old + π · G_new
        C  ← C_old + (1/π) · C_new
    This keeps (G_total ⊗ C_total) as the nearest single KF to the sum.

Trust-region radius uses the exact KFAC-weighted Frobenius norm:
    v^T Λ v ≈ tr(V_delta^T G V_delta C)
            = (G @ V_delta @ C * V_delta).sum()   [Hadamard identity]

Key differences from GAM_EWCLoRA_NCL.py (diagonal):
  - _fisher_past_delta is NOT used for the NCL update (kept empty).
  - _kfac_past stores {branch_key: (G, C)} per task-accumulated KFAC factors.
  - after_task computes G/C via forward+backward hooks on linear_a/linear_b modules.
  - In-task Fisher snapshots are disabled (no diagonal path-mixing).
  - Projector uses matrix multiply instead of elementwise Fisher inverse.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from models_LoRAbasedCL.ewclora_youyue_fitArchitechture import Learner as _SeqEWCLoRA
from optimer_PerturabtionType.gam import GAM
from utils.mechanism_eval import as_bool

logger = logging.getLogger(__name__)

# Branch configurations: (A-weight attr, B-weight attr, delta-W key suffix)
_BRANCHES = (
    ("linear_a_q", "linear_b_q", "q"),
    ("linear_a_v", "linear_b_v", "v"),
)

# Trust-region info tuples:
#   legacy   : (G, C, A_detached, B_detached, d_delta)
#   explicit : (G, C, id(A), id(B), A_detached, B_detached, d_delta)
_TrInfoKFAC = Tuple[object, ...]


class Learner(_SeqEWCLoRA):
    """SeqLoRA + LoRA KFAC Fisher + GAM with NCL-style preconditioned update."""

    def __init__(self, args):
        super().__init__(args)

        self._configured_optimizer_type = str(getattr(self, "_optimizer_type", "sgd")).lower()
        self._optimizer_type = "gam"
        self._base_optimizer_type = "gam"
        self._optimizer_type_by_task = None

        self._gam_base_optimizer = self._resolve_gam_base_optimizer(args)
        self._gam_adaptive = as_bool(args.get("gam_adaptive", False))
        self._gam_grad_reduce = str(args.get("gam_grad_reduce", "mean"))
        self._gam_perturb_eps = float(args.get("gam_perturb_eps", 1e-12))
        self._gam_grad_rho = float(args.get("gam_grad_rho", args.get("grad_rho", 0.02)))
        self._gam_grad_norm_rho = float(
            args.get("gam_grad_norm_rho", args.get("grad_norm_rho", 0.2))
        )
        self._gam_args = SimpleNamespace(
            grad_beta_1=float(args.get("gam_grad_beta_1", args.get("grad_beta_1", 1.0))),
            grad_beta_2=float(args.get("gam_grad_beta_2", args.get("grad_beta_2", 1.0))),
            grad_beta_3=float(args.get("gam_grad_beta_3", args.get("grad_beta_3", 1.0))),
            grad_gamma=float(args.get("gam_grad_gamma", args.get("grad_gamma", 0.03))),
        )

        # Do NOT fall back to ewc_lambda — NCL anchor scale should be O(1).
        self._ncl_lambda = float(args.get("ncl_lambda", 1.0))
        # Use args directly; parent sets _ewc_eta=0.0 by default which would mask 1e-4.
        self._fisher_eta = float(
            args.get("fisher_eta", args.get("ewc_eta", args.get("eta", 1e-4)))
        )
        if self._fisher_eta == 0.0:
            self._fisher_eta = 1e-4

        # KFAC damping: γ added to G+γI and C+γI before inversion.
        # Controls max eigenvalue of the inverse: λ_max(M^{-1}) ≤ 1/γ.
        # Default: sqrt(fisher_eta); increase if training is unstable.
        _kd = args.get("kfac_damping", args.get("ncl_kfac_damping", None))
        self._kfac_damping: float = (
            float(_kd) if _kd is not None
            else max(float(self._fisher_eta) ** 0.5, 1e-4)
        )

        # Fractional matrix power p for Fisher inverse: (M+γI)^{-p}.
        # p=1.0: standard KFAC inverse; p=0.5: gentler sqrt-inverse.
        self._ncl_inv_power: float = float(args.get("ncl_inv_power", 1.0))

        # Optional element-wise bound on pre-computed matrix entries after
        # the Kronecker product application (bounds per-parameter update size).
        _clip = args.get("ncl_inv_clip", None)
        self._ncl_inv_clip: Optional[float] = (
            None if (_clip is None or str(_clip).lower() in {"none", "null", ""})
            else float(_clip)
        )

        # KFAC analogue of diagonal ncl_normalize_fisher:
        # normalize each factor to average eigenvalue 1 so that
        # mean(G kron C) = 1 and kfac_damping has a stable scale.
        self._ncl_normalize_kfac: bool = as_bool(
            args.get("ncl_normalize_kfac", args.get("ncl_normalize_fisher", False))
        )
        self._kfac_norm_eps: float = float(args.get("kfac_norm_eps", 1e-12))

        # Gradient norm clipping applied to the final NCL update direction
        # (after Fisher preconditioner + anchor term) before passing to optimizer.
        # Prevents runaway anchor-term explosion when d_delta grows large.
        # Recommended: 1.0–10.0.  None = no clipping.
        _gc = args.get("ncl_grad_clip", None)
        self._ncl_grad_clip: Optional[float] = (
            None if (_gc is None or str(_gc).lower() in {"none", "null", ""})
            else float(_gc)
        )

        # Pre-computed static inverses for G/C (rebuilt in after_task).
        # {branch_key: (inv_G, inv_C)} using current damping/power settings.
        self._kfac_static_inv: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Gradient scaling for KFAC G factor (mirrors compute_all_fisher_stage1.py options):
        #   "mean"      : CE(mean) reduction, no correction (baseline)
        #   "mean_x_bs" : CE(mean) with hook gradients × batch_size (recommended)
        #   "sum"       : CE(sum) reduction
        self._kfac_grad_scale: str = str(
            args.get("kfac_grad_scale", "mean_x_bs")
        ).lower()
        # Token vs sample aggregation for 3-D (B, S, dim) Linear inputs
        self._kfac_seq_agg: str = str(args.get("kfac_seq_agg", "token")).lower()
        # Max batches for Fisher accumulation (None = full training set)
        _mb = args.get("kfac_max_batches", args.get("fisher_max_batches", None))
        self._kfac_max_batches: Optional[int] = (
            None if (_mb is None or str(_mb).lower() in {"none", "null", ""})
            else int(_mb)
        )

        # Accumulated KFAC factors per branch key: {key: (G, C)}
        self._kfac_past: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Trust-region radius.  legacy keeps the old manual path; explicit
        # enables the corrected id-matched KFAC trust-region implementation.
        _r = args.get("ncl_r_trust", args.get("ncl_r", None))
        self._ncl_r_trust: Optional[float] = (
            None if (_r is None or str(_r).lower() in {"none", "null", ""})
            else float(_r)
        )
        self._ncl_trust_mode: str = str(args.get("ncl_trust_mode", "step")).lower()
        self._ncl_trust_xi: float = float(args.get("ncl_trust_xi", 1e-12))
        self._ncl_trust_impl: str = str(args.get("ncl_trust_impl", "legacy")).lower()
        if self._ncl_trust_impl not in {"legacy", "explicit"}:
            self._ncl_trust_impl = "legacy"
        self._ncl_r_from_gam: bool = as_bool(args.get("ncl_r_from_gam", False))
        self._ncl_r_source: str = str(args.get("ncl_r_source", "gam_grad_rho")).lower()
        self._ncl_r_scale: float = float(args.get("ncl_r_scale", 1.0))
        _r_min = args.get("ncl_r_min", None)
        self._ncl_r_min: Optional[float] = (
            None if (_r_min is None or str(_r_min).lower() in {"none", "null", ""})
            else float(_r_min)
        )
        _r_max = args.get("ncl_r_max", None)
        self._ncl_r_max: Optional[float] = (
            None if (_r_max is None or str(_r_max).lower() in {"none", "null", ""})
            else float(_r_max)
        )
        self._ncl_trust_outside_action: str = str(
            args.get("ncl_trust_outside_action", "zero")
        ).lower()
        if self._ncl_trust_outside_action not in {"zero", "step", "none"}:
            self._ncl_trust_outside_action = "zero"
        self._active_ncl_r_trust: Optional[float] = self._resolve_active_ncl_radius()

        # Euclidean cumulative hard constraint:
        #   ||theta_s - eta * a_s * v_s - theta_{t-1}||_2 <= rho_GAM
        # Keeps trainable parameters inside a Euclidean ball around the task-start
        # snapshot, where the radius equals the GAM perturbation radius rho.
        # Strict guarantee holds only for SGD with no momentum/weight_decay.
        self._ncl_euclidean_trust: bool = as_bool(
            args.get("ncl_euclidean_trust", False)
        )
        self._ncl_euclidean_apply_first_task: bool = as_bool(
            args.get("ncl_euclidean_apply_first_task", False)
        )
        self._ncl_euclidean_exclude_classifier: bool = as_bool(
            args.get("ncl_euclidean_exclude_classifier", True)
        )
        self._ncl_euclidean_radius_from_gam: bool = as_bool(
            args.get("ncl_euclidean_radius_from_gam", True)
        )
        _euc_r = args.get("ncl_euclidean_radius", None)
        self._ncl_euclidean_radius: Optional[float] = (
            None if (_euc_r is None or str(_euc_r).lower() in {"none", "null", ""})
            else float(_euc_r)
        )
        self._ncl_euclidean_radius_source: str = str(
            args.get("ncl_euclidean_radius_source", "gam_grad_rho")
        ).lower()
        self._ncl_euclidean_radius_scale: float = float(
            args.get("ncl_euclidean_radius_scale", 1.0)
        )
        self._ncl_euclidean_outside_action: str = str(
            args.get("ncl_euclidean_outside_action", "zero")
        ).lower()
        if self._ncl_euclidean_outside_action not in {"zero", "none"}:
            self._ncl_euclidean_outside_action = "zero"

        # Snapshot of trainable parameters at task start (theta_{t-1}).
        # Populated by _snapshot_param_reference() at the beginning of each task.
        self._param_reference: Dict[int, torch.Tensor] = {}
        self._param_name_by_id: Dict[int, str] = {}
        self._param_reference_skipped_classifier: int = 0

        # In-task Fisher snapshots are disabled for KFAC (no diagonal path-mixing)
        self._approximate_fisher_in_task = False
        self._in_task_fisher_interval = 0
        self._in_task_fisher_omega = 0.0
        self._in_task_fisher_snapshots: List[
            Tuple[int, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = []
        self._in_task_fisher_last_metrics: Dict[str, float] = {}

        self._mechanism_eval = as_bool(args.get("mechanism_eval", False))
        self._mechanism_eps = float(args.get("mechanism_eps", 1e-12))
        self._mechanism_sums: Dict[str, float] = {}
        self._mechanism_count = 0
        self._last_mechanism_metrics: Dict[str, object] = {}

        if self._is_main_process:
            logger.info(
                "[GAM-EWCLoRA-NCL-KFAC] base=%s grad_rho=%.4g grad_norm_rho=%.4g "
                "adaptive=%s grad_reduce=%s ncl_lambda=%.4g fisher_eta=%.4g "
                "kfac_damping=%.4g inv_power=%.2g inv_clip=%s "
                "normalize_kfac=%s kfac_grad_scale=%s kfac_seq_agg=%s kfac_max_batches=%s "
                "r_trust=%s active_r=%s trust_mode=%s trust_impl=%s "
                "r_from_gam=%s r_source=%s r_scale=%.3g",
                self._gam_base_optimizer,
                self._gam_grad_rho,
                self._gam_grad_norm_rho,
                self._gam_adaptive,
                self._gam_grad_reduce,
                self._ncl_lambda,
                self._fisher_eta,
                self._kfac_damping,
                self._ncl_inv_power,
                self._ncl_inv_clip,
                self._ncl_normalize_kfac,
                self._kfac_grad_scale,
                self._kfac_seq_agg,
                self._kfac_max_batches,
                self._ncl_r_trust,
                self._active_ncl_r_trust,
                self._ncl_trust_mode,
                self._ncl_trust_impl,
                self._ncl_r_from_gam,
                self._ncl_r_source,
                self._ncl_r_scale,
            )

    # ------------------------------------------------------------------
    # GAM base optimizer resolution
    # ------------------------------------------------------------------
    def _resolve_active_ncl_radius(self) -> Optional[float]:
        if self._ncl_r_trust is not None:
            r = float(self._ncl_r_trust)
        elif self._ncl_r_from_gam:
            if self._ncl_r_source in {"gam", "gam_rho", "gam_grad_rho", "grad_rho", "rho"}:
                rho = float(self._gam_grad_rho)
            elif self._ncl_r_source in {
                "gam_norm_rho", "gam_grad_norm_rho", "grad_norm_rho", "norm_rho"
            }:
                rho = float(self._gam_grad_norm_rho)
            else:
                return None
            r = float(self._ncl_r_scale * rho / (2.0 ** 0.5))
        else:
            return None

        if self._ncl_r_min is not None:
            r = max(r, float(self._ncl_r_min))
        if self._ncl_r_max is not None:
            r = min(r, float(self._ncl_r_max))
        return r

    def _current_ncl_radius(self) -> Optional[float]:
        r = getattr(self, "_active_ncl_r_trust", None)
        if r is not None:
            return float(r)
        return self._resolve_active_ncl_radius()

    def _resolve_euclidean_radius(self) -> Optional[float]:
        """Return the Euclidean GAM-ball radius rho for the cumulative constraint.

        If ncl_euclidean_radius is set explicitly, use it directly.
        Otherwise derive from the GAM perturbation rho (no sqrt(2) factor —
        the GAM rho is already the intended ball radius, not half-diagonal).
        """
        if self._ncl_euclidean_radius is not None:
            return float(self._ncl_euclidean_radius)
        if not self._ncl_euclidean_radius_from_gam:
            return None
        src = self._ncl_euclidean_radius_source
        if src in {"gam", "gam_rho", "gam_grad_rho", "grad_rho", "rho"}:
            rho = float(self._gam_grad_rho)
        elif src in {"gam_norm_rho", "gam_grad_norm_rho", "grad_norm_rho", "norm_rho"}:
            rho = float(self._gam_grad_norm_rho)
        else:
            return None
        return float(self._ncl_euclidean_radius_scale * rho)

    def _resolve_gam_base_optimizer(self, args) -> str:
        base = str(
            args.get(
                "gam_base_optimizer",
                args.get("optimizer", self._configured_optimizer_type),
            )
        ).lower()
        if base in {"gam", "fishersam", "sam", "cflat", "rwp", "arwp"}:
            return "sgd"
        if base not in {"sgd", "adam", "adamw"}:
            logger.warning(
                "[GAM-EWCLoRA-NCL-KFAC] Unsupported gam_base_optimizer=%s; using SGD.", base
            )
            return "sgd"
        return base

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------
    def _build_optimizer(self, params, stage):
        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
        params = self._normalize_param_groups(params)
        base_name = self._gam_base_optimizer

        if base_name == "adam":
            base_optimizer = torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
            )
        elif base_name == "adamw":
            base_optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(
                    float(self.args.get("adam_beta1", self.args.get("beta1", 0.9))),
                    float(self.args.get("adam_beta2", self.args.get("beta2", 0.999))),
                ),
            )
        else:
            base_optimizer = torch.optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=weight_decay,
            )

        return GAM(
            params,
            base_optimizer=base_optimizer,
            model=self._network,
            grad_rho=float(self._gam_grad_rho),
            grad_norm_rho=float(self._gam_grad_norm_rho),
            adaptive=bool(self._gam_adaptive),
            perturb_eps=float(self._gam_perturb_eps),
            args=self._gam_args,
            grad_reduce=str(self._gam_grad_reduce),
        )

    @staticmethod
    def _normalize_param_groups(params):
        if not isinstance(params, (list, tuple)):
            return params
        normalized = []
        for group in params:
            if isinstance(group, dict):
                group_params = [p for p in group.get("params", []) if p.requires_grad]
                if not group_params:
                    continue
                new_group = dict(group)
                new_group["params"] = group_params
                normalized.append(new_group)
            elif getattr(group, "requires_grad", False):
                normalized.append(group)
        if not normalized:
            raise ValueError("No trainable parameters found for GAM-EWCLoRA-NCL-KFAC.")
        return normalized

    # ------------------------------------------------------------------
    # KFAC Fisher computation — delegates to compute_lora_kfac_fisher
    # ------------------------------------------------------------------
    def _compute_kfac_delta_fisher(
        self,
        loader,
        model: nn.Module,
        class_offset: int,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute (G, C) KFAC factors for each LoRA branch.

        Delegates to ``compute_lora_kfac_fisher`` from
        ``scripts/compute_all_fisher_stage1.py``, which:
          - Hooks the qkv parent module and correctly slices the gradient
            for q (g_out[..., :dim]) and v (g_out[..., -dim:]) branches.
          - Applies the LoRA scaling factor to the gradient.
          - Supports ``kfac_grad_scale`` (mean / mean_x_bs / sum) and
            ``kfac_seq_agg`` (token / sample).

        Key translation: ``compute_lora_kfac_fisher`` uses keys of the form
        ``"{mod_name}.delta_w_{branch}"``; we translate to ``"{mod_name}.{branch}"``
        to match the ``_iter_delta_terms`` / ``_delta_reference`` convention.

        Returns:
            {branch_key: (G, C)} where G (out_dim × out_dim) and C (in_dim × in_dim)
            are on CPU, dtype=float32.
        """
        import os as _os, sys as _sys
        _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from scripts.compute_all_fisher_stage1 import compute_lora_kfac_fisher

        task_start = int(class_offset)
        task_end = int(self._total_classes)

        lora_kfac_factors, _, _, _, _ = compute_lora_kfac_fisher(
            mode="lora_kfac_empirical",
            net=model,
            loader=loader,
            device=self._device,
            task_start=task_start,
            task_end=task_end,
            max_batches=self._kfac_max_batches,
            kfac_grad_scale=self._kfac_grad_scale,
            kfac_seq_agg=self._kfac_seq_agg,
        )

        # Translate keys: "{mod}.delta_w_q" → "{mod}.q",  "{mod}.delta_w_v" → "{mod}.v"
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
                logger.warning(
                    "[NCL-KFAC] Unexpected delta_name key=%r; skipping.", delta_name
                )
                continue
            G = factors["G"].float().cpu()
            C = factors["C"].float().cpu()
            result[key] = (G, C)
        return result

    # ------------------------------------------------------------------
    # KFAC factor accumulation (additive nearest KF)
    # ------------------------------------------------------------------
    @staticmethod
    def _additive_nearest_kf(
        G1: torch.Tensor, C1: torch.Tensor,
        G2: torch.Tensor, C2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge (G1 ⊗ C1) + (G2 ⊗ C2) → single (G_merged ⊗ C_merged).

        Uses the scale-balanced additive approximation from kfsum.py:
            π        = sqrt(tr(G1) · tr(C2)) / sqrt(tr(G2) · tr(C1))
            G_merged = G1 + π · G2
            C_merged = C1 + (1/π) · C2
        """
        tr_G1 = max(float(G1.trace().item()), 1e-30)
        tr_C1 = max(float(C1.trace().item()), 1e-30)
        tr_G2 = max(float(G2.trace().item()), 1e-30)
        tr_C2 = max(float(C2.trace().item()), 1e-30)
        pi = ((tr_G1 * tr_C2) / (tr_G2 * tr_C1)) ** 0.5
        pi = max(pi, 1e-8)
        return G1 + pi * G2, C1 + (1.0 / pi) * C2

    def _accumulate_kfac(
        self,
        old: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        new: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Merge old and new KFAC dicts using additive nearest KF per key."""
        if not old:
            return {k: (G.clone(), C.clone()) for k, (G, C) in new.items()}
        merged: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        all_keys = set(old) | set(new)
        for key in all_keys:
            if key not in old:
                G2, C2 = new[key]
                merged[key] = (G2.clone(), C2.clone())
            elif key not in new:
                G1, C1 = old[key]
                merged[key] = (G1.clone(), C1.clone())
            else:
                G1, C1 = old[key]
                G2, C2 = new[key]
                merged[key] = self._additive_nearest_kf(G1, C1, G2, C2)
        return merged

    def _normalize_kfac_pair(
        self,
        G: torch.Tensor,
        C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize KFAC factors so both have average eigenvalue 1.

        This is the Kronecker-factored equivalent of diagonal mean-Fisher
        normalization: after scaling, mean_eig(G) * mean_eig(C) = 1, hence
        the average eigenvalue of G kron C is 1.  Scaling factors separately
        also removes the G/C scale ambiguity before adding shared damping.
        """
        if not self._ncl_normalize_kfac:
            return G, C

        eps = float(self._kfac_norm_eps)
        mean_G = G.diagonal().sum() / max(int(G.shape[0]), 1)
        mean_C = C.diagonal().sum() / max(int(C.shape[0]), 1)
        one_G = torch.ones_like(mean_G)
        one_C = torch.ones_like(mean_C)
        mean_G = torch.where(torch.isfinite(mean_G) & (mean_G > eps), mean_G, one_G)
        mean_C = torch.where(torch.isfinite(mean_C) & (mean_C > eps), mean_C, one_C)
        return G / mean_G, C / mean_C

    # ------------------------------------------------------------------
    # Matrix inverse with optional fractional power and eigenvalue clipping
    # ------------------------------------------------------------------
    @staticmethod
    def _matrix_power_inv(
        M: torch.Tensor,
        damp: float,
        power: float,
        clip: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute (M + damp·I)^{-power} for symmetric PSD M via eigendecomposition.

        Always operates in float32 internally regardless of input dtype.
        Never uses torch.linalg.inv — BTGB/ATCA are near-singular when tr_G ≈ 1e-3
        over 768 dims (mean eigenvalue ≈ 1.6e-6), so bare LU inversion fails.

        Eigenvalue floor = damp ensures max inverse = 1/damp regardless of input,
        matching the guarantee: λ_min(M + damp·I) ≥ damp → ‖(M+damp·I)^{-1}‖ ≤ 1/damp.
        """
        M32 = M.detach().float()
        if not torch.isfinite(M32).all():
            # Weights have diverged (Inf/NaN from training explosion).
            # Return scaled identity so training can continue one more step.
            n = M32.shape[0]
            return (float(damp) ** (-float(power))) * torch.eye(
                n, device=M32.device, dtype=torch.float32
            )
        M_sym = 0.5 * (M32 + M32.T)   # symmetrize: numerical noise can break eigh
        eigvals, eigvecs = torch.linalg.eigh(M_sym)
        # Floor at damp (not damp*1e-2) so max inverse is bounded by 1/damp
        damped = (eigvals + float(damp)).clamp_min(float(damp))
        inv_eigvals = damped.pow(-float(power))
        if clip is not None:
            inv_eigvals = inv_eigvals.clamp_max(float(clip))
        return eigvecs @ torch.diag(inv_eigvals) @ eigvecs.T   # float32

    # ------------------------------------------------------------------
    # Pre-compute static large-matrix inverses (per task, not per batch)
    # ------------------------------------------------------------------
    def _refresh_kfac_static_inverses(self) -> None:
        """Pre-compute (G + γI)^{-p} and (C + γI)^{-p} for each branch.

        Called once per task in after_task after _kfac_past is updated.
        Stores CPU float32 tensors in _kfac_static_inv for use in
        _compute_ncl_grad_cache (avoids large-matrix inversion every batch).
        """
        self._kfac_static_inv = {}
        if not self._kfac_past:
            return
        damp = float(self._kfac_damping)
        power = float(self._ncl_inv_power)
        clip = self._ncl_inv_clip
        for key, (G_cpu, C_cpu) in self._kfac_past.items():
            G32 = 0.5 * (G_cpu + G_cpu.T).float()
            C32 = 0.5 * (C_cpu + C_cpu.T).float()
            G32, C32 = self._normalize_kfac_pair(G32, C32)
            inv_G = self._matrix_power_inv(G32, damp, power, clip)
            inv_C = self._matrix_power_inv(C32, damp, power, clip)
            self._kfac_static_inv[key] = (inv_G.cpu(), inv_C.cpu())
        if self._is_main_process:
            logger.info(
                "[NCL-KFAC] Static inverses refreshed: branches=%d damp=%.3e power=%.2g clip=%s normalize=%s",
                len(self._kfac_static_inv), damp, power, clip, self._ncl_normalize_kfac,
            )

    # ------------------------------------------------------------------
    # Euclidean cumulative constraint helpers
    # ------------------------------------------------------------------
    def _refresh_param_name_map(self) -> None:
        model = self._unwrap_network()
        self._param_name_by_id = {id(p): name for name, p in model.named_parameters()}

    @staticmethod
    def _is_classifier_param_name(name: str) -> bool:
        name = name[7:] if name.startswith("module.") else name
        if name.startswith(("fc.", "head.", "classifier.")):
            return True
        classifier_suffixes = (
            ".fc.weight",
            ".fc.bias",
            ".head.weight",
            ".head.bias",
            ".classifier.weight",
            ".classifier.bias",
        )
        return name.endswith(classifier_suffixes) or ".classifiers." in name

    def _is_euclidean_tracked_param(self, param: torch.Tensor) -> bool:
        if not self._ncl_euclidean_exclude_classifier:
            return True
        name = self._param_name_by_id.get(id(param), "")
        return not self._is_classifier_param_name(name)

    def _snapshot_param_reference(self, param_groups) -> None:
        """Snapshot trainable parameters at task start as theta_{t-1}.

        Used for the Euclidean cumulative constraint:
            ||theta_s - eta * a_s * v_s - theta_{t-1}||_2 <= rho_GAM.
        """
        self._refresh_param_name_map()
        refs: Dict[int, torch.Tensor] = {}
        skipped_classifier = 0
        for group in param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if not self._is_euclidean_tracked_param(p):
                    skipped_classifier += 1
                    continue
                if p.requires_grad:
                    refs[id(p)] = p.detach().cpu().clone()
        self._param_reference = refs
        self._param_reference_skipped_classifier = skipped_classifier

    def _compute_euclidean_cumulative_scale(
        self,
        param_groups,
        param_to_v: Dict[int, torch.Tensor],
        rho: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute scale a in [0,1] such that ||Delta - a*u||_2 <= rho.

        Delta = theta_s - theta_{t-1}  (current cumulative drift from task start).
        v is the NCL direction already stored in param_to_v.
        u is the actual SGD step vector, using each param group's own lr:
            u_i = lr_i * v_i.

        Solves the quadratic: A*a^2 + B*a + C <= 0, where
            A = ||u||^2
            B = -2 * Delta^T u
            C = ||Delta||^2 - rho^2

        If C <= 0 (currently feasible), finds the largest admissible a <= 1.
        If C > 0 (already outside ball), applies ncl_euclidean_outside_action.
        """
        eps = float(self._ncl_trust_xi)

        delta_norm_sq = 0.0
        v_norm_sq = 0.0
        step_norm_sq = 0.0
        delta_dot_v = 0.0
        delta_dot_step = 0.0
        matched = 0
        tracked = 0
        total = 0
        total_all = 0
        skipped_classifier = 0

        for group in param_groups:
            lr = float(group.get("lr", 0.0))
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                total_all += 1
                if not self._is_euclidean_tracked_param(p):
                    skipped_classifier += 1
                    continue
                total += 1
                ref = self._param_reference.get(id(p))
                if ref is None:
                    continue
                dev = p.device
                dt = p.dtype
                ref_dev = ref.to(device=dev, dtype=dt)
                delta = p.detach() - ref_dev
                delta_norm_sq += float(delta.pow(2).sum().item())
                tracked += 1

                v = param_to_v.get(id(p))
                if v is None:
                    continue
                v_dev = v.to(device=dev, dtype=dt)
                step_dev = v_dev * lr
                v_norm_sq += float(v_dev.pow(2).sum().item())
                step_norm_sq += float(step_dev.pow(2).sum().item())
                delta_dot_v += float((delta * v_dev).sum().item())
                delta_dot_step += float((delta * step_dev).sum().item())
                matched += 1

        A = step_norm_sq
        B = -2.0 * delta_dot_step
        C = delta_norm_sq - rho ** 2

        feasible = 1.0 if C <= 0.0 else 0.0
        euc_scale = 1.0

        if C <= 0.0:
            # Currently inside ball: find largest a s.t. still inside after step.
            if A > eps:
                disc = max(B ** 2 - 4.0 * A * C, 0.0)
                s_max = (-B + disc ** 0.5) / (2.0 * A + eps)
                euc_scale = min(1.0, max(0.0, s_max))
            # else: v is essentially zero; any scale is fine → keep 1.0
        else:
            # Already outside the Euclidean GAM ball.
            if self._ncl_euclidean_outside_action == "zero":
                euc_scale = 0.0
            # "none": let the step proceed unscaled

        after_norm_sq = max(
            delta_norm_sq
            - 2.0 * euc_scale * delta_dot_step
            + (euc_scale ** 2) * step_norm_sq,
            0.0,
        )

        stats = {
            "euc_scale": float(euc_scale),
            "euc_before_norm": float(delta_norm_sq ** 0.5),
            "euc_after_norm": float(after_norm_sq ** 0.5),
            "euc_budget": float(rho),
            "euc_ratio_before": float(delta_norm_sq ** 0.5 / (rho + eps)),
            "euc_ratio_after": float(after_norm_sq ** 0.5 / (rho + eps)),
            "euc_v_norm": float(v_norm_sq ** 0.5),
            "euc_step_norm": float(step_norm_sq ** 0.5),
            "euc_delta_dot_v": float(delta_dot_v),
            "euc_delta_dot_step": float(delta_dot_step),
            "euc_feasible": float(feasible),
            "euc_matched": float(matched),
            "euc_tracked": float(tracked),
            "euc_total": float(total),
            "euc_total_all": float(total_all),
            "euc_skipped_classifier": float(skipped_classifier),
        }
        return euc_scale, stats

    # ------------------------------------------------------------------
    # NCL KFAC gradient cache
    # ------------------------------------------------------------------
    def _compute_ncl_grad_cache(
        self,
        model: nn.Module,
    ) -> Tuple[
        Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        List[_TrInfoKFAC],
    ]:
        """Build per-parameter KFAC NCL cache.

        For each LoRA branch, computes the pre-inverted KFAC factors and the
        NCL reg gradient (δ-space deviation mapped to A/B space).

        A parameter cache entry: (inv_BTGB, inv_C, g_reg_A)
            v_A = inv_BTGB @ g_gam_A @ inv_C  +  ncl_lambda · g_reg_A

        B parameter cache entry: (inv_G, inv_ATCA, g_reg_B)
            v_B = inv_G @ g_gam_B @ inv_ATCA  +  ncl_lambda · g_reg_B

        KFAC chain-rule (derived in module docstring):
            F_A ≈ (B^T G B) ⊗ C  →  F_A^{-1} g_A = (B^T G B + γI)^{-1} g_A (C + γI)^{-1}
            F_B ≈ G ⊗ (A^T C A)  →  F_B^{-1} g_B = (G + γI)^{-1} g_B (A^T C A + γI)^{-1}

        Returns:
            cache    : {param_id: (inv_left, inv_right, g_reg)}
            tr_info  : trust-region norm metadata.  legacy keeps the old
                       detached-tensor tuple; explicit stores original parameter
                       ids so projector directions can be matched reliably.
        """
        if not self._kfac_past or not self._delta_reference:
            return {}, []
        # Rebuild static inverses if missing (e.g. first call after after_task)
        if not self._kfac_static_inv:
            self._refresh_kfac_static_inverses()
        if not self._kfac_static_inv:
            return {}, []

        damp = float(self._kfac_damping)
        ncl_lam = float(self._ncl_lambda)
        power = float(self._ncl_inv_power)
        clip = self._ncl_inv_clip
        cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        tr_info: List[_TrInfoKFAC] = []
        need_tr = self._current_ncl_radius() is not None

        for mod_name, module in self._iter_delta_modules(model):
            for a_attr, b_attr, branch in _BRANCHES:
                a_mod = getattr(module, a_attr, None)
                b_mod = getattr(module, b_attr, None)
                if a_mod is None or b_mod is None:
                    continue
                A: torch.Tensor = a_mod.weight   # (rank, in_dim)
                B: torch.Tensor = b_mod.weight   # (out_dim, rank)
                if not A.requires_grad and not B.requires_grad:
                    continue

                key = f"{mod_name}.{branch}"
                kfac_entry = self._kfac_past.get(key)
                delta_ref = self._delta_reference.get(key)
                static_inv = self._kfac_static_inv.get(key)
                if kfac_entry is None or delta_ref is None or static_inv is None:
                    continue

                G_cpu, C_cpu = kfac_entry
                inv_G_cpu, inv_C_cpu = static_inv
                dev = A.device

                # Always use float32; symmetrize G and C (accumulated Fisher may have
                # minor asymmetry from fp accumulation).
                G = 0.5 * (G_cpu + G_cpu.T).float().to(device=dev)    # (out_dim, out_dim)
                C = 0.5 * (C_cpu + C_cpu.T).float().to(device=dev)    # (in_dim, in_dim)
                G, C = self._normalize_kfac_pair(G, C)
                d_ref = delta_ref.float().to(device=dev)               # (out_dim, in_dim)

                with torch.no_grad():
                    A_d = A.detach().float()                  # (rank, in_dim)
                    B_d = B.detach().float()                  # (out_dim, rank)
                    delta_now = B_d @ A_d                     # (out_dim, in_dim)
                    d_delta = delta_now - d_ref               # (out_dim, in_dim)

                # ── A parameter ─────────────────────────────────────────────
                if A.requires_grad:
                    with torch.no_grad():
                        BTGB = B_d.T @ G @ B_d                  # (rank, rank)
                        BTGB = 0.5 * (BTGB + BTGB.T)           # symmetrize rank×rank
                        inv_BTGB = self._matrix_power_inv(BTGB, damp, power, clip)
                        g_reg_A = ncl_lam * (B_d.T @ d_delta)
                    cache[id(A)] = (inv_BTGB.cpu(), inv_C_cpu, g_reg_A.cpu())

                # ── B parameter ─────────────────────────────────────────────
                if B.requires_grad:
                    with torch.no_grad():
                        ATCA = A_d @ C @ A_d.T                   # (rank, rank)
                        ATCA = 0.5 * (ATCA + ATCA.T)            # symmetrize rank×rank
                        inv_ATCA = self._matrix_power_inv(ATCA, damp, power, clip)
                        g_reg_B = ncl_lam * (d_delta @ A_d.T)
                    cache[id(B)] = (inv_G_cpu, inv_ATCA.cpu(), g_reg_B.cpu())

                if need_tr and A.requires_grad and B.requires_grad:
                    if self._ncl_trust_impl == "explicit":
                        tr_info.append((
                            G.cpu(), C.cpu(),
                            id(A), id(B),
                            A_d.cpu(), B_d.cpu(),
                            d_delta.cpu(),
                        ))
                    else:
                        tr_info.append((
                            G.cpu(), C.cpu(),
                            A_d.cpu(), B_d.cpu(),
                            d_delta.cpu(),
                        ))

        return cache, tr_info

    # ------------------------------------------------------------------
    # NCL KFAC projector
    # ------------------------------------------------------------------
    def _build_ncl_projector(
        self,
        ncl_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        tr_info: List[_TrInfoKFAC],
        log_cache: Dict[str, float],
    ):
        """Return a GAM direction projector that applies the KFAC NCL update.

        For each parameter p with grad g_gam and cache entry (inv_L, inv_R, g_reg):
            v = inv_L @ g_gam @ inv_R  +  g_reg

        Trust-region uses the KFAC-weighted Frobenius norm:
            v^T Λ v  =  tr(V_delta^T G V_delta C)
                     =  (G @ V_delta @ C * V_delta).sum()    [Hadamard identity]
        where V_delta ≈ B @ v_A + v_B @ A  (chain-rule approx in ΔW space).
        """
        r_trust = self._current_ncl_radius()
        trust_mode = self._ncl_trust_mode
        trust_impl = self._ncl_trust_impl
        trust_xi = float(self._ncl_trust_xi)
        outside_action = self._ncl_trust_outside_action

        inv_clip = self._ncl_inv_clip
        grad_clip = self._ncl_grad_clip

        def projector(param_groups):
            gam_norm_sq = 0.0
            fisher_grad_norm_sq = 0.0
            reg_norm_sq = 0.0
            final_norm_sq = 0.0
            f_inv_max_val = 0.0

            # ── Pass 1: compute KFAC NCL directions ─────────────────────────
            param_to_v: Dict[int, torch.Tensor] = {}
            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad or param.grad is None:
                        continue
                    g_gam = param.grad.detach()
                    entry = ncl_cache.get(id(param))
                    if entry is not None:
                        inv_L, inv_R, g_reg = entry
                        dev, dt = g_gam.device, g_gam.dtype
                        inv_L = inv_L.to(device=dev, dtype=dt)
                        inv_R = inv_R.to(device=dev, dtype=dt)
                        g_reg = g_reg.to(device=dev, dtype=dt)
                        # KFAC-preconditioned update: F^{-p} g = inv_L @ g @ inv_R
                        v_fisher = inv_L @ g_gam @ inv_R
                        # Optional element-wise clip on the preconditioned result
                        if inv_clip is not None:
                            v_fisher = v_fisher.clamp(-inv_clip, inv_clip)
                        v = v_fisher + g_reg
                        if self._mechanism_eval:
                            fisher_grad_norm_sq += float(v_fisher.pow(2).sum().item())
                            reg_norm_sq += float(g_reg.pow(2).sum().item())
                            f_inv_max_val = max(
                                f_inv_max_val, float(v_fisher.abs().max().item())
                            )
                    else:
                        v = g_gam
                    param_to_v[id(param)] = v
                    if self._mechanism_eval:
                        gam_norm_sq += float(g_gam.pow(2).sum().item())
                        final_norm_sq += float(v.pow(2).sum().item())

            # ── Global gradient norm clipping ────────────────────────────
            # Prevents positive-feedback explosion: large F^{-1}g → large d_delta
            # → large anchor term → even larger combined step → weights diverge.
            if grad_clip is not None:
                total_norm_sq = sum(float(v.pow(2).sum().item()) for v in param_to_v.values())
                total_norm = total_norm_sq ** 0.5
                if total_norm > grad_clip:
                    clip_coef = grad_clip / (total_norm + 1e-12)
                    param_to_v = {k: v * clip_coef for k, v in param_to_v.items()}
                    if self._mechanism_eval:
                        log_cache["ncl_clip_coef"] = float(clip_coef)
                        log_cache["ncl_unclipped_norm"] = float(total_norm)

            # ── Trust-region step scaling ─────────────────────────────────
            tr_scale = 1.0
            euc_scale = 1.0
            euc_active = False
            if r_trust is not None and tr_info:
                base_lr = 1.0
                for group in param_groups:
                    lr_val = float(group.get("lr", 0.0))
                    if lr_val > 0.0:
                        base_lr = lr_val
                        break

                if trust_impl == "explicit":
                    # KFAC-weighted norm: tr(V_delta^T G V_delta C).
                    # explicit stores original param ids, fixing the detached-id mismatch.
                    vTLv = 0.0
                    deltaTLv = 0.0
                    deltaTLdelta = 0.0
                    matched = 0

                    for item in tr_info:
                        if len(item) != 7:
                            continue
                        G_cpu, C_cpu, a_id, b_id, A_cpu, B_cpu, d_delta_cpu = item
                        v_A = param_to_v.get(a_id)
                        v_B = param_to_v.get(b_id)
                        if v_A is None or v_B is None:
                            continue
                        matched += 1

                        dev = v_A.device
                        dt = v_A.dtype
                        G_dev = G_cpu.to(device=dev, dtype=dt)
                        C_dev = C_cpu.to(device=dev, dtype=dt)
                        A_dev = A_cpu.to(device=dev, dtype=dt)
                        B_dev = B_cpu.to(device=dev, dtype=dt)
                        d_delta_dev = d_delta_cpu.to(device=dev, dtype=dt)

                        v_delta = B_dev @ v_A.to(device=dev, dtype=dt) + \
                            v_B.to(device=dev, dtype=dt) @ A_dev

                        GVC = G_dev @ v_delta @ C_dev
                        vTLv += float((GVC * v_delta).sum().item())
                        deltaTLv += float((GVC * d_delta_dev).sum().item())

                        GDC = G_dev @ d_delta_dev @ C_dev
                        deltaTLdelta += float((GDC * d_delta_dev).sum().item())

                    vTLv = max(vTLv, 0.0)
                    deltaTLdelta = max(deltaTLdelta, 0.0)
                    tr_before = 0.5 * deltaTLdelta
                    tr_after = tr_before
                    tr_feasible = 1.0

                    if trust_mode == "cumulative":
                        a = 0.5 * (base_lr ** 2) * vTLv
                        b_coeff = -base_lr * deltaTLv
                        c = tr_before - r_trust ** 2
                        tr_feasible = 1.0 if c <= 0.0 else 0.0

                        if c <= 0.0:
                            if a > trust_xi:
                                disc = max(b_coeff ** 2 - 4.0 * a * c, 0.0)
                                s_max = (-b_coeff + disc ** 0.5) / (2.0 * a + trust_xi)
                                tr_scale = min(1.0, max(0.0, s_max))
                            else:
                                tr_scale = 1.0
                        elif outside_action == "zero":
                            tr_scale = 0.0
                        elif outside_action == "step":
                            v_norm = (vTLv + trust_xi) ** 0.5
                            eta_max = (2.0 ** 0.5 * r_trust) / v_norm
                            tr_scale = min(1.0, eta_max / (base_lr + trust_xi))
                        else:
                            tr_scale = 1.0

                        tr_after = 0.5 * (
                            deltaTLdelta
                            - 2.0 * base_lr * tr_scale * deltaTLv
                            + (base_lr * tr_scale) ** 2 * vTLv
                        )
                    else:
                        v_norm = (vTLv + trust_xi) ** 0.5
                        eta_max = (2.0 ** 0.5 * r_trust) / v_norm
                        tr_scale = min(1.0, eta_max / (base_lr + trust_xi))
                        tr_after = 0.5 * (base_lr * tr_scale) ** 2 * vTLv

                    log_cache.update({
                        "tr_scale": float(tr_scale),
                        "tr_vTLv": float(vTLv),
                        "tr_deltaTLv": float(deltaTLv),
                        "tr_deltaTLdelta": float(deltaTLdelta),
                        "tr_before": float(tr_before),
                        "tr_after": float(max(tr_after, 0.0)),
                        "tr_budget": float(r_trust ** 2),
                        "tr_r": float(r_trust),
                        "tr_base_lr": float(base_lr),
                        "tr_matched": float(matched),
                        "tr_total": float(len(tr_info)),
                        "tr_feasible": float(tr_feasible),
                    })
                else:
                    # legacy branch intentionally keeps the previous behavior.
                    vTLv = 0.0
                    bTLd = 0.0
                    DTLd = 0.0

                    for G_cpu, C_cpu, A_cpu, B_cpu, d_delta_cpu in tr_info:
                        v_A = param_to_v.get(id(A_cpu))
                        v_B = param_to_v.get(id(B_cpu))
                        if v_A is None or v_B is None:
                            continue
                        dev = v_A.device
                        G_dev = G_cpu.to(device=dev, dtype=v_A.dtype)
                        C_dev = C_cpu.to(device=dev, dtype=v_A.dtype)
                        A_dev = A_cpu.to(device=dev, dtype=v_A.dtype)
                        B_dev = B_cpu.to(device=dev, dtype=v_A.dtype)
                        d_delta_dev = d_delta_cpu.to(device=dev, dtype=v_A.dtype)

                        v_delta = B_dev @ v_A.to(dev) + v_B.to(dev) @ A_dev
                        GVC = G_dev @ v_delta @ C_dev
                        vTLv += float((GVC * v_delta).sum().item())

                        if trust_mode == "cumulative":
                            bTLd += float((GVC * d_delta_dev).sum().item())
                            GDC = G_dev @ d_delta_dev @ C_dev
                            DTLd += float((GDC * d_delta_dev).sum().item())

                    if trust_mode == "cumulative" and tr_info:
                        a = 0.5 * vTLv
                        b_coeff = bTLd * base_lr
                        c = 0.5 * DTLd - r_trust ** 2
                        if c <= 0.0 and a > trust_xi:
                            disc = max(b_coeff ** 2 - 4.0 * a * c, 0.0)
                            eta_max = (-b_coeff + disc ** 0.5) / (2.0 * a)
                            tr_scale = min(1.0, max(0.0, eta_max / (base_lr + trust_xi)))
                    else:
                        v_norm = (vTLv + trust_xi) ** 0.5
                        eta_max = (2.0 ** 0.5 * r_trust) / v_norm
                        tr_scale = min(1.0, eta_max / (base_lr + trust_xi))

                    log_cache["tr_scale"] = float(tr_scale)
                    log_cache["tr_vTLv"] = float(vTLv)

            # ── Euclidean cumulative hard scaling ────────────────────────
            # Constraint: ||theta_s - lr*a*v - theta_{t-1}||_2 <= rho_GAM
            # Applied after KFAC trust-region; final scale = min(tr_scale, euc_scale).
            if self._ncl_euclidean_trust:
                rho_euc = self._resolve_euclidean_radius()
                has_prior_task = bool(self._kfac_past)
                euc_allowed = has_prior_task or self._ncl_euclidean_apply_first_task
                if rho_euc is not None and self._param_reference and euc_allowed:
                    euc_scale, euc_stats = self._compute_euclidean_cumulative_scale(
                        param_groups=param_groups,
                        param_to_v=param_to_v,
                        rho=float(rho_euc),
                    )
                    euc_active = True
                    log_cache.update(euc_stats)
                    log_cache["combined_scale"] = float(
                        min(float(tr_scale), float(euc_scale))
                    )
                    log_cache["euc_active"] = 1.0
                    log_cache["euc_exclude_classifier"] = float(
                        self._ncl_euclidean_exclude_classifier
                    )
                else:
                    log_cache["euc_active"] = 0.0
                    log_cache["euc_has_prior_task"] = float(has_prior_task)
                    log_cache["euc_apply_first_task"] = float(
                        self._ncl_euclidean_apply_first_task
                    )

            # ── Pass 2: write (scaled) NCL directions to param.grad ──────
            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad or param.grad is None:
                        continue
                    v = param_to_v.get(id(param))
                    if v is None:
                        continue
                    scale = float(tr_scale)
                    if euc_active and self._is_euclidean_tracked_param(param):
                        scale = min(scale, float(euc_scale))
                    if scale != 1.0:
                        v = v * scale
                    param.grad.data.copy_(v)

            if self._mechanism_eval:
                _eps = 1e-12
                _ratio = (fisher_grad_norm_sq ** 0.5) / (reg_norm_sq ** 0.5 + _eps)
                log_cache.update({
                    "gam_grad_norm": gam_norm_sq ** 0.5,
                    "ncl_fisher_grad_norm": fisher_grad_norm_sq ** 0.5,
                    "ncl_reg_norm": reg_norm_sq ** 0.5,
                    "ncl_ratio": _ratio,        # healthy range 0.1–10; >100 means Fisher inverse exploding
                    "ncl_f_inv_max": f_inv_max_val,
                    "final_grad_norm": final_norm_sq ** 0.5,
                })

        return projector

    # ------------------------------------------------------------------
    # Main per-batch NCL KFAC step
    # ------------------------------------------------------------------
    def _gam_ncl_step(
        self,
        model: nn.Module,
        optimizer: GAM,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        class_offset: int,
    ) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        """One batch: GAM on CE loss, then apply KFAC NCL preconditioner."""
        log_cache: Dict[str, float] = {"ce_clean": 0.0, "gam": 0.0, "ncl_reg_norm": 0.0}

        ncl_cache, tr_info = self._compute_ncl_grad_cache(model)

        optimizer.set_direction_projector(
            self._build_ncl_projector(ncl_cache, tr_info, log_cache)
        )

        _call_count = [0]

        def closure():
            _call_count[0] += 1
            optimizer.zero_grad()
            outputs = self._network(inputs)
            ce_loss, _, _ = self._task_loss(outputs["logits"], targets, class_offset)
            if _call_count[0] == 1:
                log_cache["ce_clean"] = float(ce_loss.detach().item())
            else:
                log_cache["gam"] = float(ce_loss.detach().item())
            ce_loss.backward()
            return outputs, ce_loss.detach()

        try:
            outputs, loss_value = optimizer.step(closure=closure)
        finally:
            optimizer.set_direction_projector(None)
            optimizer.zero_grad()

        ce_value = float(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
        log_cache["ce_clean"] = ce_value
        log_cache["ncl_lambda"] = float(self._ncl_lambda)
        log_cache["has_kfac"] = float(bool(self._kfac_past))
        log_cache["ncl_params"] = float(len(ncl_cache))
        active_r = self._current_ncl_radius()
        if active_r is not None:
            log_cache["ncl_active_r_trust"] = float(active_r)

        return outputs["logits"].detach(), ce_value, log_cache

    # ------------------------------------------------------------------
    # Training loop
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
        # Euclidean constraint only applies from Task 1 onwards (protecting previous
        # tasks).  Task 0 has no prior reference — clearing keeps _param_reference
        # empty so the projector's guard `if self._param_reference` skips it.
        if self._cur_task > 0:
            self._snapshot_param_reference(optimizer.param_groups)
        else:
            self._param_reference = {}
        self._in_task_fisher_snapshots = []
        self._in_task_fisher_last_metrics = {}
        self._active_ncl_r_trust = self._resolve_active_ncl_radius()

        if self._is_main_process:
            has_k = bool(self._kfac_past)
            k_info = ""
            if has_k:
                keys = list(self._kfac_past.keys())
                G0, C0 = self._kfac_past[keys[0]]
                k_info = (
                    f"KFAC_branches={len(keys)} "
                    f"G_shape={tuple(G0.shape)} C_shape={tuple(C0.shape)} "
                    f"tr_G={float(G0.trace().item()):.3e} tr_C={float(C0.trace().item()):.3e}"
                )
            self._log(
                f"[GAM-EWCLoRA-NCL-KFAC][TaskStart] task={self._cur_task} "
                f"ncl_lambda={self._ncl_lambda:.3g} kfac_damping={self._kfac_damping:.3g} "
                f"trust_impl={self._ncl_trust_impl} trust_mode={self._ncl_trust_mode} "
                f"r_active={self._active_ncl_r_trust} r_from_gam={self._ncl_r_from_gam} "
                f"r_source={self._ncl_r_source} r_scale={self._ncl_r_scale:.3g} "
                f"has_kfac={has_k} {k_info}"
            )

        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        info = ""

        for epoch in prog_bar:
            self._network.train()
            total_loss = 0.0
            ce_losses = 0.0
            gam_losses = 0.0
            ncl_reg_norms = 0.0
            loss_count = 0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                logits, loss_value, log_dict = self._gam_ncl_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=inputs,
                    targets=targets,
                    class_offset=class_offset,
                )

                total_loss += float(loss_value)
                ce_losses += float(log_dict.get("ce_clean", 0.0))
                gam_losses += float(log_dict.get("gam", 0.0))
                ncl_reg_norms += float(log_dict.get("ncl_reg_norm", 0.0))
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
            train_acc = 0.0 if total == 0 else round(100.0 * correct / total, 2)
            info = (
                f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                f"Loss {total_loss / denom:.3f}, Train_accy {train_acc:.2f}, "
                f"CE {ce_losses / denom:.3f}, GAM {gam_losses / denom:.3f}, "
                f"NCL_reg_norm {ncl_reg_norms / denom:.3e}"
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
        logits, loss_value, log_dict = self._gam_ncl_step(
            model=self._unwrap_network(),
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            class_offset=class_offset,
        )
        if self._mechanism_eval:
            self._record_mechanism_stats(log_dict)
        return logits, loss_value

    # ------------------------------------------------------------------
    # In-task Fisher: disabled for KFAC (no diagonal path-mixing)
    # ------------------------------------------------------------------
    def _should_capture_in_task_fisher(self, epoch_idx: int) -> bool:
        return False

    def _fisher_trace(self, fisher: Optional[Dict[str, torch.Tensor]]) -> float:
        if not fisher:
            return 0.0
        return float(sum(
            float(v.detach().float().clamp_min(0).sum().item())
            for v in fisher.values()
        ))

    # ------------------------------------------------------------------
    # Mechanism stats (unchanged from NCL.py)
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
            try:
                self._mechanism_sums[key] = self._mechanism_sums.get(key, 0.0) + float(value)
            except (TypeError, ValueError):
                continue
        self._mechanism_count += 1

    def _summarize_mechanism_stats(self) -> Dict[str, float]:
        if self._mechanism_count <= 0:
            return {}
        denom = float(self._mechanism_count)
        return {f"train_{k}_mean": v / denom for k, v in sorted(self._mechanism_sums.items())}

    # ------------------------------------------------------------------
    # after_task: compute KFAC factors and accumulate
    # ------------------------------------------------------------------
    def after_task(self):
        base_model = self._unwrap_network()

        new_kfac = self._compute_kfac_delta_fisher(
            loader=self.train_loader,
            model=base_model,
            class_offset=int(self._known_classes),
        )
        self._kfac_past = self._accumulate_kfac(self._kfac_past, new_kfac)
        self._refresh_kfac_static_inverses()

        if self._is_main_process:
            keys = list(new_kfac.keys())
            if keys:
                G0, C0 = new_kfac[keys[0]]
                self._log(
                    f"[GAM-EWCLoRA-NCL-KFAC][KFACStats] task={self._cur_task} "
                    f"branches={len(keys)} "
                    f"new_tr_G={float(G0.trace().item()):.3e} "
                    f"new_tr_C={float(C0.trace().item()):.3e}"
                )
            accum_keys = list(self._kfac_past.keys())
            if accum_keys:
                G_acc, C_acc = self._kfac_past[accum_keys[0]]
                self._log(
                    f"[GAM-EWCLoRA-NCL-KFAC][KFACStats] task={self._cur_task} "
                    f"accum_branches={len(accum_keys)} "
                    f"accum_tr_G={float(G_acc.trace().item()):.3e} "
                    f"accum_tr_C={float(C_acc.trace().item()):.3e}"
                )

        self._delta_reference = {}
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._in_task_fisher_snapshots = []

        if self._mechanism_eval:
            self._last_mechanism_metrics = {
                "task": int(self._cur_task),
                "method_type": "gam_ewclora_ncl_kfac",
                "ncl_lambda": float(self._ncl_lambda),
                "kfac_damping": float(self._kfac_damping),
                "ncl_normalize_kfac": bool(self._ncl_normalize_kfac),
                "ncl_trust_impl": self._ncl_trust_impl,
                "ncl_trust_mode": self._ncl_trust_mode,
                "ncl_r_trust": self._ncl_r_trust,
                "ncl_active_r_trust": self._active_ncl_r_trust,
                "ncl_r_from_gam": bool(self._ncl_r_from_gam),
                "ncl_r_source": self._ncl_r_source,
                "ncl_r_scale": float(self._ncl_r_scale),
                "gam_base_optimizer": self._gam_base_optimizer,
                "gam_grad_rho": float(self._gam_grad_rho),
                "gam_grad_norm_rho": float(self._gam_grad_norm_rho),
                **self._in_task_fisher_last_metrics,
                **self._summarize_mechanism_stats(),
            }
            if self._is_main_process:
                self._log(f"[GAM-EWCLoRA-NCL-KFAC][Mechanism] {self._last_mechanism_metrics}")

    def get_mechanism_metrics(self) -> Dict[str, object]:
        return dict(self._last_mechanism_metrics)


GAMEWCLoRANCLKFAC = Learner
EWCLoRA = Learner
