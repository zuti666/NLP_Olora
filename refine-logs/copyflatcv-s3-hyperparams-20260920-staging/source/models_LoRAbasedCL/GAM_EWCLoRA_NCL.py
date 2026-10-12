"""SeqLoRA EWCLoRA with GAM + NCL-style update (diagonal Fisher, no EWC lambda loss).

This replaces the lambda-EWC penalty term with a Natural Continual Learning (NCL)
preconditioned update.

Standard EWC (GAM_EWCLoRA_lambda.py):
    g_final = g_gam + λ · F · (Δ - Δ_ref)_param
    θ ← θ - lr · g_final

NCL (this file):
    g_composite = g_gam + F · (Δ - Δ_ref)_param      # λ=1; reg term has natural NCL scale
    g_ncl       = F_param^{-1} · g_composite          # Fisher inverse preconditioner
                = F_param^{-1} · g_gam  +  (Δ - Δ_ref)_param
    θ ← θ - lr · g_ncl

The NCL insight: F^{-1} * (F * (Δ - Δ_ref)) = (Δ - Δ_ref), so the Bayesian
convergence term is preserved in pure form while the current-task gradient is
rotated into directions that do not destroy previously acquired knowledge.

Diagonal Fisher in A/B LoRA parameter space is approximated from the stored
delta-W Fisher via the Gauss-Newton / chain-rule mapping:

    F_A[j,k] = Σ_i  F_delta[i,k] · B[i,j]²          shape: (rank, in_dim)
    F_B[i,j] = Σ_k  F_delta[i,k] · A[j,k]²          shape: (out_dim, rank)

NCL regularisation gradient (no extra Fisher factor):

    g_reg_A = B^T @ (Δ_now - Δ_ref)                  shape: (rank, in_dim)
    g_reg_B = (Δ_now - Δ_ref) @ A^T                  shape: (out_dim, rank)

Final gradient applied through GAM's base optimiser:

    param.grad ← g_gam / (F_param + η)  +  ncl_lambda · g_reg_param

Fisher is still accumulated per task in ``after_task`` (identical to the lambda
version), so the prior precision Λ_k = Λ_{k-1} + F_k grows over tasks.

Key differences from GAM_EWCLoRA_lambda.py:
  - No _ewc_lambda in loss; no dual-ascent; no safe-lambda clipping.
  - EWC loss never enters the closure — only CE loss is backward'd.
  - Fisher inverse applied element-wise to GAM gradient inside the projector.
  - New hyperparameter ``ncl_lambda`` (default 1.0) scales (Δ - Δ_ref) term.
  - ``fisher_eta`` still used as floor for numerical stability in F^{-1}.
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


class Learner(_SeqEWCLoRA):
    """SeqLoRA + delta-W Fisher + GAM with NCL-style preconditioned update."""

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

        # NCL anchor-term scale. Default 1.0 — do NOT fall back to ewc_lambda,
        # which can be O(1000) and would overwhelm the anchor term.
        self._ncl_lambda = float(args.get("ncl_lambda", 1.0))
        # Fisher inverse damping floor.  eta is interpreted relative to the
        # (optionally normalized) Fisher scale.  With per-element delta-W
        # Fisher ≈ 1e-6 and eta=1e-4, all f_inv hit the clip (Fisher unused).
        # If ncl_normalize_fisher=True, Fisher is rescaled to mean=1 first,
        # so eta=0.1 means "10% of average importance → max amplification".
        self._fisher_eta = float(
            args.get("fisher_eta", args.get("ewc_eta", args.get("eta", 1e-4)))
        )
        if self._fisher_eta == 0.0:
            self._fisher_eta = 1e-4

        # Normalize accumulated Fisher before f_inv computation.
        # Problem: delta-W Fisher per-element ≈ 1e-6, eta=1e-4 (100x larger)
        # → ALL f_inv = 1/eta → clipped → Fisher geometry completely unused.
        # Fix: divide F_d by its mean so that mean(F_d_norm) = 1, then set
        # eta relative to this normalized scale (e.g. eta=0.1 means the floor
        # is 10% of the average element, giving f_inv range [1, clip]).
        self._ncl_normalize_fisher: bool = as_bool(
            args.get("ncl_normalize_fisher", False)
        )

        # Fisher inverse clipping: bound F_i^{-1} to [0, ncl_inv_clip].
        # Directly caps the max preconditioning gain regardless of eta.
        # With normalized Fisher (mean=1) and eta=0.1: clip=5 → max 5x CE
        # amplification for the least-important directions.
        _clip = args.get("ncl_inv_clip", None)
        self._ncl_inv_clip: Optional[float] = (
            None if (_clip is None or str(_clip).lower() in {"none", "null", ""})
            else float(_clip)
        )

        # Fractional Fisher inverse power p ∈ (0, 1].
        # p=1.0: original NCL (F+η)^{-1}·g  (can be aggressive)
        # p=0.5: gentler sqrt-inverse; trades off plasticity vs stability
        self._ncl_inv_power: float = float(args.get("ncl_inv_power", 1.0))

        # Trust-region radius r.  When set, the learning rate is scaled each
        # step so the update stays within the Fisher ball of radius r.
        #
        # ncl_trust_mode="step":
        #   η_s = min(η, sqrt(2)*r / sqrt(v^T Λ v + ξ))
        #   Guarantees: 0.5 * δ^T Λ δ ≤ r²  for this step.
        #
        # ncl_trust_mode="cumulative":
        #   Solves quadratic a η² + b η + c ≤ 0 to keep the parameter
        #   inside the flat-valid Fisher ball around the task reference:
        #   0.5 * (θ - θ_{t-1})^T Λ (θ - θ_{t-1}) ≤ r²
        #
        # Set ncl_r_trust=None (default) to skip explicit TR enforcement and
        # treat γ as a free learning-rate hyperparameter (original NCL).
        _r = args.get("ncl_r_trust", args.get("ncl_r", None))
        self._ncl_r_trust: Optional[float] = (
            None if (_r is None or str(_r).lower() in {"none", "null", ""})
            else float(_r)
        )
        self._ncl_trust_mode: str = str(args.get("ncl_trust_mode", "step")).lower()
        self._ncl_trust_xi: float = float(args.get("ncl_trust_xi", 1e-12))

        # Global gradient norm clipping applied to the composite NCL direction
        # (F^{-1}g_gam + ncl_lambda * g_reg) before writing to param.grad.
        # Prevents positive-feedback explosion: at later tasks the accumulated
        # Fisher amplifies both CE and anchor components, driving NCL_reg_norm
        # from ~0.4 to ~4e5 in two epochs.  With clip=5 and lr=0.01 the max
        # effective parameter step norm is 0.05 per batch, which is safe.
        _gc = args.get("ncl_grad_clip", None)
        self._ncl_grad_clip: Optional[float] = (
            None if (_gc is None or str(_gc).lower() in {"none", "null", ""})
            else float(_gc)
        )

        # In-task Fisher snapshots (identical logic to lambda version)
        self._approximate_fisher_in_task = as_bool(
            args.get(
                "approximte_fisher_inTask",
                args.get("approximate_fisher_inTask",
                         args.get("approximate_fisher_in_task", True)),
            )
        )
        self._in_task_fisher_interval = int(
            args.get(
                "in_task_fisher_interval",
                args.get("approximate_fisher_interval",
                         args.get("approximte_fisher_interval", 10)),
            )
        )
        self._in_task_fisher_omega = min(
            1.0,
            max(
                0.0,
                float(
                    args.get(
                        "in_task_fisher_omega",
                        args.get("path_fisher_omega",
                                 args.get("approximate_fisher_omega",
                                          args.get("approximte_fisher_omega", 0.5))),
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
        self._mechanism_sums: Dict[str, float] = {}
        self._mechanism_count = 0
        self._last_mechanism_metrics: Dict[str, object] = {}

        if self._is_main_process:
            logger.info(
                "[GAM-EWCLoRA-NCL] base=%s grad_rho=%.4g grad_norm_rho=%.4g "
                "adaptive=%s grad_reduce=%s ncl_lambda=%.4g fisher_eta=%.4g "
                "inv_clip=%s inv_power=%.2g "
                "r_trust=%s trust_mode=%s "
                "approx_fisher_inTask=%s interval=%d omega=%.3g",
                self._gam_base_optimizer,
                self._gam_grad_rho,
                self._gam_grad_norm_rho,
                self._gam_adaptive,
                self._gam_grad_reduce,
                self._ncl_lambda,
                self._fisher_eta,
                self._ncl_inv_clip,
                self._ncl_inv_power,
                self._ncl_r_trust,
                self._ncl_trust_mode,
                self._approximate_fisher_in_task,
                self._in_task_fisher_interval,
                self._in_task_fisher_omega,
            )

    # ------------------------------------------------------------------
    # GAM base optimizer resolution
    # ------------------------------------------------------------------
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
                "[GAM-EWCLoRA-NCL] Unsupported gam_base_optimizer=%s; using SGD.", base
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
            raise ValueError("No trainable parameters found for GAM-EWCLoRA-NCL.")
        return normalized

    # ------------------------------------------------------------------
    # NCL gradient cache: Fisher inverse and (Δ - Δ_ref) reg per param
    # ------------------------------------------------------------------
    def _compute_ncl_grad_cache(
        self,
        model: nn.Module,
    ) -> Tuple[
        Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        """Build per-parameter (fisher_inv, ncl_reg_grad) cache for NCL update.

        For each LoRA A/B weight, computes:

            Fisher in A-param space (chain rule):
                F_A[j,k] = Σ_i  F_delta[i,k] · B[i,j]²
            Fisher in B-param space (chain rule):
                F_B[i,j] = Σ_k  F_delta[i,k] · A[j,k]²

            NCL regularisation gradient (δ-space deviation mapped to A/B):
                g_reg_A = B^T @ (Δ_now - Δ_ref)
                g_reg_B = (Δ_now - Δ_ref) @ A^T

        Returns:
            cache    : {param_id: (fisher_inv, ncl_lambda * g_reg)}
            tr_info  : list of (F_d, A, B, d_delta) for trust-region norm computation.
                       Each entry corresponds to one delta-W module × branch.
                       Only populated when self._ncl_r_trust is not None.
        """
        if not self._fisher_past_delta or not self._delta_reference:
            return {}, []

        eta = float(self._fisher_eta)
        ncl_lam = float(self._ncl_lambda)
        inv_power = float(self._ncl_inv_power)
        inv_clip = self._ncl_inv_clip
        cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        tr_info: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        need_tr = self._ncl_r_trust is not None

        for mod_name, module in self._iter_delta_modules(model):
            for a_attr, b_attr, branch in _BRANCHES:
                a_mod = getattr(module, a_attr, None)
                b_mod = getattr(module, b_attr, None)
                if a_mod is None or b_mod is None:
                    continue
                A: torch.Tensor = a_mod.weight  # (rank, in_dim)
                B: torch.Tensor = b_mod.weight  # (out_dim, rank)
                if not A.requires_grad and not B.requires_grad:
                    continue

                key = f"{mod_name}.{branch}"
                F_delta = self._fisher_past_delta.get(key)
                delta_ref = self._delta_reference.get(key)
                if F_delta is None or delta_ref is None:
                    continue

                dev = A.device
                dt = A.dtype   # model dtype (may be bfloat16); cast back only at the end
                # Always use float32 for Fisher computations.
                # bfloat16 (7-bit mantissa) causes 0×Inf=NaN when large Fisher path-average
                # values overflow to Inf on conversion, then multiply with zero B_sq entries.
                F_d = F_delta.float().to(device=dev).clamp_min(0.0)   # (out_dim, in_dim) f32
                d_ref = delta_ref.float().to(device=dev)               # (out_dim, in_dim) f32

                # Fisher scale fix: per-element delta-W Fisher ≈ 1e-6, far below
                # any reasonable eta → all f_inv hit the clip → Fisher unused.
                # Normalize so mean(F_d) = 1 to make eta & clip meaningful.
                if self._ncl_normalize_fisher:
                    F_d_mean = F_d.mean().clamp_min(1e-12)
                    F_d = F_d / F_d_mean

                with torch.no_grad():
                    delta_now = B.detach().float() @ A.detach().float()   # f32
                    d_delta = delta_now - d_ref                            # f32

                # ── A weight ────────────────────────────────────────────
                if A.requires_grad:
                    with torch.no_grad():
                        B_sq = B.detach().float().pow(2)       # (out_dim, rank) f32
                        F_A = B_sq.T @ F_d                     # (rank, in_dim) f32
                        F_A_floor = (F_A + eta).clamp_min(eta)
                        f_inv_A = (F_A_floor.pow(-inv_power) if inv_power != 1.0
                                   else 1.0 / F_A_floor)
                        if inv_clip is not None:
                            f_inv_A = f_inv_A.clamp_max(inv_clip)
                        g_reg_A = ncl_lam * (B.detach().float().T @ d_delta)
                    cache[id(A)] = (f_inv_A, g_reg_A)

                # ── B weight ────────────────────────────────────────────
                if B.requires_grad:
                    with torch.no_grad():
                        A_sq = A.detach().float().pow(2)       # (rank, in_dim) f32
                        F_B = F_d @ A_sq.T                     # (out_dim, rank) f32
                        F_B_floor = (F_B + eta).clamp_min(eta)
                        f_inv_B = (F_B_floor.pow(-inv_power) if inv_power != 1.0
                                   else 1.0 / F_B_floor)
                        if inv_clip is not None:
                            f_inv_B = f_inv_B.clamp_max(inv_clip)
                        g_reg_B = ncl_lam * (d_delta @ A.detach().float().T)
                    cache[id(B)] = (f_inv_B, g_reg_B)

                # Collect module info for trust-region norm: (F_d, A, B, d_delta)
                if need_tr and A.requires_grad and B.requires_grad:
                    tr_info.append((
                        F_d,
                        A.detach(),
                        B.detach(),
                        d_delta,
                    ))

        return cache, tr_info

    # ------------------------------------------------------------------
    # NCL projector
    # ------------------------------------------------------------------
    def _build_ncl_projector(
        self,
        ncl_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        tr_info: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        log_cache: Dict[str, float],
    ):
        """Return a GAM direction projector that applies the NCL update.

        For each parameter p with grad g_gam:
            v = g_gam / (F_param + η)  +  ncl_lambda · (Δ - Δ_ref)_param

        If ncl_r_trust is set, the step is scaled so the Fisher-weighted update
        norm stays within the trust-region radius r:

          Step-level (mode="step"):
              scale = min(1,  sqrt(2)*r / (lr * sqrt(v^T Λ v + ξ)))

          Cumulative (mode="cumulative"):
              solve  a·η² + b·η + c ≤ 0  for max admissible η, then scale.

        In both cases v^T Λ v is approximated in delta-W space:
            v_delta ≈ B @ v_A + v_B @ A
            v^T Λ v ≈ Σ F_delta · v_delta²
        """
        r_trust = self._ncl_r_trust
        trust_mode = self._ncl_trust_mode
        trust_xi = float(self._ncl_trust_xi)

        grad_clip = self._ncl_grad_clip

        def projector(param_groups):
            gam_norm_sq = 0.0
            fisher_grad_norm_sq = 0.0   # |F^{-p} g_gam|²  — preconditioned task grad
            reg_norm_sq = 0.0           # |λ · g_reg|²      — anchor term
            final_norm_sq = 0.0
            f_inv_max_val = 0.0         # max elementwise Fisher inverse value

            # ── Pass 1: compute NCL directions, store them ───────────────
            # We need all v_A / v_B before we can compute v^T Λ v.
            param_to_v: Dict[int, torch.Tensor] = {}
            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad or param.grad is None:
                        continue
                    g_gam = param.grad.detach()
                    entry = ncl_cache.get(id(param))
                    if entry is not None:
                        f_inv, g_reg = entry
                        f_inv = f_inv.to(device=g_gam.device, dtype=g_gam.dtype)
                        g_reg = g_reg.to(device=g_gam.device, dtype=g_gam.dtype)
                        v_fisher = f_inv * g_gam   # preconditioned plasticity term
                        v = v_fisher + g_reg
                        if self._mechanism_eval:
                            fisher_grad_norm_sq += float(v_fisher.pow(2).sum().item())
                            reg_norm_sq += float(g_reg.pow(2).sum().item())
                            f_inv_max_val = max(f_inv_max_val, float(f_inv.max().item()))
                    else:
                        v = g_gam
                    param_to_v[id(param)] = v
                    if self._mechanism_eval:
                        gam_norm_sq += float(g_gam.pow(2).sum().item())
                        final_norm_sq += float(v.pow(2).sum().item())

            # ── Global gradient norm clipping ────────────────────────────
            if grad_clip is not None:
                total_norm_sq = sum(
                    float(v.pow(2).sum().item()) for v in param_to_v.values()
                )
                total_norm = total_norm_sq ** 0.5
                if total_norm > grad_clip:
                    clip_coef = grad_clip / (total_norm + 1e-12)
                    param_to_v = {k: v * clip_coef for k, v in param_to_v.items()}
                    if self._mechanism_eval:
                        log_cache["ncl_clip_coef"] = float(clip_coef)
                        log_cache["ncl_unclipped_norm"] = float(total_norm)

            # ── Trust-region step scaling ─────────────────────────────────
            tr_scale = 1.0
            if r_trust is not None and tr_info:
                # Get base lr from first param group with a positive lr
                base_lr = 1.0
                for group in param_groups:
                    lr_val = float(group.get("lr", 0.0))
                    if lr_val > 0.0:
                        base_lr = lr_val
                        break

                # v^T Λ v in delta-W space: Σ F_d · (B @ v_A + v_B @ A)²
                vTLv = 0.0
                bTLd = 0.0   # Δ^T Λ v  (for cumulative mode)
                DTLd = 0.0   # Δ^T Λ Δ  (for cumulative mode)
                for F_d, A, B, d_delta in tr_info:
                    # Find v_A and v_B tensors from param_to_v
                    # A/B are detached refs — need to find them by matching id
                    # We use a best-effort scan: match by shape and stored A/B
                    v_A = param_to_v.get(id(A))
                    v_B = param_to_v.get(id(B))
                    if v_A is None or v_B is None:
                        continue
                    dev = F_d.device
                    F_d_dev = F_d.to(device=dev)
                    # Approximate v in delta-W space
                    v_delta = B.to(dev) @ v_A.to(dev) + v_B.to(dev) @ A.to(dev)
                    vTLv += float((F_d_dev * v_delta.pow(2)).sum().item())
                    if trust_mode == "cumulative":
                        d_delta_dev = d_delta.to(dev)
                        bTLd += float((F_d_dev * d_delta_dev * v_delta).sum().item())
                        DTLd += float((F_d_dev * d_delta_dev.pow(2)).sum().item())

                if trust_mode == "cumulative" and tr_info:
                    # Solve a η² + b η + c ≤ 0
                    a = 0.5 * vTLv
                    b_coeff = bTLd * base_lr
                    c = 0.5 * DTLd - r_trust ** 2
                    if c <= 0.0 and a > trust_xi:
                        # Current point is feasible; find max admissible step
                        disc = b_coeff ** 2 - 4.0 * a * c
                        disc = max(disc, 0.0)
                        eta_max = (-b_coeff + disc ** 0.5) / (2.0 * a)
                        tr_scale = min(1.0, max(0.0, eta_max / (base_lr + trust_xi)))
                    # else: already infeasible or flat Fisher — don't clip
                else:
                    # Step-level: η_s = min(η, sqrt(2)*r / sqrt(v^T Λ v))
                    v_norm = (vTLv + trust_xi) ** 0.5
                    eta_max = (2.0 ** 0.5 * r_trust) / v_norm
                    tr_scale = min(1.0, eta_max / (base_lr + trust_xi))

                log_cache["tr_scale"] = float(tr_scale)
                log_cache["tr_vTLv"] = float(vTLv)

            # ── Pass 2: write (scaled) NCL directions to param.grad ──────
            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad or param.grad is None:
                        continue
                    v = param_to_v.get(id(param))
                    if v is None:
                        continue
                    if tr_scale != 1.0:
                        v = v * tr_scale
                    param.grad.data.copy_(v)

            if self._mechanism_eval:
                _eps = 1e-12
                _ratio = (fisher_grad_norm_sq ** 0.5) / (reg_norm_sq ** 0.5 + _eps)
                log_cache.update({
                    "gam_grad_norm": gam_norm_sq ** 0.5,
                    "ncl_fisher_grad_norm": fisher_grad_norm_sq ** 0.5,
                    "ncl_reg_norm": reg_norm_sq ** 0.5,
                    "ncl_ratio": _ratio,        # |F^{-p}g_gam| / |λ·g_reg|; target 0.1–10
                    "ncl_f_inv_max": f_inv_max_val,
                    "final_grad_norm": final_norm_sq ** 0.5,
                })

        return projector

    # ------------------------------------------------------------------
    # Main per-batch NCL step
    # ------------------------------------------------------------------
    def _gam_ncl_step(
        self,
        model: nn.Module,
        optimizer: GAM,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        class_offset: int,
    ) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        """One batch: GAM on CE loss, then apply NCL Fisher preconditioner.

        No EWC loss is computed — the NCL reg term enters only through the
        gradient projector applied after GAM's perturbation step.
        """
        log_cache: Dict[str, float] = {"ce_clean": 0.0, "gam": 0.0, "ncl_reg_norm": 0.0}

        # Precompute NCL caches (no backward needed)
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
        log_cache["has_fisher"] = float(bool(self._fisher_past_delta))
        log_cache["ncl_params"] = float(len(ncl_cache))

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
        self._in_task_fisher_snapshots = []
        self._in_task_fisher_last_metrics = {}

        if self._is_main_process:
            has_f = bool(self._fisher_past_delta)
            f_info = ""
            if has_f:
                f_vals = list(self._fisher_past_delta.values())
                f_max = float(max(v.max().item() for v in f_vals))
                f_sum = float(sum(v.sum().item() for v in f_vals))
                f_numel = sum(int(v.numel()) for v in f_vals)
                f_info = (
                    f"F_tensors={len(f_vals)} numel={f_numel} "
                    f"F_max={f_max:.3e} F_sum={f_sum:.3e}"
                )
            self._log(
                f"[GAM-EWCLoRA-NCL][TaskStart] task={self._cur_task} "
                f"ncl_lambda={self._ncl_lambda:.3g} fisher_eta={self._fisher_eta:.3g} "
                f"has_fisher={has_f} {f_info}"
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
    # In-task Fisher snapshots (unchanged lifecycle from lambda version)
    # ------------------------------------------------------------------
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
            "linear_a", "linear_b", "lora_a", "lora_b",
            "w_as", "w_bs", "scaling_factor", "wrapped_param",
        )
        snapshot: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if any(m in name.lower() for m in markers):
                snapshot[name] = param.detach().cpu().clone()
        return snapshot

    def _capture_in_task_fisher_snapshot(
        self,
        train_loader,
        model: nn.Module,
        optimizer: GAM,
        epoch_idx: int,
        class_offset: int,
    ) -> int:
        was_training = bool(model.training)
        logger.info(
            "[GAM-EWCLoRA-NCL][InTaskFisher] Computing delta-W snapshot "
            "task=%d epoch=%d", self._cur_task, epoch_idx,
        )
        try:
            fisher = super()._compute_delta_fisher(train_loader, model, class_offset)
        finally:
            optimizer.zero_grad()
            model.train(was_training)

        fisher_snapshot = {k: v.detach().cpu().clone() for k, v in fisher.items()}
        lora_snapshot = self._capture_lora_weight_snapshot(model)
        self._in_task_fisher_snapshots.append((int(epoch_idx), fisher_snapshot, lora_snapshot))
        logger.info(
            "[GAM-EWCLoRA-NCL][InTaskFisher] Saved snapshot task=%d epoch=%d "
            "count=%d fisher_trace=%.6g",
            self._cur_task, epoch_idx,
            len(self._in_task_fisher_snapshots),
            self._fisher_trace(fisher_snapshot),
        )
        return len(self._in_task_fisher_snapshots)

    def _fisher_trace(self, fisher: Optional[Dict[str, torch.Tensor]]) -> float:
        if not fisher:
            return 0.0
        return float(sum(
            float(v.detach().float().clamp_min(0).sum().item())
            for v in fisher.values()
        ))

    def _match_delta_tensor(self, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        ref = reference.detach().cpu()
        value_cpu = value.detach().cpu().to(dtype=ref.dtype)
        if tuple(value_cpu.shape) == tuple(ref.shape):
            return value_cpu
        matched = torch.zeros_like(ref)
        slices = tuple(
            slice(0, min(int(value_cpu.shape[d]), int(ref.shape[d])))
            for d in range(min(value_cpu.ndim, ref.ndim))
        )
        if len(slices) == value_cpu.ndim == ref.ndim:
            matched[slices] = value_cpu[slices]
        return matched

    def _mean_in_task_fisher(self, model: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        if not self._in_task_fisher_snapshots:
            return None
        mean_fisher: Dict[str, torch.Tensor] = {}
        for key, delta_now in self._iter_delta_terms(model):
            acc, count = None, 0
            for _, fsnap, _ in self._in_task_fisher_snapshots:
                val = fsnap.get(key)
                if val is None:
                    continue
                val = self._match_delta_tensor(val, delta_now)
                acc = val.clone() if acc is None else acc + val
                count += 1
            if acc is not None and count > 0:
                mean_fisher[key] = acc / float(count)
        return mean_fisher or None

    def _mix_in_task_path_fisher(
        self,
        task_fisher: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        snapshot_epochs = [ep for ep, _, _ in self._in_task_fisher_snapshots]
        path_fisher = self._mean_in_task_fisher(model)
        final_trace = self._fisher_trace(task_fisher)
        path_trace = self._fisher_trace(path_fisher)
        lora_tensors = sum(len(ls) for _, _, ls in self._in_task_fisher_snapshots)

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
                "in_task_fisher_snapshots": len(snapshot_epochs),
                "in_task_fisher_final_trace": float(final_trace),
                "in_task_fisher_path_trace": float(path_trace),
                "in_task_fisher_mixed_trace": float(final_trace),
                "in_task_lora_snapshot_tensors": int(lora_tensors),
            }
            return task_fisher

        omega = float(self._in_task_fisher_omega)
        assert path_fisher is not None  # guaranteed by use_path check above
        mixed: Dict[str, torch.Tensor] = {}
        for key, delta_now in self._iter_delta_terms(model):
            fv = task_fisher.get(key)
            pv = path_fisher.get(key)
            if fv is None and pv is None:
                continue
            if fv is None:
                mixed[key] = self._match_delta_tensor(pv, delta_now).clone()
            elif pv is None:
                mixed[key] = self._match_delta_tensor(fv, delta_now).clone()
            else:
                ft = self._match_delta_tensor(fv, delta_now)
                pt = self._match_delta_tensor(pv, delta_now)
                mixed[key] = (1.0 - omega) * ft + omega * pt

        mixed_trace = self._fisher_trace(mixed)
        self._in_task_fisher_last_metrics = {
            "approximte_fisher_inTask": bool(self._approximate_fisher_in_task),
            "in_task_fisher_used_path": True,
            "in_task_fisher_interval": int(self._in_task_fisher_interval),
            "in_task_fisher_omega": float(omega),
            "in_task_fisher_snapshots": len(snapshot_epochs),
            "in_task_fisher_final_trace": float(final_trace),
            "in_task_fisher_path_trace": float(path_trace),
            "in_task_fisher_mixed_trace": float(mixed_trace),
            "in_task_lora_snapshot_tensors": int(lora_tensors),
        }
        logger.info(
            "[GAM-EWCLoRA-NCL][InTaskFisher] Mixed final/path Fisher "
            "task=%d snapshots=%d omega=%.3g final_trace=%.6g mixed_trace=%.6g",
            self._cur_task, len(snapshot_epochs), omega, final_trace, mixed_trace,
        )
        return mixed

    # ------------------------------------------------------------------
    # Mechanism stats
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
    # after_task: accumulate Fisher (identical lifecycle to lambda version)
    # ------------------------------------------------------------------
    def after_task(self):
        base_model = self._unwrap_network()
        new_fisher = super()._compute_delta_fisher(
            loader=self.train_loader,
            model=base_model,
            class_offset=int(self._known_classes),
        )
        new_fisher = self._mix_in_task_path_fisher(new_fisher, base_model)
        self._fisher_past_delta = super()._accumulate_fisher(self._fisher_past_delta, new_fisher)

        if self._is_main_process:
            f_vals = list(new_fisher.values())
            if f_vals:
                nf_max = float(max(v.max().item() for v in f_vals))
                nf_sum = float(sum(v.sum().item() for v in f_vals))
                nf_numel = sum(int(v.numel()) for v in f_vals)
                self._log(
                    f"[GAM-EWCLoRA-NCL][FisherStats] task={self._cur_task} "
                    f"new_fisher tensors={len(f_vals)} numel={nf_numel} "
                    f"max={nf_max:.3e} sum={nf_sum:.3e}"
                )
            accum_vals = list(self._fisher_past_delta.values())
            if accum_vals:
                af_max = float(max(v.max().item() for v in accum_vals))
                af_sum = float(sum(v.sum().item() for v in accum_vals))
                af_numel = sum(int(v.numel()) for v in accum_vals)
                self._log(
                    f"[GAM-EWCLoRA-NCL][FisherStats] task={self._cur_task} "
                    f"accum_fisher tensors={len(accum_vals)} numel={af_numel} "
                    f"max={af_max:.3e} sum={af_sum:.3e}"
                )

        self._delta_reference = {}
        self._known_classes = self._total_classes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._in_task_fisher_snapshots = []

        if self._mechanism_eval:
            self._last_mechanism_metrics = {
                "task": int(self._cur_task),
                "method_type": "gam_ewclora_ncl",
                "ncl_lambda": float(self._ncl_lambda),
                "fisher_eta": float(self._fisher_eta),
                "gam_base_optimizer": self._gam_base_optimizer,
                "gam_grad_rho": float(self._gam_grad_rho),
                "gam_grad_norm_rho": float(self._gam_grad_norm_rho),
                **self._in_task_fisher_last_metrics,
                **self._summarize_mechanism_stats(),
            }
            if self._is_main_process:
                self._log(f"[GAM-EWCLoRA-NCL][Mechanism] {self._last_mechanism_metrics}")

    def get_mechanism_metrics(self) -> Dict[str, object]:
        return dict(self._last_mechanism_metrics)


GAMEWCLoRANCL = Learner
EWCLoRA = Learner
