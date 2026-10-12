from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KFACFisherSAM(torch.optim.Optimizer):
    """
    KFAC-based Fisher-SAM with a SAM-compatible API.

    Perturbation:
        eps = gamma * F^{-1} g / sqrt(g^T F^{-1} g)

    KFAC block approximation for trainable Linear/Conv2d weights:
        F_m ~= G_m kron A_m
        F_m^{-1} g_m ~= G_m^{-1} g_m A_m^{-1}

    With kfac_target="lora_delta_w", the KFAC object is the seqlora ViT
    effective LoRA increment Delta W=B@A for q/v branches instead of the
    individual A/B parameter coordinates.

    Non-covered parameters fall back to diagonal Fisher-SAM by default. This
    matters for ViT models because LayerNorm, class/position tokens, and many
    adapter parameters are not Linear/Conv2d matrix weights.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer,
        rho: Optional[float] = None,
        gamma: Optional[float] = None,
        eta: float = 1.0,
        damping: float = 1e-3,
        stat_decay: float = 0.95,
        TCov: int = 1,
        TInv: int = 10,
        batch_averaged: Optional[bool] = None,
        token_averaged: bool = False,
        kfac_grad_scale: Optional[str] = None,
        kfac_seq_agg: str = "token",
        kfac_target: str = "module",
        fallback: str = "diag",
        fallback_inverse_mode: str = "paper",
        adaptive: bool = False,
        eps: float = 1e-12,
        max_kfac_dim: Optional[int] = None,
        verbose: bool = False,
        **base_optimizer_kwargs,
    ):
        if gamma is None:
            gamma = 0.1 if rho is None else rho
        if rho is None:
            rho = gamma

        gamma = float(gamma)
        rho = float(rho)
        eta = float(eta)
        damping = float(damping)
        eps = float(eps)

        if gamma < 0.0:
            raise ValueError(f"Invalid gamma, should be non-negative: {gamma}")
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        if eta < 0.0:
            raise ValueError(f"Invalid eta, should be non-negative: {eta}")
        if damping <= 0.0:
            raise ValueError(f"Invalid damping, should be positive: {damping}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps, should be positive: {eps}")
        if TCov <= 0:
            raise ValueError(f"Invalid TCov, should be positive: {TCov}")
        if TInv <= 0:
            raise ValueError(f"Invalid TInv, should be positive: {TInv}")
        if fallback not in {"none", "diag", "sam"}:
            raise ValueError("fallback must be one of {'none', 'diag', 'sam'}.")
        if fallback_inverse_mode not in {"paper", "damping"}:
            raise ValueError("fallback_inverse_mode must be one of {'paper', 'damping'}.")

        if kfac_grad_scale is None:
            # Preserve the old batch_averaged=True meaning: CE(mean) hook grads
            # are multiplied by batch size before forming G.
            kfac_grad_scale = "mean_x_bs" if batch_averaged is not False else "mean"
        kfac_grad_scale = str(kfac_grad_scale).lower()
        if kfac_grad_scale not in {"mean", "mean_x_bs", "sum"}:
            raise ValueError(
                f"Unknown kfac_grad_scale={kfac_grad_scale!r}. "
                "Use 'mean', 'mean_x_bs', or 'sum'."
            )

        kfac_seq_agg = str(kfac_seq_agg).lower()
        if kfac_seq_agg not in {"token", "sample"}:
            raise ValueError(
                f"Unknown kfac_seq_agg={kfac_seq_agg!r}. Use 'token' or 'sample'."
            )
        kfac_target = str(kfac_target).lower()
        if kfac_target in {"lora", "lora_delta", "lora_delta_weight"}:
            kfac_target = "lora_delta_w"
        if kfac_target not in {"module", "lora_delta_w"}:
            raise ValueError("kfac_target must be 'module' or 'lora_delta_w'.")

        self.model = model
        self.gamma = gamma
        self.rho = rho
        self.eta = eta
        self.damping = damping
        self.stat_decay = float(stat_decay)
        self.TCov = int(TCov)
        self.TInv = int(TInv)
        self.kfac_grad_scale = kfac_grad_scale
        self.kfac_seq_agg = kfac_seq_agg
        self.kfac_target = kfac_target
        self.token_averaged = bool(token_averaged)
        self.fallback = fallback
        self.fallback_inverse_mode = fallback_inverse_mode
        self.adaptive = bool(adaptive)
        self.eps = eps
        self.max_kfac_dim = None if max_kfac_dim is None else int(max_kfac_dim)
        self.verbose = bool(verbose)

        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise ValueError("KFACFisherSAM received no trainable parameters.")

        defaults = dict(
            rho=rho,
            gamma=gamma,
            eta=eta,
            damping=damping,
            adaptive=bool(adaptive),
            **base_optimizer_kwargs,
        )
        super().__init__(params, defaults)

        if isinstance(base_optimizer, torch.optim.Optimizer):
            self.base_optimizer = base_optimizer
        else:
            self.base_optimizer = base_optimizer(self.param_groups, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self._set_param_group_defaults(
            rho=rho,
            gamma=gamma,
            eta=eta,
            damping=damping,
            adaptive=bool(adaptive),
        )

        self.modules: list[nn.Module] = []
        self.handles = []
        self._module_include_bias: Dict[nn.Module, bool] = {}
        self._module_shapes: Dict[nn.Module, Tuple[int, int]] = {}

        self.m_aa: Dict[nn.Module, torch.Tensor] = {}
        self.m_gg: Dict[nn.Module, torch.Tensor] = {}
        self.Q_a: Dict[nn.Module, torch.Tensor] = {}
        self.Q_g: Dict[nn.Module, torch.Tensor] = {}
        self.d_a: Dict[nn.Module, torch.Tensor] = {}
        self.d_g: Dict[nn.Module, torch.Tensor] = {}

        self.lora_modules: list[nn.Module] = []
        self.lora_specs: Dict[Tuple[nn.Module, str], Dict[str, Any]] = {}
        self._lora_module_keys: Dict[nn.Module, list[Tuple[nn.Module, str]]] = {}
        self.lora_m_aa: Dict[Tuple[nn.Module, str], torch.Tensor] = {}
        self.lora_m_gg: Dict[Tuple[nn.Module, str], torch.Tensor] = {}
        self.lora_Q_a: Dict[Tuple[nn.Module, str], torch.Tensor] = {}
        self.lora_Q_g: Dict[Tuple[nn.Module, str], torch.Tensor] = {}
        self.lora_d_a: Dict[Tuple[nn.Module, str], torch.Tensor] = {}
        self.lora_d_g: Dict[Tuple[nn.Module, str], torch.Tensor] = {}
        self._lora_current_a: Dict[nn.Module, torch.Tensor] = {}
        self._lora_current_grad_dw: Dict[Tuple[nn.Module, str], torch.Tensor] = {}

        self.acc_stats = False
        self.steps = 0
        self._current_batch_size = 1
        self.last_clean_loss = None
        self.last_fsam_loss = None

        self._prepare_model()

    def _set_param_group_defaults(self, **defaults: Any) -> None:
        for group in self.param_groups:
            for key, value in defaults.items():
                group.setdefault(key, value)

    def _first_param(self) -> torch.Tensor:
        for group in self.param_groups:
            for p in group["params"]:
                return p
        raise ValueError("KFACFisherSAM received an empty parameter list.")

    def _has_any_gradient(self) -> bool:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    return True
        return False

    @staticmethod
    def _extract_loss(value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, (list, tuple)):
            for item in reversed(value):
                loss = KFACFisherSAM._extract_loss(item)
                if torch.is_tensor(loss):
                    return loss
        return None

    def _backward_if_closure_only_returned_loss(self, closure_result) -> None:
        if self._has_any_gradient():
            return
        loss = self._extract_loss(closure_result)
        if torch.is_tensor(loss) and loss.requires_grad:
            loss.backward()

    def _prepare_model(self) -> None:
        if self.kfac_target == "lora_delta_w":
            self._prepare_lora_model()
            return

        count = 0
        for module in self.model.modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            if isinstance(module, nn.Conv2d) and int(module.groups) != 1:
                if self.verbose:
                    print(f"[KFAC-FSAM] skip grouped Conv2d: {module}")
                continue

            weight = getattr(module, "weight", None)
            if weight is None or not weight.requires_grad:
                continue

            if isinstance(module, nn.Linear):
                out_dim = int(module.weight.shape[0])
                in_dim = int(module.weight.shape[1])
            else:
                out_dim = int(module.weight.shape[0])
                in_dim = int(module.weight[0].numel())

            include_bias = bool(
                getattr(module, "bias", None) is not None and module.bias.requires_grad
            )
            a_dim = in_dim + int(include_bias)
            g_dim = out_dim

            if self.max_kfac_dim is not None and max(a_dim, g_dim) > self.max_kfac_dim:
                if self.verbose:
                    print(
                        "[KFAC-FSAM] skip large block "
                        f"A={a_dim}, G={g_dim}, module={module}"
                    )
                continue

            self.modules.append(module)
            self._module_include_bias[module] = include_bias
            self._module_shapes[module] = (out_dim, a_dim)

            h1 = module.register_forward_pre_hook(self._save_input)
            h2 = module.register_full_backward_hook(self._save_grad_output)
            self.handles.extend([h1, h2])

            if self.verbose:
                print(
                    f"[KFAC-FSAM] keep module {count}: {module} "
                    f"(A={a_dim}, G={g_dim}, include_bias={include_bias})"
                )
            count += 1

        if self.verbose:
            print(f"[KFAC-FSAM] total KFAC modules: {len(self.modules)}")

    @staticmethod
    def _is_lora_qkv_module(module: nn.Module) -> bool:
        return all(
            hasattr(module, attr)
            for attr in ("qkv", "linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v", "dim")
        )

    @staticmethod
    def _lora_scale_value(module: nn.Module, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scale_container = getattr(module, "scaling_factor", None)
        if scale_container is None:
            return torch.ones((), device=device, dtype=dtype)
        try:
            scale_module = scale_container[0]
        except Exception:
            scale_module = scale_container
        param = getattr(scale_module, "param", None)
        if param is not None:
            return param.detach().to(device=device, dtype=dtype).reshape(-1)[0]
        return torch.ones((), device=device, dtype=dtype)

    def _prepare_lora_model(self) -> None:
        count = 0
        for module in self.model.modules():
            if not self._is_lora_qkv_module(module):
                continue

            module_keys = []
            dim = int(getattr(module, "dim"))
            for branch in ("q", "v"):
                a_module = getattr(module, f"linear_a_{branch}", None)
                b_module = getattr(module, f"linear_b_{branch}", None)
                if a_module is None or b_module is None:
                    continue
                a_param = getattr(a_module, "weight", None)
                b_param = getattr(b_module, "weight", None)
                if a_param is None or b_param is None:
                    continue
                if not (a_param.requires_grad or b_param.requires_grad):
                    continue

                a_dim = int(a_param.shape[1])
                g_dim = int(b_param.shape[0])
                if self.max_kfac_dim is not None and max(a_dim, g_dim) > self.max_kfac_dim:
                    if self.verbose:
                        print(
                            "[KFAC-FSAM] skip large LoRA Delta-W block "
                            f"A={a_dim}, G={g_dim}, branch={branch}, module={module}"
                        )
                    continue

                key = (module, branch)
                self.lora_specs[key] = {
                    "module": module,
                    "branch": branch,
                    "a_param": a_param,
                    "b_param": b_param,
                    "a_dim": a_dim,
                    "g_dim": g_dim,
                    "qkv_dim": dim,
                }
                module_keys.append(key)

                if self.verbose:
                    print(
                        f"[KFAC-FSAM] keep LoRA Delta-W block {count}: "
                        f"branch={branch} (A={a_dim}, G={g_dim})"
                    )
                count += 1

            if module_keys:
                self.lora_modules.append(module)
                self._lora_module_keys[module] = module_keys
                h1 = module.register_forward_hook(self._save_lora_input)
                h2 = module.register_full_backward_hook(self._save_lora_grad_output)
                self.handles.extend([h1, h2])

        if not self.lora_specs:
            raise ValueError(
                "kfac_target='lora_delta_w' was requested, but no seqlora train-time "
                "qkv LoRA branches were found."
            )
        if self.verbose:
            print(f"[KFAC-FSAM] total LoRA Delta-W KFAC blocks: {len(self.lora_specs)}")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @staticmethod
    def _ema_update(
        old: Optional[torch.Tensor],
        new: torch.Tensor,
        decay: float,
    ) -> torch.Tensor:
        if old is None:
            return new.detach().clone()
        old.mul_(decay).add_(new.detach(), alpha=1.0 - decay)
        return old

    def _append_bias_column(self, a: torch.Tensor, module: nn.Module) -> torch.Tensor:
        if not self._module_include_bias.get(module, False):
            return a
        ones = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
        return torch.cat([a, ones], dim=1)

    @torch.no_grad()
    def _compute_linear_cov_a(self, a: torch.Tensor, module: nn.Linear) -> torch.Tensor:
        if a.dim() == 3 and self.kfac_seq_agg == "sample":
            a = a.sum(dim=1)
        elif a.dim() > 2:
            a = a.reshape(-1, a.shape[-1])
        a = self._append_bias_column(a, module)
        return a.t() @ (a / max(1, a.size(0)))

    @torch.no_grad()
    def _compute_linear_cov_g(self, g: torch.Tensor, module: nn.Linear) -> torch.Tensor:
        batch_size = int(g.shape[0]) if g.dim() >= 1 else int(self._current_batch_size)
        num_tokens = int(g.shape[1]) if g.dim() > 2 else 1

        if self.kfac_grad_scale == "mean_x_bs":
            g = g * float(batch_size)
        if self.token_averaged and g.dim() > 2:
            g = g * float(num_tokens)

        if g.dim() == 3 and self.kfac_seq_agg == "sample":
            g = g.sum(dim=1)
        elif g.dim() > 2:
            g = g.reshape(-1, g.shape[-1])

        return g.t() @ (g / max(1, g.size(0)))

    @torch.no_grad()
    def _compute_conv_cov_a(self, a: torch.Tensor, module: nn.Conv2d) -> torch.Tensor:
        patches = F.unfold(
            a,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        ).transpose(1, 2)
        patches = patches.reshape(-1, patches.shape[-1])
        patches = self._append_bias_column(patches, module)
        return patches.t() @ (patches / max(1, patches.size(0)))

    @torch.no_grad()
    def _compute_conv_cov_g(self, g: torch.Tensor, module: nn.Conv2d) -> torch.Tensor:
        batch_size = int(g.shape[0])
        if self.kfac_grad_scale == "mean_x_bs":
            g = g * float(batch_size)
        g = g.permute(0, 2, 3, 1).reshape(-1, g.shape[1])
        return g.t() @ (g / max(1, g.size(0)))

    @torch.no_grad()
    def _compute_cov_a(self, a: torch.Tensor, module: nn.Module) -> Optional[torch.Tensor]:
        if isinstance(module, nn.Linear):
            return self._compute_linear_cov_a(a, module)
        if isinstance(module, nn.Conv2d):
            return self._compute_conv_cov_a(a, module)
        return None

    @torch.no_grad()
    def _compute_cov_g(self, g: torch.Tensor, module: nn.Module) -> Optional[torch.Tensor]:
        if isinstance(module, nn.Linear):
            return self._compute_linear_cov_g(g, module)
        if isinstance(module, nn.Conv2d):
            return self._compute_conv_cov_g(g, module)
        return None

    @torch.no_grad()
    def _prepare_lora_cov_matrix(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().float()
        if x.dim() == 1:
            return x.reshape(1, -1)
        if x.dim() == 3 and self.kfac_seq_agg == "sample":
            return x.sum(dim=1)
        if x.dim() > 2:
            return x.reshape(-1, x.shape[-1])
        return x.reshape(-1, x.shape[-1])

    @torch.no_grad()
    def _prepare_lora_exact_matrix(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().float()
        if x.dim() == 1:
            return x.reshape(1, -1)
        if x.dim() > 2:
            return x.reshape(-1, x.shape[-1])
        return x.reshape(-1, x.shape[-1])

    @torch.no_grad()
    def _compute_lora_cov_a(self, a: torch.Tensor) -> torch.Tensor:
        a_mat = self._prepare_lora_cov_matrix(a)
        return a_mat.t() @ (a_mat / max(1, a_mat.size(0)))

    @torch.no_grad()
    def _compute_lora_cov_g(self, g: torch.Tensor) -> torch.Tensor:
        batch_size = int(g.shape[0]) if g.dim() >= 1 else int(self._current_batch_size)
        num_tokens = int(g.shape[1]) if g.dim() > 2 else 1

        if self.kfac_grad_scale == "mean_x_bs":
            g = g * float(batch_size)
        if self.token_averaged and g.dim() > 2:
            g = g * float(num_tokens)

        g_mat = self._prepare_lora_cov_matrix(g)
        return g_mat.t() @ (g_mat / max(1, g_mat.size(0)))

    def _save_lora_input(self, module: nn.Module, inputs, _output) -> None:
        if not self.acc_stats:
            return
        if len(inputs) == 0 or inputs[0] is None:
            return

        a = inputs[0].detach()
        if a.dim() >= 1:
            self._current_batch_size = int(a.shape[0])
        self._lora_current_a[module] = a

        if self.steps % self.TCov != 0:
            return

        aa = self._compute_lora_cov_a(a)
        for key in self._lora_module_keys.get(module, []):
            self.lora_m_aa[key] = self._ema_update(
                self.lora_m_aa.get(key, None),
                aa,
                self.stat_decay,
            )

    def _save_lora_grad_output(self, module: nn.Module, grad_input, grad_output) -> None:
        if not self.acc_stats:
            return
        if len(grad_output) == 0 or grad_output[0] is None:
            return

        g_out = grad_output[0].detach().float()
        dim = int(getattr(module, "dim"))
        scale = self._lora_scale_value(module, g_out.device, g_out.dtype)
        a_raw = self._lora_current_a.get(module, None)
        a_exact = self._prepare_lora_exact_matrix(a_raw) if a_raw is not None else None

        for key in self._lora_module_keys.get(module, []):
            branch = key[1]
            if branch == "q":
                g_branch = g_out[..., :dim] * scale
            else:
                g_branch = g_out[..., -dim:] * scale

            if a_exact is not None:
                g_exact = self._prepare_lora_exact_matrix(g_branch)
                if g_exact.shape[0] == a_exact.shape[0]:
                    self._lora_current_grad_dw[key] = g_exact.t() @ a_exact

            if self.steps % self.TCov != 0:
                continue
            gg = self._compute_lora_cov_g(g_branch)
            self.lora_m_gg[key] = self._ema_update(
                self.lora_m_gg.get(key, None),
                gg,
                self.stat_decay,
            )

    def _save_input(self, module: nn.Module, inputs) -> None:
        if not self.acc_stats:
            return
        if self.steps % self.TCov != 0:
            return
        if len(inputs) == 0 or inputs[0] is None:
            return

        a = inputs[0].detach()
        if a.dim() >= 1:
            self._current_batch_size = int(a.shape[0])
        aa = self._compute_cov_a(a, module)
        if aa is None:
            return

        self.m_aa[module] = self._ema_update(
            self.m_aa.get(module, None),
            aa,
            self.stat_decay,
        )

    def _save_grad_output(self, module: nn.Module, grad_input, grad_output) -> None:
        if not self.acc_stats:
            return
        if self.steps % self.TCov != 0:
            return
        if len(grad_output) == 0 or grad_output[0] is None:
            return

        g = grad_output[0].detach()
        gg = self._compute_cov_g(g, module)
        if gg is None:
            return

        self.m_gg[module] = self._ema_update(
            self.m_gg.get(module, None),
            gg,
            self.stat_decay,
        )

    def _update_inv(self, module: nn.Module) -> bool:
        if module not in self.m_aa or module not in self.m_gg:
            return False

        aa = self.m_aa[module].float()
        gg = self.m_gg[module].float()

        aa = 0.5 * (aa + aa.t())
        gg = 0.5 * (gg + gg.t())

        d_a, Q_a = torch.linalg.eigh(aa)
        d_g, Q_g = torch.linalg.eigh(gg)

        self.d_a[module] = torch.clamp(d_a, min=self.eps)
        self.d_g[module] = torch.clamp(d_g, min=self.eps)
        self.Q_a[module] = Q_a
        self.Q_g[module] = Q_g
        return True

    def _get_matrix_form_grad(self, module: nn.Module) -> Optional[torch.Tensor]:
        if module.weight is None or module.weight.grad is None:
            return None

        if isinstance(module, nn.Linear):
            grad_mat = module.weight.grad.detach()
        elif isinstance(module, nn.Conv2d):
            grad_mat = module.weight.grad.detach().reshape(module.weight.grad.shape[0], -1)
        else:
            return None

        if self._module_include_bias.get(module, False):
            if module.bias is None or module.bias.grad is None:
                return None
            grad_mat = torch.cat([grad_mat, module.bias.grad.detach().reshape(-1, 1)], dim=1)

        expected_shape = self._module_shapes.get(module, None)
        if expected_shape is not None and tuple(grad_mat.shape) != tuple(expected_shape):
            return None

        return grad_mat

    def _get_kfac_inverse_grad(
        self,
        module: nn.Module,
        grad_mat: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if module not in self.Q_a or module not in self.Q_g:
            if not self._update_inv(module):
                return None

        Q_a = self.Q_a[module].to(device=grad_mat.device).float()
        Q_g = self.Q_g[module].to(device=grad_mat.device).float()
        d_a = self.d_a[module].to(device=grad_mat.device).float()
        d_g = self.d_g[module].to(device=grad_mat.device).float()

        grad_mat_f = grad_mat.float()
        if Q_a.shape[0] != grad_mat_f.shape[1] or Q_g.shape[0] != grad_mat_f.shape[0]:
            return None

        v1 = Q_g.t() @ grad_mat_f @ Q_a
        v2 = v1 / (d_g.unsqueeze(1) * d_a.unsqueeze(0) + self.damping)
        v = Q_g @ v2 @ Q_a.t()
        return v.to(dtype=grad_mat.dtype)

    def _update_lora_inv(self, key: Tuple[nn.Module, str]) -> bool:
        if key not in self.lora_m_aa or key not in self.lora_m_gg:
            return False

        aa = self.lora_m_aa[key].float()
        gg = self.lora_m_gg[key].float()

        aa = 0.5 * (aa + aa.t())
        gg = 0.5 * (gg + gg.t())

        d_a, Q_a = torch.linalg.eigh(aa)
        d_g, Q_g = torch.linalg.eigh(gg)

        self.lora_d_a[key] = torch.clamp(d_a, min=self.eps)
        self.lora_d_g[key] = torch.clamp(d_g, min=self.eps)
        self.lora_Q_a[key] = Q_a
        self.lora_Q_g[key] = Q_g
        return True

    def _get_lora_kfac_inverse_grad(
        self,
        key: Tuple[nn.Module, str],
        grad_mat: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if key not in self.lora_Q_a or key not in self.lora_Q_g:
            if not self._update_lora_inv(key):
                return None

        Q_a = self.lora_Q_a[key].to(device=grad_mat.device).float()
        Q_g = self.lora_Q_g[key].to(device=grad_mat.device).float()
        d_a = self.lora_d_a[key].to(device=grad_mat.device).float()
        d_g = self.lora_d_g[key].to(device=grad_mat.device).float()

        grad_mat_f = grad_mat.float()
        if Q_a.shape[0] != grad_mat_f.shape[1] or Q_g.shape[0] != grad_mat_f.shape[0]:
            return None

        v1 = Q_g.t() @ grad_mat_f @ Q_a
        v2 = v1 / (d_g.unsqueeze(1) * d_a.unsqueeze(0) + self.damping)
        v = Q_g @ v2 @ Q_a.t()
        return v.to(dtype=grad_mat.dtype)

    def _split_lora_delta_to_params(
        self,
        key: Tuple[nn.Module, str],
        delta_direction: torch.Tensor,
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        spec = self.lora_specs[key]
        a_param = spec["a_param"]
        b_param = spec["b_param"]
        a_weight = a_param.detach().to(device=delta_direction.device, dtype=delta_direction.dtype)
        b_weight = b_param.detach().to(device=delta_direction.device, dtype=delta_direction.dtype)

        out: Dict[torch.nn.Parameter, torch.Tensor] = {}
        if a_param.requires_grad:
            out[a_param] = b_weight.t() @ delta_direction
        if b_param.requires_grad:
            out[b_param] = delta_direction @ a_weight.t()
        return out

    def _split_matrix_to_params(
        self,
        module: nn.Module,
        matrix: torch.Tensor,
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        out: Dict[torch.nn.Parameter, torch.Tensor] = {}
        if self._module_include_bias.get(module, False):
            out[module.weight] = matrix[:, :-1].reshape_as(module.weight)
            out[module.bias] = matrix[:, -1].reshape_as(module.bias)
        else:
            out[module.weight] = matrix.reshape_as(module.weight)
        return out

    def _fallback_inverse_grad(self, p: torch.Tensor, group) -> torch.Tensor:
        g = p.grad.detach()
        if self.fallback == "sam":
            direction = g
        elif self.fallback_inverse_mode == "paper":
            direction = g / (1.0 + group["eta"] * g.pow(2) + group["damping"])
        else:
            direction = g / (g.pow(2) + group["damping"])

        if group.get("adaptive", False):
            direction = direction * p.detach().pow(2)
        return direction

    def _compute_lora_kfac_fsam_direction(
        self,
    ) -> Tuple[Dict[torch.nn.Parameter, torch.Tensor], torch.Tensor]:
        updates: Dict[torch.nn.Parameter, torch.Tensor] = {}
        kfac_updated_params: set[torch.nn.Parameter] = set()

        first_param = self._first_param()
        device = first_param.device
        norm_sq = torch.zeros((), device=device, dtype=torch.float32)

        for key in self.lora_specs:
            if key not in self.lora_m_aa or key not in self.lora_m_gg:
                continue
            if (
                self.steps % self.TInv == 0
                or key not in self.lora_Q_a
                or key not in self.lora_Q_g
            ):
                if not self._update_lora_inv(key):
                    continue

            grad_dw = self._lora_current_grad_dw.get(key, None)
            if grad_dw is None:
                continue

            inv_grad_dw = self._get_lora_kfac_inverse_grad(key, grad_dw)
            if inv_grad_dw is None:
                continue

            term = (grad_dw * inv_grad_dw).sum().detach()
            norm_sq.add_(term.to(device=device, dtype=torch.float32))

            split = self._split_lora_delta_to_params(key, inv_grad_dw)
            updates.update(split)
            kfac_updated_params.update(split.keys())

        if self.fallback != "none":
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None or p in kfac_updated_params:
                        continue
                    direction = self._fallback_inverse_grad(p, group)
                    updates[p] = direction
                    term = (p.grad.detach() * direction).sum().detach()
                    norm_sq.add_(term.to(device=device, dtype=torch.float32))

        norm = torch.sqrt(torch.clamp(norm_sq, min=self.eps))
        return updates, norm

    def _compute_kfac_fsam_direction(
        self,
    ) -> Tuple[Dict[torch.nn.Parameter, torch.Tensor], torch.Tensor]:
        if self.kfac_target == "lora_delta_w":
            return self._compute_lora_kfac_fsam_direction()

        updates: Dict[torch.nn.Parameter, torch.Tensor] = {}
        kfac_updated_params: set[torch.nn.Parameter] = set()

        first_param = self._first_param()
        device = first_param.device
        norm_sq = torch.zeros((), device=device, dtype=torch.float32)

        for module in self.modules:
            if module not in self.m_aa or module not in self.m_gg:
                continue
            if self.steps % self.TInv == 0 or module not in self.Q_a or module not in self.Q_g:
                if not self._update_inv(module):
                    continue

            grad_mat = self._get_matrix_form_grad(module)
            if grad_mat is None:
                continue

            inv_grad_mat = self._get_kfac_inverse_grad(module, grad_mat)
            if inv_grad_mat is None:
                continue

            term = (grad_mat * inv_grad_mat).sum().detach()
            norm_sq.add_(term.to(device=device, dtype=torch.float32))

            split = self._split_matrix_to_params(module, inv_grad_mat)
            updates.update(split)
            kfac_updated_params.update(split.keys())

        if self.fallback != "none":
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None or p in kfac_updated_params:
                        continue
                    direction = self._fallback_inverse_grad(p, group)
                    updates[p] = direction
                    term = (p.grad.detach() * direction).sum().detach()
                    norm_sq.add_(term.to(device=device, dtype=torch.float32))

        norm = torch.sqrt(torch.clamp(norm_sq, min=self.eps))
        return updates, norm

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        updates, norm = self._compute_kfac_fsam_direction()
        if len(updates) == 0:
            raise RuntimeError(
                "No KFAC-FSAM direction was computed. "
                "Check gradients, trainable Linear/Conv2d modules, or fallback setting."
            )

        for group in self.param_groups:
            scale = group["gamma"] / (norm + self.eps)
            for p in group["params"]:
                if p not in updates:
                    continue
                self.state[p]["old_p"] = p.detach().clone(memory_format=torch.preserve_format)
                e_w = updates[p] * scale.to(device=p.device, dtype=p.dtype)
                self.state[p]["e_w"] = e_w
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" not in state:
                    continue
                p.copy_(state["old_p"])

        self.base_optimizer.step()
        self._clear_perturbation_state()

        if zero_grad:
            self.zero_grad()

        self.steps += 1

    @torch.no_grad()
    def restore(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "old_p" not in state:
                    continue
                p.copy_(state["old_p"])
        self._clear_perturbation_state()

    def _clear_perturbation_state(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state.pop("old_p", None)
                state.pop("e_w", None)

    def step(
        self,
        closure: Optional[Callable[[], Any]] = None,
        zero_grad: bool = True,
    ):
        """
        SAM-compatible convenience wrapper.

        The closure may either call backward internally, as the existing SAM/GAM
        training loops in this repository do, or simply return a loss tensor.
        The return value is the perturbed-pass closure result, matching the
        DiagonalFisherSAM wrapper behavior.
        """
        if closure is None:
            raise RuntimeError("KFACFisherSAM requires a closure.")

        closure = torch.enable_grad()(closure)

        self.zero_grad()
        if self.kfac_target == "lora_delta_w":
            self._lora_current_a.clear()
            self._lora_current_grad_dw.clear()
        self.acc_stats = True
        try:
            clean_result = closure()
            self._backward_if_closure_only_returned_loss(clean_result)
        finally:
            self.acc_stats = False

        clean_loss = self._extract_loss(clean_result)
        self.last_clean_loss = clean_loss.detach() if torch.is_tensor(clean_loss) else None

        self.first_step(zero_grad=True)
        fsam_result = closure()
        self._backward_if_closure_only_returned_loss(fsam_result)

        fsam_loss = self._extract_loss(fsam_result)
        self.last_fsam_loss = fsam_loss.detach() if torch.is_tensor(fsam_loss) else None

        self.second_step(zero_grad=zero_grad)
        return fsam_result

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        state = self.base_optimizer.state_dict()
        state["kfac_fsam_steps"] = self.steps
        return state

    def load_state_dict(self, state_dict):
        self.steps = int(state_dict.pop("kfac_fsam_steps", 0))
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups


KFACFisherSAMViT = KFACFisherSAM
FisherSAMKFAC = KFACFisherSAM
FSAMKFAC = KFACFisherSAM

__all__ = [
    "KFACFisherSAM",
    "KFACFisherSAMViT",
    "FisherSAMKFAC",
    "FSAMKFAC",
]
