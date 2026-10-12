"""Efficient multi-probe FisherShape optimizer.

This is a conservative extension of :mod:`fishershape` for LoRA continual
learning experiments.  It combines two ideas from efficient SAM variants:

* LookSAM-style periodic sharpness computation: only compute the expensive
  perturbed gradient every ``probe_interval`` steps, and optionally reuse the
  cached sharpness component on intermediate clean-gradient steps.
* Lookbehind/multistep-style aggregation: when a probe is computed, average
  several perturbed gradients/finite-difference estimates before composing the
  final FisherShape direction.

The optimizer keeps the original FisherShape semantics when
``probe_interval=1`` and ``probe_repeats=1``.
"""

from __future__ import annotations

import torch

from .fishershape import FisherShape
from .util import disable_running_stats, enable_running_stats


class EfficientFisherShape(FisherShape):
    def __init__(
        self,
        params,
        base_optimizer,
        model,
        rho=0.05,
        alpha=0.2,
        alpha_mode="fixed",
        sam_equivalent=False,
        adaptive=False,
        perturb_eps=1e-12,
        rho_scheduler=None,
        grad_reduce="mean",
        probe_interval=1,
        probe_repeats=1,
        probe_mode="multiradius",
        reuse_mode="look",
        reuse_scale=1.0,
        cache_momentum=0.0,
        **kwargs,
    ):
        super().__init__(
            params,
            base_optimizer=base_optimizer,
            model=model,
            rho=rho,
            alpha=alpha,
            alpha_mode=alpha_mode,
            sam_equivalent=sam_equivalent,
            adaptive=adaptive,
            perturb_eps=perturb_eps,
            rho_scheduler=rho_scheduler,
            grad_reduce=grad_reduce,
            **kwargs,
        )
        self.probe_interval = max(1, int(probe_interval))
        self.probe_repeats = max(1, int(probe_repeats))
        self.probe_mode = str(probe_mode).lower()
        if self.probe_mode not in {"multiradius", "repeat"}:
            raise ValueError(
                'probe_mode must be "multiradius" or "repeat", '
                f"got {self.probe_mode!r}"
            )

        self.reuse_mode = str(reuse_mode).lower()
        if self.reuse_mode not in {"look", "sgd", "none"}:
            raise ValueError(
                'reuse_mode must be "look", "sgd", or "none", '
                f"got {self.reuse_mode!r}"
            )
        self.reuse_scale = float(reuse_scale)
        self.cache_momentum = min(1.0, max(0.0, float(cache_momentum)))
        self._step_idx = 0
        self._cached_component_norm = 0.0
        self._cached_g0_norm = 0.0

    @torch.no_grad()
    def _save_g0_and_norm(self):
        norm_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    self.state[p].pop("g_0", None)
                    continue
                g_0 = p.grad.detach().clone()
                self.state[p]["g_0"] = g_0
                if self.adaptive:
                    norm_sq += float(torch.sum((g_0 * torch.abs(p)) ** 2).item())
                else:
                    norm_sq += float(torch.sum(g_0 ** 2).item())
        return norm_sq ** 0.5

    @torch.no_grad()
    def _apply_probe_from_clean(self, scale: float):
        for group in self.param_groups:
            for p in group["params"]:
                g_0 = self.state[p].get("g_0")
                if g_0 is None:
                    continue
                e_w = g_0 * torch.as_tensor(scale, device=p.device, dtype=p.dtype)
                if self.adaptive:
                    e_w = e_w * torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def _accumulate_probe_estimate(self, s_probe: float, s_ref: float):
        for group in self.param_groups:
            for p in group["params"]:
                g_0 = self.state[p].get("g_0")
                if g_0 is None or p.grad is None:
                    continue
                if self.probe_mode == "multiradius":
                    # Convert every finite difference to a common reference
                    # displacement s_ref so compose_direction(s_ref) can reuse
                    # the original FisherShape implementation.
                    probe_diff = (p.grad.detach() - g_0) * (s_ref / max(s_probe, self.perturb_eps))
                else:
                    probe_diff = p.grad.detach() - g_0

                if "probe_diff_sum" not in self.state[p]:
                    self.state[p]["probe_diff_sum"] = probe_diff.clone()
                else:
                    self.state[p]["probe_diff_sum"].add_(probe_diff)
                self.state[p]["probe_count"] = self.state[p].get("probe_count", 0) + 1

    @torch.no_grad()
    def _write_averaged_probe_gradient(self):
        for group in self.param_groups:
            for p in group["params"]:
                g_0 = self.state[p].get("g_0")
                if g_0 is None:
                    continue
                diff_sum = self.state[p].pop("probe_diff_sum", None)
                count = int(self.state[p].pop("probe_count", 0))
                if diff_sum is None or count <= 0:
                    p.grad = g_0.clone()
                else:
                    p.grad = g_0 + diff_sum / float(count)

    @torch.no_grad()
    def _cache_component_from_current_direction(self):
        component_norm_sq = 0.0
        g0_norm_sq = 0.0
        momentum = float(self.cache_momentum)
        for group in self.param_groups:
            for p in group["params"]:
                g_0 = self.state[p].get("g_0_cache")
                if g_0 is None or p.grad is None:
                    continue
                component = p.grad.detach() - g_0
                if momentum > 0.0 and "fs_component_cache" in self.state[p]:
                    component = (
                        momentum * self.state[p]["fs_component_cache"].to(component)
                        + (1.0 - momentum) * component
                    )
                self.state[p]["fs_component_cache"] = component.detach().clone()
                component_norm_sq += float(torch.sum(component ** 2).item())
                g0_norm_sq += float(torch.sum(g_0 ** 2).item())
                del self.state[p]["g_0_cache"]
        self._cached_component_norm = component_norm_sq ** 0.5
        self._cached_g0_norm = g0_norm_sq ** 0.5

    @torch.no_grad()
    def _reuse_cached_component(self, current_g0_norm: float):
        if self.reuse_mode in {"sgd", "none"}:
            self.last_stats = {
                "fs_used_probe": 0.0,
                "fs_probe_interval": float(self.probe_interval),
                "fs_probe_repeats": float(self.probe_repeats),
                "fs_reuse_mode_look": 0.0,
                "fs_reuse_scale": 0.0,
                "fs_cached_component_norm": float(self._cached_component_norm),
                "fs_cached_g0_norm": float(self._cached_g0_norm),
                "fs_g0_norm": float(current_g0_norm),
                "fs_rf_norm": 0.0,
                "fs_cos_g0_rf": 0.0,
                "fs_s0": 0.0,
                "fs_alpha_eff": 0.0,
            }
            return

        scale = self.reuse_scale
        if self._cached_g0_norm > 0.0:
            scale *= current_g0_norm / (self._cached_g0_norm + self.perturb_eps)

        reused_norm_sq = 0.0
        g0_dot_reused = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                cached = self.state[p].get("fs_component_cache")
                if cached is None:
                    continue
                component = cached.to(device=p.device, dtype=p.dtype) * scale
                g0_dot_reused += float(torch.sum(p.grad.detach() * component).item())
                reused_norm_sq += float(torch.sum(component ** 2).item())
                p.grad.add_(component)

        reused_norm = reused_norm_sq ** 0.5
        denom = current_g0_norm * reused_norm + self.perturb_eps
        self.last_stats = {
            "fs_used_probe": 0.0,
            "fs_probe_interval": float(self.probe_interval),
            "fs_probe_repeats": float(self.probe_repeats),
            "fs_reuse_mode_look": 1.0,
            "fs_reuse_scale": float(scale),
            "fs_cached_component_norm": float(self._cached_component_norm),
            "fs_cached_g0_norm": float(self._cached_g0_norm),
            "fs_g0_norm": float(current_g0_norm),
            "fs_rf_norm": float(reused_norm / max(self.rho, self.perturb_eps)),
            "fs_cos_g0_rf": float(g0_dot_reused / denom),
            "fs_s0": 0.0,
            "fs_alpha_eff": 0.0,
        }

    def step(self, closure=None):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        use_probe = (self._step_idx % self.probe_interval) == 0

        with self.maybe_no_sync():
            outputs, loss_value = get_grad()
            g0_norm = self._save_g0_and_norm()
            s_ref = self.rho / (g0_norm + self.perturb_eps)

            if use_probe:
                disable_running_stats(self.model)
                for probe_idx in range(self.probe_repeats):
                    if self.probe_mode == "multiradius" and self.probe_repeats > 1:
                        radius_scale = float(probe_idx + 1) / float(self.probe_repeats)
                    else:
                        radius_scale = 1.0
                    s_probe = s_ref * radius_scale

                    self._apply_probe_from_clean(s_probe)
                    get_grad()
                    self._accumulate_probe_estimate(s_probe, s_ref)
                    self.unperturb()

                self._write_averaged_probe_gradient()
                # Preserve g0 after compose_direction deletes state[p]["g_0"],
                # so the LookSAM-style component can be cached.
                for group in self.param_groups:
                    for p in group["params"]:
                        g_0 = self.state[p].get("g_0")
                        if g_0 is not None:
                            self.state[p]["g_0_cache"] = g_0.detach().clone()
                self.compose_direction(s_ref)
                self.last_stats.update(
                    {
                        "fs_used_probe": 1.0,
                        "fs_probe_interval": float(self.probe_interval),
                        "fs_probe_repeats": float(self.probe_repeats),
                        "fs_reuse_mode_look": 1.0 if self.reuse_mode == "look" else 0.0,
                        "fs_reuse_scale": float(self.reuse_scale),
                    }
                )
                self._cache_component_from_current_direction()
                self.last_stats.update(
                    {
                        "fs_cached_component_norm": float(self._cached_component_norm),
                        "fs_cached_g0_norm": float(self._cached_g0_norm),
                    }
                )
            else:
                # Clean-gradient step.  Either keep pure SGD or add the cached
                # sharpness component distilled on the latest probe step.
                for group in self.param_groups:
                    for p in group["params"]:
                        self.state[p].pop("g_0", None)
                self._reuse_cached_component(g0_norm)

            if callable(self._direction_projector):
                self._direction_projector(self.param_groups)

        self._sync_grad()
        self.base_optimizer.step()
        enable_running_stats(self.model)
        self._step_idx += 1

        return outputs, loss_value

    def __repr__(self):
        return (
            f"EfficientFisherShape({self.base_optimizer.__class__.__name__}, "
            f"rho={self.rho}, alpha={self.alpha}, "
            f"sam_equivalent={self.sam_equivalent}, "
            f"probe_interval={self.probe_interval}, "
            f"probe_repeats={self.probe_repeats}, "
            f"reuse_mode={self.reuse_mode})"
        )
