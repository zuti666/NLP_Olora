"""Periodic stochastic multi-directional Fisher shaping.

This optimizer implements the second efficient Factor-FS design.  Unlike
``fishershape_efficient``, which probes several radii along one clean-gradient
direction, this implementation probes a *fixed* radius along gradients from
different current-task microbatches.

For every paired microbatch closure j it computes

    g0_j = grad L(theta; B_j)
    g1_j = grad L(theta + rho * unit(g0_j); B_j)
    r_j  = (||g0_j|| + eps) / rho * (g1_j - g0_j)

and aggregates ``r_bar = sum_j w_j r_j``.  The default ratio-mode direction is

    d = g_bar + c * kappa * ||g_bar|| * unit(r_bar),

where ``g_bar = sum_j w_j g0_j`` and
``kappa = ||sum_j w_j unit(r_j)||`` measures agreement between stochastic
shaping estimates.  On non-probe steps the latest shaping component can be
reused, or the optimizer can fall back to the clean gradient.

The learner supplies paired closures so that each clean/perturbed gradient
uses exactly the same microbatch.  EWC is deliberately absent here; the
learner injects its clean-point gradient through ``set_direction_projector``
after the current-task direction has been composed.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch

from .fishershape import FisherShape
from .util import disable_running_stats, enable_running_stats


GradMap = Dict[torch.nn.Parameter, torch.Tensor]


class StochasticMultiDirectionalFisherShape(FisherShape):
    """Fixed-radius, microbatch-direction FisherShape with periodic reuse."""

    def __init__(
        self,
        params,
        base_optimizer,
        model,
        rho=0.05,
        alpha=0.2,
        alpha_mode="ratio",
        sam_equivalent=False,
        adaptive=False,
        perturb_eps=1e-12,
        rho_scheduler=None,
        grad_reduce="mean",
        num_directions=2,
        probe_interval=1,
        reuse_mode="look",
        reuse_scale=1.0,
        confidence_mode="consistency",
        match_probe_rng=True,
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
        if adaptive:
            raise ValueError(
                "StochasticMultiDirectionalFisherShape requires adaptive=False: "
                "the method assumes a fixed Euclidean perturbation radius rho."
            )

        self.num_directions = max(1, int(num_directions))
        self.probe_interval = max(1, int(probe_interval))
        self.reuse_mode = str(reuse_mode).lower()
        if self.reuse_mode not in {"look", "sgd", "none"}:
            raise ValueError(
                'reuse_mode must be "look", "sgd", or "none", '
                f"got {self.reuse_mode!r}"
            )
        self.reuse_scale = float(reuse_scale)

        self.confidence_mode = str(confidence_mode).lower()
        if self.confidence_mode not in {"consistency", "none"}:
            raise ValueError(
                'confidence_mode must be "consistency" or "none", '
                f"got {self.confidence_mode!r}"
            )
        self.match_probe_rng = bool(match_probe_rng)

        self._step_idx = 0
        self._cached_component_norm = 0.0
        self._cached_g0_norm = 0.0
        self._cached_consistency = 0.0
        self._cached_confidence = 0.0

    def will_probe(self) -> bool:
        """Return whether the next optimizer step is a multi-direction probe."""
        return (self._step_idx % self.probe_interval) == 0

    def _trainable_params(self) -> Iterable[torch.nn.Parameter]:
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    yield param

    @torch.no_grad()
    def _clone_grads(self) -> GradMap:
        return {
            param: param.grad.detach().clone()
            for param in self._trainable_params()
            if param.grad is not None
        }

    @staticmethod
    def _map_norm(grads: GradMap) -> float:
        return sum(float(grad.pow(2).sum().item()) for grad in grads.values()) ** 0.5

    @staticmethod
    def _add_scaled_(target: GradMap, source: GradMap, scale: float) -> None:
        for param, value in source.items():
            if param not in target:
                target[param] = value.detach().clone().mul_(scale)
            else:
                target[param].add_(value, alpha=scale)

    @torch.no_grad()
    def _perturb_from(self, grads: GradMap, grad_norm: float) -> None:
        scale = float(self.rho) / (float(grad_norm) + self.perturb_eps)
        for param, grad in grads.items():
            displacement = grad * scale
            param.add_(displacement)
            self.state[param]["mdfs_displacement"] = displacement

    @torch.no_grad()
    def _restore_clean_point(self) -> None:
        for param in self._trainable_params():
            displacement = self.state[param].pop("mdfs_displacement", None)
            if displacement is not None:
                param.sub_(displacement)

    @staticmethod
    def _capture_rng_state():
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return cpu_state, cuda_states

    @staticmethod
    def _restore_rng_state(state) -> None:
        cpu_state, cuda_states = state
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    @staticmethod
    def _detach_output(output):
        if torch.is_tensor(output):
            return output.detach()
        if isinstance(output, dict):
            return {
                key: StochasticMultiDirectionalFisherShape._detach_output(value)
                for key, value in output.items()
            }
        if isinstance(output, tuple):
            return tuple(
                StochasticMultiDirectionalFisherShape._detach_output(value)
                for value in output
            )
        if isinstance(output, list):
            return [
                StochasticMultiDirectionalFisherShape._detach_output(value)
                for value in output
            ]
        return output

    @staticmethod
    def _merge_outputs(outputs: Sequence):
        """Merge clean microbatch outputs while preserving their input order."""
        if not outputs:
            return None
        first = outputs[0]
        if torch.is_tensor(first):
            if first.ndim == 0:
                return first
            return torch.cat(list(outputs), dim=0)
        if isinstance(first, dict):
            merged = {}
            for key in first:
                values = [output[key] for output in outputs if key in output]
                if len(values) != len(outputs):
                    continue
                if all(torch.is_tensor(value) and value.ndim > 0 for value in values):
                    try:
                        merged[key] = torch.cat(values, dim=0)
                        continue
                    except RuntimeError:
                        pass
                merged[key] = values[0]
            return merged
        return first

    @staticmethod
    def _normalise_weights(count: int, weights: Optional[Sequence[float]]) -> List[float]:
        if count <= 0:
            raise ValueError("At least one probe closure is required")
        if weights is None:
            return [1.0 / float(count)] * count
        if len(weights) != count:
            raise ValueError(
                f"probe_weights has length {len(weights)}, expected {count}"
            )
        positive = [max(0.0, float(value)) for value in weights]
        total = sum(positive)
        if total <= 0.0:
            raise ValueError("probe_weights must have positive total mass")
        return [value / total for value in positive]

    @torch.no_grad()
    def _write_direction(self, clean_grad: GradMap, component: GradMap) -> None:
        for param in self._trainable_params():
            clean = clean_grad.get(param)
            shape = component.get(param)
            if clean is None and shape is None:
                param.grad = None
            elif clean is None:
                param.grad = shape.detach().clone()
            elif shape is None:
                param.grad = clean.detach().clone()
            else:
                param.grad = clean + shape

    @torch.no_grad()
    def _cache_component(
        self,
        component: GradMap,
        clean_norm: float,
        consistency: float,
        confidence: float,
    ) -> None:
        component_norm_sq = 0.0
        for param in self._trainable_params():
            value = component.get(param)
            if value is None:
                self.state[param].pop("mdfs_component_cache", None)
                continue
            self.state[param]["mdfs_component_cache"] = value.detach().clone()
            component_norm_sq += float(value.pow(2).sum().item())
        self._cached_component_norm = component_norm_sq ** 0.5
        self._cached_g0_norm = float(clean_norm)
        self._cached_consistency = float(consistency)
        self._cached_confidence = float(confidence)

    @torch.no_grad()
    def _reuse_cached_component(self, clean_norm: float) -> None:
        if self.reuse_mode in {"sgd", "none"} or self._cached_component_norm <= 0.0:
            scale = 0.0
        else:
            scale = float(self.reuse_scale)
            # Ratio-mode shaping is defined relative to the current clean
            # gradient norm, so preserve that ratio between probe steps.
            if self.alpha_mode == "ratio" or self.sam_equivalent:
                scale *= float(clean_norm) / (
                    self._cached_g0_norm + self.perturb_eps
                )

        reused_norm_sq = 0.0
        clean_dot_reused = 0.0
        for param in self._trainable_params():
            if param.grad is None:
                continue
            cached = self.state[param].get("mdfs_component_cache")
            if cached is None or scale == 0.0:
                continue
            component = cached.to(device=param.device, dtype=param.dtype) * scale
            clean_dot_reused += float((param.grad.detach() * component).sum().item())
            reused_norm_sq += float(component.pow(2).sum().item())
            param.grad.add_(component)

        reused_norm = reused_norm_sq ** 0.5
        denom = float(clean_norm) * reused_norm + self.perturb_eps
        self.last_stats = {
            "fs_used_probe": 0.0,
            "fs_probe_interval": float(self.probe_interval),
            "fs_num_directions": float(self.num_directions),
            "fs_direction_consistency": float(self._cached_consistency),
            "fs_confidence": float(self._cached_confidence),
            "fs_reuse_mode_look": 1.0 if scale != 0.0 else 0.0,
            "fs_reuse_scale": float(scale),
            "fs_cached_component_norm": float(self._cached_component_norm),
            "fs_cached_g0_norm": float(self._cached_g0_norm),
            "fs_g0_norm": float(clean_norm),
            "fs_rf_norm": 0.0,
            "fs_shaping_norm": float(reused_norm),
            "fs_cos_g0_rf": float(clean_dot_reused / denom),
            "fs_s0": 0.0,
            "fs_probe_scale_min": 0.0,
            "fs_probe_scale_max": 0.0,
            "fs_alpha_eff": 0.0,
            "fs_probe_backward_count": 0.0,
        }

    def _multi_direction_probe(
        self,
        probe_closures: Sequence,
        probe_weights: Optional[Sequence[float]],
    ):
        closures = list(probe_closures)[: self.num_directions]
        selected_weights = (
            None
            if probe_weights is None
            else list(probe_weights)[: len(closures)]
        )
        weights = self._normalise_weights(len(closures), selected_weights)

        clean_sum: GradMap = {}
        r_sum: GradMap = {}
        q_sum: GradMap = {}
        diff_sum: GradMap = {}
        clean_outputs = []
        clean_loss = None
        probe_scales = []

        disable_running_stats(self.model)
        try:
            for closure, weight in zip(closures, weights):
                rng_before = self._capture_rng_state() if self.match_probe_rng else None
                outputs_j, loss_j = closure()
                rng_after_clean = self._capture_rng_state() if self.match_probe_rng else None
                g0_j = self._clone_grads()
                g0_norm_j = self._map_norm(g0_j)
                probe_scales.append(
                    float(self.rho) / (g0_norm_j + self.perturb_eps)
                )

                clean_outputs.append(self._detach_output(outputs_j))
                weighted_loss = loss_j.detach() * float(weight)
                clean_loss = weighted_loss if clean_loss is None else clean_loss + weighted_loss
                self._add_scaled_(clean_sum, g0_j, float(weight))

                self._perturb_from(g0_j, g0_norm_j)
                try:
                    if rng_before is not None:
                        self._restore_rng_state(rng_before)
                    closure()
                    g1_j = self._clone_grads()
                finally:
                    self._restore_clean_point()
                    if rng_after_clean is not None:
                        self._restore_rng_state(rng_after_clean)

                diff_j: GradMap = {}
                for param in set(g0_j) | set(g1_j):
                    g0_value = g0_j.get(param)
                    g1_value = g1_j.get(param)
                    if g0_value is None:
                        diff_j[param] = g1_value.detach().clone()
                    elif g1_value is None:
                        diff_j[param] = -g0_value
                    else:
                        diff_j[param] = g1_value - g0_value
                self._add_scaled_(diff_sum, diff_j, float(weight))

                r_scale = (g0_norm_j + self.perturb_eps) / max(
                    float(self.rho), self.perturb_eps
                )
                r_j = {param: value * r_scale for param, value in diff_j.items()}
                r_norm_j = self._map_norm(r_j)
                self._add_scaled_(r_sum, r_j, float(weight))
                if r_norm_j > 0.0:
                    self._add_scaled_(
                        q_sum,
                        r_j,
                        float(weight) / (r_norm_j + self.perturb_eps),
                    )
        finally:
            self._restore_clean_point()

        clean_norm = self._map_norm(clean_sum)
        r_norm = self._map_norm(r_sum)
        mean_probe_scale = sum(
            weight * scale for weight, scale in zip(weights, probe_scales)
        )
        consistency = self._map_norm(q_sum)
        consistency = min(1.0, max(0.0, consistency))
        confidence = consistency if self.confidence_mode == "consistency" else 1.0

        clean_r_dot = 0.0
        for param, clean_value in clean_sum.items():
            r_value = r_sum.get(param)
            if r_value is not None:
                clean_r_dot += float((clean_value * r_value).sum().item())

        if self.sam_equivalent:
            # M=1 is exact SAM.  For M>1 this is the sample-weighted average
            # of the M SAM correction terms and is reported as such.
            component = diff_sum
            alpha_eff = 0.0
        elif self.alpha_mode == "ratio":
            alpha_eff = (
                float(self.alpha)
                * float(confidence)
                * clean_norm
                / (r_norm + self.perturb_eps)
            )
            component = {param: value * alpha_eff for param, value in r_sum.items()}
        else:
            alpha_eff = float(self.alpha) * float(confidence)
            component = {param: value * alpha_eff for param, value in r_sum.items()}

        shaping_norm = self._map_norm(component)
        self._write_direction(clean_sum, component)
        self._cache_component(component, clean_norm, consistency, confidence)

        denom = clean_norm * r_norm + self.perturb_eps
        self.last_stats = {
            "fs_used_probe": 1.0,
            "fs_probe_interval": float(self.probe_interval),
            "fs_num_directions": float(len(closures)),
            "fs_direction_consistency": float(consistency),
            "fs_confidence": float(confidence),
            "fs_reuse_mode_look": 1.0 if self.reuse_mode == "look" else 0.0,
            "fs_reuse_scale": float(self.reuse_scale),
            "fs_cached_component_norm": float(self._cached_component_norm),
            "fs_cached_g0_norm": float(self._cached_g0_norm),
            "fs_g0_norm": float(clean_norm),
            "fs_rf_norm": float(r_norm),
            "fs_shaping_norm": float(shaping_norm),
            "fs_cos_g0_rf": float(clean_r_dot / denom),
            # Compatibility field: for PMD-FS this is the weighted mean of
            # the per-direction scales s_j=rho/(||g0_j||+eps).
            "fs_s0": float(mean_probe_scale),
            "fs_probe_scale_min": float(min(probe_scales)),
            "fs_probe_scale_max": float(max(probe_scales)),
            "fs_alpha_eff": float(alpha_eff),
            "fs_probe_backward_count": float(2 * len(closures)),
        }
        return self._merge_outputs(clean_outputs), clean_loss

    def step(
        self,
        closure=None,
        *,
        probe_closures: Optional[Sequence] = None,
        probe_weights: Optional[Sequence[float]] = None,
    ):
        get_grad = closure if closure is not None else self.forward_backward_func
        use_probe = self.will_probe()

        try:
            with self.maybe_no_sync():
                if use_probe:
                    closures = probe_closures if probe_closures is not None else [get_grad]
                    outputs, loss_value = self._multi_direction_probe(
                        closures,
                        probe_weights,
                    )
                else:
                    outputs, loss_value = get_grad()
                    clean_norm = float(self._grad_norm().item())
                    self._reuse_cached_component(clean_norm)

                if callable(self._direction_projector):
                    self._direction_projector(self.param_groups)

            self._sync_grad()
            self.base_optimizer.step()
            self._step_idx += 1
            return outputs, loss_value
        finally:
            self._restore_clean_point()
            enable_running_stats(self.model)

    def __repr__(self):
        return (
            "StochasticMultiDirectionalFisherShape("
            f"{self.base_optimizer.__class__.__name__}, rho={self.rho}, "
            f"alpha={self.alpha}, num_directions={self.num_directions}, "
            f"probe_interval={self.probe_interval}, reuse_mode={self.reuse_mode}, "
            f"confidence_mode={self.confidence_mode})"
        )
