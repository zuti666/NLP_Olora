"""Minimal native g12, g02, 2gm and g13 updates with transactional probes.

The midpoint is formed in actual trainable factor/FC coordinates. Every CE
read replays R0, and final submission advances random state to g0's R1 only.
Unused readouts are absent in normal training; audit_all is an isolated gate.
"""
import math
import torch

from audit_core import rng_get, rng_set
from state_transaction import attention_backend, capture_state


FORMULAS = {"sum12": "g1+g2", "base02": "g0+g2",
            "two_midpoint": "2*gm", "sum13": "g1+g3"}


def _clone(values):
    return tuple(v.detach().clone() for v in values)


def _combine(a, b, a_scale=1.0, b_scale=1.0):
    return tuple(a_scale*x+b_scale*y for x, y in zip(a, b))


def _inner(a, b):
    value = float(sum((x.detach().double()*y.detach().double()).sum()
                      for x, y in zip(a, b)).item())
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite direction inner product")
    return value


def vector_norm(values):
    return math.sqrt(max(0.0, _inner(values, values)))


def vector_cosine(a, b):
    denominator = vector_norm(a)*vector_norm(b)
    return _inner(a, b)/denominator if denominator > 1e-24 else None


def normalized(values, eps=1e-12):
    denominator = vector_norm(values)+float(eps)
    return tuple(v/denominator for v in values)


def direction_name(direction):
    if direction not in FORMULAS:
        raise ValueError("Unknown direction: " + str(direction))
    return FORMULAS[direction]


def validate_factors(config):
    """Validate the new explicit interface; old estimator flags cannot override it."""
    direction = config.get("direction", config.get("update_direction"))
    direction_name(direction)
    for key in ("direction", "update_direction", "component_rule"):
        if key in config and config[key] != direction:
            raise ValueError(key + " conflicts with the explicit direction")
    obsolete = ("first_location_estimator", "second_residual_estimator",
                "residual_coefficient", "second_probe_sign", "second_difference_sign",
                "second_difference_weight")
    if any(key in config for key in obsolete):
        raise ValueError("Legacy F/B estimator fields are not supported in this protocol")
    for key in ("branch", "probe_branch"):
        if key in config and config[key] != "P":
            raise ValueError("The common first location must be branch P")
    if "first_probe_sign" in config and config["first_probe_sign"] != 1:
        raise ValueError("The common first probe is positive")
    return direction


def expected_budget(config):
    direction = validate_factors(config)
    g2 = int(direction != "two_midpoint")
    gm = int(direction == "two_midpoint")
    g3 = int(direction == "sum13")
    return dict(train_ce_per_update=2+g2+gm+g3,
                diagnostic_ce_per_update=0,
                first_plus_ce_per_update=1, first_minus_ce_per_update=0,
                g3_plus_ce_per_update=g3, g3_minus_ce_per_update=0,
                local_plus_ce_per_update=1+g3, local_minus_ce_per_update=0,
                g0_ce_per_update=1, g1_ce_per_update=1,
                g2_ce_per_update=g2, midpoint_ce_per_update=gm,
                gm_ce_per_update=gm, g3_ce_per_update=g3)


class FourProbeStep:
    """Read only needed probes, restore theta0, submit exactly one ordinary SGD."""

    def __init__(self, model, params, geometry, config):
        self.model, self.params, self.geometry = model, tuple(params), geometry
        self.config = dict(config)
        self.rho, self.radius = float(config["rho"]), float(config["radius"])
        self.eps = float(config.get("eps", 1e-12))
        self.direction = validate_factors(config)
        self.formula, self.branch = direction_name(self.direction), "P"
        self.budget = expected_budget(config)
        self.backend = config.get("backend", "native")
        if self.backend not in ("native", "math"):
            raise ValueError("Unknown attention backend")
        if not all(math.isfinite(v) for v in (self.rho, self.radius, self.eps)):
            raise ValueError("Probe radii and epsilon must be finite")
        if self.rho < 0 or self.radius < 0 or self.eps <= 0:
            raise ValueError("Probe radii must be nonnegative and epsilon positive")
        if not self.params or len({id(p) for p in self.params}) != len(self.params):
            raise ValueError("Empty or duplicate trainable parameters")
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        if trainable != {id(p) for p in self.params}:
            raise ValueError("params must contain every trainable model parameter")
        if {id(p) for p in geometry.params} != trainable:
            raise ValueError("Effective map does not cover all trainables")
        for module in model.modules():
            if bool(getattr(module, "_register_delta_hook", False)):
                raise ValueError("Pure direction training must disable delta-gradient hooks")
            if (getattr(module, "external_delta_w_q", None) is not None or
                    getattr(module, "external_delta_w_v", None) is not None):
                raise ValueError("Pure direction training cannot use external delta weights")
        self._geo_index = [next(i for i, p in enumerate(self.params) if p is q)
                           for q in geometry.params]
        self._pending = None

    def _geometry_order(self, values):
        return tuple(values[i] for i in self._geo_index)

    def _validate(self, values, label):
        if len(values) != len(self.params):
            raise ValueError(label + " tensor count mismatch")
        for p, v in zip(self.params, values):
            if v is None or v.shape != p.shape or v.device != p.device or v.dtype != p.dtype:
                raise ValueError(label + " tensor shape/device/dtype mismatch")
        if not bool(torch.stack([torch.isfinite(v).all() for v in values]).all()):
            raise FloatingPointError(label + " has a nonfinite value")

    @torch.no_grad()
    def _set_point(self, values):
        for p, v in zip(self.params, values):
            p.copy_(v)

    def _restore(self, state, rng):
        state["transaction"].restore()
        rng_set(rng)

    def _evaluate(self, values, inputs, targets, closure, state):
        self._restore(state, state["rng0"])
        self._set_point(values)
        with torch.enable_grad(), attention_backend(self.model, self.backend):
            loss, logits = closure(inputs, targets)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("Closure loss must be finite scalar")
            grads = torch.autograd.grad(loss, self.params, create_graph=False,
                                        retain_graph=False, allow_unused=False)
        state["gradient_calls"] += 1
        self._validate(grads, "ordinary CE gradient")
        return float(loss.detach().item()), logits.detach(), _clone(grads), rng_get()

    @torch.no_grad()
    def _point(self, base, direction, scale):
        return tuple(v+float(scale)*d for v, d in zip(base, direction))

    def _probe_audit(self, base, point, gradient, radius):
        displacement = _combine(point, base, 1., -1.)
        length, norm = vector_norm(displacement), vector_norm(gradient)
        expected = radius*norm/(norm+self.eps)
        alignment = vector_cosine(displacement, gradient)
        tolerance = max(1e-8, radius*1e-4)
        degenerate = expected <= 1e-24
        if (abs(length-expected) > tolerance or
                (not degenerate and (alignment is None or abs(alignment-1.) > 1e-4))):
            raise AssertionError("Actual point violates global normalized probe")
        return dict(requested_sign=1, parameter_length=length,
                    target_radius=radius, expected_parameter_length=expected,
                    alignment_cosine=alignment, length_error=abs(length-expected),
                    absolute_tolerance=tolerance, cosine_tolerance=1e-4,
                    degenerate=degenerate, passed=True)

    def compute(self, inputs, targets, loss_closure, current_lr, audit_all=False):
        lr = float(current_lr)
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("current_lr must be finite and positive")
        if self._pending is not None:
            raise RuntimeError("Previous direction must be applied or discarded first")
        transaction = capture_state(self.model)
        value_map = {id(p): v for p, v in transaction.trainable}
        state = {"base": tuple(value_map[id(p)] for p in self.params),
                 "rng0": transaction.rng, "transaction": transaction,
                 "gradient_calls": 0}
        try:
            l0, logits0, g0, rng1 = self._evaluate(state["base"], inputs, targets, loss_closure, state)
            first = self._point(state["base"], normalized(g0, self.eps), self.rho)
            first_audit = self._probe_audit(state["base"], first, g0, self.rho)
            l1, _, g1, _ = self._evaluate(first, inputs, targets, loss_closure, state)
            q = _combine(g1, g0, 1., -1.)
            outer = self._point(state["base"], normalized(q, self.eps), self.radius)
            outer_audit = self._probe_audit(state["base"], outer, q, self.radius)
            # Use already-rounded actual endpoint tensors. This averages A/B
            # factors and FC coordinates, never their effective BA matrices.
            midpoint = tuple((a+b)*0.5 for a, b in zip(first, outer))
            points = {"phi0": state["base"], "phi1": first, "phi2": outer,
                      "phim": midpoint}
            gradients, losses = {"g0": g0, "g1": g1}, {"g0": l0, "g1": l1}
            need2 = self.direction != "two_midpoint" or audit_all
            needm = self.direction == "two_midpoint" or audit_all
            need3 = self.direction == "sum13" or audit_all
            if need2:
                loss, _, g2, _ = self._evaluate(outer, inputs, targets, loss_closure, state)
                gradients["g2"], losses["g2"] = g2, loss
            if needm:
                loss, _, gm, _ = self._evaluate(midpoint, inputs, targets, loss_closure, state)
                gradients["gm"], losses["gm"] = gm, loss
            third_audit = None
            if need3:
                third = self._point(outer, normalized(gradients["g2"], self.eps), self.rho)
                third_audit = self._probe_audit(outer, third, gradients["g2"], self.rho)
                loss, _, g3, _ = self._evaluate(third, inputs, targets, loss_closure, state)
                gradients["g3"], losses["g3"], points["phi3"] = g3, loss, third
            if self.direction == "sum12":
                U = _combine(g1, gradients["g2"])
            elif self.direction == "base02":
                U = _combine(g0, gradients["g2"])
            elif self.direction == "two_midpoint":
                U = tuple(2.*v for v in gradients["gm"])
            else:
                U = _combine(g1, gradients["g3"])
            expected_calls = 5 if audit_all else self.budget["train_ce_per_update"]
            if state["gradient_calls"] != expected_calls:
                raise AssertionError("Ordinary CE count violates the native/audit budget")
            self._validate(U, "final update direction")
            self._restore(state, state["rng0"])
            restoration = transaction.check()
            rng_set(rng1)
            for p, u in zip(self.params, U):
                p.grad = u.detach().clone()
            delta1 = _combine(first, state["base"], 1., -1.)
            delta2 = _combine(outer, state["base"], 1., -1.)
            m = _combine(midpoint, state["base"], 1., -1.)
            d = _combine(outer, first, .5, -.5)
            ideal_midpoint = self._point(state["base"], _combine(delta1, delta2, .5, .5), 1.)
            midpoint_rounding = vector_norm(_combine(midpoint, ideal_midpoint, 1., -1.))
            e12 = None
            if "gm" in gradients and "g2" in gradients:
                e12 = tuple(.5*(a+b)-c for a, b, c in zip(g1, gradients["g2"], gradients["gm"]))
            norms = {key: vector_norm(values) for key, values in gradients.items()}
            norms.update(q=vector_norm(q), U=vector_norm(U), m=vector_norm(m), d=vector_norm(d))
            if e12 is not None:
                norms["e12"] = vector_norm(e12)
            displacement_points = {key: value for key, value in points.items() if key != "phi0"}
            record = dict(
                branch="P", direction=self.direction, formula=self.formula,
                backend=self.backend, first_probe_sign=1,
                first_location_formula="g1-g0", first_location_formula_exact=True,
                first_probe_parameter_audit=first_audit,
                first_location_parameter_audit=first_audit,
                outer_location_parameter_audit=outer_audit,
                second_probe_parameter_audit=third_audit,
                g3_used=self.direction == "sum13", midpoint_used=self.direction == "two_midpoint",
                midpoint_definition="actual rounded theta1/theta2 trainable A/B+FC coordinate midpoint",
                midpoint_not_effective_BA_average=True,
                midpoint_rounding_parameter_error=midpoint_rounding,
                first_location_only_g1_needed=True,
                ordinary_gradient_calls=state["gradient_calls"],
                native_expected_gradient_calls=self.budget["train_ce_per_update"],
                minimal_expected_gradient_calls=self.budget["train_ce_per_update"],
                expected_budget=dict(self.budget), audit_all=bool(audit_all),
                additional_diagnostic_gradient_calls=expected_calls-self.budget["train_ce_per_update"],
                diagnostic_only_extra_readouts=bool(audit_all), sgd_updates=0,
                fisher_calls=0, weight_merge_calls=0, gradient_projection_calls=0,
                hvp_calls=0, create_graph_calls=0, opposite_g1_gradient_calls=0,
                g0_gradient_calls=1, g1_gradient_calls=1,
                g2_gradient_calls=int(need2), midpoint_gradient_calls=int(needm),
                gm_gradient_calls=int(needm), g3_gradient_calls=int(need3),
                first_positive_gradient_calls=1, first_negative_gradient_calls=0,
                g1_positive_gradient_calls=1, g1_negative_gradient_calls=0,
                g3_positive_gradient_calls=int(need3), g3_negative_gradient_calls=0,
                local_positive_gradient_calls=1+int(need3), local_negative_gradient_calls=0,
                rho=self.rho, radius=self.radius, eps=self.eps,
                loss0=l0, loss1=l1, loss2=losses.get("g2"),
                loss_midpoint=losses.get("gm"), loss3=losses.get("g3"),
                delta_loss1=l1-l0,
                delta_loss2=losses["g2"]-l0 if need2 else None,
                delta_loss3=losses["g3"]-losses["g2"] if need3 else None,
                norms=norms,
                cosines={"U_"+key: vector_cosine(U, value) for key, value in gradients.items()},
                midpoint_geometry={"m_parameter_norm": vector_norm(m),
                                   "d_parameter_norm": vector_norm(d),
                                   "delta1_delta2_cosine": vector_cosine(delta1, delta2)},
                projection=None, centered_response=None,
                probe_displacements={name: self.geometry.actual_delta(
                    self._geometry_order(state["base"]), self._geometry_order(point))
                    for name, point in displacement_points.items()},
                base_parameters_exact=True, base_buffers_exact=True, state_restoration=restoration,
                forward_rng_rule="R0 replay for every read; final RNG=R1 from g0")
            record["context_metrics"] = dict(
                gradient_norms=dict(norms), cosines=dict(record["cosines"]),
                midpoint_geometry=dict(record["midpoint_geometry"]),
                q_norm=norms["q"], q_inverse_outer_scale=self.radius/(norms["q"]+self.eps),
                q_degenerate=norms["q"] <= 1e-12,
                g0_dot_U=_inner(g0, U), nominal_parameter_step_norm=lr*norms["U"],
                probe_displacements=dict(record["probe_displacements"]),
                measured_from_existing_context_only=True, additional_gradient_calls=0,
                e12_measured=e12 is not None)
            context = dict(points=points, grads=gradients, q=q, U=U, e12=e12,
                           rng0=state["rng0"], rng1=rng1)
            self._pending = dict(base=state["base"], U=U, lr=lr, transaction=transaction)
            return dict(loss0=l0, logits0=logits0, U=U, record=record, context=context)
        except BaseException:
            self._restore(state, state["rng0"])
            self._pending = None
            raise

    def diagnose_sides(self, inputs, targets, loss_closure, result, current_lr):
        """A compatibility no-op: diagnostics never silently add gradient reads."""
        if self._pending is None:
            raise RuntimeError("Context diagnostics require a pending update")
        entry = capture_state(self.model)
        return dict(diagnostic_gradient_calls=0, diagnostic_forward_count=0,
                    diagnostic_sgd_updates=0, fisher_calls=0, weight_merge_calls=0,
                    gradient_projection_calls=0, hvp_calls=0, create_graph_calls=0,
                    measurement_sources="existing_training_context_only",
                    extra_side_readout_disabled=True, state_restoration=entry.check())

    def discard(self):
        """Cancel pending update and restore its entire pre-compute transaction."""
        if self._pending is not None:
            self._pending["transaction"].restore()
            self._pending["transaction"].check()
            self._pending = None

    @torch.no_grad()
    def apply_update(self, current_lr=None, optimizer=None):
        if self._pending is None:
            raise RuntimeError("compute must precede apply_update")
        pending = self._pending
        base, U = pending["base"], pending["U"]
        lr = pending["lr"] if current_lr is None else float(current_lr)
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("SGD learning rate must be positive")
        if not all(torch.equal(p, v) for p, v in zip(self.params, base)):
            raise RuntimeError("Base changed between compute and actual update")
        if optimizer is not None:
            if not isinstance(optimizer, torch.optim.SGD):
                raise ValueError("Only ordinary torch.optim.SGD is supported")
            groups = optimizer.param_groups
            if {id(p) for g in groups for p in g["params"]} != {id(p) for p in self.params}:
                raise ValueError("SGD must cover exactly every trainable parameter")
            if any(any(float(g.get(k, 0)) != 0 for k in ("momentum", "weight_decay", "dampening"))
                   or bool(g.get("nesterov", False)) or bool(g.get("maximize", False)) for g in groups):
                raise ValueError("SGD must have zero momentum, dampening, and weight decay")
            if any(abs(float(g["lr"])-lr) > 1e-12*max(abs(lr), 1.) for g in groups):
                raise ValueError("All SGD groups must share the nominal LR")
        try:
            for p, u in zip(self.params, U):
                p.grad = u.detach().clone()
            if optimizer is None:
                for p, u in zip(self.params, U):
                    p.add_(u, alpha=-lr)
            else:
                optimizer.step()
            self._validate(tuple(p.detach() for p in self.params), "updated parameters")
            expected = tuple(v.add(u, alpha=-lr) for v, u in zip(base, U))
            if not all(torch.equal(p, v) for p, v in zip(self.params, expected)):
                raise AssertionError("Actual natural SGD differs from base-lr*U")
            actual = self.geometry.actual_delta(self._geometry_order(base))
            self._pending = None
            return dict(matched=False, nominal_lr=lr, scale=lr, step_scale=lr,
                        scale_over_lr=1., actual=actual, sgd_updates=1,
                        natural_sgd_exact=True)
        except BaseException:
            pending["transaction"].restore()
            pending["transaction"].check()
            self._pending = None
            raise
