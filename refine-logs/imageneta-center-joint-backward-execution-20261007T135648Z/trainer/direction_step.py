"""Explicit forward/backward location and residual estimators, natural SGD.

Only the requested local gradients are evaluated.  In particular BB contains
no positive local readout, and centered residuals use five training CE calls.
"""
import math
import torch

from audit_core import rng_get, rng_set
from state_transaction import attention_backend, capture_state


def _clone(values):
    return tuple(v.detach().clone() for v in values)


def _combine(a, b, a_scale=1.0, b_scale=1.0):
    return tuple(a_scale*x + b_scale*y for x, y in zip(a, b))


def _inner(a, b):
    # One global reduction, covering every trainable LoRA and FC parameter.
    reduction = sum((x.detach().double()*y.detach().double()).sum()
                    for x, y in zip(a, b))
    value = float(reduction.item())
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite direction inner product")
    return value


def vector_norm(values):
    return math.sqrt(max(0.0, _inner(values, values)))


def vector_cosine(a, b):
    denominator = vector_norm(a)*vector_norm(b)
    return _inner(a, b)/denominator if denominator > 1e-24 else None


def normalized(values, eps=1e-12):
    denominator = vector_norm(values) + float(eps)
    return tuple(v/denominator for v in values)


def direction_name(first_location_estimator, second_residual_estimator,
                   residual_coefficient=1, update_base="g0"):
    family = "F" if first_location_estimator == "forward" else "B"
    if second_residual_estimator == "centered":
        sign = "+" if residual_coefficient == 1 else "-"
        return "{}{}(g3{}plus-g3{}minus)/2".format(update_base, sign, family, family)
    if second_residual_estimator == "forward":
        return "{}+g3{}plus-g2{}".format(update_base, family, family)
    return "{}+g2{}-g3{}minus".format(update_base, family, family)


def validate_factors(config):
    """Require v3 factors and validate any supplied derived legacy aliases."""
    first = config["first_location_estimator"]
    second = config["second_residual_estimator"]
    coefficient = config["residual_coefficient"]
    anchor = config["update_base"]
    if first not in ("forward", "backward_corrected"):
        raise ValueError("Unknown first_location_estimator")
    if second not in ("forward", "backward_corrected", "centered"):
        raise ValueError("Unknown second_residual_estimator")
    if type(coefficient) is not int or coefficient not in (-1, 1):
        raise ValueError("residual_coefficient must be the integer +/-1")
    if coefficient == -1 and second != "centered":
        raise ValueError("Negative residual_coefficient is reserved for centered")
    if anchor != "g0":
        raise ValueError("This registered v3 design requires update_base=g0")
    name = direction_name(first, second, coefficient, anchor)
    for key in ("direction", "update_direction"):
        if key in config and config[key] != name:
            raise ValueError(key + " does not match explicit estimators")
    aliases = {"first_probe_sign": 1 if first == "forward" else -1}
    if second == "centered":
        if any(key in config for key in ("second_probe_sign", "second_difference_sign",
                                         "second_difference_weight")):
            raise ValueError("Centered estimator has no single-side legacy factors")
    else:
        aliases.update(second_probe_sign=1 if second == "forward" else -1,
                       second_difference_sign=1 if second == "forward" else -1,
                       second_difference_weight=1.)
    for key, expected in aliases.items():
        if key in config and (isinstance(config[key], bool) or config[key] != expected):
            raise ValueError(key + " conflicts with the explicit estimators")
    for key in ("branch", "probe_branch"):
        if key in config and config[key] != ("P" if first == "forward" else "C"):
            raise ValueError(key + " conflicts with first_location_estimator")
    return first, second, coefficient, anchor


def expected_budget(config):
    """Pure mathematical per-update budget; no additional diagnostic probes."""
    first, second, _, _ = validate_factors(config)
    fp, fm = int(first == "forward"), int(first == "backward_corrected")
    gp = int(second in ("forward", "centered"))
    gm = int(second in ("backward_corrected", "centered"))
    return dict(train_ce_per_update=2+fp+fm+gp+gm,
                diagnostic_ce_per_update=0,
                first_plus_ce_per_update=fp, first_minus_ce_per_update=fm,
                g3_plus_ce_per_update=gp, g3_minus_ce_per_update=gm,
                local_plus_ce_per_update=fp+gp, local_minus_ce_per_update=fm+gm)


class FourProbeStep:
    """Compute requested local probes; restore base; submit one SGD step."""

    def __init__(self, model, params, geometry, config):
        self.model = model
        self.params = tuple(params)
        self.geometry = geometry
        self.config = dict(config)
        self.rho = float(self.config["rho"])
        self.radius = float(self.config["radius"])
        self.eps = float(self.config.get("eps", 1e-12))
        self.first_location_estimator, self.second_residual_estimator, self.residual_coefficient, self.update_base = validate_factors(self.config)
        self.first_probe_sign = 1 if self.first_location_estimator == "forward" else -1
        self.second_probe_sign = (1 if self.second_residual_estimator == "forward" else
                                  -1 if self.second_residual_estimator == "backward_corrected" else None)
        self.second_difference_sign = (self.second_probe_sign if self.second_probe_sign is not None else None)
        self.branch = "P" if self.first_location_estimator == "forward" else "C"
        self.direction = direction_name(self.first_location_estimator, self.second_residual_estimator,
                                        self.residual_coefficient, self.update_base)
        self.budget = expected_budget(self.config)
        self.backend = self.config.get("backend", "native")
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
            if getattr(module, "external_delta_w_q", None) is not None or getattr(module, "external_delta_w_v", None) is not None:
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
                raise ValueError(label + " missing tensor or shape/device/dtype mismatch")
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
        return tuple(v + float(scale)*d for v, d in zip(base, direction))

    def _third_probe_audit(self, outer, third, g2, sign):
        displacement = _combine(third, outer, 1., -1.)
        length, norm2 = vector_norm(displacement), vector_norm(g2)
        expected = self.rho*norm2/(norm2+self.eps)
        alignment = vector_cosine(displacement, g2)
        # FP32 addition may round tiny component displacements. Audit the actual
        # tensors, using an explicit tolerance; never silently change rho.
        tolerance = max(1e-8, self.rho*1e-4)
        degenerate = expected <= 1e-24
        passed = (abs(length-expected) <= tolerance and
                  (degenerate or alignment is not None and
                   abs(alignment-sign) <= 1e-4))
        if not passed:
            raise AssertionError("Actual third point violates signed normalized probe")
        return dict(requested_sign=sign, parameter_length=length,
                    target_radius=self.rho, expected_parameter_length=expected,
                    alignment_cosine=alignment, length_error=abs(length-expected),
                    absolute_tolerance=tolerance, cosine_tolerance=1e-4,
                    degenerate=degenerate, passed=True)

    def compute(self, inputs, targets, loss_closure, current_lr):
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
            # Preserve the old FF/FB floating-point operation order exactly.
            first_point = self._point(state["base"], normalized(g0, self.eps),
                                      self.first_probe_sign*self.rho)
            first_audit = self._third_probe_audit(state["base"], first_point, g0,
                                                   self.first_probe_sign)
            l1, _, g1, _ = self._evaluate(first_point, inputs, targets, loss_closure, state)
            q = (_combine(g1, g0, 1., -1.) if self.first_probe_sign == 1 else
                 _combine(g0, g1, 1., -1.))
            outer = self._point(state["base"], normalized(q, self.eps), self.radius)
            l2, _, g2, _ = self._evaluate(outer, inputs, targets, loss_closure, state)
            unit2 = normalized(g2, self.eps)
            thirds, gradients3, losses3, audits3 = {}, {}, {}, {}
            signs = (1, -1) if self.second_residual_estimator == "centered" else (self.second_probe_sign,)
            for sign in signs:
                side = "plus" if sign == 1 else "minus"
                point = self._point(outer, unit2, sign*self.rho)
                audit = self._third_probe_audit(outer, point, g2, sign)
                loss, _, gradient, _ = self._evaluate(point, inputs, targets, loss_closure, state)
                thirds[side], gradients3[side], losses3[side], audits3[side] = point, gradient, loss, audit
            if state["gradient_calls"] != self.budget["train_ce_per_update"]:
                raise AssertionError("Ordinary training CE count violates explicit estimator budget")
            selected = "minus" if self.second_probe_sign == -1 else "plus"
            g3, l3, third_point = gradients3[selected], losses3[selected], thirds[selected]
            # delta32 retains its precise legacy meaning: selected g3 minus g2.
            delta32 = _combine(g3, g2, 1., -1.)
            centered_response, s2, e2 = None, None, None
            if self.second_residual_estimator == "centered":
                # Do not divide by rho: 0.5 times the two-readout difference.
                s2 = tuple(0.5*(plus-minus) for plus, minus in
                           zip(gradients3["plus"], gradients3["minus"]))
                e2 = tuple(0.5*(plus+minus-2.*middle) for plus, minus, middle in
                           zip(gradients3["plus"], gradients3["minus"], g2))
                residual = s2
                U = _combine(g0, residual, 1., self.residual_coefficient)
                corrected_plus = _combine(gradients3["plus"], g2, 1., -1.)
                corrected_minus = _combine(g2, gradients3["minus"], 1., -1.)
                centered_response = dict(s2_norm=vector_norm(s2), e2_norm=vector_norm(e2),
                    s2_e2_cosine=vector_cosine(s2, e2),
                    corrected_sides_cosine=vector_cosine(corrected_plus, corrected_minus),
                    e2_to_s2_norm_ratio=vector_norm(e2)/max(vector_norm(s2), self.eps),
                    formula="s2=(g3plus-g3minus)/2; e2=(g3plus+g3minus-2*g2)/2",
                    measured_from_training_context=True, additional_gradient_calls=0)
            else:
                residual = (delta32 if self.second_probe_sign == 1 else
                            tuple(-value for value in delta32))
                # In particular FB equals old U=g0-(g3minus-g2), bit for bit.
                U = _combine(g0, delta32, 1., self.second_difference_sign)
            self._validate(U, "final update direction")
            self._restore(state, state["rng0"])
            restoration = transaction.check()
            rng_set(rng1)
            for p, u in zip(self.params, U):
                p.grad = u.detach().clone()
            label = "g1_plus" if self.first_probe_sign == 1 else "g1_minus"
            norms = {k: vector_norm(v) for k, v in
                     {"g0": g0, label: g1, "g2": g2, "g3": g3, "q": q,
                      "delta32": delta32, "residual": residual, "U": U}.items()}
            norms.update({"g3_"+side: vector_norm(values) for side, values in gradients3.items()})
            if s2 is not None:
                norms.update(s2=vector_norm(s2), e2=vector_norm(e2))
            displacement_points = {"phi1": first_point, "phi2": outer, "phi3": third_point}
            displacement_points.update({"phi3_"+side: point for side, point in thirds.items()})
            record = dict(
                branch=self.branch, direction=self.direction, backend=self.backend, formula=self.direction,
                first_location_estimator=self.first_location_estimator,
                second_residual_estimator=self.second_residual_estimator,
                residual_coefficient=self.residual_coefficient,
                first_probe_sign=self.first_probe_sign, second_probe_sign=self.second_probe_sign,
                second_difference_sign=self.second_difference_sign, update_base=self.update_base,
                first_probe_parameter_audit=first_audit,
                first_location_parameter_audit=first_audit,
                first_location_formula_exact=True,
                first_location_formula="g1_plus-g0" if self.first_probe_sign == 1 else "g0-g1_minus",
                second_difference_weight=1., second_probe_parameter_audit=audits3[selected],
                second_probe_parameter_audits=audits3, centered_response=centered_response,
                second_residual_formula_exact=True,
                ordinary_gradient_calls=state["gradient_calls"], sgd_updates=0,
                fisher_calls=0, weight_merge_calls=0, gradient_projection_calls=0,
                g3_gradient_calls=len(signs), opposite_g1_gradient_calls=0, hvp_calls=0, create_graph_calls=0,
                first_positive_gradient_calls=int(self.first_probe_sign == 1),
                first_negative_gradient_calls=int(self.first_probe_sign == -1),
                g1_positive_gradient_calls=int(self.first_probe_sign == 1),
                g1_negative_gradient_calls=int(self.first_probe_sign == -1),
                g3_positive_gradient_calls=int("plus" in gradients3),
                g3_negative_gradient_calls=int("minus" in gradients3),
                local_positive_gradient_calls=self.budget["local_plus_ce_per_update"],
                local_negative_gradient_calls=self.budget["local_minus_ce_per_update"],
                additional_diagnostic_gradient_calls=0,
                rho=self.rho, radius=self.radius, eps=self.eps,
                loss0=l0, loss1=l1, first_gradient_label=label, loss2=l2, loss3=l3,
                loss3_positive=losses3.get("plus"), loss3_negative=losses3.get("minus"),
                delta_loss1=l1-l0, delta_loss2=l2-l0, delta_loss3=l3-l2,
                norms=norms,
                cosines={"delta32_g0": vector_cosine(delta32, g0), "delta32_g1": vector_cosine(delta32, g1),
                         "delta32_g2": vector_cosine(delta32, g2), "delta32_g3": vector_cosine(delta32, g3),
                         "residual_g0": vector_cosine(residual, g0), "residual_g1": vector_cosine(residual, g1),
                         "U_g0": vector_cosine(U, g0), "U_g1": vector_cosine(U, g1),
                         "U_g3": vector_cosine(U, g3)},
                projection=None,
                residual_to_base_norm_ratio=vector_norm(residual)/max(vector_norm(g0), self.eps),
                probe_displacements={name: self.geometry.actual_delta(
                    self._geometry_order(state["base"]), self._geometry_order(point))
                    for name, point in displacement_points.items()},
                base_parameters_exact=True, base_buffers_exact=True, state_restoration=restoration,
                forward_rng_rule="R0 replay for every probe; final RNG=R1 from g0")
            points = {"phi0": state["base"], "phi1": first_point, "phi2": outer, "phi3": third_point}
            points.update({"phi3_"+side: point for side, point in thirds.items()})
            points.update({"phi3"+side: point for side, point in thirds.items()})
            grads = {"g0": g0, "g1": g1, label: g1, "g2": g2, "g3": g3}
            grads.update({"g3_"+side: gradient for side, gradient in gradients3.items()})
            context = dict(points=points, grads=grads, delta32=delta32,
                           residual=residual, q=q, U=U, s2=s2, e2=e2,
                           rng0=state["rng0"], rng1=rng1)
            self._pending = dict(base=state["base"], U=U, lr=lr, transaction=transaction)
            return dict(loss0=l0, logits0=logits0, U=U, record=record, context=context)
        except BaseException:
            self._restore(state, state["rng0"])
            self._pending = None
            raise

    def diagnose_sides(self, inputs, targets, loss_closure, result, current_lr):
        """Report already-read training sides, without any forward or gradient.

        A single-side estimator has no opposite-side measurement.  This method
        deliberately cannot silently request one, so fully backward BB stays
        free of positive local readouts even when called by analysis code.
        """
        if self._pending is None:
            raise RuntimeError("Context diagnostics require a pending update")
        entry = capture_state(self.model)
        output = dict(diagnostic_gradient_calls=0, diagnostic_forward_count=0,
                      diagnostic_g3_positive_gradient_calls=0,
                      diagnostic_g3_negative_gradient_calls=0,
                      diagnostic_sgd_updates=0, fisher_calls=0, weight_merge_calls=0,
                      gradient_projection_calls=0, hvp_calls=0, create_graph_calls=0,
                      measurement_sources="existing_training_context_only",
                      extra_side_readout_disabled=True,
                      centered_response=result["record"]["centered_response"])
        output["state_restoration"] = entry.check()
        return output

    def discard(self):
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
