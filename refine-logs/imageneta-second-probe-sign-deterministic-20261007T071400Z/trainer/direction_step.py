"""Fixed positive P geometry, signed second probe, one natural SGD update."""
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


def direction_name(update_base, second_probe_sign, second_difference_sign):
    return "{}_probe2_{}_residual_{}".format(
        update_base, "plus" if second_probe_sign == 1 else "minus",
        "plus" if second_difference_sign == 1 else "minus")


def validate_factors(config):
    """Reject omitted or mismatched factors; no legacy fallback is permitted."""
    first = config["first_probe_sign"]
    probe, difference = config["second_probe_sign"], config["second_difference_sign"]
    anchor = config["update_base"]
    weight = config["second_difference_weight"]
    if type(first) is not int or first != 1:
        raise ValueError("first_probe_sign must be the integer +1")
    if any(type(v) is not int or v not in (-1, 1) for v in (probe, difference)):
        raise ValueError("Second probe and residual signs must be integers +/-1")
    if anchor not in ("g0", "g1") or isinstance(weight, bool) or float(weight) != 1.:
        raise ValueError("update_base=g0|g1 and second_difference_weight=1 required")
    if config["direction"] != direction_name(anchor, probe, difference):
        raise ValueError("update_direction does not match explicit factors")
    return first, probe, difference, anchor


class FourProbeStep:
    """Compute g0/g1/g2/g3; restore base; submit exactly one SGD step."""

    def __init__(self, model, params, geometry, config):
        self.model = model
        self.params = tuple(params)
        self.geometry = geometry
        self.config = dict(config)
        self.rho = float(self.config["rho"])
        self.radius = float(self.config["radius"])
        self.eps = float(self.config.get("eps", 1e-12))
        self.branch = self.config["branch"]
        self.direction = self.config["direction"]
        self.backend = self.config.get("backend", "native")
        if self.branch != "P":
            raise ValueError("The first and outer probes must retain positive P geometry")
        self.first_probe_sign, self.second_probe_sign, self.second_difference_sign, self.update_base = validate_factors(self.config)
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

    def _third_probe_audit(self, outer, third, g2):
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
                   abs(alignment-self.second_probe_sign) <= 1e-4))
        if not passed:
            raise AssertionError("Actual third point violates signed normalized probe")
        return dict(requested_sign=self.second_probe_sign, parameter_length=length,
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
            first_point = self._point(state["base"], normalized(g0, self.eps), self.rho)
            l1, _, g1, _ = self._evaluate(first_point, inputs, targets, loss_closure, state)
            # The outer location is constructed from the ORIGINAL base.
            q = _combine(g1, g0, 1., -1.)
            outer = self._point(state["base"], normalized(q, self.eps), self.radius)
            l2, _, g2, _ = self._evaluate(outer, inputs, targets, loss_closure, state)
            third_point = self._point(outer, normalized(g2, self.eps), self.second_probe_sign*self.rho)
            third_audit = self._third_probe_audit(outer, third_point, g2)
            l3, _, g3, _ = self._evaluate(third_point, inputs, targets, loss_closure, state)
            if state["gradient_calls"] != 4:
                raise AssertionError("Exactly four ordinary gradient probes required")
            delta32 = _combine(g3, g2, 1., -1.)
            anchor = g0 if self.update_base == "g0" else g1
            U = _combine(anchor, delta32, 1., self.second_difference_sign)
            self._validate(U, "final update direction")
            self._restore(state, state["rng0"])
            restoration = transaction.check()
            rng_set(rng1)
            for p, u in zip(self.params, U):
                p.grad = u.detach().clone()
            label = "g1_plus"
            record = dict(
                branch=self.branch, direction=self.direction, backend=self.backend,
                formula="{}{}(g3{}-g2)".format(self.update_base, "+" if self.second_difference_sign == 1 else "-", "+" if self.second_probe_sign == 1 else "-"),
                first_probe_sign=1, second_probe_sign=self.second_probe_sign,
                second_difference_sign=self.second_difference_sign, update_base=self.update_base,
                second_difference_weight=1., second_probe_parameter_audit=third_audit,
                second_residual_formula_exact=True,
                ordinary_gradient_calls=state["gradient_calls"], sgd_updates=0,
                fisher_calls=0, weight_merge_calls=0, gradient_projection_calls=0,
                g3_gradient_calls=1, opposite_g1_gradient_calls=0, hvp_calls=0, create_graph_calls=0,
                g3_positive_gradient_calls=int(self.second_probe_sign == 1),
                g3_negative_gradient_calls=int(self.second_probe_sign == -1),
                rho=self.rho, radius=self.radius, eps=self.eps,
                loss0=l0, loss1=l1, first_gradient_label=label, loss2=l2, loss3=l3,
                delta_loss1=l1-l0, delta_loss2=l2-l0, delta_loss3=l3-l2,
                norms={k: vector_norm(v) for k, v in {"g0": g0, label: g1, "g2": g2, "g3": g3, "q": q, "delta32": delta32, "U": U}.items()},
                cosines={"delta32_g0": vector_cosine(delta32, g0), "delta32_g1": vector_cosine(delta32, g1),
                         "delta32_g2": vector_cosine(delta32, g2), "delta32_g3": vector_cosine(delta32, g3),
                         "U_g0": vector_cosine(U, g0), "U_g1": vector_cosine(U, g1),
                         "U_g3": vector_cosine(U, g3)},
                projection=None,
                residual_to_base_norm_ratio=vector_norm(delta32)/max(vector_norm(anchor), self.eps),
                probe_displacements={name: self.geometry.actual_delta(
                    self._geometry_order(state["base"]), self._geometry_order(point))
                    for name, point in {"phi1": first_point, "phi2": outer, "phi3": third_point}.items()},
                base_parameters_exact=True, base_buffers_exact=True, state_restoration=restoration,
                forward_rng_rule="R0 replay for every probe; final RNG=R1 from g0")
            context = dict(points={"phi0": state["base"], "phi1": first_point, "phi2": outer, "phi3": third_point},
                           grads={"g0": g0, "g1": g1, label: g1, "g2": g2, "g3": g3},
                           delta32=delta32, q=q, U=U, rng0=state["rng0"], rng1=rng1)
            self._pending = dict(base=state["base"], U=U, lr=lr)
            return dict(loss0=l0, logits0=logits0, U=U, record=record, context=context)
        except BaseException:
            self._restore(state, state["rng0"])
            self._pending = None
            raise

    def diagnose_sides(self, inputs, targets, loss_closure, result, current_lr):
        """One opposite-side gradient at the same pre-update snapshot.

        Reuse the training g0/g1/g2 AND the selected g3, and measure only the
        opposite third point. GPU backward reductions need not be bitwise
        reproducible when repeated; no same-side recomputation is performed.
        Restore the pending U gradients, base, buffers, caches, modes, and
        final R1 exactly, even if the isolated opposite-side probe raises.
        """
        if self._pending is None:
            raise RuntimeError("Side diagnostics require a pending computed update")
        entry = capture_state(self.model)
        context = result["context"]
        state = dict(transaction=entry, rng0=context["rng0"], gradient_calls=0)
        outer, g2 = context["points"]["phi2"], context["grads"]["g2"]
        selected_name = "plus" if self.second_probe_sign == 1 else "minus"
        opposite_sign = -self.second_probe_sign
        opposite_name = "plus" if opposite_sign == 1 else "minus"
        # Clone the already-computed training difference, rather than replacing
        # it with a re-evaluated gradient or changing the pending training U.
        selected_difference = _clone(context["delta32"])
        sides = {selected_name: dict(loss=result["record"]["loss3"], delta=selected_difference)}
        try:
            point = self._point(outer, normalized(g2, self.eps), opposite_sign*self.rho)
            loss, _, grad, _ = self._evaluate(point, inputs, targets, loss_closure, state)
            sides[opposite_name] = dict(loss=loss, delta=_combine(grad, g2, 1., -1.))
            plus, minus = sides["plus"]["delta"], sides["minus"]["delta"]
            corrected_minus = tuple(-v for v in minus)
            candidates = {}
            for base_label in ("g0", "g1"):
                anchor = context["grads"][base_label]
                for name, delta in (("plus", plus), ("minus_corrected", corrected_minus)):
                    candidate = _combine(anchor, delta)
                    after = tuple(v.add(u, alpha=-float(current_lr)) for v, u in zip(context["points"]["phi0"], candidate))
                    candidates[base_label+"_"+name] = self.geometry.actual_delta(
                        self._geometry_order(context["points"]["phi0"]), self._geometry_order(after))
            selected = sides[selected_name]["delta"]
            if not all(torch.equal(v, w) for v, w in zip(selected, context["delta32"])):
                raise AssertionError("Training selected-side clone must remain exact")
            output = dict(diagnostic_gradient_calls=state["gradient_calls"], diagnostic_forward_count=1,
                          diagnostic_g3_positive_gradient_calls=int(opposite_sign == 1),
                          diagnostic_g3_negative_gradient_calls=int(opposite_sign == -1),
                          training_selected_probe_sign=self.second_probe_sign,
                          diagnostic_opposite_probe_sign=opposite_sign,
                          selected_side_reused_from_training=True,
                          selected_side_gradient_source="training_context_clone",
                          selected_side_new_gradient_calls=0,
                          opposite_side_new_gradient_calls=1,
                          measurement_sources={selected_name:"training_g3", opposite_name:"isolated_opposite_probe"},
                          selected_side_reference_exact=True,
                          selected_side_exact_kind="training_reference_clone_equality_not_independent_replay",
                          diagnostic_requested_side_gradient_calls=0,
                          diagnostic_opposite_side_gradient_calls=1,
                          diagnostic_sgd_updates=0, fisher_calls=0, weight_merge_calls=0,
                          gradient_projection_calls=0, hvp_calls=0, create_graph_calls=0,
                          cos_positive_negative_corrected=vector_cosine(plus, corrected_minus),
                          positive_negative_norm_ratio=vector_norm(plus)/max(vector_norm(minus), self.eps),
                          positive_norm=vector_norm(plus), negative_norm=vector_norm(minus),
                          loss3_positive=sides["plus"]["loss"], loss3_negative=sides["minus"]["loss"],
                          delta_loss3_positive=sides["plus"]["loss"]-result["record"]["loss2"],
                          delta_loss3_negative=sides["minus"]["loss"]-result["record"]["loss2"],
                          candidate_effective_steps=candidates, selected_side_exact=True)
        finally:
            entry.restore()
            restored = entry.check()
        output["state_restoration"] = restored
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
            self._set_point(base)
            self._pending = None
            raise
