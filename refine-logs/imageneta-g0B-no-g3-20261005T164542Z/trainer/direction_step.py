"""Three ordinary gradient probes and one natural SGD update.

P/g0_plus_B uses g0 + g2 - alpha*g1+, with one global FP64 projection.
C/A uses g0 + g2 - g1-. No unused opposite probe or g3 is evaluated.
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


class ThreeProbeStep:
    """Compute g0, one selected g1 and g2; restore; submit one SGD step."""

    def __init__(self, model, params, geometry, config):
        self.model = model
        self.params = tuple(params)
        self.geometry = geometry
        self.config = dict(config)
        self.rho = float(self.config["rho"])
        self.radius = float(self.config["radius"])
        self.eps = float(self.config.get("eps", 1e-12))
        self.projection_eps = 1e-12
        self.branch = self.config["branch"]
        self.direction = self.config["direction"]
        self.backend = self.config.get("backend", "native")
        if (self.branch, self.direction) not in (("P", "g0_plus_B"), ("C", "A")):
            raise ValueError("Only P/g0_plus_B and C/A are supported")
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
            sign = 1.0 if self.branch == "P" else -1.0
            first_point = self._point(state["base"], normalized(g0, self.eps), sign*self.rho)
            l1, _, g1, _ = self._evaluate(first_point, inputs, targets, loss_closure, state)
            # Both branches construct the outer point from the ORIGINAL base.
            q = _combine(g1, g0, sign, -sign)
            outer = self._point(state["base"], normalized(q, self.eps), self.radius)
            l2, _, g2, _ = self._evaluate(outer, inputs, targets, loss_closure, state)
            if state["gradient_calls"] != 3:
                raise AssertionError("Exactly three ordinary gradient probes required")
            c21 = _combine(g2, g1, 1, -1)
            projection = None
            if self.direction == "g0_plus_B":
                numerator = _inner(g1, c21)
                denominator = _inner(g1, g1)
                alpha = numerator/(denominator+self.projection_eps)
                if not math.isfinite(alpha):
                    raise FloatingPointError("Nonfinite global projection coefficient")
                perp = _combine(c21, g1, 1, -alpha)
                U = tuple(a+b-alpha*c for a, b, c in zip(g0, g2, g1))
                orthogonality_dot = _inner(g1, perp)
                expected_dot = numerator*self.projection_eps/(denominator+self.projection_eps)
                scale = vector_norm(g1)*vector_norm(c21)
                projection = dict(
                    scope="all_trainable_LoRA_and_seen_FC", reduction_dtype="float64",
                    tensor_count=len(self.params), parameter_count=sum(p.numel() for p in self.params),
                    alpha=alpha, dot_g1_c21=numerator, norm_g1_squared=denominator,
                    denominator_epsilon=self.projection_eps, dot_g1_perp=orthogonality_dot,
                    expected_dot_from_epsilon=expected_dot,
                    orthogonality_relative_error=abs(orthogonality_dot)/scale if scale > 0 else 0.,
                    projection_relative_error=abs(orthogonality_dot-expected_dot)/scale if scale > 0 else 0.,
                    norm_perp=vector_norm(perp), norm_parallel=abs(alpha)*vector_norm(g1),
                    g1_zero=denominator == 0.)
                if projection["projection_relative_error"] > 1e-5:
                    raise AssertionError("Global projection consistency failed")
            else:
                # Preserve the historical A_C floating-point operation order.
                U = _combine(g0, c21)
            self._validate(U, "final update direction")
            self._restore(state, state["rng0"])
            restoration = transaction.check()
            rng_set(rng1)
            for p, u in zip(self.params, U):
                p.grad = u.detach().clone()
            label = "g1_plus" if self.branch == "P" else "g1_minus"
            record = dict(
                branch=self.branch, direction=self.direction, backend=self.backend,
                formula="g0+g2-alpha*g1_plus" if self.direction == "g0_plus_B" else "g0+g2-g1_minus",
                ordinary_gradient_calls=state["gradient_calls"], sgd_updates=0,
                fisher_calls=0, weight_merge_calls=0, gradient_projection_calls=int(projection is not None),
                g3_gradient_calls=0, opposite_g1_gradient_calls=0, hvp_calls=0, create_graph_calls=0,
                rho=self.rho, radius=self.radius, eps=self.eps,
                loss0=l0, loss1=l1, first_gradient_label=label, loss2=l2,
                delta_loss1=l1-l0, delta_loss2=l2-l0,
                norms={k: vector_norm(v) for k, v in {"g0": g0, label: g1, "g2": g2, "q": q, "c21": c21, "U": U}.items()},
                cosines={"c21_g0": vector_cosine(c21, g0), "c21_g1": vector_cosine(c21, g1),
                         "U_g0": vector_cosine(U, g0), "U_g1": vector_cosine(U, g1)},
                projection=projection,
                probe_displacements={name: self.geometry.actual_delta(
                    self._geometry_order(state["base"]), self._geometry_order(point))
                    for name, point in {"phi1": first_point, "phi2": outer}.items()},
                base_parameters_exact=True, base_buffers_exact=True, state_restoration=restoration,
                forward_rng_rule="R0 replay for every probe; final RNG=R1 from g0")
            context = dict(points={"phi0": state["base"], "phi1": first_point, "phi2": outer},
                           grads={"g0": g0, "g1": g1, label: g1, "g2": g2},
                           c21=c21, q=q, U=U, rng0=state["rng0"], rng1=rng1)
            self._pending = dict(base=state["base"], U=U, lr=lr)
            return dict(loss0=l0, logits0=logits0, U=U, record=record, context=context)
        except BaseException:
            self._restore(state, state["rng0"])
            self._pending = None
            raise

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
