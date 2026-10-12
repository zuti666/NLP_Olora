"""Independent four-probe, pure-CE direction interventions.

Every ordinary arm evaluates exactly g0, g1+, g1-, and the selected g2.
Probes are temporary exact snapshot displacements; no GAM projector, Fisher,
weight merge, optimizer momentum, or additional gradient normalization enters
the final direction.  The one actual update is made by ``apply_update``.
"""
import math
import torch

from audit_core import rng_get, rng_set
from effective_step import solve_positive_scale
from response_hvp import attention_backend, capture_state


def _clone(values):
    return tuple(v.detach().clone() for v in values)


def _combine(a, b, a_scale=1.0, b_scale=1.0):
    return tuple(a_scale*x + b_scale*y for x,y in zip(a,b))


def _inner(a, b):
    result = sum((x.detach().double()*y.detach().double()).sum() for x,y in zip(a,b))
    value = float(result.item())
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite direction inner product")
    return value


def vector_norm(values):
    return math.sqrt(max(0.0, _inner(values, values)))


def vector_cosine(a, b):
    denominator = vector_norm(a)*vector_norm(b)
    return _inner(a,b)/denominator if denominator > 1e-24 else None


def normalized(values, eps=1e-12):
    denominator = vector_norm(values) + float(eps)
    return tuple(v/denominator for v in values)


class FourProbeStep:
    """One clean four-probe calculation plus one optional exact SGD update.

    ``loss_closure(inputs, targets)`` returns ``(scalar_loss, logits)`` and must
    not call backward or step.  ``params`` are all trainable model parameters;
    geometry is an EffectiveStep covering exactly those parameters, in any
    order.  ``compute`` returns loss0/logits0/U/record/context and writes U to
    parameter.grad.  The context is transient tensor state, not a JSON record.

    An optional observer(context) is called after all four probes, at restored
    base parameters and RNG R0.  It may return None, U, or (U, JSON-report).
    This gives separately audited HVP code an explicit detached-vector hook.
    """

    def __init__(self, model, params, geometry, config):
        self.model = model
        self.params = tuple(params)
        self.geometry = geometry
        self.config = dict(config)
        self.rho = float(self.config.get("rho", .05))
        self.radius = float(self.config.get("radius", .25))
        self.eps = float(self.config.get("eps", 1e-12))
        self.branch = self.config.get("branch", "P")
        self.direction = self.config.get("direction", "A")
        self.backend = self.config.get("backend", "native")
        if self.branch not in ("P", "D", "C", "R"):
            raise ValueError("Unknown probe branch: " + str(self.branch))
        if self.direction not in ("g0", "g1_plus", "g1_minus", "g2", "A",
                                   "residual_only", "hstart", "hmid"):
            raise ValueError("Unknown final direction: " + str(self.direction))
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
            if bool(getattr(module,"_register_delta_hook",False)):
                raise ValueError("Pure direction training must disable delta-gradient hooks")
            if getattr(module,"external_delta_w_q",None) is not None or getattr(module,"external_delta_w_v",None) is not None:
                raise ValueError("Pure direction training cannot use external delta weights")
        self._geo_index = [next(i for i,p in enumerate(self.params) if p is q)
                           for q in geometry.params]
        self._pending = None

    def _geometry_order(self, values):
        return tuple(values[i] for i in self._geo_index)

    def _validate(self, values, label):
        if len(values) != len(self.params):
            raise ValueError(label + " tensor count mismatch")
        for p,v in zip(self.params,values):
            if v is None or v.shape != p.shape or v.device != p.device:
                raise ValueError(label + " missing tensor or shape/device mismatch")
        if not bool(torch.stack([torch.isfinite(v).all() for v in values]).all()):
            raise FloatingPointError(label + " has a nonfinite value")

    @torch.no_grad()
    def _set_point(self, values):
        for p,v in zip(self.params,values):
            p.copy_(v)

    @torch.no_grad()
    def _restore(self, state, rng):
        # The HVP and ordinary probes share one implementation for registered
        # buffers, modes, all gradients, LoRA nonregistered controls/caches,
        # frozen-parameter immutability and RNG transactions.
        state["transaction"].restore()
        rng_set(rng)

    def _evaluate(self, values, inputs, targets, closure, state):
        self._restore(state, state["rng0"])
        self._set_point(values)
        with torch.enable_grad(), attention_backend(self.model,self.backend):
            loss, logits = closure(inputs,targets)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("Closure loss must be finite scalar")
            grads = torch.autograd.grad(loss, self.params, create_graph=False,
                                        retain_graph=False, allow_unused=False)
        self._validate(grads, "ordinary CE gradient")
        # RNG R1 is always from g0, not from the final probe or an observer.
        return float(loss.detach().item()), logits.detach(), _clone(grads), rng_get()

    @torch.no_grad()
    def _point(self, base, direction, scale):
        # This is the actual rounded parameter point which will be evaluated.
        return tuple(v + float(scale)*d for v,d in zip(base,direction))

    def compute(self, inputs, targets, loss_closure, current_lr, observer=None):
        lr = float(current_lr)
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("current_lr must be finite and positive")
        if self._pending is not None:
            raise RuntimeError("Previous direction must be applied or discarded first")
        transaction = capture_state(self.model)
        value_map = {id(p):v for p,v in transaction.trainable}
        grad_map = {id(p):g for p,g in transaction.grads}
        old_grads = tuple(grad_map[id(p)] for p in self.params)
        state = {"base":tuple(value_map[id(p)] for p in self.params),
                 "buffers":transaction.buffers,"modes":transaction.modes,
                 "rng0":transaction.rng,"transaction":transaction}
        rng1 = None
        try:
            l0, logits0, g0, rng1 = self._evaluate(state["base"],inputs,targets,loss_closure,state)
            n0 = normalized(g0,self.eps)
            plus = self._point(state["base"],n0,self.rho)
            minus = self._point(state["base"],n0,-self.rho)
            lp, _, gp, _ = self._evaluate(plus,inputs,targets,loss_closure,state)
            lm, _, gm, _ = self._evaluate(minus,inputs,targets,loss_closure,state)
            cplus = _combine(gp,g0,1,-1)
            cminus = _combine(gm,g0,1,-1)
            first = gp if self.branch in ("P","R") else gm
            q = cplus if self.branch in ("P","R") else cminus
            if self.branch in ("C","R"):
                q = tuple(-v for v in q)
            outer = self._point(state["base"],normalized(q,self.eps),self.radius)
            l2, _, g2, _ = self._evaluate(outer,inputs,targets,loss_closure,state)
            c21 = _combine(g2,first,1,-1)
            directions = {"g0":g0,"g1_plus":gp,"g1_minus":gm,"g2":g2,
                          "A":_combine(g0,c21),"residual_only":c21}
            if self.direction in ("hstart","hmid") and observer is None:
                raise ValueError("HVP direction requires an audited observer")
            U = directions.get(self.direction, directions["A"])
            chosen_first = plus if self.branch in ("P","R") else minus
            context = {"model":self.model,"params":self.params,"geometry":self.geometry,
                       "points":{"phi0":state["base"],"phi1_plus":plus,"phi1_minus":minus,
                                 "phi1":chosen_first,"phi2":outer},
                       "grads":{"g0":g0,"g1_plus":gp,"g1_minus":gm,"g1":first,"g2":g2},
                       "c10":_combine(first,g0,1,-1),"c21":c21,"q":q,
                       "d12":_combine(outer,chosen_first,1,-1),"U":U,
                       "rng_state":state["rng0"],"rng0":state["rng0"],"rng1":rng1,
                       "backend":self.backend,"config":dict(self.config),
                       "inputs":inputs,"targets":targets,"loss_closure":loss_closure,
                       "closure":lambda: loss_closure(inputs,targets)[0]}
            extra = {}
            if observer is not None:
                self._restore(state,state["rng0"])
                result = observer(context)
                if result is not None:
                    if isinstance(result,tuple) and len(result)==2 and isinstance(result[1],dict):
                        U, extra = result
                    else:
                        U = result
                    U = _clone(U)
                    context["U"] = U
            self._validate(U,"final update direction")
            self._restore(state,rng1)
            if not all(torch.equal(p,v) for p,v in zip(self.params,state["base"])):
                raise AssertionError("Base parameters did not restore exactly")
            if not all(torch.equal(b,v) for b,v in state["buffers"]):
                raise AssertionError("Base buffers did not restore exactly")
            # Everything except RNG is back at the entry state. Temporarily
            # restore R0 for the shared exact state check, then advance to R1.
            rng_set(state["rng0"])
            restoration = transaction.check()
            rng_set(rng1)
            for p,u in zip(self.params,U):
                p.grad = u.detach().clone()
            record = {"branch":self.branch,"direction":self.direction,"backend":self.backend,
                      "ordinary_gradient_calls":4,"sgd_updates":0,
                      "fisher_calls":0,"weight_merge_calls":0,"gradient_projection_calls":0,
                      "rho":self.rho,"radius":self.radius,"eps":self.eps,
                      "loss0":l0,"loss1_plus":lp,"loss1_minus":lm,"loss2":l2,
                      "delta_loss1_plus":lp-l0,"delta_loss1_minus":lm-l0,
                      "delta_gradient_norm2":vector_norm(g2)-vector_norm(g0),
                      "norms":{k:vector_norm(v) for k,v in
                               {"g0":g0,"g1_plus":gp,"g1_minus":gm,"g2":g2,
                                "c10_plus":cplus,"c10_minus":cminus,"q":q,"c21":c21,"U":U}.items()},
                      "cosines":{"qP_qC":vector_cosine(cplus,tuple(-x for x in cminus)),
                                 "qD_qR":vector_cosine(cminus,tuple(-x for x in cplus)),
                                 "c21_g0":vector_cosine(c21,g0),"c21_g1":vector_cosine(c21,first),
                                 "U_g0":vector_cosine(U,g0),"U_g1_plus":vector_cosine(U,gp)},
                      "probe_displacements":{
                          name:self.geometry.actual_delta(self._geometry_order(state["base"]),
                                                          self._geometry_order(point))
                          for name,point in {"phi1_plus":plus,"phi1_minus":minus,"phi2":outer}.items()},
                      "base_parameters_exact":True,"base_buffers_exact":True,
                      "state_restoration":restoration,
                      "forward_rng_rule":"R0 replay for every probe; final RNG=R1 from g0",
                      "observer":extra}
            self._pending = {"base":state["base"],"g1_plus":gp,"U":U,"lr":lr}
            return {"loss0":l0,"logits0":logits0,"U":U,"record":record,"context":context}
        except BaseException:
            self._restore(state,state["rng0"])
            for p,g in zip(self.params,old_grads):
                p.grad = None if g is None else g.clone()
            self._pending = None
            raise

    def discard(self):
        """Drop a calculated direction without making an update."""
        self._pending = None

    @torch.no_grad()
    def apply_update(self, current_lr=None, matched=False, optimizer=None):
        """Submit one SGD update, with optional positive-g1 effective matching.

        Reference target and final verification are measured from actual rounded
        parameter values, using FP64 BA Gram products.  Match failures restore
        the base and raise; there is no silent nominal-learning-rate fallback.
        """
        if self._pending is None:
            raise RuntimeError("compute must precede apply_update")
        pending = self._pending
        base,U,gp = pending["base"],pending["U"],pending["g1_plus"]
        lr = pending["lr"] if current_lr is None else float(current_lr)
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("SGD learning rate must be positive")
        if not all(torch.equal(p,v) for p,v in zip(self.params,base)):
            raise RuntimeError("Base changed between compute and actual update")
        geo_base = self._geometry_order(base)
        saved_lrs = None
        if optimizer is not None:
            if not isinstance(optimizer,torch.optim.SGD):
                raise ValueError("Only ordinary torch.optim.SGD is supported")
            groups = optimizer.param_groups
            if {id(p) for g in groups for p in g["params"]} != {id(p) for p in self.params}:
                raise ValueError("Actual SGD must cover exactly every trainable parameter")
            if any(float(g.get("momentum",0)) != 0 or float(g.get("weight_decay",0)) != 0
                   or bool(g.get("nesterov",False)) or bool(g.get("maximize",False))
                   for g in groups):
                raise ValueError("Actual SGD must have no momentum, weight decay, or maximize")
            saved_lrs = [g["lr"] for g in groups]
            if any(abs(float(v)-lr) > 1e-12*max(abs(lr),1.0) for v in saved_lrs):
                raise ValueError("All SGD groups must share the declared nominal LR")
        try:
            for p,v,g in zip(self.params,base,gp):
                p.copy_(v)
                p.add_(g,alpha=-lr)
            reference = self.geometry.actual_delta(geo_base)
            self._set_point(base)
            target = reference["joint_norm"]
            natural_poly = self.geometry.polynomial(self._geometry_order(U),geo_base)
            g1_identity = self.direction == "g1_plus" and all(torch.equal(a,b) for a,b in zip(U,gp))
            solve = {"matched":False,"scale":lr}
            scale = lr
            if matched:
                if target <= 0:
                    raise ValueError("Cannot match zero effective g1 reference target")
                if g1_identity:
                    solve = {"matched":True,"scale":lr,"identity":True,
                             "target":target,"relative_error":0.0}
                else:
                    solve = solve_positive_scale(natural_poly,target,initial_scale=lr,
                        max_scale=float(self.config.get("match_max_scale",10.0)),
                        rtol=1e-10,max_iterations=90)
                    scale = solve["scale"]
            if optimizer is None:
                for p,u in zip(self.params,U):
                    p.add_(u,alpha=-scale)
            else:
                for p,u in zip(self.params,U):
                    p.grad = u.detach().clone()
                for group in optimizer.param_groups:
                    group["lr"] = scale
                optimizer.step()
            actual = self.geometry.actual_delta(geo_base)
            error = abs(actual["joint_norm"]-target)/target if target > 0 else None
            tolerance = float(self.config.get("match_actual_rtol",1e-3))
            if matched and (error is None or error > tolerance):
                raise ArithmeticError("Rounded FP32 effective matching failed: " + repr(error))
            self._pending = None
            return {"matched":bool(matched),"nominal_lr":lr,"scale":scale,"step_scale":scale,
                    "scale_over_lr":scale/lr,"reference":reference,"actual":actual,
                    "actual_relative_error":error,"actual_match_relative_error":error,
                    "tolerance":tolerance,"g1_identity":g1_identity,
                    "root":solve,"sgd_updates":1,
                    "natural_predicted_norm":natural_poly.norm(lr),
                    "effective_norm_over_g1_reference":actual["joint_norm"]/target if target>0 else None}
        except BaseException:
            self._set_point(base)
            self._pending = None
            raise
        finally:
            if saved_lrs is not None:
                for group,saved_lr in zip(optimizer.param_groups,saved_lrs):
                    group["lr"] = saved_lr
