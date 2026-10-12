"""Effective unscaled LoRA BA + FC step geometry, without dense BA products.

Snapshots contain actual factor values. All norm measurements use float64
low-rank Gram products; the training tensors and native GAM path are untouched.
Call capture/polynomial AFTER native unperturb and BEFORE zero-momentum SGD.
"""
from dataclasses import dataclass
import math
import torch


def _finite_scalar(x, label):
    x = float(x)
    if not math.isfinite(x):
        raise FloatingPointError(label + " is nonfinite")
    return x


def _inner(x, y, u, v):
    # <XY,UV> = sum((X^T U) * (Y V^T)); no out-by-in tensor.
    return ((x.T @ u) * (y @ v.T)).sum()


def _nonnegative(x, scale, label):
    x, scale = _finite_scalar(x, label), _finite_scalar(scale, label + " scale")
    if x < -1e-11 * max(abs(scale), 1e-300):
        raise FloatingPointError(label + " is materially negative: " + repr(x))
    return max(0.0, x)


@dataclass(frozen=True)
class StepPolynomial:
    c2: float
    c3: float
    c4: float
    fc2: float = 0.0

    def __post_init__(self):
        for key in ("c2", "c3", "c4", "fc2"):
            _finite_scalar(getattr(self, key), key)
        if self.c2 < 0 or self.c4 < 0 or self.fc2 < 0:
            raise ValueError("Norm polynomial has negative quadratic/quartic coefficient")

    def squared(self, scale):
        s = _finite_scalar(scale, "step scale")
        if s < 0:
            raise ValueError("Step scale must be nonnegative")
        terms = (s*s*self.c2, s*s*s*self.c3, s*s*s*s*self.c4)
        return _nonnegative(sum(terms), sum(abs(t) for t in terms), "effective squared norm")

    def norm(self, scale):
        return math.sqrt(self.squared(scale))

    def as_dict(self):
        return {"c2": self.c2, "c3": self.c3, "c4": self.c4, "fc2": self.fc2}


def solve_positive_scale(poly, target, initial_scale=0.04, max_scale=10.0,
                         rtol=1e-10, max_iterations=90):
    """Smallest positive root in (0,max_scale], with no monotonicity assumption.

Split the quartic at ALL positive stationary points, then bisect the first
interval that reaches target. No bracket/zero direction/nonfinite inputs fail.
initial_scale is validated and reported; it never truncates the root search.
"""
    target = _finite_scalar(target, "target effective norm")
    maximum = _finite_scalar(max_scale, "max_scale")
    initial = _finite_scalar(initial_scale, "initial_scale")
    if target <= 0 or maximum <= 0 or initial <= 0 or rtol <= 0:
        raise ValueError("Target, bounds, initial scale, and tolerance must be positive")
    if poly.c2 == 0 and poly.c4 == 0:
        raise ValueError("Cannot match positive target with zero effective direction")
    critical = []
    # f'(s)/s = 2*c2 + 3*c3*s + 4*c4*s^2.
    if poly.c4 > 0:
        aa, bb, cc = 4*poly.c4, 3*poly.c3, 2*poly.c2
        discriminant = bb*bb - 4*aa*cc
        if discriminant >= 0:
            root_d = math.sqrt(discriminant)
            q = -.5 * (bb + math.copysign(root_d, bb))
            roots = (q / aa, cc / q) if q != 0 else (0.0,)
            critical = sorted(set(x for x in roots if math.isfinite(x) and 0 < x < maximum))
    elif poly.c3 != 0:
        x = -2*poly.c2/(3*poly.c3)
        if 0 < x < maximum:
            critical = [x]
    cuts = [0.0] + critical + [maximum]
    target_squared = target*target
    bracket = None
    for lo, hi in zip(cuts, cuts[1:]):
        f_lo, f_hi = poly.squared(lo)-target_squared, poly.squared(hi)-target_squared
        if f_lo == 0 and lo > 0:
            bracket = (lo, lo)
            break
        if f_lo <= 0 <= f_hi:
            bracket = (lo, hi)
            break
    if bracket is None:
        raise ValueError("No positive effective-step root within explicit max_scale=" + str(maximum))
    lo, hi = bracket
    iterations = 0
    for iterations in range(1, max_iterations+1):
        mid = (lo+hi)/2
        value = poly.norm(mid)
        if abs(value-target)/target <= rtol:
            lo = hi = mid
            break
        if value < target:
            lo = mid
        else:
            hi = mid
    result = (lo+hi)/2
    error = abs(poly.norm(result)-target)/target
    if error > max(rtol*2, 1e-9):
        raise ArithmeticError("Effective root did not converge: " + str(error))
    return {"scale": result, "target": target, "predicted_norm": poly.norm(result),
            "relative_error": error, "iterations": iterations, "bracket": list(bracket),
            "stationary_points": critical, "smallest_positive_root": True,
            "max_scale": maximum, "initial_scale": initial, "matched": True}


class EffectiveStep:
    """Factor parameter geometry. Pairs are (A_parameter,B_parameter)."""
    def __init__(self, pairs, fc_parameters, all_trainable=None):
        self.pairs = list(pairs)
        self.fc_parameters = list(fc_parameters)
        self.params = [p for a,b in self.pairs for p in (a,b)] + self.fc_parameters
        if not self.pairs or not self.fc_parameters:
            raise ValueError("Expected LoRA factors and full FC parameters")
        if len({id(p) for p in self.params}) != len(self.params):
            raise ValueError("Duplicate effective-map parameter")
        for a,b in self.pairs:
            if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
                raise ValueError("Invalid BA factor dimensions")
        if all_trainable is not None and {id(p) for p in self.params} != {id(p) for p in all_trainable}:
            raise ValueError("Effective map must cover every trainable exactly: all LoRA A/B and FC")

    @classmethod
    def from_model(cls, model):
        if hasattr(model, "module"):
            model = model.module
        backbone = model.backbone
        if bool(getattr(backbone, "learn_alpha", False)):
            raise ValueError("Only unscaled LoRA BA is supported")
        scales = list(getattr(backbone,"wrapped_param",[]))
        if scales and any(type(module).__name__ != "IdentityScale" for module in scales):
            raise ValueError("All frozen LoRA wrappers must use IdentityScale")
        for name,module in backbone.named_modules():
            if "scale" in type(module).__name__.lower() and type(module).__name__ != "IdentityScale":
                # Source uses IdentityScale; never silently ignore trainable/nonunit scaling.
                if any(True for _ in module.parameters()):
                    raise ValueError("Unsupported LoRA scaling: " + name)
        pairs = [(a.weight,b.weight) for a,b in zip(backbone.w_As,backbone.w_Bs)]
        if len(backbone.w_As) != len(backbone.w_Bs):
            raise ValueError("Mismatched A/B lists")
        return cls(pairs, [p for p in model.fc.parameters() if p.requires_grad],
                   [p for p in model.parameters() if p.requires_grad])

    @torch.no_grad()
    def capture(self):
        return tuple(p.detach().clone() for p in self.params)

    def _validate(self, values):
        if len(values) != len(self.params):
            raise ValueError("Snapshot/direction tensor count mismatch")
        flags = []
        for p,value in zip(self.params,values):
            if value is None or value.shape != p.shape:
                raise ValueError("Missing gradient or invalid tensor shape")
            flags.append(torch.isfinite(value).all())
        if not bool(torch.stack(flags).all()):
            raise FloatingPointError("Nonfinite effective-map factors/direction")

    @torch.no_grad()
    def polynomial(self, grads=None, snapshot=None):
        """For theta -> theta-s*grads, return c2*s²+c3*s³+c4*s⁴.

        Optional grads/snapshot are in self.params order; default is current
        actual p.grad/current p. This is algebra before floating-point SGD.
        """
        values = list(self.params) if snapshot is None else list(snapshot)
        directions = [p.grad for p in self.params] if grads is None else list(grads)
        self._validate(values)
        self._validate(directions)
        terms = []
        for i in range(len(self.pairs)):
            a,b = (v.detach().double() for v in values[2*i:2*i+2])
            da,db = (v.detach().double() for v in directions[2*i:2*i+2])
            l2 = _inner(b,da,b,da) + _inner(db,a,db,a) + 2*_inner(b,da,db,a)
            lq = _inner(b,da,db,da) + _inner(db,a,db,da)
            q2 = _inner(db,da,db,da)
            terms.append(torch.stack((l2,-2*lq,q2)))
        coefficients = torch.stack(terms).sum(0)
        fc2 = sum(v.detach().double().square().sum() for v in directions[2*len(self.pairs):])
        c2,c3,c4,fc = torch.cat((coefficients,fc2.reshape(1))).cpu().tolist()
        c2 = _nonnegative(c2,abs(c2),"LoRA linear norm") + fc
        c4 = _nonnegative(c4,abs(c4),"LoRA quadratic norm")
        return StepPolynomial(c2,c3,c4,fc)

    @torch.no_grad()
    def actual_delta(self, before, after=None):
        """Actual tensor update, evaluated as ΔBA=B_old ΔA+ΔB A_new.

        FP64 Gram calculation equals explicit FP64 BA_after-BA_before. It does
        not use polynomial predictions or rounded FP32 dense matrix products.
        """
        after = list(self.params) if after is None else list(after)
        self._validate(before)
        self._validate(after)
        lora_terms = []
        for i in range(len(self.pairs)):
            a0,b0 = (v.detach().double() for v in before[2*i:2*i+2])
            a1,b1 = (v.detach().double() for v in after[2*i:2*i+2])
            da,db = a1-a0,b1-b0
            t1,t2,cross = _inner(b0,da,b0,da),_inner(db,a1,db,a1),_inner(b0,da,db,a1)
            lora_terms.append(torch.stack((t1+t2+2*cross,t1+t2+2*cross.abs())))
        ls, bound = torch.stack(lora_terms).sum(0).cpu().tolist()
        ls = _nonnegative(ls,bound,"actual LoRA squared delta")
        fc = sum((v.detach().double()-u.detach().double()).square().sum()
                 for u,v in zip(before[2*len(self.pairs):],after[2*len(self.pairs):]))
        raw = sum((v.detach().double()-u.detach().double()).square().sum() for u,v in zip(before,after))
        fs,ps = torch.stack((fc,raw)).cpu().tolist()
        return {"joint_norm": math.sqrt(_nonnegative(ls+fs,ls+fs,"joint delta")),
                "lora_norm": math.sqrt(ls), "fc_norm": math.sqrt(fs), "parameter_norm": math.sqrt(ps),
                "measurement": "actual_factor_values_float64_Gram_unscaled_BA_plus_all_FC"}
