"""Five native-budget gradient rules for either LoRA or ordinary full FT.

Every virtual read restores the full entry transaction and replays R0.  Only
the base read's R1 is advanced, and exactly one ordinary torch SGD update is
committed from the base.  Probe norms use FP64 reductions and FP32 coordinates
as in the reused single200 runs.  GAM's intrinsic decomposition separately
uses the original source's native-dtype norm/cosine epsilon ordering.
"""
import hashlib
import math

import torch

from state_transaction import State, backend_context, rng_equal, rng_get, rng_set


METHODS = ('sgd', 'sam', 'mirror', 'g2', 'gam')
READS = dict(zip(METHODS, (1, 2, 2, 3, 4)))
FORMULAS = dict(zip(METHODS, (
    'g0', 'g1_plus', 'g0+(g0-g1_minus)', 'g2',
    'h_plus-gamma*(h_minus-cosine*norm(h_minus)*h_plus/(norm(h_plus)+eps))')))


def _clone(values):
    return tuple(value.detach().clone() for value in values)


def _digest(values):
    h = hashlib.sha256()
    for value in values:
        v = value.detach().contiguous()
        h.update(str((tuple(v.shape), str(v.dtype))).encode())
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def _dot(a, b):
    value = float(torch.stack([(x.double()*y.double()).sum()
                               for x, y in zip(a, b)]).sum().item())
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite global inner product')
    return value


def _norm(values):
    return math.sqrt(max(0., _dot(values, values)))


def _cos(a, b, eps):
    den = _norm(a)*_norm(b)
    return _dot(a, b)/den if den > eps else None


def _finite(values, label):
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError(label+' contains a nonfinite tensor')


def _native_norm(values):
    # Matches source GAM._grad_norm: sequential per-parameter native sums.
    total = None
    for value in values:
        term = torch.sum(value**2)
        total = term if total is None else total+term
    return torch.sqrt(total)


@torch.no_grad()
def gam_decompose(g0, g1, g2, g3, gamma=.1, eps=1e-12):
    """Original beta1=beta2=beta3=1 decomposition, without SGV or HVP.

    Keep the two source epsilon placements.  A dot/norm-squared shortcut is
    mathematically close but is not the exact implemented source formula.
    """
    h_minus = tuple(a+b for a, b in zip(g0, g2))
    h_plus = tuple(a+b for a, b in zip(g1, g3))
    inner = 0.0
    for minus, plus in zip(h_minus, h_plus):
        inner += torch.sum(minus*plus)
    new_norm, old_norm = _native_norm(h_plus), _native_norm(h_minus)
    cosine = inner/(new_norm*old_norm+eps)
    vertical = tuple(minus-cosine*old_norm*plus/(new_norm+eps)
                     for minus, plus in zip(h_minus, h_plus))
    # add_ with alpha matches source p.grad.data.add_(vertical,alpha=-gamma).
    U = tuple(plus.clone().add_(orthogonal, alpha=-float(gamma))
              for plus, orthogonal in zip(h_plus, vertical))
    _finite(U, 'GAM decomposition')
    return U, dict(h_minus=h_minus, h_plus=h_plus, vertical=vertical,
                   cosine=float(cosine), native_h_plus_norm=float(new_norm),
                   native_h_minus_norm=float(old_norm))


class FiveMethodStep:
    """Native gradient budget: SGD 1, SAM/mirror 2, g2 3, GAM 4."""
    def __init__(self, model, params, config):
        self.model, self.params = model, list(params)
        if not self.params or len({id(p) for p in self.params}) != len(self.params):
            raise ValueError('Nonempty unique trainable parameter list required')
        if {id(p) for p in self.params} != {id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError('Scope must cover exactly all trainable parameters')
        if any(p.dtype != torch.float32 for p in self.params):
            raise ValueError('Registered actual coordinates are FP32')
        self.method = str(config['method'])
        if self.method not in METHODS:
            raise ValueError('Unknown registered method')
        self.rho = float(config.get('rho', .05))
        self.radius = float(config.get('radius', 1.))
        self.gamma = float(config.get('gamma', .1))
        self.eps = float(config.get('eps', 1e-12))
        self.backend = config.get('backend', 'native')
        if self.backend != 'native':
            raise ValueError('Registered backend must be native')
        if any(not math.isfinite(v) or v <= 0 for v in (self.rho, self.radius, self.eps)):
            raise ValueError('rho, radius, eps must be positive finite')
        if not math.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError('gamma must be finite and nonnegative')
        self._layout = self._layout_now()
        self._pending = None
        self.real_updates = 0

    def _layout_now(self):
        return tuple((id(p), tuple(p.shape), str(p.device), str(p.dtype), p.requires_grad)
                     for p in self.params)

    def _check_layout(self):
        if (self._layout_now() != self._layout or
                {id(p) for p in self.params} != {id(p) for p in self.model.parameters() if p.requires_grad}):
            raise ValueError('Trainable layout changed; create a new stepper')

    def compute(self, inputs, targets, loss_closure, current_lr):
        """Return loss0/logits0/U/record/context; do not perform an update."""
        if self._pending is not None:
            raise RuntimeError('Pending compute must be applied or discarded first')
        self._check_layout()
        eta = float(current_lr)
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError('Positive finite current_lr required')
        entry = State(self.model)
        saved = {id(p): value for p, value in entry.trainable}
        base, rng0 = tuple(saved[id(p)] for p in self.params), entry.rng
        calls, points, gradients, losses, unused = 0, {}, {}, {}, {}

        def probe(label, point):
            nonlocal calls
            entry.restore()
            with torch.no_grad():
                for p, value in zip(self.params, point):
                    p.copy_(value)
                for p in self.model.parameters():
                    p.grad = None
            with torch.enable_grad(), backend_context(self.model, self.backend):
                loss, logits = loss_closure(inputs, targets)
                if loss.ndim != 0 or not bool(torch.isfinite(loss)) or not loss.requires_grad:
                    raise ValueError('Closure must retain a finite scalar first-order loss')
                raw = torch.autograd.grad(loss, self.params, allow_unused=True, create_graph=False)
            if any(p.grad is not None for p in self.model.parameters()):
                raise AssertionError('Closure must not call backward')
            values = tuple(torch.zeros_like(p) if g is None else g.detach().clone()
                           for p, g in zip(self.params, raw))
            _finite(values, 'Ordinary gradient '+label)
            if not all(torch.equal(p, value) for p, value in zip(self.params, point)):
                raise AssertionError('Closure mutated trainable parameters')
            calls += 1
            points[label], gradients[label], losses[label] = point, values, float(loss.detach())
            unused[label] = sum(g is None for g in raw)
            return logits.detach().clone(), values, rng_get()

        def shifted(origin, direction, signed_radius):
            scale = signed_radius/(_norm(direction)+self.eps)
            return tuple(value.clone().add_(gradient, alpha=scale)
                         for value, gradient in zip(origin, direction))

        q, projection = None, None
        try:
            logits0, g0, rng1 = probe('g0', base)
            if self.method == 'mirror':
                _, gminus, _ = probe('g1_minus', shifted(base, g0, -self.rho))
                U = tuple(a+(a-b) for a, b in zip(g0, gminus))
            elif self.method == 'sgd':
                U = _clone(g0)
            else:
                _, g1, _ = probe('g1', shifted(base, g0, self.rho))
                if self.method == 'sam':
                    U = _clone(g1)
                else:
                    q = tuple(a-b for a, b in zip(g1, g0))
                    # GAM outer relocation is from theta0, never theta1.
                    _, g2, _ = probe('g2', shifted(base, q, self.radius))
                    if self.method == 'g2':
                        U = _clone(g2)
                    else:
                        _, g3, _ = probe('g3', shifted(points['g2'], g2, self.rho))
                        U, projection = gam_decompose(g0, g1, g2, g3, self.gamma, self.eps)
            _finite(U, 'Final update')
            if calls != READS[self.method]:
                raise AssertionError('Native gradient budget mismatch')
        except BaseException:
            entry.restore()
            entry.check()
            self._pending = None
            raise
        entry.restore()
        restored = entry.check()
        rng_set(rng1)
        if not rng_equal(rng_get(), rng1):
            raise AssertionError('Only g0 R1 may advance training RNG')
        norms = {name: _norm(value) for name, value in gradients.items()}
        norms.update(U=_norm(U), q=_norm(q) if q is not None else None)
        record = dict(
            method=self.method, direction=self.method, formula=FORMULAS[self.method],
            native_probe_skipping=True, native_expected_gradient_calls=READS[self.method],
            rho=self.rho, radius=self.radius, gamma_GAM=self.gamma if self.method == 'gam' else 0.,
            grad_beta_1=1., grad_beta_2=1., grad_beta_3=1.,
            probe_norm_reduction='FP64 global; point add_ in actual FP32',
            gam_decomposition_norm_reduction='native dtype sequential source sum' if projection else None,
            final_update_origin='base', outer_origin='base' if q is not None else None,
            first_probe_sign=-1 if self.method == 'mirror' else (1 if self.method != 'sgd' else None),
            final_update_normalized=False, current_lr=eta,
            state_restore_checks=restored, only_base_R1_rng_advance=True,
            gradient_norms=norms, g_grad_norm=norms['g0'], U_grad_norm=norms['U'],
            u_g_ratio=norms['U']/norms['g0'] if norms['g0'] > self.eps else None,
            g_dot_U=_dot(g0, U), g_U_cosine=_cos(g0, U, self.eps),
            loss0=losses['g0'], probe_losses=losses,
            base_parameter_sha256=_digest(base), g0_sha256=_digest(g0),
            U_sha256=_digest(U), final_gradient_sha256=_digest(U),
            closure_count=calls, gradient_calls=calls, gradient_reads=calls,
            ordinary_gradient_calls=calls, training_ce_gradient_passes=calls,
            training_forward_count=calls, sgd_updates=0, real_sgd_update=False,
            fisher_estimation_passes=0, fisher_calls=0,
            weight_merge_updates=0, weight_merge_calls=0,
            projection_calls=int(self.method == 'gam'),
            gradient_projection_calls=int(self.method == 'gam'),
            gam_intrinsic_projection_calls=int(self.method == 'gam'),
            extra_gradient_projection_calls=0,
            hvp_calls=0, qr_calls=0, create_graph_calls=0,
            gradient_clipping_calls=0, final_norm_rescale_calls=0,
            ewc_calls=0, replay_calls=0, sgv_calls=0,
            unused_trainable_tensors_per_read=unused,
            history_used=False, history_available=False, history_weight=0.)
        for name in ('g1', 'g1_minus', 'g2', 'g3'):
            record[name+'_sha256'] = _digest(gradients[name]) if name in gradients else None
        for name, point in points.items():
            record[name+'_parameter_sha256'] = _digest(point)
        if 'g1' in points:
            record['actual_inner_displacement_norm'] = _norm(tuple(a-b for a, b in zip(points['g1'], base)))
        if 'g1_minus' in points:
            record['actual_inner_displacement_norm'] = _norm(tuple(a-b for a, b in zip(points['g1_minus'], base)))
        if 'g2' in points:
            record['actual_outer_displacement_norm'] = _norm(tuple(a-b for a, b in zip(points['g2'], base)))
            record['q_sha256'] = _digest(q)
        if 'g3' in points:
            record['actual_terminal_displacement_norm'] = _norm(tuple(a-b for a, b in zip(points['g3'], points['g2'])))
        if projection:
            record['projection'] = {key: value for key, value in projection.items()
                                    if key not in ('h_minus', 'h_plus', 'vertical')}
            record['projection']['source_formula'] = 'cosine=dot/(new_norm*old_norm+eps); vertical=h_minus-cosine*old_norm*h_plus/(new_norm+eps)'
        context = dict(base=base, probe_points=points, probe_gradients=gradients,
                       g0=g0, g1=gradients.get('g1'), g1_minus=gradients.get('g1_minus'),
                       g2=gradients.get('g2'), g3=gradients.get('g3'), q=q, U=U,
                       rng0=rng0, rng1=rng1, entry_state=entry, projection=projection)
        self._pending = dict(eta=eta, base=base, U=U, record=record, context=context)
        return dict(loss0=losses['g0'], logits0=logits0, U=U, record=record, context=context)

    @torch.no_grad()
    def apply_update(self, current_lr, optimizer):
        if self._pending is None:
            raise RuntimeError('compute must precede one committed SGD update')
        self._check_layout()
        pending, eta = self._pending, float(current_lr)
        if not math.isfinite(eta) or eta != pending['eta']:
            raise ValueError('Learning rate changed between compute and update')
        if type(optimizer) is not torch.optim.SGD:
            raise ValueError('Only torch.optim.SGD is supported')
        opt_params = [p for group in optimizer.param_groups for p in group['params']]
        if len(opt_params) != len(self.params) or {id(p) for p in opt_params} != {id(p) for p in self.params}:
            raise ValueError('SGD parameter scope mismatch')
        for group in optimizer.param_groups:
            if float(group['lr']) != eta:
                raise ValueError('All SGD groups must use current_lr')
            if (any(float(group.get(key, 0.)) != 0. for key in ('momentum', 'dampening', 'weight_decay')) or
                    any(bool(group.get(key, False)) for key in ('nesterov', 'maximize', 'differentiable', 'foreach', 'fused'))):
                raise ValueError('Natural SGD requires zero momentum/decay and ordinary single-tensor updates')
        if any(optimizer.state.get(p) for p in opt_params):
            raise ValueError('Ordinary SGD must not retain optimizer moments')
        if not all(torch.equal(p, base) for p, base in zip(self.params, pending['base'])):
            raise AssertionError('Parameters changed between compute and real update')
        before_commit = State(self.model)
        try:
            expected = tuple(base.clone().add_(U, alpha=-eta)
                             for base, U in zip(pending['base'], pending['U']))
            for p, U in zip(self.params, pending['U']):
                p.grad = U.clone()
            optimizer.step()
            if not all(torch.equal(p, value) for p, value in zip(self.params, expected)):
                raise AssertionError('Real torch SGD differs from the single-step FP32 specification')
            _finite(self.params, 'Updated parameters')
            delta = tuple(p.detach()-base for p, base in zip(self.params, pending['base']))
            update = dict(sgd_updates=1, real_sgd_update=True,
                          actual_parameter_step_norm=_norm(delta),
                          nominal_parameter_step_norm=eta*_norm(pending['U']),
                          updated_parameter_sha256=_digest(self.params),
                          final_gradient_sha256=_digest(pending['U']),
                          exact_torch_sgd_update=True, history_used=False,
                          optimizer_state_entries=len(optimizer.state))
        except BaseException:
            before_commit.restore()
            before_commit.check()
            raise
        self.real_updates += 1
        pending['record'].update(update)
        self._pending = None
        return update

    def discard(self):
        if self._pending is not None:
            self._pending['context']['entry_state'].restore()
            self._pending['context']['entry_state'].check()
            self._pending = None
