"""Four ordinary CE probes; P/C/CD select an outer point and submit only g2."""
import copy
import math
import torch

from audit_core import rng_get, rng_set, rng_equal
from response_hvp import attention_backend, capture_state, _cache_name


def _clone(values):
    return tuple(v.detach().clone() for v in values)


def _combine(a, b, a_scale=1.0, b_scale=1.0):
    return tuple(a_scale*x + b_scale*y for x, y in zip(a, b))


def _inner(a, b):
    result = sum((x.detach().double()*y.detach().double()).sum() for x, y in zip(a, b))
    value = float(result.item())
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite direction inner product")
    return value


def vector_norm(values):
    return math.sqrt(max(0.0, _inner(values, values)))


def vector_cosine(a, b):
    denominator = vector_norm(a)*vector_norm(b)
    return _inner(a, b)/denominator if denominator > 0 else None


def normalized(values, eps=1e-12):
    denominator = vector_norm(values) + float(eps)
    return tuple(v/denominator for v in values)


def _identity_tree(value):
    """Retain every mutable entry object so rollback also restores references."""
    if isinstance(value, dict):
        return value, {k: _identity_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return value, [_identity_tree(v) for v in value]
    if torch.is_tensor(value):
        return value, value.detach().clone()
    return value, copy.deepcopy(value)


@torch.no_grad()
def _restore_tree(node):
    original, saved = node
    if isinstance(original, dict):
        original.clear()
        original.update({k: _restore_tree(v) for k, v in saved.items()})
    elif isinstance(original, list):
        original[:] = [_restore_tree(v) for v in saved]
    elif isinstance(original, tuple):
        for value in saved:
            _restore_tree(value)
    elif torch.is_tensor(original) and not isinstance(original, torch.nn.Parameter):
        original.copy_(saved)
    return original


class FullState:
    """Ordinary-probe transaction, including original gradient/cache objects."""
    def __init__(self, model):
        self.state = capture_state(model)
        self.trainable = self.state.trainable
        self.buffers = self.state.buffers
        self.modes = self.state.modes
        self.rng = self.state.rng
        self.grad_refs = [(p, p.grad, None if p.grad is None else p.grad.detach().clone())
                          for p in model.parameters()]
        self.controls = [(m, {k: _identity_tree(v) for k, v in m.__dict__.items()
                              if _cache_name(k) or k in ('momentum', 'fused_attn')})
                         for m in model.modules()]

    @torch.no_grad()
    def restore(self):
        self.state.restore()
        for p, original, saved in self.grad_refs:
            if original is not None:
                original.copy_(saved)
            p.grad = original
        for module, attrs in self.controls:
            for key, node in attrs.items():
                setattr(module, key, _restore_tree(node))

    def check(self):
        checks = self.state.check()
        checks['gradient_references_exact'] = all(p.grad is original for p, original, _ in self.grad_refs)
        checks['cache_references_exact'] = all(getattr(m, k) is node[0]
                                              for m, attrs in self.controls for k, node in attrs.items())
        if not all(checks.values()):
            raise AssertionError("Ordinary-probe transaction restore failed: " + repr(checks))
        return checks


class FourProbeStep:
    """Every branch reads g0,g+,g-,g2 from theta0 and common forward RNG R0.

    compute temporarily exposes g2 in parameter.grad and advances RNG to R1
    from g0. discard and all failures restore the full entry. apply_update
    commits exactly one ordinary SGD update from theta0, preserving R1.
    Radii may be zero for mathematical CPU acceptance; the trainer separately
    enforces the registered production rho=.05/r=.25 configuration.
    """
    def __init__(self, model, params, geometry, config):
        self.model, self.params, self.geometry = model, tuple(params), geometry
        self.config = dict(config)
        self.rho = float(self.config.get('rho', .05))
        self.radius = float(self.config.get('radius', .25))
        self.eps = float(self.config.get('eps', 1e-12))
        self.branch = self.config.get('branch', 'P')
        self.direction = self.config.get('direction', 'g2')
        self.backend = self.config.get('backend', 'math')
        if self.branch not in ('P', 'C', 'CD') or self.direction != 'g2' or self.backend != 'math':
            raise ValueError('Registered branches are P/C/CD, final direction g2, backend math')
        if any(not math.isfinite(v) for v in (self.rho, self.radius, self.eps)) or self.rho < 0 or self.radius < 0 or self.eps != 1e-12:
            raise ValueError('Finite nonnegative radii and eps=1e-12 required')
        if self.config.get('step_mode', 'natural') != 'natural':
            raise ValueError('Only natural SGD is registered')
        forbidden = ('hvp', 'create_graph', 'fisher', 'weight_merge', 'projector', 'normalize_final_gradient')
        if any(bool(self.config.get(k, False)) for k in forbidden):
            raise ValueError('Forbidden method option')
        if not self.params or len({id(p) for p in self.params}) != len(self.params):
            raise ValueError('Empty or duplicate trainable parameters')
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        if trainable != {id(p) for p in self.params} or {id(p) for p in geometry.params} != trainable:
            raise ValueError('Parameters and effective map must cover every trainable')
        for module in model.modules():
            if bool(getattr(module, '_register_delta_hook', False)):
                raise ValueError('Delta gradient hooks must be disabled')
            if any(getattr(module, k, None) is not None for k in ('external_delta_w_q', 'external_delta_w_v')):
                raise ValueError('External delta weights must be absent')
        self._geo_index = [next(i for i, p in enumerate(self.params) if p is q) for q in geometry.params]
        self._pending = None
        self.total_ordinary_gradient_calls = 0
        self.total_closure_calls = 0

    def _geometry_order(self, values):
        return tuple(values[i] for i in self._geo_index)

    def _validate(self, values, label):
        if len(values) != len(self.params):
            raise ValueError(label + ' tensor count mismatch')
        for p, v in zip(self.params, values):
            if v is None or v.shape != p.shape or v.device != p.device or v.dtype != p.dtype:
                raise ValueError(label + ' shape/device/dtype mismatch')
            if not bool(torch.isfinite(v).all()):
                raise FloatingPointError(label + ' is nonfinite')

    @torch.no_grad()
    def _set_point(self, values):
        for p, v in zip(self.params, values):
            p.copy_(v)

    @torch.no_grad()
    def _point(self, base, direction, scale):
        return tuple(v + float(scale)*d for v, d in zip(base, direction))

    def _evaluate(self, point, inputs, targets, closure, transaction):
        transaction.restore()
        self._set_point(point)
        with torch.enable_grad(), attention_backend(self.model, self.backend):
            self.total_closure_calls += 1
            loss, logits = closure(inputs, targets)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Closure must return finite scalar loss and logits')
            self.total_ordinary_gradient_calls += 1
            grads = torch.autograd.grad(loss, self.params, create_graph=False, retain_graph=False, allow_unused=False)
        self._validate(grads, 'ordinary CE gradient')
        return loss.detach().clone(), logits.detach().clone(), _clone(grads), rng_get()

    def compute(self, inputs, targets, loss_closure, current_lr, observer=None):
        if self._pending is not None:
            raise RuntimeError('Previous direction must be applied or discarded first')
        if observer is not None:
            raise ValueError('Observers that alter g2 or add HVP are forbidden')
        lr = float(current_lr)
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError('Positive finite learning rate required')
        transaction = FullState(self.model)
        value_map = {id(p): v for p, v in transaction.trainable}
        base = tuple(value_map[id(p)] for p in self.params)
        before_ce = self.total_ordinary_gradient_calls
        before_closures = self.total_closure_calls
        try:
            l0, logits0, g0, rng1 = self._evaluate(base, inputs, targets, loss_closure, transaction)
            n0 = normalized(g0, self.eps)
            plus = self._point(base, n0, self.rho)
            minus = self._point(base, n0, -self.rho)
            lp, logitsp, gp, _ = self._evaluate(plus, inputs, targets, loss_closure, transaction)
            lm, logitsm, gm, _ = self._evaluate(minus, inputs, targets, loss_closure, transaction)
            # Keep the registered P/C operation order; CD explicitly combines raw sides.
            qP = _combine(gp, g0, 1, -1)
            qC = tuple(-v for v in _combine(gm, g0, 1, -1))
            qCD = tuple(.5*(p-m) for p, m in zip(gp, gm))
            s = qCD
            e = tuple(.5*(p+m)-z for p, m, z in zip(gp, gm, g0))
            q = {'P': qP, 'C': qC, 'CD': qCD}[self.branch]
            outer = self._point(base, normalized(q, self.eps), self.radius)
            l2, logits2, g2, _ = self._evaluate(outer, inputs, targets, loss_closure, transaction)
            U = g2
            norms = {k: vector_norm(v) for k, v in dict(g0=g0,g1_plus=gp,g1_minus=gm,
                     g2=g2,qP=qP,qC=qC,qCD=qCD,s=s,e=e,q=q,U=U).items()}
            dtype_eps = torch.finfo(self.params[0].dtype).eps
            low_s_threshold = 10*dtype_eps*max(norms['g1_plus'], norms['g1_minus'], norms['g0'])
            points = {'phi0': base, 'phi1_plus': plus, 'phi1_minus': minus, 'phi2': outer}
            context = dict(model=self.model,params=self.params,geometry=self.geometry,base=base,
                points=points,outer=outer,grads=dict(g0=g0,g1_plus=gp,g1_minus=gm,g1=None,g2=g2),
                g0=g0,g1_plus=gp,g1_minus=gm,g2=g2,qP=qP,qC=qC,qCD=qCD,s=s,e=e,q=q,
                c21=None,U=U,rng0=transaction.rng,rng1=rng1,post_rng=rng1,
                losses=dict(g0=l0,g1_plus=lp,g1_minus=lm,g2=l2),
                logits=dict(g0=logits0,g1_plus=logitsp,g1_minus=logitsm,g2=logits2),
                backend=self.backend,config=dict(self.config),inputs=inputs,targets=targets,loss_closure=loss_closure)
            transaction.restore()
            restoration = transaction.check()
            ce_calls = self.total_ordinary_gradient_calls-before_ce
            closure_calls = self.total_closure_calls-before_closures
            if ce_calls != 4 or closure_calls != 4:
                raise AssertionError('Exactly four ordinary gradients and closures required')
            record = dict(branch=self.branch,direction='g2',backend=self.backend,
                ordinary_gradient_calls=ce_calls,closure_calls=closure_calls,sgd_updates=0,fisher_calls=0,weight_merge_calls=0,
                gradient_projection_calls=0,hvp_calls=0,create_graph_calls=0,
                rho=self.rho,radius=self.radius,eps=self.eps,
                loss0=float(l0),loss1_plus=float(lp),loss1_minus=float(lm),loss2=float(l2),
                delta_loss1_plus=float(lp-l0),delta_loss1_minus=float(lm-l0),
                delta_gradient_norm2=norms['g2']-norms['g0'],
                delta_gradient_norm2_sign=1 if norms['g2']>norms['g0'] else -1 if norms['g2']<norms['g0'] else 0,
                delta_gradient_norm2_absolute=abs(norms['g2']-norms['g0']),norms=norms,
                cosines=dict(qP_qC=vector_cosine(qP,qC),qCD_qP=vector_cosine(qCD,qP),
                             qCD_qC=vector_cosine(qCD,qC),g0_g2=vector_cosine(g0,g2)),
                asymmetry_e_over_s=norms['e']/(norms['s']+self.eps),low_s=norms['s']<=low_s_threshold,
                low_s_threshold=low_s_threshold,low_s_machine_epsilon=dtype_eps,
                low_s_rule='norm(s)<=10*eps_machine*max(norm(g+),norm(g-),norm(g0)); FP64 norm accumulation',
                low_s_interpretation='floating point cancellation reference only; no update modification',
                c21=None,g1_anchor=None,c21_applicable=False,
                probe_displacements={name: self.geometry.actual_delta(self._geometry_order(base),self._geometry_order(point))
                                     for name,point in points.items() if name!='phi0'},
                parameter_probe_displacements={name: vector_norm(_combine(point,base,1,-1))
                                               for name,point in points.items() if name!='phi0'},
                outer_nominal_normalized_distance=self.radius*norms['q']/(norms['q']+self.eps),
                base_parameters_exact=True,base_buffers_exact=True,state_restoration=restoration,
                forward_rng_rule='Each probe restores full theta0/R0; success advances only to R1 from g0')
            for p, u in zip(self.params, U):
                p.grad = u.detach().clone()
            rng_set(rng1)
            self._pending = dict(base=base,U=U,lr=lr,transaction=transaction,rng1=rng1)
            return dict(loss0=float(l0),logits0=logits0,U=U,record=record,context=context)
        except BaseException:
            transaction.restore()
            transaction.check()
            self._pending = None
            raise

    def discard(self):
        """Restore theta0, old gradients/cache identities/modes/buffers and R0."""
        if self._pending is not None:
            transaction = self._pending['transaction']
            self._pending = None
            transaction.restore()
            transaction.check()

    @torch.no_grad()
    def apply_update(self, current_lr=None, matched=False, optimizer=None):
        if self._pending is None:
            raise RuntimeError('compute must precede apply_update')
        pending = self._pending
        transaction = pending['transaction']
        base, U = pending['base'], pending['U']
        optimizer_state = None
        try:
            lr = pending['lr'] if current_lr is None else float(current_lr)
            if matched or not math.isfinite(lr) or lr <= 0 or lr != pending['lr']:
                raise ValueError('Only the declared positive natural learning rate is allowed')
            if not all(torch.equal(p,v) for p,v in zip(self.params,base)):
                raise RuntimeError('Base changed between compute and update')
            if not isinstance(optimizer,torch.optim.SGD):
                raise ValueError('One ordinary torch.optim.SGD optimizer required')
            groups = optimizer.param_groups
            actual_params = [p for group in groups for p in group['params']]
            if len(actual_params)!=len(self.params) or {id(p) for p in actual_params}!={id(p) for p in self.params}:
                raise ValueError('SGD must cover every trainable exactly once')
            if any(float(g.get(k,0))!=0 for g in groups for k in ('momentum','dampening','weight_decay')) or any(bool(g.get(k,False)) for g in groups for k in ('nesterov','maximize','differentiable')):
                raise ValueError('Ordinary zero-momentum/decay SGD required')
            if any(float(g['lr'])!=lr for g in groups):
                raise ValueError('Every SGD group must use the declared natural LR')
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            # Reestablish full theta0 before the unique commit, then expose g2/R1.
            transaction.restore()
            transaction.check()
            for p,u in zip(self.params,U):
                p.grad = u.detach().clone()
            rng_set(pending['rng1'])
            expected = tuple(v.clone().add_(u,alpha=-lr) for v,u in zip(base,U))
            optimizer.step()
            self._validate(tuple(p.detach() for p in self.params),'updated parameters')
            exact = all(torch.equal(p,v) for p,v in zip(self.params,expected))
            if not exact or not rng_equal(rng_get(),pending['rng1']):
                raise AssertionError('Unique natural SGD update differs from theta0-lr*g2 or changes RNG')
            actual = self.geometry.actual_delta(self._geometry_order(base))
            self._pending = None
            return dict(matched=False,nominal_lr=lr,scale=lr,step_scale=lr,scale_over_lr=1.,
                        actual=actual,parameter_update_norm=vector_norm(_combine(expected,base,1,-1)),
                        sgd_updates=1,base_sgd_exact=exact,natural_g2_only=True)
        except BaseException:
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            transaction.restore()
            transaction.check()
            self._pending = None
            raise

    def verify_zero_update_replay(self, inputs, targets, loss_closure, current_lr):
        """Two full four-probe repeats at one exact entry; strict bitwise gate."""
        entry = FullState(self.model)
        repeats = []
        before_ce = self.total_ordinary_gradient_calls
        before_closures = self.total_closure_calls
        try:
            for _ in range(2):
                result = self.compute(inputs,targets,loss_closure,current_lr)
                repeats.append(result['context'])
                self.discard()
                entry.check()
            a,b = repeats
            same = lambda xs,ys: len(xs)==len(ys) and all(torch.equal(x,y) for x,y in zip(xs,ys))
            checks = {k: same(a[k],b[k]) for k in ('g0','g1_plus','g1_minus','q','outer','g2','U')}
            checks['losses'] = all(torch.equal(a['losses'][k],b['losses'][k]) for k in a['losses'])
            checks['logits'] = all(torch.equal(a['logits'][k],b['logits'][k]) for k in a['logits'])
            checks['R1'] = rng_equal(a['rng1'],b['rng1'])
            if self.total_ordinary_gradient_calls-before_ce != 8 or self.total_closure_calls-before_closures != 8:
                raise AssertionError('Replay must perform exactly eight ordinary gradients')
            receipt = dict(ordinary_gradient_calls=self.total_ordinary_gradient_calls-before_ce,closure_calls=8,sgd_updates=0,zero_updates=True,strict_bitwise=True,
                           checks=checks,entry_state_restored=True,passed=all(checks.values()))
            if not receipt['passed']:
                error = AssertionError('Strict first-batch four-probe replay failed: '+repr(checks))
                error.replay_receipt = receipt
                raise error
            return receipt
        finally:
            self.discard()
            entry.restore()
            entry.check()
