"""Three ordinary CE probes and one natural odd/even SGD update.

All temporary evaluations start from one full entry state and common RNG R0.
The symmetric normalized probes share theta0 and R0; no outer probe or final normalization.
"""
import copy
import math
import torch

from audit_core import rng_get, rng_set, rng_equal
from response_hvp import attention_backend
from direction_step import FullState, _clone, _combine, normalized, vector_norm, vector_cosine

METHOD = 'odd_even_local_v1'
BRANCH = 'local_symmetric'
DIRECTIONS = ('g1_plus','backward_reflected','central','g0','g1_minus','even_average','even_reflected')
DIRECTION = 'backward_reflected'
FORMULAS = dict(g1_plus='g1_plus', backward_reflected='g0+(g0-g1_minus)',
                central='g0+0.5*(g1_plus-g1_minus)', g0='g0', g1_minus='g1_minus',
                even_average='0.5*(g1_plus+g1_minus)',
                even_reflected='g0+(g0-0.5*(g1_plus+g1_minus))')


class OddEvenStep:
    """Read g0, gplus and gminus from one entry; select one registered U.

    compute exposes detached U in .grad and R1 from the g0 evaluation. discard
    and failures restore full entry state, including original gradient objects.
    apply_update restores theta0 and commits exactly one ordinary SGD step.
    rho=0 is accepted only for mathematical tests; trainer fixes rho=.05.
    """
    def __init__(self, model, params, geometry, config):
        self.model, self.params, self.geometry = model, tuple(params), geometry
        self.config = dict(config)
        self.rho = float(self.config.get('rho', .05))
        self.eps = float(self.config.get('eps', 1e-12))
        self.branch = self.config.get('branch', BRANCH)
        self.direction = self.config.get('direction', DIRECTION)
        self.backend = self.config.get('backend', 'math')
        if self.branch != BRANCH or self.direction not in DIRECTIONS or self.backend != 'math':
            raise ValueError('Only local_symmetric, seven registered directions and math attention are registered')
        if self.config.get('method_name', METHOD) != METHOD:
            raise ValueError('Unsupported method name')
        if any(not math.isfinite(v) for v in (self.rho, self.eps)) or self.rho < 0 or self.eps != 1e-12:
            raise ValueError('Finite nonnegative rho and eps=1e-12 required')
        if self.config.get('step_mode', 'natural') != 'natural':
            raise ValueError('Only natural SGD is registered')
        if 'radius' in self.config or self.config.get('outer_radius') is not None:
            raise ValueError('This method has no outer radius')
        allowed = {'rho','eps','branch','direction','backend','method_name','step_mode','outer_radius'}
        if set(self.config)-allowed:
            raise ValueError('Unknown or forbidden method options: '+repr(sorted(set(self.config)-allowed)))
        forbidden = ('hvp', 'create_graph', 'fisher', 'weight_merge', 'projector',
                     'gradient_projection', 'normalize_final_gradient', 'matched')
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
            raise ValueError('Observers and extra model passes are forbidden')
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
            unit = normalized(g0, self.eps)
            plus = self._point(base, unit, self.rho)
            minus = self._point(base, unit, -self.rho)
            lp, logitsp, gp, _ = self._evaluate(plus, inputs, targets, loss_closure, transaction)
            lm, logitsm, gm, _ = self._evaluate(minus, inputs, targets, loss_closure, transaction)
            # Diagnostic decomposition; updates use their registered FP32 expression.
            odd = tuple(.5*(p-m) for p,m in zip(gp,gm))
            even = tuple(.5*((p-z)+(m-z)) for z,p,m in zip(g0,gp,gm))
            correction = tuple(z-m for z,m in zip(g0,gm))
            if self.direction == 'g1_plus':
                U = _clone(gp)
            elif self.direction == 'backward_reflected':
                U = tuple(z+c for z,c in zip(g0,correction))
            elif self.direction == 'central':
                U = tuple(z+.5*(p-m) for z,p,m in zip(g0,gp,gm))
            elif self.direction == 'g0':
                U = _clone(g0)
            elif self.direction == 'g1_minus':
                U = _clone(gm)
            elif self.direction == 'even_average':
                U = tuple(.5*(p+m) for p,m in zip(gp,gm))
            elif self.direction == 'even_reflected':
                U = tuple(z+(z-.5*(p+m)) for z,p,m in zip(g0,gp,gm))
            else:
                raise AssertionError('Unregistered update direction')
            self._validate(U, 'registered odd/even update direction')
            norms = {k: vector_norm(v) for k,v in dict(g0=g0,g1_plus=gp,g1_minus=gm,s=odd,e=even,correction=correction,U=U).items()}
            points = {'phi0': base, 'phi1_plus': plus, 'phi1_minus': minus}
            context = dict(model=self.model,params=self.params,geometry=self.geometry,base=base,
                points=points,g0=g0,g1_plus=gp,g1_minus=gm,s=odd,e=even,correction=correction,U=U,
                grads=dict(g0=g0,g1_plus=gp,g1_minus=gm),rng0=transaction.rng,rng1=rng1,post_rng=rng1,
                losses=dict(g0=l0,g1_plus=lp,g1_minus=lm),logits=dict(g0=logits0,g1_plus=logitsp,g1_minus=logitsm),
                backend=self.backend,config=dict(self.config),inputs=inputs,targets=targets)
            transaction.restore()
            restoration = transaction.check()
            ce_calls = self.total_ordinary_gradient_calls-before_ce
            closure_calls = self.total_closure_calls-before_closures
            if ce_calls != 3 or closure_calls != 3:
                raise AssertionError('Exactly three ordinary gradients and closures required')
            record = dict(method_name=METHOD,branch=self.branch,direction=self.direction,backend=self.backend,
                ordinary_gradient_calls=ce_calls,closure_calls=closure_calls,sgd_updates=0,
                fisher_calls=0,weight_merge_calls=0,gradient_projection_calls=0,hvp_calls=0,create_graph_calls=0,
                rho=self.rho,outer_radius=None,eps=self.eps,
                update_formula=FORMULAS[self.direction],probe_formula='theta0 +/- rho*g0/(norm(g0)+eps)',
                loss0=float(l0),loss1_plus=float(lp),loss1_minus=float(lm),
                delta_loss1_plus=float(lp-l0),delta_loss1_minus=float(lm-l0),norms=norms,
                cosines=dict(g0_g1_plus=vector_cosine(g0,gp),g0_g1_minus=vector_cosine(g0,gm),
                             g1_plus_g1_minus=vector_cosine(gp,gm),g0_s=vector_cosine(g0,odd),
                             g0_e=vector_cosine(g0,even),s_e=vector_cosine(odd,even),
                             g0_U=vector_cosine(g0,U),g1_plus_U=vector_cosine(gp,U),g1_minus_U=vector_cosine(gm,U)),
                correction_over_g0=norms['correction']/(norms['g0']+self.eps),
                s_over_g0=norms['s']/(norms['g0']+self.eps),e_over_g0=norms['e']/(norms['g0']+self.eps),
                e_over_s=norms['e']/(norms['s']+self.eps),
                probe_displacements={k:self.geometry.actual_delta(self._geometry_order(base),self._geometry_order(point))
                                     for k,point in points.items() if k!='phi0'},
                parameter_probe_displacements={k:vector_norm(_combine(point,base,1,-1))
                                               for k,point in points.items() if k!='phi0'},
                probe_nominal_normalized_distance=self.rho*norms['g0']/(norms['g0']+self.eps),
                base_parameters_exact=True,base_buffers_exact=True,state_restoration=restoration,
                forward_rng_rule='Each probe restores full theta0/R0; success advances only to R1 from g0')
            for p,u in zip(self.params,U):
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
            # Reestablish full theta0 before the unique commit, then expose U/R1.
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
                raise AssertionError('Unique natural SGD update differs from theta0-lr*U or changes RNG')
            actual = self.geometry.actual_delta(self._geometry_order(base))
            self._pending = None
            return dict(matched=False,nominal_lr=lr,scale=lr,step_scale=lr,scale_over_lr=1.,
                        actual=actual,parameter_update_norm=vector_norm(_combine(expected,base,1,-1)),
                        sgd_updates=1,base_sgd_exact=exact,natural_odd_even_only=True,update_direction=self.direction)
        except BaseException:
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            transaction.restore()
            transaction.check()
            self._pending = None
            raise

    def verify_zero_update_replay(self, inputs, targets, loss_closure, current_lr):
        """Two isolated three-probe repeats at one exact entry; strict bitwise gate."""
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
            checks = {k: same(a[k],b[k]) for k in ('g0','g1_plus','g1_minus','s','e','correction','U')}
            checks['plus_point'] = same(a['points']['phi1_plus'],b['points']['phi1_plus'])
            checks['minus_point'] = same(a['points']['phi1_minus'],b['points']['phi1_minus'])
            checks['losses'] = all(torch.equal(a['losses'][k],b['losses'][k]) for k in a['losses'])
            checks['logits'] = all(torch.equal(a['logits'][k],b['logits'][k]) for k in a['logits'])
            checks['R1'] = rng_equal(a['rng1'],b['rng1'])
            ce_calls = self.total_ordinary_gradient_calls-before_ce
            closure_calls = self.total_closure_calls-before_closures
            if ce_calls != 6 or closure_calls != 6:
                raise AssertionError('Replay must perform exactly six ordinary gradients and closures')
            receipt = dict(method_name=METHOD,ordinary_gradient_calls=ce_calls,closure_calls=closure_calls,
                           sgd_updates=0,zero_updates=True,strict_bitwise=True,checks=checks,
                           entry_state_restored=True,passed=all(checks.values()))
            if not receipt['passed']:
                error = AssertionError('Strict first-batch odd/even replay failed: '+repr(checks))
                error.replay_receipt = receipt
                raise error
            return receipt
        finally:
            self.discard()
            entry.restore()
            entry.check()
