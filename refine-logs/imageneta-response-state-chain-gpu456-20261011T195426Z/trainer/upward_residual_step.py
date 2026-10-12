"""Eight upward-side residual rules, with only the probes each rule needs.

All virtual reads restore the COMPLETE entry state and replay R0.  Construction
starts at the same base (the outer point is not accumulated from the SAM point).
Only the base read's R1 is committed; apply_update performs one natural SGD.
"""
import math
import torch

from response_step import ResponseStep, _clone, _digest, _norm, _dot, _cos, _finite
from state_transaction import State, backend_context, rng_equal, rng_get, rng_set

DIRECTIONS = ('g0', 'g1', 'g2_minus_s', 'g2', 'g2_plus_s',
              'g0_plus_d', 'g1_plus_d', 'g3')
READS = dict(zip(DIRECTIONS, (1, 2, 3, 3, 3, 4, 4, 4)))
FORMULAS = dict(zip(DIRECTIONS, ('g0', 'g1', 'g2-(g1-g0)', 'g2',
    'g2+(g1-g0)', 'g0+(g3-g2)', 'g1+(g3-g2)', 'g3')))

class UpwardResidualStep(ResponseStep):
    def __init__(self, model, params, geometry, config):
        super().__init__(model, params, geometry, dict(method='sgd',
            history_weight=0., eps=config.get('eps', 1e-12),
            backend=config.get('backend', 'native'), radius_mode='sgd_step', radius_scale=1.))
        self.direction = config['direction']
        if self.direction not in DIRECTIONS:
            raise ValueError('Unregistered upward residual direction')
        self.rho, self.radius = float(config['rho']), float(config['radius'])
        if any(not math.isfinite(v) or v <= 0 for v in (self.rho, self.radius)):
            raise ValueError('Positive finite upward probe radii required')

    def compute(self, inputs, targets, loss_closure, current_lr):
        if self._pending is not None:
            raise RuntimeError('Pending compute must be applied before another read')
        self._check_layout()
        eta = float(current_lr)
        if not math.isfinite(eta) or eta < 0:
            raise ValueError('Finite nonnegative current_lr required')
        entry = State(self.model)
        base, rng0 = _clone(self.params), entry.rng
        calls, losses, points, gradients = 0, {}, {}, {}

        def probe(label, point):
            nonlocal calls
            entry.restore()
            with torch.no_grad():
                for p, value in zip(self.params, point): p.copy_(value)
                for p in self.model.parameters(): p.grad = None
            with backend_context(self.model, self.backend):
                loss, logits = loss_closure(inputs, targets)
                if loss.ndim != 0 or not bool(torch.isfinite(loss)) or not loss.requires_grad:
                    raise ValueError('Closure must return a finite first-order scalar loss')
                raw = torch.autograd.grad(loss, self.params, allow_unused=True, create_graph=False)
            if any(p.grad is not None for p in self.model.parameters()):
                raise AssertionError('Closure must not call backward')
            values = tuple(torch.zeros_like(p) if g is None else g.detach().clone()
                           for p, g in zip(self.params, raw))
            _finite(values, 'Upward probe '+label)
            if not all(torch.equal(p, value) for p, value in zip(self.params, point)):
                raise AssertionError('Closure mutated trainable parameters')
            calls += 1
            losses[label], points[label], gradients[label] = float(loss.detach()), point, values
            return logits.detach().clone(), values, rng_get()

        def shifted(origin, gradient, radius):
            # N(g) = g/(||g||+eps), matching the registered GAM probe definition.
            scale = radius / (_norm(gradient)+self.eps)
            return tuple(v.clone().add_(g, alpha=scale) for v, g in zip(origin, gradient))

        s = d = None
        try:
            logits0, g0, rng1 = probe('g0', base)
            if READS[self.direction] >= 2:
                _, g1, _ = probe('g1', shifted(base, g0, self.rho))
                s = tuple(a-b for a,b in zip(g1,g0))
            if READS[self.direction] >= 3:
                _, g2, _ = probe('g2', shifted(base, s, self.radius))
            if READS[self.direction] >= 4:
                _, g3, _ = probe('g3', shifted(points['g2'], g2, self.rho))
                d = tuple(a-b for a,b in zip(g3,g2))
            if self.direction in ('g0','g1','g2','g3'):
                U = _clone(gradients[self.direction])
            elif self.direction == 'g2_minus_s': U = tuple(a-b for a,b in zip(g2,s))
            elif self.direction == 'g2_plus_s': U = tuple(a+b for a,b in zip(g2,s))
            elif self.direction == 'g0_plus_d': U = tuple(a+b for a,b in zip(g0,d))
            elif self.direction == 'g1_plus_d': U = tuple(a+b for a,b in zip(g1,d))
            _finite(U, 'Upward final update')
        finally:
            entry.restore()
            restored = entry.check()
        rng_set(rng1)
        if not rng_equal(rng_get(),rng1): raise AssertionError('Only g0 R1 may advance RNG')
        if calls != READS[self.direction]: raise AssertionError('Native skip-probe count mismatch')
        norms={name:_norm(value) for name,value in gradients.items()}
        norms.update(g=norms['g0'],U=_norm(U),final=_norm(U),
                     response=_norm(s) if s is not None else 0.)
        h = eta*norms['g0']
        record = dict(method=self.direction,direction=self.direction,formula=FORMULAS[self.direction],
            native_probe_skipping=True,probe_family='upward_sam_response_outer_point',
            rho=self.rho,r=self.radius,inner_rho=self.rho,outer_R=self.radius,
            gamma_GAM=0.,residual_coefficient=1.,final_update_normalized=False,
            h=h,current_lr=eta,history_used=False,history_available=False,history_weight=0.,
            cache_source=None,cache_source_description='disabled',history_reset_count=self.history_reset_count,
            fallback=False,fallback_reason=None,fallback_count=0,only_base_R1_rng_advance=True,
            state_restore_checks=restored,gradient_norms=norms,g_grad_norm=norms['g0'],
            U_grad_norm=norms['U'],u_g_ratio=norms['U']/norms['g0'] if norms['g0']>self.eps else None,
            g_dot_U=_dot(g0,U),g_U_cosine=_cos(g0,U,self.eps),
            loss0=losses['g0'],probe_losses=losses,base_parameter_sha256=_digest(base),
            g0_sha256=_digest(g0),U_sha256=_digest(U),final_gradient_sha256=_digest(U),
            closure_count=calls,gradient_calls=calls,gradient_reads=calls,
            ordinary_gradient_calls=calls,training_ce_gradient_passes=calls,
            training_forward_count=calls,normal_gradient_calls=READS[self.direction],
            training_scope='all trainable LoRA plus full current FC',
            fisher_estimation_passes=0,weight_merge_updates=0,projection_calls=0,hvp_calls=0,
            qr_calls=0,create_graph_calls=0,gradient_clipping_calls=0,final_norm_rescale_calls=0,
            first_probe_sign=1.,outer_origin='base',final_update_origin='base')
        for name in ('g1','g2','g3'):
            record[name+'_sha256']=_digest(gradients[name]) if name in gradients else None
        if s is not None:
            record.update(s_norm=_norm(s),s_sha256=_digest(s))
        if d is not None:
            record.update(d_norm=_norm(d),d_sha256=_digest(d))
        if 'g1' in points:
            record['actual_inner_displacement_norm']=_norm(tuple(a-b for a,b in zip(points['g1'],base)))
        if 'g2' in points:
            record['actual_outer_displacement_norm']=_norm(tuple(a-b for a,b in zip(points['g2'],base)))
        if 'g3' in points:
            record['actual_terminal_displacement_norm']=_norm(tuple(a-b for a,b in zip(points['g3'],points['g2'])))
        context=dict(base=base,probe_points=points,probe_gradients=gradients,
            g0=g0,g1=gradients.get('g1'),g2=gradients.get('g2'),g3=gradients.get('g3'),
            s=s,d=d,U=U,rng0=rng0,rng1=rng1,entry_state=entry,loss0=losses['g0'])
        geometry_before=self.geometry.capture() if self.geometry is not None else None
        self._pending=dict(eta=eta,base=base,g0=g0,U=U,record=record,context=context,geometry_before=geometry_before)
        return dict(loss0=losses['g0'],logits0=logits0,U=U,record=record,context=context)
