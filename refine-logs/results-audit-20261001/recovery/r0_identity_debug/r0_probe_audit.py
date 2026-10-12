"""Audited native GAM probes. No training state is committed."""
import hashlib
from types import SimpleNamespace
import torch
import torch.nn.functional as F


def compare(A, xs, ys, exact=False):
    av, bv = A.flatten(xs), A.flatten(ys)
    finite = bool(torch.isfinite(av).all() and torch.isfinite(bv).all())
    result = dict(finite=finite, max_abs=float((av-bv).abs().max()),
                  relative_error=A.norm(av-bv)/max(A.norm(bv),1e-20),
                  all_exact=all(torch.equal(x,y) for x,y in zip(xs,ys)),
                  per_tensor_allclose=all(torch.allclose(x,y,rtol=1e-5,atol=1e-7) for x,y in zip(xs,ys)))
    result['passed'] = finite and result['all_exact' if exact else 'per_tensor_allclose']
    if not result['passed']:
        raise AssertionError('Native self-spec failed: '+str(result))
    return result


def audited_probes(A, model, geometry, anchor, x, y, start, end, rho, radius, metric_x, reference_logits, GAM):
    anchor.restore(); model.train()
    params=geometry.params
    checks=[]; points=[]; shifts=[]; own=[]; rngs=[]; norms=[]
    def norm_native(gs):
        value=None
        for g in gs:
            term=torch.sum(g**2)
            value=term if value is None else value+term
        return torch.sqrt(value)
    class CheckedGAM(GAM):
        @torch.no_grad()
        def perturb_weights(self, perturb_idx):
            gs=[p.grad.detach().clone() for p in params]
            n=norm_native(gs); scale=self.grad_rho/(n+self.perturb_eps)
            expected=[p.detach()+g*scale.to(p) for p,g in zip(params,gs)]
            super().perturb_weights(perturb_idx)
            checks.append(dict(operation='perturb_'+str(perturb_idx),**compare(A,params,expected,True)))
            norms.append(dict(operation='perturb_'+str(perturb_idx),norm=float(n),scale=float(scale)))
        @torch.no_grad()
        def unperturb(self, perturb_key):
            expected=[p.detach()-self.state[p][perturb_key] for p in params]
            super().unperturb(perturb_key)
            checks.append(dict(operation='unperturb_'+perturb_key,**compare(A,params,expected,True)))
        @torch.no_grad()
        def grad_norm_ascent(self):
            gs=[p.grad.detach()-self.state[p]['g_0'] for p in params]
            n=norm_native(gs); scale=self.grad_norm_rho/(n+self.perturb_eps)
            expected=[p.detach()+g*scale.to(p) for p,g in zip(params,gs)]
            super().grad_norm_ascent()
            checks.append(dict(operation='norm_ascent',**compare(A,params,expected,True)))
            norms.append(dict(operation='norm_ascent',norm=float(n),scale=float(scale)))
        @torch.no_grad()
        def gradient_decompose(self,args=None):
            expected=[self.state[p]['g_1']+p.grad.detach() for p in params]
            super().gradient_decompose(args)
            checks.append(dict(operation='gamma0_beta111_selected_gradient',**compare(A,[p.grad for p in params],expected,True)))
    base=torch.optim.SGD(params,lr=0.0,momentum=0.0,weight_decay=0.0)
    opt=CheckedGAM(params,base_optimizer=base,model=model,grad_rho=rho,grad_norm_rho=radius,
                   adaptive=False,perturb_eps=1e-12,
                   args=SimpleNamespace(grad_beta_1=1.,grad_beta_2=1.,grad_beta_3=1.,grad_gamma=0.))
    native_step=base.step
    def checked_step(*a,**kw):
        before=[p.detach().clone() for p in params]
        native_step(*a,**kw)
        checks.append(dict(operation='base_SGD_lr0',**compare(A,params,before,True)))
    base.step=checked_step
    try:
        loss=F.cross_entropy(A.logits(model,x)[:,start:end],y-start)
        clean=A.flatten([g.detach() for g in torch.autograd.grad(loss,params)])
        del loss
        def closure():
            point=geometry.displacement_metrics()
            point['kl_all_seen']=A.kl_from_logits(reference_logits,A.isolated_logits(model,metric_x))
            shifts.append(geometry.parameter_delta().detach().clone())
            opt.zero_grad(set_to_none=True)
            # Capture exactly the state/RNG entering this real closure, then restore
            # after an independent autograd.grad evaluation before real backward.
            snapshot=A.Snapshot(model)
            state=A.rng_get()
            rngs.append(dict(cpu=A.tensor_digest(state[2]),cuda=[A.tensor_digest(v) for v in state[3]]))
            ref_loss=F.cross_entropy(A.logits(model,x)[:,start:end],y-start)
            ref=[g.detach().clone() for g in torch.autograd.grad(ref_loss,params)]
            del ref_loss
            snapshot.restore(); snapshot.check(); del snapshot
            out=A.logits(model,x)
            actual_loss=F.cross_entropy(out[:,start:end],y-start)
            actual_loss.backward()
            own.append(compare(A,[p.grad for p in params],ref))
            point['probe_batch_ce']=float(actual_loss.detach())
            points.append(point)
            return out.detach(),actual_loss.detach()
        opt.step(closure)
        grads={'g%d'%k:A.flatten([opt.state[p]['g_%d'%k].detach().clone() for p in params]) for k in range(4)}
        if not all(bool(torch.isfinite(v).all()) for v in grads.values()):
            raise FloatingPointError('Nonfinite native gradient')
        stats=dict(radius=radius,rho=rho,closure_count=len(points),ce_gradient_passes=5,
                   independent_self_spec_ce_passes=4,points=points,
                   clean_cache_vs_g0_maxabs=float((clean-grads['g0']).abs().max()),
                   native_self_spec=dict(passed=True,operation_checks=checks,closure_CE_checks=own,
                       closure_rng_hashes=rngs,norms=norms,
                       final_native_roundtrip=geometry.displacement_metrics()))
        return grads,stats,shifts
    finally:
        anchor.restore(); anchor.check()
