"""Transactional offline diagnostics: exact loss Hessian, not Fisher or GGN.

All coordinates are the complete trainable LoRA factors and full classifier.
Every vector used in an HVP is detached, including the second vector H g0.
The diagnostic never commits an optimizer update. Explicit attention and loss
microbatching are allowed only after agreeing with the original gradient.
"""
import contextlib
import math
import time
import torch

GRADIENT_GATE_REL = 2e-4
GRADIENT_GATE_COS = .99999
HVP_GATE_REL = .08
FD_DISTANCES = (1e-3, 3e-3, 1e-2)
DEGENERATE_FP32_MULTIPLIER = 32


def dot(a, b):
    return float(sum((x.detach().double()*y.detach().double()).sum()
                     for x,y in zip(a,b)).item())


def norm(a):
    return math.sqrt(max(0.,dot(a,a)))


def difference(a,b):
    return tuple(x-y for x,y in zip(a,b))


def scaled(a,s):
    return tuple(x*float(s) for x in a)


def comparison(a,b,reference_norm=None):
    na,nb=norm(a),norm(b)
    threshold=DEGENERATE_FP32_MULTIPLIER*torch.finfo(torch.float32).eps*max(reference_norm or 0.,1e-12)
    degenerate=na<=threshold or nb<=threshold
    return dict(a_norm=na,b_norm=nb,cosine=None if degenerate else max(-1.,min(1.,dot(a,b)/(na*nb))),
        relative_error=None if degenerate else norm(difference(a,b))/(nb+1e-12),degenerate=degenerate,
        degeneracy_threshold=threshold)


def _clone_gradients(model,params,loss):
    gradients=torch.autograd.grad(loss,params,create_graph=False,allow_unused=True)
    return tuple(torch.zeros_like(p) if g is None else g.detach().clone() for p,g in zip(params,gradients))


class DiagnosticEngine:
    """Context attributes: model, geometry, loss_fn, State, backend_context.

    loss_fn(inputs, targets) is EXACT native mean training CE. All stochastic
    masks must be disabled in the inherited ViT protocol for microbatch means.
    Data augmentation is already materialized by the fixed-batch context.
    """
    def __init__(self,context,microbatch=16):
        self.context=context;self.model=context.model;self.geometry=context.geometry
        self.params=list(self.geometry.params);self.microbatch=int(microbatch)
        if self.microbatch<=0:raise ValueError('microbatch must be positive')
        for module in self.model.modules():
            if isinstance(module,torch.nn.modules.dropout._DropoutNd) and module.training and module.p!=0:
                raise ValueError('Loss microbatch Hessian requires native zero-dropout protocol')
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) and module.training:
                raise ValueError('Batch-dependent training normalization disallows exact mean microbatching')
            if module.training and float(getattr(module,'drop_prob',0.) or 0.)!=0:
                raise ValueError('Stochastic-depth masks disallow exact mean microbatch Hessian')
        self.state=context.State(self.model)
        self.snap=tuple(p.detach().clone() for p in self.params)
        self.reads=0;self.hvp_calls=0

    def restore(self):
        self.state.restore()

    def close(self):
        self.restore();return self.state.check()

    @torch.no_grad()
    def move(self,displacement):
        self.restore()
        for p,v,d in zip(self.params,self.snap,displacement):p.copy_(v+d)

    def gradient(self,inputs,targets,backend='native',displacement=None,chunked=False):
        self.restore()
        if displacement is not None:self.move(displacement)
        out=[torch.zeros_like(p) for p in self.params];size=targets.numel()
        batch=self.microbatch if chunked else size
        with self.context.backend_context(self.model,backend):
            for start in range(0,size,batch):
                stop=min(size,start+batch)
                loss=self.context.loss_fn(inputs[start:stop],targets[start:stop])
                self.reads+=1
                gs=_clone_gradients(self.model,self.params,loss)
                for a,g in zip(out,gs):a.add_(g,alpha=(stop-start)/size)
        self.restore()
        if not all(bool(torch.isfinite(g).all()) for g in out):raise FloatingPointError('Nonfinite offline gradient')
        return tuple(out)

    def hvp(self,inputs,targets,vector):
        """H(theta) @ detached(vector), weighted mean of exact microbatch HVPs."""
        self.restore();size=targets.numel();out=[torch.zeros_like(p) for p in self.params]
        vec=tuple(v.detach() for v in vector)
        with self.context.backend_context(self.model,'math'):
            for start in range(0,size,self.microbatch):
                stop=min(size,start+self.microbatch)
                loss=self.context.loss_fn(inputs[start:stop],targets[start:stop]);self.reads+=1
                grads=torch.autograd.grad(loss,self.params,create_graph=True,allow_unused=True)
                directional=sum((g*v).sum() for g,v in zip(grads,vec) if g is not None and g.requires_grad)
                if not torch.is_tensor(directional) or not directional.requires_grad:
                    second=[None]*len(self.params)
                else:
                    second=torch.autograd.grad(directional,self.params,allow_unused=True)
                for a,p,h in zip(out,self.params,second):
                    if h is not None:a.add_(h.detach(),alpha=(stop-start)/size)
                self.hvp_calls+=1
                del loss,grads,directional,second
        self.restore()
        if not all(bool(torch.isfinite(g).all()) for g in out):raise FloatingPointError('Nonfinite exact HVP')
        return tuple(out)

    def finite_difference_hvp(self,inputs,targets,vector,distance):
        nv=norm(vector)
        if nv<=1e-12:return None
        step=distance/nv
        plus=self.gradient(inputs,targets,'math',scaled(vector,step),True)
        minus=self.gradient(inputs,targets,'math',scaled(vector,-step),True)
        return scaled(difference(plus,minus),1/(2*step))

    def numerical_gate(self,inputs,targets):
        native=self.gradient(inputs,targets)
        repeat=self.gradient(inputs,targets)
        repeat_exact=all(torch.equal(x,y) for x,y in zip(native,repeat))
        explicit=self.gradient(inputs,targets,'math',chunked=True)
        agreement=comparison(explicit,native,norm(native))
        first_order_ok=(repeat_exact and not agreement['degenerate'] and
            agreement['relative_error']<=GRADIENT_GATE_REL and agreement['cosine']>=GRADIENT_GATE_COS)
        receipt=dict(repeated_native_gradient_exact=repeat_exact,first_order_backend_agreement=agreement,
            gradient_gate=dict(relative_error_max=GRADIENT_GATE_REL,cosine_min=GRADIENT_GATE_COS),
            hvp_gate=dict(relative_error_max=HVP_GATE_REL,required_adjacent_grid_points=2,absolute_normalized_displacements=list(FD_DISTANCES)),
            curvature='exact native mean CE loss Hessian at base trainable parameters',
            second_hvp_vector_detached=True,microbatch=self.microbatch,
            microbatch_mean_same_gradient_checked=True,backend='explicit_math_after_native_first_order_agreement',
            first_order_gate_passed=first_order_ok,status='failed_first_order_gate' if not first_order_ok else 'checking_hvp')
        if norm(native)<=1e-12 and repeat_exact and norm(explicit)<=1e-12:
            receipt.update(status='degenerate_gradient',accepted=False,numerical_implementation_failed=False,
                g0_norm=norm(native),q_norm=None,p_norm=None)
            return native,None,None,receipt
        if not first_order_ok:
            receipt.update(accepted=False,numerical_implementation_failed=True)
            return native,None,None,receipt
        q=self.hvp(inputs,targets,native)
        if norm(q)<=1e-12:
            receipt.update(status='degenerate_curvature',accepted=False,numerical_implementation_failed=False,
                q_norm=norm(q),g0_norm=norm(native),p_norm=None)
            return native,q,None,receipt
        p=self.hvp(inputs,targets,tuple(v.detach() for v in q))
        grids={};hvp_ok=True
        for name,vector,exact in (('H_g0',native,q),('H_detached_H_g0',q,p)):
            records=[]
            for h in FD_DISTANCES:
                fd=self.finite_difference_hvp(inputs,targets,vector,h)
                row=dict(distance=h,**comparison(fd,exact,norm(exact)))
                records.append(row)
            valid=[not row['degenerate'] and row['relative_error']<=HVP_GATE_REL for row in records]
            passed=any(a and b for a,b in zip(valid,valid[1:]))
            grids[name]=dict(grid=records,passed=passed);hvp_ok &= passed
        receipt.update(finite_difference=grids,accepted=bool(hvp_ok),
            status='accepted' if hvp_ok else 'failed_hvp_finite_difference_gate',numerical_implementation_failed=not bool(hvp_ok),
            q_norm=norm(q),p_norm=norm(p),g0_norm=norm(native),
            frozen_H_quadratic_form=.5*dot(native,q),
            quadratic_form_can_be_negative=True)
        return native,q,p,receipt

    def native_candidates(self,inputs,targets,lr,scale=1.,g0=None):
        if g0 is None:g0=self.gradient(inputs,targets)
        g0n=norm(g0)
        if g0n<=1e-12:return dict(status='degenerate_g0',g0=g0)
        gm=self.gradient(inputs,targets,displacement=scaled(g0,-lr*scale))
        response=difference(g0,gm);rn=norm(response)
        threshold=DEGENERATE_FP32_MULTIPLIER*torch.finfo(torch.float32).eps*max(g0n,1e-12)
        if rn<=threshold:
            return dict(status='degenerate_first_response',g0=g0,inner_gradient=gm,response=response,
                response_norm=rn,response_degeneracy_threshold=threshold)
        g2=self.gradient(inputs,targets,displacement=scaled(response,.25*scale/rn))
        g1=self.gradient(inputs,targets,displacement=scaled(g0,.05/g0n))
        return dict(status='accepted',g0=g0,g1=g1,g2=g2,inner_gradient=gm,response=response,
            response_norm=rn,response_degeneracy_threshold=threshold,inner_radius=lr*scale*g0n,
            outer_radius=.25*scale,sam_inner_radius=.05,scale=scale)


def correction_diagnostic(engine,inputs,targets,lr):
    started=time.perf_counter();g0,q,p,gate=engine.numerical_gate(inputs,targets)
    record=dict(numerical_gate=gate,scales=[],submitted_training_updates=0)
    if gate.get('accepted'):
        for scale in (1.,.5):
            candidate=engine.native_candidates(inputs,targets,lr,scale,g0)
            row=dict(scale=scale,status=candidate['status'],degenerate=True,cosine=None,
                relative_error=None,delta_native_norm=None,delta_quadratic_norm=None)
            for key in ('inner_radius','outer_radius','response_norm','response_degeneracy_threshold'):
                if key in candidate:row[key]=candidate[key]
            if candidate['status']=='accepted':
                dn=difference(candidate['g2'],g0);dq=scaled(p,.25*scale/norm(q))
                row.update(comparison(dn,dq,norm(g0)),g0_norm=norm(g0),g2_norm=norm(candidate['g2']),
                    delta_native_norm=norm(dn),delta_quadratic_norm=norm(dq),
                    expression='delta_quad = outer_radius * H.detach(H.g0) / ||H.g0||')
                if row['degenerate']:row['status']='degenerate_correction'
            record['scales'].append(row)
    elif not gate.get('numerical_implementation_failed',True):
        record['scales']=[dict(scale=scale,status=gate['status'],degenerate=True,
            cosine=None,relative_error=None,g0_norm=norm(g0),delta_native_norm=None,
            delta_quadratic_norm=None) for scale in (1.,.5)]
    record.update(elapsed_seconds=time.perf_counter()-started,gradient_forward_calls=engine.reads,
        exact_hvp_microbatch_calls=engine.hvp_calls,restoration=engine.close())
    return record


@torch.no_grad()
def finite_lora_parts(geometry,grads,step):
    """Exact finite-step LoRA effective displacement and factor-balance split.

    Unscaled BA is checked by EffectiveStep.from_model. FC is included in Phi,
    but is explicitly separate from LoRA and Q statistics. No model mutation.
    """
    snap=geometry.capture();directions=list(grads);parts=[];delta_effective=[]
    for i,(ap,bp) in enumerate(geometry.pairs):
        a,b=[v.detach().double() for v in snap[2*i:2*i+2]]
        ga,gb=[v.detach().double() for v in directions[2*i:2*i+2]]
        da,db=-step*ga,-step*gb
        ta,tb,tx=b@da,db@a,db@da
        delta=ta+tb+tx
        first=b.T@db+db.T@b-da@a.T-a@da.T
        second=db.T@db-da@da.T
        q0=b.T@b-a@a.T
        q1=(b+db).T@(b+db)-(a+da)@(a+da).T
        err=torch.linalg.vector_norm((q1-q0)-(first+second)).item()
        actual_err=torch.linalg.vector_norm((b+db)@(a+da)-b@a-delta).item()
        nA=torch.linalg.vector_norm(ta).item();nB=torch.linalg.vector_norm(tb).item()
        cosine=None if min(nA,nB)<=1e-30 else float((ta*tb).sum()/(nA*nB))
        parts.append(dict(layer_index=i,Q0_norm=torch.linalg.vector_norm(q0).item(),
            delta_Q_norm=torch.linalg.vector_norm(first+second).item(),
            delta_Q_first_order_norm=torch.linalg.vector_norm(first).item(),
            delta_Q_second_order_norm=torch.linalg.vector_norm(second).item(),
            delta_Q_identity_error=err,effective_identity_error=actual_err,
            T_A_norm=nA,T_B_norm=nB,T_cross_norm=torch.linalg.vector_norm(tx).item(),
            effective_delta_norm=torch.linalg.vector_norm(delta).item(),
            T_A_T_B_cosine=cosine,pi_A=nA/(nA+nB+1e-12)))
        delta_effective.append(delta)
    fc=[-step*v.detach().double() for v in directions[2*len(geometry.pairs):]]
    delta_effective.extend(fc)
    return dict(layers=parts,effective_norm=norm(delta_effective),
        lora_effective_norm=norm(delta_effective[:len(geometry.pairs)]),fc_norm=norm(fc),
        scale=step),tuple(delta_effective)


def matched_effective_diagnostic(engine,inputs,targets,lr,solver):
    started=time.perf_counter();cand=engine.native_candidates(inputs,targets,lr)
    record=dict(status=cand['status'],submitted_training_updates=0,native_lr=lr,
        effective_map='concatenated all unscaled LoRA BA and full classifier tensors',
        selection_rule='smallest positive quartic effective-norm root; no test-based selection')
    if cand['status']=='accepted':
        sam,ds=finite_lora_parts(engine.geometry,cand['g1'],lr)
        g2raw,dr=finite_lora_parts(engine.geometry,cand['g2'],lr)
        polynomial=engine.geometry.polynomial(cand['g2'])
        if sam['effective_norm']<=1e-12:
            record.update(status='degenerate_sam_effective_displacement',sam=sam,g2_native=g2raw)
        else:
            try:
                solve=solver(polynomial,sam['effective_norm'],initial_scale=lr,max_scale=10.,rtol=1e-10)
            except (ValueError,ArithmeticError) as error:
                record.update(status='effective_scalar_match_unavailable',sam=sam,g2_native=g2raw,
                    scalar_solver_exception_type=type(error).__name__,scalar_solver_exception=str(error))
                record.update(elapsed_seconds=time.perf_counter()-started,restoration=engine.close())
                return record
            matched,dm=finite_lora_parts(engine.geometry,cand['g2'],solve['scale'])
            record.update(status='accepted',sam=sam,g2_native=g2raw,g2_matched=matched,
                native_effective_comparison=comparison(dr,ds,sam['effective_norm']),
                matched_effective_comparison=comparison(dm,ds,sam['effective_norm']),
                scalar_solver=solve,matched_lr_ratio=solve['scale']/lr,
                norm_match_relative_error=abs(matched['effective_norm']-sam['effective_norm'])/sam['effective_norm'],
                inner_sgd_probe_radius=cand['inner_radius'],outer_response_radius=.25,sam_radius=.05)
    record.update(elapsed_seconds=time.perf_counter()-started,restoration=engine.close())
    return record
