"""Same-point HVP response for real updates; frozen source is never edited.

All HVPs use math SDPA and a detached, fixed native-normalized direction.
The diagnostic module supplies its audited state, CE and autograd primitives;
unsupported second derivatives are errors here, never a finite-FD fallback.
"""
import importlib.util
import math
from pathlib import Path
import torch

_path=Path(__file__).resolve().parents[1]/'diagnostics/curvature_audit.py'
_spec=importlib.util.spec_from_file_location('fdhvp_curvature_primitives',_path)
curvature=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(curvature)
State=curvature.State
backend_context=curvature.backend_context
compare=curvature.compare
norm=curvature.norm


def validate_settings(rule,first,second,coefficient,backend):
    if first not in ('fd','hvp') or second not in ('fd','hvp'):
        raise ValueError('component_first_response and component_second_response must be fd or hvp')
    if backend not in ('native','math'):raise ValueError('Backend must be native or math')
    if not math.isfinite(coefficient) or coefficient<=0:raise ValueError('Positive finite correction coefficient required')
    if (first=='hvp' or second=='hvp') and (rule!='ug' or backend!='math'):
        raise ValueError('HVP response training requires rule=ug and common math SDPA backend')
    if coefficient!=1. and rule not in ('us','ug'):
        raise ValueError('Correction coefficient applies only to US or UG')


def native_scaled(gradient,radius):
    """Exact nonadaptive frozen GAM: sequential FP32 sum then multiply scale."""
    n=None
    for g in gradient:
        term=torch.sum(g**2);n=term if n is None else n+term
    n=torch.sqrt(n)
    if not bool(torch.isfinite(n)):raise FloatingPointError('Nonfinite native norm')
    scale=radius/(n+1e-12)
    return [g.detach()*scale.to(g) for g in gradient]


def fixed_unit(gradient):
    return curvature.native_unit(gradient)


def backend_state():
    return {'flash':torch.backends.cuda.flash_sdp_enabled(),'math':torch.backends.cuda.math_sdp_enabled(),
            'mem_efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),'cudnn':torch.backends.cuda.cudnn_sdp_enabled()}


def require_math_backend():
    flags=backend_state()
    if flags!={'flash':False,'math':True,'mem_efficient':False,'cudnn':False}:
        raise AssertionError('HVP/FD math control backend was not actually enforced')
    return flags


def require_deterministic_model(model):
    bad=[name for name,m in model.named_modules() if
         (isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.p>0) or float(getattr(m,'drop_prob',0.) or 0.)>0]
    if bad:raise ValueError('HVP training requires disabled dropout/drop-path: '+repr(bad))


def same_point_hvp(frame,native_gradient,direction,x,y,start,end,rho):
    """Replay at actual native point, and restore the *entry* state exactly.

frame is captured before the original closure; caller entry is after closure,
so restoring frame alone would incorrectly rewind training RNG/gradients.
"""
    if not math.isfinite(rho) or rho<=0:raise ValueError('Positive rho required')
    if any(v.requires_grad for v in direction):raise ValueError('HVP direction must already be detached')
    curvature.finite(list(direction),'fixed HVP direction')
    expected_direction=fixed_unit(native_gradient)
    if len(direction)!=len(expected_direction) or not all(torch.equal(v,w) for v,w in zip(direction,expected_direction)):
        raise AssertionError('HVP direction differs from fixed native-normalized local gradient')
    live=State(frame.model)
    try:
        with backend_context('math'):
            flags=require_math_backend()
            loss,g,h=curvature.gradient_at(frame,x,y,start,end,backend='math',need_hvp=True,direction=direction)
        agreement=curvature.assert_same_gradient(g,native_gradient)
        correction=[rho*v for v in h]
        curvature.finite(correction,'rho HVP correction')
        scaled_exact=all(torch.equal(c,rho*v) for c,v in zip(correction,h))
        if not scaled_exact:raise AssertionError('HVP correction omitted or altered rho')
        report={'status':'available','backend':'math','direction_detached':True,'fallback_used':False,
            'backend_flags':flags,'direction_matches_native_unit_exact':True,'direction_norm':norm(direction),
            'same_point_gradient':agreement,'same_point_ce':loss,'hvp_norm':norm(h),
            'rho_hvp_norm':norm(correction),'rho':rho,'rho_multiplication_exact':scaled_exact,
            'hvp_forward_count':1,'first_derivative_passes':1,'second_derivative_passes':1}
        return correction,report
    finally:
        live.restore();restored=live.check()
        if 'report' in locals():report['entry_state_restore']=restored


def direction(state,rule,coefficient=1.,second_response='fd'):
    if rule=='u0':return state['g_0'].detach().clone()
    if rule=='us':
        # This identity is intentionally exact in floating point at lambda=1.
        if coefficient==1.:return state['g_1'].detach().clone()
        return state['g_0']+coefficient*(state['g_1']-state['g_0'])
    if rule=='ug':
        correction=(state['g_3']-state['g_2']) if second_response=='fd' else state['rho_hvp_2']
        return state['g_0']+coefficient*correction
    if rule=='uo':return state['g_2'].detach().clone()
    if rule=='ufull':return state['g_3'].detach().clone()
    raise ValueError('Unknown component rule '+str(rule))
