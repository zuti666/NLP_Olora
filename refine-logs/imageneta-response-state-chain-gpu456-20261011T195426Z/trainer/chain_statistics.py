"""Passive exact actual-update decompositions, with low-rank FP64 Gram norms."""
import math
import torch
from effective_step import _inner

def norm(x):return float(x.double().norm())
def _product_norm(x,y):return math.sqrt(max(0.,float(_inner(x,y,x,y))))
def _sum_product_norm(terms):
    return math.sqrt(max(0.,sum(float(_inner(x,y,u,v)) for x,y in terms for u,v in terms)))
def quantiles(values):
    x=torch.tensor(values,dtype=torch.float64)
    q=torch.quantile(x,torch.tensor([.25,.5,.75],dtype=torch.float64)).tolist()
    return dict(q25=q[0],median=q[1],q75=q[2],iqr=q[2]-q[0])
@torch.no_grad()
def factor_epoch(geometry,start):
    rows=[]
    for i,(ap,bp) in enumerate(geometry.pairs):
        a,b=ap.double(),bp.double();a0,b0=(x.double() for x in start[2*i:2*i+2])
        q=b.T@b-a@a.T;q0=b0.T@b0-a0@a0.T
        denominator=float(a.square().sum()+b.square().sum())
        denominator0=float(a0.square().sum()+b0.square().sum())
        qhat=norm(q)/(denominator+1e-12);qhat0=norm(q0)/(denominator0+1e-12)
        rows.append(dict(layer=i,block=i//2,lora_branch='q' if i%2==0 else 'v',a_norm=norm(a),b_norm=norm(b),q_fro=norm(q),qhat=qhat,qhat_start=qhat0,qhat_change=qhat-qhat0,q_change_fro=norm(q-q0),effective_delta_from_start=_sum_product_norm([(b,a),(-b0,a0)])))
    return dict(layers=rows,cross_layer_qhat=quantiles([x['qhat'] for x in rows]),cross_layer_qhat_change=quantiles([x['qhat_change'] for x in rows]),cross_layer_q_change_fro=quantiles([x['q_change_fro'] for x in rows]),effective_displacement=geometry.actual_delta(start),all_layers_predeclared=True)
@torch.no_grad()
def update_decomposition(geometry,before):
    rows=[]
    for i,(ap,bp) in enumerate(geometry.pairs):
        a,b=(x.double() for x in before[2*i:2*i+2]);an,bn=ap.double(),bp.double();da,db=an-a,bn-b
        first=b.T@db+db.T@b-da@a.T-a@da.T
        second=db.T@db-da@da.T
        actual=bn.T@bn-an@an.T-(b.T@b-a@a.T)
        ta=(b,da);tb=(db,a);tx=(db,da)
        na,nb,nx=(_product_norm(*term) for term in (ta,tb,tx));dot=float(_inner(*ta,*tb));cos=dot/(na*nb) if na*nb>1e-20 else None
        residual=norm(actual-first-second)
        # Dense products are unnecessary; this low-rank identity is exact in FP64.
        dw=_sum_product_norm([ta,tb,tx])
        rows.append(dict(layer=i,block=i//2,lora_branch='q' if i%2==0 else 'v',delta_q_fro=norm(actual),delta_q_first_fro=norm(first),delta_q_second_fro=norm(second),delta_q_identity_error=residual,delta_q_identity_relative_error=residual/(norm(actual)+1e-12),delta_q_identity_state_scale_error=residual/(norm(b.T@b-a@a.T)+norm(bn.T@bn-an@an.T)+1e-12),T_A_norm=na,T_B_norm=nb,T_cross_norm=nx,T_A_T_B_cosine=cos,pi_A=na/(na+nb+1e-12),actual_delta_W_norm=dw,delta_A_norm=norm(da),delta_B_norm=norm(db)))
    result=dict(layers=rows,actual_update=geometry.actual_delta(before),cross_layer_pi_A=quantiles([x['pi_A'] for x in rows]),cross_layer_q_first_fro=quantiles([x['delta_q_first_fro'] for x in rows]),cross_layer_q_second_fro=quantiles([x['delta_q_second_fro'] for x in rows]),head_update_norm=geometry.actual_delta(before)['fc_norm'],measurement='actual committed factor tensors, FP64 lowrank Gram; no added model forward/backward')
    return result
