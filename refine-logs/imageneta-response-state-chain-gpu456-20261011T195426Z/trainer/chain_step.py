"""Native SAM / descending response g2 followed by unchanged SGD at epoch26."""
import time
import torch
from upward_residual_step import UpwardResidualStep
from independent_response_step import IndependentResponseStep
from response_step import _digest
READS={'sgd':1,'sam':2,'sf_g2':3}
FORMULAS={'sgd':'g0','sam':'g1=gradL(theta+0.05*N(g0))','sf_g2':'g2=gradL(theta+0.25*N(g0-gradL(theta-eta*g0)))'}
def active_method(branch,epoch,switch_after=25):
    if branch not in ('S','R') or type(epoch) is not int or epoch<1 or epoch>50 or switch_after not in (1,25):
        raise ValueError('Registered branchS/R, epochs1..50, formal25/smoke1 switch required')
    return ('sam' if branch=='S' else 'sf_g2') if epoch<=switch_after else 'sgd'
def cosine_lr(initial,epoch):
    import math
    return initial*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/50)))
class ChainStep:
    def __init__(self,model,params,geometry,cfg):
        self.branch=cfg['chain_branch'];self.switch=cfg['chain_switch_after_epoch'];self.active=None;self.pending=None;self.epoch=0
        self.kernels={n:UpwardResidualStep(model,params,geometry,dict(direction=d,rho=.05,radius=1.,eps=1e-12,backend='native')) for n,d in (('sgd','g0'),('sam','g1'))}
        self.kernels['sf_g2']=IndependentResponseStep(model,params,geometry,dict(method='current_g2',history_weight=0.,eps=1e-12,backend='native',radius_mode='independent_inner_outer_v1',radius_scale=1.,radius_variant='fixed_outer',response_probe_variant='S_F',inner_probe_radius_mode='sgd_step',outer_probe_radius_mode='fixed_l2',fixed_inner_probe_radius=None,fixed_outer_probe_radius=.25))
    def set_epoch(self,epoch):
        if self.pending is not None:raise RuntimeError('Cannot switch pending batch')
        self.epoch=epoch;self.active=active_method(self.branch,epoch,self.switch)
    def compute(self,*args,**kw):
        if self.active is None or self.pending is not None:raise RuntimeError('Set epoch and complete preceding batch first')
        if torch.cuda.is_available():torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter();r=self.kernels[self.active].compute(*args,**kw)
        if torch.cuda.is_available():torch.cuda.synchronize()
        r['record'].update(active_method=self.active,chain_branch=self.branch,gradient_formula=FORMULAS[self.active],normal_gradient_calls=READS[self.active],native_probe_skipping=True,native_probe_compute_seconds=time.perf_counter()-t,native_training_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0)
        self.pending=self.active;return r
    def apply_update(self,*args,**kw):
        if self.pending!=self.active:raise RuntimeError('One compute must precede one commit')
        if torch.cuda.is_available():torch.cuda.synchronize()
        t=time.perf_counter();r=self.kernels[self.active].apply_update(*args,**kw)
        if torch.cuda.is_available():torch.cuda.synchronize()
        r.update(native_sgd_commit_seconds=time.perf_counter()-t,native_training_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,actual_post_parameter_sha256=_digest(self.kernels[self.active].params))
        self.pending=None;return r
    def state_dict(self):return dict(schema='ChainStep_v1',branch=self.branch,epoch=self.epoch,active=self.active,pending=self.pending,history=None,continuous_optimizer_switch=True)
