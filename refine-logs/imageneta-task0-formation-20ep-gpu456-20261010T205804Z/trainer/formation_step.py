"""Native-probe Task0 method schedules; one continuous ordinary SGD optimizer.

Method selection changes only which first-order vector is submitted. The model,
optimizer, scheduler, tensor references and RNG are never replaced at epoch11.
Every constituent kernel independently commits only g0/R1 and restores R0 for
virtual reads. No historical gradient is stored by any kernel.
"""
import time
import torch
from upward_residual_step import UpwardResidualStep
from independent_response_step import IndependentResponseStep
from response_step import _digest

CONDITIONS=('sgd','sam','gam_residual','sf_g2','sf_g2_then_sgd','sgd_then_sf_g2')
READS={'sgd':1,'sam':2,'gam_residual':4,'sf_g2':3}
FORMULAS={'sgd':'g0','sam':'g1 at theta+0.05*N(g0)',
          'gam_residual':'g0+(g3-g2); ascending rho=0.05, outer r=1',
          'sf_g2':'g2 at theta+0.25*N(g0-gradL(theta-eta*g0))'}

def active_method(condition,task,epoch):
    if condition not in CONDITIONS or task not in (0,1) or type(epoch) is not int or not 1<=epoch<=20:
        raise ValueError('Registered condition, task0/1, and epoch1..20 required')
    if task==1:return 'sgd'
    if condition=='sf_g2_then_sgd':return 'sf_g2' if epoch<=10 else 'sgd'
    if condition=='sgd_then_sf_g2':return 'sgd' if epoch<=10 else 'sf_g2'
    return condition

def cosine_lr(initial_lr,epoch):
    """Learning rate USED in epoch (scheduler steps once after each epoch)."""
    import math
    if epoch not in range(1,21):raise ValueError('Epoch1..20 only')
    return initial_lr*(.1+.9*.5*(1.+math.cos(math.pi*(epoch-1)/20)))

class FormationStep:
    def __init__(self,model,params,geometry,cfg):
        self.condition=cfg['formation_condition_id'];self.task=cfg['_formation_task']
        self.epoch=0;self.active=None;self.pending=None
        common=dict(rho=.05,radius=1.,eps=1e-12,backend='native')
        self.kernels={name:UpwardResidualStep(model,params,geometry,dict(common,direction=direction))
            for name,direction in (('sgd','g0'),('sam','g1'),('gam_residual','g0_plus_d'))}
        self.kernels['sf_g2']=IndependentResponseStep(model,params,geometry,
            dict(method='current_g2',history_weight=0.,eps=1e-12,backend='native',
                 radius_mode='independent_inner_outer_v1',radius_scale=1.,radius_variant='fixed_outer',
                 response_probe_variant='S_F',inner_probe_radius_mode='sgd_step',outer_probe_radius_mode='fixed_l2',
                 fixed_inner_probe_radius=None,fixed_outer_probe_radius=.25))
    def set_epoch(self,epoch):
        if self.pending is not None:raise RuntimeError('Cannot switch a pending SGD batch')
        self.epoch=epoch;self.active=active_method(self.condition,self.task,epoch)
    def compute(self,*a,**kw):
        if self.active is None or self.pending is not None:raise RuntimeError('Set epoch before one clean batch')
        if torch.cuda.is_available():torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();out=self.kernels[self.active].compute(*a,**kw)
        if torch.cuda.is_available():torch.cuda.synchronize()
        out['record'].update(formation_condition_id=self.condition,active_method=self.active,
            formation_epoch=self.epoch,formation_task=self.task,gradient_formula=FORMULAS[self.active],
            native_probe_skipping=True,normal_gradient_calls=READS[self.active],
            gam_radii_active=self.active in ('sam','gam_residual'),sf_outer_radius=.25 if self.active=='sf_g2' else None,
            native_probe_compute_seconds=time.perf_counter()-started,
            native_training_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0)
        self.pending=self.active;return out
    def apply_update(self,*a,**kw):
        if self.pending!=self.active:raise RuntimeError('Missing matching compute')
        if torch.cuda.is_available():torch.cuda.synchronize()
        started=time.perf_counter();out=self.kernels[self.active].apply_update(*a,**kw)
        if torch.cuda.is_available():torch.cuda.synchronize()
        out['native_sgd_commit_seconds']=time.perf_counter()-started
        out['native_training_peak_allocated_bytes']=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        out['actual_post_parameter_sha256']=_digest(self.kernels[self.active].params)
        self.pending=None;return out
    def state_dict(self):
        return dict(schema='FormationStep_v1',condition=self.condition,task=self.task,epoch=self.epoch,
                    active=self.active,pending=self.pending,kernels={n:k.state_dict() for n,k in self.kernels.items()},
                    history=None,continuous_optimizer_switch=True)
