"""Training-instance-only final direction extension of the historical GAM path.

No loader, seeding, model forward, scheduler, evaluation, or task-end replacement.
Use ``with install(factory, selector, audit_path, source_root): native_main(...)``.
The four native probes and native decomposition still execute; only the final
direction is selected before the existing clean projector and ordinary SGD.
"""
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import pickle
import random
from types import MethodType

import numpy as np
import torch

SELECTORS=('g0','g2','ug','g1_control','native_control')
_MISSING=object()


def sha_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensors_sha(values):
    h=hashlib.sha256()
    for value in values:
        value=value.detach().contiguous().cpu()
        h.update(str((tuple(value.shape),str(value.dtype))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def rng_sha():
    """Read RNG states; never draw a random value or restore/reseed state."""
    h=hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=4))
    h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        for state in torch.cuda.get_rng_state_all():h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()


def finite(values,label):
    if not values or any(v is None or not bool(torch.isfinite(v).all()) for v in values):
        raise FloatingPointError('Missing/nonfinite '+label)


def stats(values):
    finite(values,'audited tensors')
    squared=sum(float(v.detach().double().square().sum()) for v in values)
    if not math.isfinite(squared):raise FloatingPointError('Nonfinite audit norm')
    return {'sha256':tensors_sha(values),'l2_norm':math.sqrt(squared)}


def exact(actual,expected,label):
    if len(actual)!=len(expected) or not all(torch.equal(a,b) for a,b in zip(actual,expected)):
        raise AssertionError(label+' exact own-spec mismatch')


def set_temporary(obj,name,value,restores):
    previous=obj.__dict__.get(name,_MISSING)
    restores.append((obj,name,previous));setattr(obj,name,value)


def restore_methods(restores):
    for obj,name,value in reversed(restores):
        if value is _MISSING:delattr(obj,name)
        else:setattr(obj,name,value)


class BatchAudit:
    def __init__(self,session,learner,opt,inputs,targets,class_offset):
        self.s,self.l,self.opt=session,learner,opt
        if not isinstance(opt,session.GAM):raise TypeError('Expected canonical frozen GAM training optimizer')
        if not isinstance(opt.base_optimizer,torch.optim.SGD):raise TypeError('Historical protocol requires SGD')
        args=opt.args
        if (args.grad_beta_1,args.grad_beta_2,args.grad_beta_3,args.grad_gamma)!=(1.,1.,0.,0.):
            raise AssertionError('Extension must use historical gamma0 g1 base configuration')
        if opt.adaptive or opt.grad_rho!=.05 or opt.grad_norm_rho!=1. or getattr(args,'use_sgv',False):
            raise AssertionError('Historical nonadaptive rho=.05/r=1/no-SGV protocol differs')
        if float(learner._lambda_flat)!=1.:raise AssertionError('Historical lambda_flat must remain 1')
        self.params=[p for group in opt.param_groups for p in group['params']]
        if not self.params or len({id(p) for p in self.params})!=len(self.params):raise AssertionError('Invalid parameter scope')
        if any(not p.requires_grad for p in self.params):raise AssertionError('Optimizer includes frozen parameters')
        self.model=learner._network
        names={id(p):name for name,p in self.model.named_parameters()}
        if {id(p) for p in self.model.parameters() if p.requires_grad}!={id(p) for p in self.params}:
            raise AssertionError('Training optimizer omits trainable parameters')
        task=int(learner._cur_task)
        if class_offset!=20*task or int(learner._total_classes)!=20*(task+1):raise AssertionError('Historical current-task class slice differs')
        self.row={'schema_version':1,'selector':session.selector,'step_index':session.steps,'task':task,
            'task_step_index':session.task_steps.get(task,0),'class_offset':class_offset,'seen_classes':learner._total_classes,
            'batch_samples':int(targets.numel()),'input_sha256':tensors_sha([inputs]),'target_sha256':tensors_sha([targets]),
            'parameter_names':[names[id(p)] for p in self.params],'rng_before_batch_sha256':rng_sha(),
            'native_decomposition_called':0,'closure_count':0,'forward_count':0,'sgd_count':0,
            'selected_formula':{'g0':'g0','g2':'g2','ug':'g0 + (g3 - g2)','g1_control':'native g1 passthrough',
                                'native_control':'original unwrapped decomposition g1'}[session.selector],
            'decomposition_method_wrapped':session.selector!='native_control',
            'decomposition_call_evidence':('unmodified native step call site and four exact gradient caches before projector'
                if session.selector=='native_control' else 'direct wrapped invocation of original method')}
        self.gradients=[];self.point_hashes=[];self.clean=None;self.selected=None;self.projected=None;self.restores=[];self.forward_hook=None

    def install(self):
        opt=self.opt;learner=self.l
        native_step=opt.step;native_decompose=opt.gradient_decompose
        native_projector=learner._build_direction_projector;native_sgd=opt.base_optimizer.step
        def count_forward(*_):self.row['forward_count']+=1
        self.forward_hook=self.model.register_forward_hook(count_forward)
        def step(_opt,closure=None):
            if closure is None:raise AssertionError('Expected native training closure')
            def wrapped():
                self.point_hashes.append(tensors_sha(self.params))
                output=closure();self.row['closure_count']+=1
                values=[None if p.grad is None else p.grad.detach().clone() for p in self.params]
                finite(values,'native probe gradients');self.gradients.append(values)
                if not bool(torch.isfinite(output[1])):raise FloatingPointError('Nonfinite native probe loss')
                return output
            return native_step(closure=wrapped)
        def decompose(_opt,*a,**kw):
            before=rng_sha()
            # Native control leaves GAM.gradient_decompose entirely untouched.
            # Its output/cache is observed when the following projector runs.
            result=None if self.s.selector=='native_control' else native_decompose(*a,**kw)
            self.row['native_decomposition_called']+=1
            if len(self.gradients)!=4:raise AssertionError('Expected all four gradients before selection')
            for i,observed in enumerate(self.gradients):
                exact([opt.state[p]['g_'+str(i)] for p in self.params],observed,'native g'+str(i)+' cache')
            expected=[]
            with torch.no_grad():
                for p in self.params:
                    state=opt.state[p]
                    value=(state['g_0'] if self.s.selector=='g0' else state['g_2'] if self.s.selector=='g2'
                           else state['g_0']+(state['g_3']-state['g_2']) if self.s.selector=='ug' else state['g_1'])
                    expected.append(value.detach().clone())
                    # Control truly preserves native p.grad, including arithmetic/bytes.
                    if self.s.selector not in ('g1_control','native_control'):p.grad.copy_(value)
            finite(expected,'selected direction');exact([p.grad for p in self.params],expected,'selected direction')
            self.selected=expected
            self.row.update(gradients={f'g{i}':stats(g) for i,g in enumerate(self.gradients)},
                probe_parameter_sha256=self.point_hashes,selected_expected=stats(expected),
                selected_actual=stats([p.grad for p in self.params]),selected_exact=True,
                four_native_gradient_caches_exact=True)
            if rng_sha()!=before:raise AssertionError('Direction selection/decomposition changed RNG')
            self.row['selection_rng_unchanged']=True
            return result
        def build_projector(_learner,clean,fisher,delta=None):
            if fisher or delta or float(learner._lambda_flat)!=1.:raise AssertionError('Fisher/penalty cache or altered lambda_flat')
            if set(clean)!={id(p) for p in self.params}:raise AssertionError('Clean gradient scope mismatch')
            self.clean=[clean[id(p)].detach().clone() for p in self.params];finite(self.clean,'clean gradients')
            callback=native_projector(clean,fisher,delta)
            def project(groups):
                before=rng_sha()
                if self.s.selector=='native_control':decompose(opt)
                if self.selected is None:raise AssertionError('Projector invoked before direction selection')
                exact([p.grad for p in self.params],self.selected,'direction reaching native projector')
                expected=[c+1.*(g-c) for c,g in zip(self.clean,self.selected)]
                result=callback(groups)
                finite([p.grad for p in self.params],'projected gradients')
                exact([p.grad for p in self.params],expected,'native clean+(selected-clean) projector')
                self.projected=[p.grad.detach().clone() for p in self.params]
                self.row.update(clean_gradient=stats(self.clean),projected_expected=stats(expected),
                    projected_actual=stats(self.projected),projector_exact=True,fisher_cache_empty=True,delta_cache_empty=True)
                if rng_sha()!=before:raise AssertionError('Native projector/audit changed RNG')
                self.row['projector_rng_unchanged']=True
                return result
            return project
        def sgd(_base,*a,**kw):
            before_rng=rng_sha()
            if self.projected is None:raise AssertionError('SGD invoked before native projector')
            exact([p.grad for p in self.params],self.projected,'gradient reaching actual SGD')
            rates=[];before=[p.detach().clone() for p in self.params];expected=[]
            for group in opt.base_optimizer.param_groups:
                eta=float(group['lr']);rates.append(eta)
                if not math.isfinite(eta) or eta<=0:raise AssertionError('Invalid historical learning rate')
                if any(float(group.get(k,0))!=0. for k in ('momentum','weight_decay','dampening')) or group.get('nesterov',False) or group.get('maximize',False):
                    raise AssertionError('Historical SGD momentum/decay changed')
                expected.extend([p.detach().clone().add_(p.grad,alpha=-eta) for p in group['params']])
            self.row['sgd_before_parameter_sha256']=tensors_sha(before)
            self.row['actual_sgd_gradient']=stats([p.grad for p in self.params])
            result=native_sgd(*a,**kw);self.row['sgd_count']+=1
            finite(self.params,'updated training parameters');exact(self.params,expected,'actual SGD from native restored point')
            self.row.update(lr=rates,sgd_expected_parameter_sha256=tensors_sha(expected),
                sgd_actual_parameter_sha256=tensors_sha(self.params),sgd_exact=True,
                parameter_update_l2=math.sqrt(sum(float((p.detach().double()-b.double()).square().sum()) for p,b in zip(self.params,before))))
            if rng_sha()!=before_rng:raise AssertionError('SGD/audit changed RNG')
            self.row['sgd_rng_unchanged']=True
            return result
        set_temporary(opt,'step',MethodType(step,opt),self.restores)
        if self.s.selector!='native_control':
            set_temporary(opt,'gradient_decompose',MethodType(decompose,opt),self.restores)
        set_temporary(learner,'_build_direction_projector',MethodType(build_projector,learner),self.restores)
        set_temporary(opt.base_optimizer,'step',MethodType(sgd,opt.base_optimizer),self.restores)

    def finish(self):
        if (self.row['closure_count'],self.row['forward_count'],self.row['sgd_count'],self.row['native_decomposition_called'])!=(4,5,1,1):
            raise AssertionError('Historical clean+four-closure/one-decomposition/one-SGD count changed')
        self.row.update(rng_after_batch_sha256=rng_sha(),status='passed')
        return self.row

    def uninstall(self):
        if self.forward_hook is not None:self.forward_hook.remove()
        restore_methods(self.restores)


class Session:
    def __init__(self,factory,selector,audit_path,source_root):
        if selector not in SELECTORS:raise ValueError('Unknown legacy direction selector')
        self.factory,self.selector,self.path=factory,selector,Path(audit_path)
        self.source=Path(source_root).resolve()
        module=importlib.import_module('optimer_PerturabtionType.gam');self.GAM=module.GAM
        if Path(inspect.getfile(self.GAM)).resolve()!=self.source/'optimer_PerturabtionType/gam.py':
            raise AssertionError('GAM import is not the pinned canonical frozen source')
        self.steps=0;self.task_steps={};self.samples=0;self.factory_calls=0;self.restores=[];self.failed=None
        self.summary_path=self.path.with_name(self.path.stem+'_summary.json')
        self.identity={'selector':selector,'legacy_direction_sha256':sha_file(__file__),
            'source_gam_sha256':sha_file(inspect.getfile(self.GAM)),'source_root':str(self.source),
            'audit_path':str(self.path),'protocol':'native historical loader/global RNG/5CE/mechanism/SGD; training-instance direction hook only'}

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.summary_path.exists():raise FileExistsError(self.summary_path)
        self.handle=self.path.open('x');original=self.factory.get_model
        def create(*a,**kw):
            learner=original(*a,**kw);self.factory_calls+=1
            if self.factory_calls==1:self.bind(learner)
            return learner
        set_temporary(self.factory,'get_model',create,self.restores)
        return self

    def bind(self,learner):
        native_batch=learner._step_batch_with_log
        def batch(_learner,optimizer,inputs,targets,class_offset):
            audit=BatchAudit(self,learner,optimizer,inputs,targets,class_offset)
            try:
                audit.install()
                result=native_batch(optimizer,inputs,targets,class_offset)
                row=audit.finish();self.handle.write(json.dumps(row,allow_nan=False)+'\n');self.handle.flush()
                self.steps+=1;self.samples+=row['batch_samples'];self.task_steps[row['task']]=row['task_step_index']+1
                return result
            except BaseException as exc:
                self.failed={'error':repr(exc),'partial_audit':audit.row}
                raise
            finally:audit.uninstall()
        set_temporary(learner,'_step_batch_with_log',MethodType(batch,learner),self.restores)

    def __exit__(self,kind,value,tb):
        restore_methods(self.restores);self.handle.close()
        okay=kind is None and self.steps>0
        summary={**self.identity,'status':'completed' if okay else 'failed','steps':self.steps,
            'samples':self.samples,'task_steps':self.task_steps,'factory_calls':self.factory_calls,
            'native_ce_forward_passes':5*self.steps,'native_gradient_closures':4*self.steps,
            'actual_sgd_steps':self.steps,'extra_training_forwards':0,'RNG_reseeds_or_restores':0,
            'all_recorded_steps_passed':True,'factory_and_bound_methods_restored':True,
            'audit_sha256':sha_file(self.path),'failure':self.failed or (repr(value) if kind else None)}
        with self.summary_path.open('x') as handle:json.dump(summary,handle,indent=2,allow_nan=False);handle.write('\n')
        if kind is None and not okay:raise AssertionError('No native training batches audited')
        return False


def install(factory,selector,audit_path,source_root):
    return Session(factory,selector,audit_path,source_root)
