#!/usr/bin/env python3
"""CPU-only positive/negative regression tests for effective BA+FC matching."""
import json
import math
from pathlib import Path
import sys
import torch
from effective_step import EffectiveStep, StepPolynomial, solve_positive_scale


def formation_regression():
    import ast
    import copy
    import tempfile
    from types import SimpleNamespace,MethodType
    import torch.nn.functional as F
    import formation_task0 as formation
    import audit_core as audit
    source = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')
    sys.path.insert(0,str(source))
    from optimer_PerturabtionType.gam import GAM
    tree = ast.parse((source/'models_LoRAbasedCL/as1_normfisher_gam_intask.py').read_text())
    learner_class = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Learner')
    method = next(n for n in learner_class.body if isinstance(n,ast.FunctionDef) and n.name=='_build_direction_projector')
    env = {'torch':torch}
    exec(compile('from __future__ import annotations\n'+ast.unparse(method),'<frozen-native-projector>','exec'),env)
    native_projector = env['_build_direction_projector']
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.w_As = torch.nn.ModuleList([torch.nn.Linear(5,2,bias=False)])
            self.backbone.w_Bs = torch.nn.ModuleList([torch.nn.Linear(2,5,bias=False)])
            self.backbone.w_meta = [{'layer':0,'channel':'q'}]
            self.backbone.task_id = 0
            self.fc = torch.nn.Linear(5,20)
        def forward(self,x):
            return {'logits':self.fc(x+ self.backbone.w_Bs[0](self.backbone.w_As[0](x)))}
    torch.manual_seed(18)
    initial = Toy().state_dict()
    inputs = [torch.randn(7,5) for _ in range(3)]
    targets = [torch.randint(20,(7,)) for _ in range(3)]
    states, reports, negatives = {},{},[]
    with tempfile.TemporaryDirectory(prefix='formation_cpu_') as tmp:
        for arm in formation.ARMS:
            cli = SimpleNamespace(source=str(source),output_dir=str(Path(tmp)/arm),max_epochs=1,max_batches=3)
            raw = {'formation_arm':arm,'seed':2,'step_reference_dir':str(Path(tmp)/'s')}
            cfg = formation.resolve_config(raw,cli)
            model = Toy(); model.load_state_dict(initial)
            args = SimpleNamespace(grad_beta_1=cfg['gam_grad_beta_1'],grad_beta_2=1.,
                                   grad_beta_3=cfg['gam_grad_beta_3'],grad_gamma=0.,use_sgv=False)
            sgd = torch.optim.SGD(model.parameters(),lr=cfg['init_lr'],momentum=0.,weight_decay=0.)
            opt = GAM(model.parameters(),sgd,model,grad_rho=.05,grad_norm_rho=cfg['gam_grad_norm_rho'],adaptive=False,args=args)
            learner = SimpleNamespace(_unwrap_network=lambda:model,_lambda_flat=1.,_mechanism_eps=1e-12,
                _mechanism_grad_stats=False,_build_optimizer=lambda params,stage:opt,
                _step_batch_with_log=lambda *a:None,_run_regularized_train=lambda *a,**k:None)
            learner._build_direction_projector = MethodType(native_projector,learner)
            c = formation.FormationController(learner,None,cfg,cli)
            c.install(); c.instrument_optimizer(opt)
            for i,(x,y) in enumerate(zip(inputs,targets)):
                c.epoch,c.batch_index = 1,i
                c.current_batch_identity = {'input_sha256':audit.tensor_digest(x),'target_sha256':audit.tensor_digest(y),
                    'indices_sha256':str(i),'indices':list(range(7)),'data_seed':200001}
                opt.zero_grad()
                loss = F.cross_entropy(model(x)['logits'],y)
                clean = torch.autograd.grad(loss,list(model.parameters()))
                cache = {id(p):g.detach().clone() for p,g in zip(model.parameters(),clean)}
                opt.set_direction_projector(learner._build_direction_projector(cache,{},{}))
                def closure():
                    opt.zero_grad()
                    out = model(x)
                    loss = F.cross_entropy(out['logits'],y)
                    loss.backward()
                    return out,loss.detach()
                opt.step(closure)
            c.trace_file.close()
            states[arm] = copy.deepcopy(model.state_dict())
            reports[arm] = {'steps':len(c.trace_rows),'all_self_spec':all(r['self_spec_passed'] for r in c.trace_rows),
                'max_match_error':max((r['actual_match_relative_error'] or 0 for r in c.trace_rows)),
                'all_four_closure':all(r['closure_count']==4 for r in c.trace_rows)}
            if arm == 's':
                reference_cfg = {k:v for k,v in cfg.items() if k!='device'}
                audit.atomic_json(Path(tmp)/arm/'effective_config.json',reference_cfg)
                audit.atomic_json(Path(tmp)/arm/'step_trace_manifest.json',{'schema_version':1,'seed':2,
                    'trace_sha256':audit.digest(c.trace_path),'steps':3})
                audit.atomic_json(Path(tmp)/arm/'final_summary.json',{'status':'completed','formation_arm':'s','seed':2,
                    'step_trace_sha256':audit.digest(c.trace_path),'counters':{'training_batches':3},'source_gam_sha256':c.source_sha,
                    'smoke':True,'completed_epochs':1,'checkpoint_epochs':[0,1],
                    'finite_check':{'passed':True},'task_check':{'passed':True},'step_checks':{'passed':True},
                    'checkpoint_checks':{'0':True,'1':True},'trainer_sha256':audit.digest(formation.__file__),
                    'effective_step_sha256':audit.digest(Path(formation.__file__).with_name('effective_step.py')),
                    'base_trainer_sha256':audit.digest(formation.base.__file__)})
        # The paired control differs only by expected floating projector arithmetic.
        maximum = max(float((states['sum'][k]-states['mean_lr2'][k]).abs().max()) for k in states['sum'])
        assert maximum < 2e-7,maximum
        reference = formation.StepReference(Path(tmp)/'s',2)
        wrong = copy.deepcopy(reference.rows[(1,0)]['identity']); wrong['input_sha256']='wrong'
        negatives.append(must_reject(lambda:reference.get(1,0,wrong),'augmentation_pairing'))
        negatives.append(must_reject(lambda:reference.get(9,0,wrong),'missing_reference_step'))
        negatives.append(must_reject(lambda:formation.StepReference(Path(tmp)/'s',1),'reference_seed'))
        formal_protocol = {'epochs':50,'batches_per_epoch':10,'smoke':False,'config':cfg,'code_identity':{}}
        negatives.append(must_reject(lambda:formation.StepReference(Path(tmp)/'s',2,protocol=formal_protocol),'formal_rejects_smoke_reference'))
        missing_protocol = {'epochs':1,'batches_per_epoch':4,'smoke':True,'config':cfg,'code_identity':{}}
        negatives.append(must_reject(lambda:formation.StepReference(Path(tmp)/'s',2,protocol=missing_protocol),'incomplete_reference_sequence'))
        with (Path(tmp)/'s'/'step_trace.jsonl').open('a') as f:
            f.write(' ')
        negatives.append(must_reject(lambda:formation.StepReference(Path(tmp)/'s',2),'reference_hash_corruption'))
    schedules = {}
    for seed in (0,1,2):
        for arm in formation.ARMS:
            cli = SimpleNamespace(output_dir='/tmp/no_write_config_test',max_epochs=1,max_batches=1)
            cfg = formation.resolve_config({'formation_arm':arm,'seed':seed,'step_reference_dir':'/tmp/reference'},cli)
            assert cfg['epochs']==50 and cfg['init_epoch']==1 and cfg['seed']==seed
            p = torch.nn.Parameter(torch.ones(1))
            opt = torch.optim.SGD([p],lr=cfg['init_lr'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'],eta_min=cfg['init_lr']*.1)
            lrs = [opt.param_groups[0]['lr']]
            for _ in range(50):
                opt.step(); scheduler.step(); lrs.append(opt.param_groups[0]['lr'])
            schedules[(seed,arm)] = lrs
            assert math.isclose(lrs[-1],.004*formation.ARMS[arm][1],rel_tol=1e-12)
        assert all(math.isclose(a,2*b,rel_tol=1e-12) for a,b in zip(schedules[(seed,'mean_lr2')],schedules[(seed,'sum')]))
    return {'native_frozen_GAM_and_projector':True,'arms':reports,'sum_vs_mean_lr2_max_abs':maximum,
            'schedule_cases':len(schedules),'T_max50_including_smoke':True,'negative_cases_rejected':negatives}


def explicit(before, after, count):
    pieces = [(after[2*i+1].double() @ after[2*i].double() -
               before[2*i+1].double() @ before[2*i].double()).flatten() for i in range(count)]
    pieces += [(v.double()-u.double()).flatten() for u,v in zip(before[2*count:],after[2*count:])]
    return float(torch.cat(pieces).norm())


def must_reject(fn, label):
    try:
        fn()
    except (ValueError, FloatingPointError, ArithmeticError):
        return label
    raise AssertionError("Negative case not rejected: " + label)


def run():
    torch.set_num_threads(1)
    torch.manual_seed(832)
    results, negatives = [], []
    for dtype in (torch.float32, torch.float64):
        for zero_b in (False, True):
            pairs = [(torch.nn.Parameter(torch.randn(3,11,dtype=dtype)),
                      torch.nn.Parameter(torch.zeros(9,3,dtype=dtype) if zero_b else torch.randn(9,3,dtype=dtype)))
                     for _ in range(3)]
            fc = [torch.nn.Parameter(torch.randn(4,9,dtype=dtype)),torch.nn.Parameter(torch.randn(4,dtype=dtype))]
            geom = EffectiveStep(pairs,fc)
            for p in geom.params:
                p.grad = torch.randn_like(p)
            before = geom.capture()
            poly = geom.polynomial()
            for scale in (.04,.004,1e-8,0.):
                after = [p.detach().clone().add_(p.grad,alpha=-scale) for p in geom.params]
                actual = geom.actual_delta(before,after)["joint_norm"]
                dense = explicit(before,after,len(pairs))
                assert math.isclose(actual,dense,rel_tol=3e-8,abs_tol=3e-14),(actual,dense)
                # Polynomial is the exact-real-algebra prediction; actual includes SGD rounding.
                after64 = [p.double()-scale*p.grad.double() for p in geom.params]
                predicted_dense = explicit(before,after64,len(pairs))
                assert math.isclose(poly.norm(scale),predicted_dense,rel_tol=3e-8,abs_tol=3e-14)
                results.append({"dtype":str(dtype),"zero_b":zero_b,"scale":scale,
                                "gram_vs_dense_abs_error":abs(actual-dense)})
            target = geom.actual_delta(before,[p.detach().clone().add_(p.grad,alpha=-.04) for p in geom.params])["joint_norm"]
            solved = solve_positive_scale(poly,target)
            actual = geom.actual_delta(before,[p.detach().clone().add_(p.grad,alpha=-solved["scale"]) for p in geom.params])
            assert abs(actual["joint_norm"]-target)/target < 1e-3
    # Nonmonotonic BA path: A=B=1,dA=dB=1 => delta=-2s+s².
    poly = StepPolynomial(4.,-4.,1.)
    root = solve_positive_scale(poly,.75,max_scale=4.)
    assert abs(root["scale"]-.5) < 1e-8 and len(root["stationary_points"]) == 2,root
    # FC included in target; LoRA-only normalization would fail this comparison.
    assert geom.polynomial().fc2 > 0
    negatives.append(must_reject(lambda:solve_positive_scale(StepPolynomial(0,0,0),1),"zero_direction"))
    negatives.append(must_reject(lambda:solve_positive_scale(StepPolynomial(1,0,0),20,max_scale=1),"unreachable_bound"))
    negatives.append(must_reject(lambda:solve_positive_scale(poly,float('nan')),"nonfinite_target"))
    negatives.append(must_reject(lambda:StepPolynomial(1,float('nan'),1),"nonfinite_coefficient"))
    negatives.append(must_reject(lambda:solve_positive_scale(poly,0),"zero_target"))
    negatives.append(must_reject(lambda:EffectiveStep(pairs,fc,[*geom.params,torch.nn.Parameter(torch.ones(1))]),"scope_omission"))
    saved = geom.params[0].grad.clone()
    geom.params[0].grad.fill_(float('nan'))
    negatives.append(must_reject(lambda:geom.polynomial(),"nonfinite_gradient"))
    geom.params[0].grad.copy_(saved)
    return {"status":"passed","device":"cpu","explicit_dense_cases":len(results),
            "cases":results,"nonmonotonic_smallest_root":root,"negative_cases_rejected":negatives}


if __name__ == "__main__":
    report = run()
    report['formation_regression'] = formation_regression()
    if len(sys.argv)>1:
        Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
