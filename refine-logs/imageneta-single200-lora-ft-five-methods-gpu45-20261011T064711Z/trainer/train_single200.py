#!/usr/bin/env python3
"""Single supervised ImageNet-A 200-class LoRA / full-FT five-method study.

No continual-learning training hooks are called. The source learner is used
only to reproduce its exact pretrained LoRA/head initialization; the complete
5981-image training set is optimized as one fixed classification problem.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
import audit_core as audit
from scope_helper import full_ft_from_lora_origin, parameter_hash, scope_receipt
from state_transaction import State, rng_get, rng_set, rng_equal

PROTOCOL = 'imageneta_single200_lora_fullft_five_methods_native_fp32_exact_restore_v1'
METHODS = ('sgd','sam','mirror','g2','gam')
READS = dict(zip(METHODS,(1,2,2,3,4)))
FORBIDDEN_FALSE = ('class_shuffle','approximate_fisher_in_task','in_task_weight_merge',
    'task_endpoint_weight_merge','as1_delta_w_penalty','compute_full_fisher','mechanism_eval',
    'mechanism_grad_stats','flat_eval','feature_flat_eval','feature_cka_eval','feature_proto_eval',
    'attention_probe_eval','use_sgv','gam_use_sgv','ewc_normalize_fisher',
    'ewc_normalize_accumulated_fisher','fisher_split','ewc_dual_ascent','final_update_norm_matching')


def resolve_config(raw, cli):
    cfg = copy.deepcopy(raw)
    scope, method = cfg.get('train_scope'), cfg.get('method')
    if scope not in ('lora','full_ft') or method not in METHODS:
        raise ValueError('Explicit supported train_scope and method required')
    fixed = dict(dataset='imageneta',seed=2,init_cls=200,increment=200,nb_tasks=1,
        epochs=50,init_epoch=50,batch_size=128,lora_rank=10,rho=.05,r=1.,
        momentum=0.,weight_decay=0.,strict_deterministic=True,attention_backend='native',
        ewc_lambda=0.,ewc_skip_task_fisher=True,memory_size=0,memory_per_class=0,
        history_weight=0.,min_lr_ratio=.1,model_pretrained_tag='augreg2_in21k_ft_in1k')
    for name, expected in fixed.items():
        if cfg.get(name) != expected or (name == 'seed' and isinstance(cfg.get(name),bool)):
            raise ValueError('Frozen setting mismatch: '+name)
    for name in FORBIDDEN_FALSE:
        if cfg.get(name,False) is not False:
            raise ValueError('Forbidden module or protocol flag: '+name)
    expected_lr = .04 if scope == 'lora' else .001
    if cfg.get('lr') != expected_lr or cfg.get('gamma') != (.1 if method == 'gam' else 0.):
        raise ValueError('Frozen scope learning rate or intrinsic GAM gamma mismatch')
    if cli.max_epochs not in (0,1) or cli.max_batches not in (0,2) or bool(cli.max_epochs) != bool(cli.max_batches):
        raise ValueError('Only full 50 epochs or one-epoch/two-batch smoke allowed')
    cfg.update(protocol=PROTOCOL,actual_optimizer='torch.optim.SGD',
        single_task_full_200_classes=True,training_problem='supervised_single200',
        total_sessions=1,init_lr=expected_lr,cosine_min_lr_ratio=.1,
        gam_grad_gamma=cfg['gamma'],gam_grad_beta_1=1.,gam_grad_beta_2=1.,gam_grad_beta_3=1.,
        gam_grad_rho=cfg['rho'],gam_grad_norm_rho=cfg['r'],
        probe_gradient_count=READS[method],compute_g3_probe=method == 'gam',
        g3_used_in_final=method == 'gam',ordinary_gradient_calls=READS[method],
        continual_learning=False,replay=False,history_source=None,
        gradient_normalization='probe construction only',
        final_update_normalized=False,hessian_vector_products=False)
    return cfg


def source_integrity(source):
    manifest_path = Path(__file__).resolve().parents[1]/'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    wrong = [name for name, sha in manifest['files'].items()
             if audit.digest(source/name) != sha]
    if wrong:
        raise AssertionError('Pinned source changed: '+repr(wrong))
    return dict(passed=True,file_count=len(manifest['files']),
                manifest_sha256=audit.digest(manifest_path),source=str(source))


def initialize_model_and_data(cfg, source, output, device):
    """Historical exact setup order: seed → DataManager → factory → FC."""
    sys.path.insert(0,str(source));os.chdir(source)
    from utils import factory
    from utils.random_reproduce import set_random
    from utils.data_manager import DataManager
    source_cfg=copy.deepcopy(cfg)
    source_cfg.update(device=[device],filepath=str(output/'initialization_only')+os.sep)
    set_random(cfg['seed'])
    dm=DataManager('imageneta',False,cfg['seed'],200,200,source_cfg)
    learner=factory.get_model(cfg['model_name'],source_cfg)
    model=learner._unwrap_network()
    model.backbone.to(device)
    model.update_fc(200)
    model.to(device)
    origin_identity=dict(task=0,seed=cfg['seed'],class_slice=[0,200],fc_shape=[200,768],
        task_initial_parameter_sha256=parameter_hash(model),
        frozen_origin_sha256=parameter_hash(model,True),
        new_fc_rows_sha256={name:audit.tensor_digest(v) for name,v in model.fc.state_dict().items()},
        response_layout='single200',data_seed_rule='seed*100000+epoch (epoch 1-based)',
        forward_rng_rule='R0 replay; one g0/R1 advance; passive evaluation restored')
    reference=Path(cfg['reference_init_output'])
    expected=json.loads((reference/'task_identities.json').read_text())[0]
    parity_keys=('task_initial_parameter_sha256','frozen_origin_sha256','new_fc_rows_sha256')
    origin_identity['reference_initialization_exact']=all(origin_identity[k] == expected[k] for k in parity_keys)
    if not origin_identity['reference_initialization_exact']:
        raise AssertionError('Historical LoRA/head initialization parity failed: '+repr(origin_identity))
    if cfg['train_scope'] == 'full_ft':
        original_cfg=model.backbone.base_vit.pretrained_cfg
        model=full_ft_from_lora_origin(model).to(device)
    else:
        original_cfg=model.backbone.lora_vit.pretrained_cfg
    pretrained_tag=original_cfg.get('tag')
    if pretrained_tag != cfg['model_pretrained_tag']:
        raise AssertionError('Unexpected pretrained ViT tag: '+repr(original_cfg))
    del learner
    actual=scope_receipt(model,cfg['train_scope'])
    actual.update(pretrained_cfg=original_cfg,model_pretrained_tag=pretrained_tag,
        pretrained_backbone_sha256=parameter_hash(model.backbone.base_vit if cfg['train_scope']=='lora' else model.backbone),
        initial_fc_sha256={name:audit.tensor_digest(v) for name,v in model.fc.state_dict().items()},
        actual_initial_parameter_sha256=parameter_hash(model))
    if actual['initial_fc_sha256'] != origin_identity['new_fc_rows_sha256']:
        raise AssertionError('Training scope altered paired classifier initialization')
    train=dm.get_dataset(np.arange(200),source='train',mode='train')
    test=dm.get_dataset(np.arange(200),source='test',mode='test')
    if len(train)!=5981 or len(test)!=1519 or list(map(int,dm._class_order))!=list(range(200)):
        raise AssertionError('Complete fixed class order ImageNet-A split required')
    dataset=dict(full_train_samples=5981,full_test_samples=1519,nb_classes=200,
        class_order=list(range(200)),task_increments=[200],batches_per_epoch=47,
        train_targets_sha256=audit.tensor_digest(torch.as_tensor(dm._train_targets)),
        test_targets_sha256=audit.tensor_digest(torch.as_tensor(dm._test_targets)))
    return model,dm,train,test,origin_identity,actual,dataset


def worker_init(_):
    seed=torch.initial_seed()%(2**32);np.random.seed(seed);random.seed(seed)


def epoch_batches(dm,dataset,cfg,epoch,max_batches):
    """Independent exact historical sampler/augmentation stream."""
    seed=cfg['seed']*100000+epoch
    loader=dm.build_dataloader(dataset,batch_size=128,shuffle=True,
        num_workers=cfg.get('train_num_workers',8),generator=torch.Generator().manual_seed(seed),
        worker_init_fn=worker_init,persistent_workers=False)
    train_rng=rng_get();audit.seed_all(seed);data_rng=rng_get();rng_set(train_rng)
    iterator=None
    for batch in range(min(len(loader),max_batches) if max_batches else len(loader)):
        train_rng=rng_get();rng_set(data_rng)
        try:
            if iterator is None: iterator=iter(loader)
            ids,inputs,targets=next(iterator);data_rng=rng_get()
        finally:
            rng_set(train_rng)
        yield batch,ids,inputs,targets,dict(input_sha256=audit.tensor_digest(inputs),
            target_sha256=audit.tensor_digest(targets),indices_sha256=audit.tensor_digest(ids),
            indices=ids.tolist(),data_seed=seed)


@contextlib.contextmanager
def observational(model):
    state=State(model)
    try:
        yield
    finally:
        state.restore();state.check()


def evaluate(model,loader,device):
    count=correct=0;ce=0.;group_count=[0]*10;group_correct=[0]*10
    with observational(model):
        model.eval()
        with torch.no_grad():
            for _,inputs,targets in loader:
                inputs,targets=inputs.to(device),targets.to(device)
                logits=model(inputs)['logits']
                if logits.shape[-1]!=200:raise AssertionError('Classifier must expose all 200 classes')
                ce+=float(F.cross_entropy(logits,targets,reduction='sum'))
                hit=logits.argmax(-1).eq(targets);count+=targets.numel();correct+=int(hit.sum())
                for group in range(10):
                    mask=(targets>=20*group)&(targets<20*(group+1))
                    group_count[group]+=int(mask.sum());group_correct[group]+=int((mask&hit).sum())
    if count!=1519 or any(n<=0 for n in group_count) or not math.isfinite(ce):
        raise AssertionError('Test evaluation incomplete/nonfinite')
    groups=[100.*c/n for c,n in zip(group_correct,group_count)]
    return dict(test_accuracy=100.*correct/count,test_ce=ce/count,
        test_task_macro_accuracy=sum(groups)/10,test_group_accuracies=groups,
        test_group_counts=group_count,test_samples=count,classification_classes=200,
        passive_state_and_rng_restored=True,test_used_for_selection=False)


def cuda_witness(output):
    """Kernel witness also records the actual worker and GPU assignment."""
    from cuda_witness import main as witness_main
    previous=sys.argv
    state=rng_get()
    try:
        sys.argv=[previous[0],'--output',str(output/'cuda_witness.json')]
        witness_main()
    finally:
        sys.argv=previous;rng_set(state)
    witness=json.loads((output/'cuda_witness.json').read_text())
    expected_uuid=os.environ.get('SINGLE200_EXPECTED_GPU_UUID')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if expected_uuid and visible!=expected_uuid:
        raise AssertionError('Expected assigned GPU UUID is not the sole visible GPU')
    if torch.cuda.device_count()!=1 or torch.cuda.current_device()!=0:
        raise AssertionError('Each worker must expose exactly its one assigned GPU')
    stat=Path('/proc/self/stat').read_text();tail=stat[stat.rfind(')')+2:].split()
    witness.update(pid=os.getpid(),process_starttime=tail[19],pid_namespace=os.readlink('/proc/self/ns/pid'),
        expected_physical_gpu=os.environ.get('SINGLE200_EXPECTED_GPU'),expected_uuid=expected_uuid,
        cuda_visible_devices=visible,cuda_logical_device=0,visible_device_count=1,
        assigned_device_match=True)
    audit.atomic_json(output/'cuda_witness.json',witness)
    return witness


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','config','output-dir'):parser.add_argument('--'+name,required=True)
    for name in ('max-epochs','max-batches'):parser.add_argument('--'+name,type=int,default=0)
    parser.add_argument('--dry-run',action='store_true')
    cli=parser.parse_args()
    source=Path(cli.source).resolve();config=Path(cli.config).resolve();out=Path(cli.output_dir).resolve()
    cfg=resolve_config(json.loads(config.read_text()),cli)
    if cli.dry_run:
        print(json.dumps(dict(status='valid',protocol=PROTOCOL,train_scope=cfg['train_scope'],
            method=cfg['method'],epochs=cli.max_epochs or 50,batches=cli.max_batches or 47,
            ordinary_gradient_calls=READS[cfg['method']]),allow_nan=False));return
    # Queue wrappers may have already produced the CUDA witness in this directory.
    if out.exists() and any(p.name!='cuda_witness.json' for p in out.iterdir()):
        raise FileExistsError('Training output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Strict CUBLAS workspace required')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    os.environ.setdefault('HF_HUB_OFFLINE','1');torch.cuda.set_device(0)
    witness=cuda_witness(out)
    integrity=source_integrity(source)
    started=time.time();epochs=[];steps=0
    counters=dict(training_batches=0,training_samples=0,training_ce_gradient_passes=0,
        training_forward_count=0,training_sgd_updates=0,diagnostic_loss_forward_count=0,
        fisher_estimation_passes=0,weight_merge_updates=0,hvp_calls=0,create_graph_calls=0,
        gradient_clipping_calls=0,final_norm_rescale_calls=0,projection_calls=0,
        forbidden_work_attempts=0,training_hvp_passes=0,training_create_graph_passes=0)
    summary=dict(status='running',protocol=PROTOCOL,train_scope=cfg['train_scope'],method=cfg['method'],
        seed=2,smoke=bool(cli.max_epochs),completed_epochs=0,completed_steps=0,
        counters=counters,epochs_per_task=cli.max_epochs or 50,completed_tasks=0,
        full_partition_tasks=1,total_dataset_classes=200,r=1.,rho=.05,
        selection_rule='fixed epoch50 endpoint; test never selects checkpoints',
        BWT=None,forgetting=None,AAA=None,input_config_sha256=audit.digest(config),
        trainer_sha256=audit.digest(__file__),cuda_witness=witness,source_integrity=integrity)
    audit.atomic_json(out/'final_summary.json',summary)
    trace=out/'step_trace.jsonl'
    try:
        model,dm,train,test,identity,model_receipt,dataset=initialize_model_and_data(
            cfg,source,out,torch.device('cuda:0'))
        audit.atomic_json(out/'initial_identity.json',identity)
        audit.atomic_json(out/'task_identities.json',[identity])
        audit.atomic_json(out/'model_scope_receipt.json',model_receipt)
        audit.atomic_json(out/'dataset_partition_receipt.json',dataset)
        # Do not preserve stale CL/dynamic-radius labels as the effective algorithm.
        effective=copy.deepcopy(cfg)
        for name in ('upward_direction','response_method','component_rule','update_direction',
                     'response_radius_mode','study_role','study_condition_id','study_dev_split_path',
                     'response_algorithm_family','response_lambda_mode','cl_eval_task_size'):
            effective.pop(name,None)
        effective.update(model_scope_receipt=model_receipt,device=['cuda:0'])
        audit.atomic_json(out/'effective_config.json',effective)
        params=[p for p in model.parameters() if p.requires_grad]
        trainable_names=[name for name,p in model.named_parameters() if p.requires_grad]
        optimizer=torch.optim.SGD(params,lr=cfg['lr'],momentum=0.,weight_decay=0.)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=cfg['lr']*.1)
        from five_method_step import FiveMethodStep
        stepper=FiveMethodStep(model,params,dict(method=cfg['method'],rho=.05,radius=1.,
            eps=1e-12,gamma=cfg['gamma'],backend='native'))
        summary['algorithm_sha256']=audit.digest(Path(__file__).with_name('five_method_step.py'))
        loader=dm.build_dataloader(test,batch_size=128,shuffle=False,
            num_workers=cfg.get('train_num_workers',8))
        reference_trace=Path(cfg['reference_init_output'])/'step_trace.jsonl'
        with reference_trace.open() as reference_handle:
            reference_first=json.loads(reference_handle.readline())
        expected_batches=cli.max_batches or 47
        with trace.open('x') as handle:
            for epoch in range(1,(cli.max_epochs or 50)+1):
                model.train();samples=correct=0;loss_sum=0.;order=hashlib.sha256()
                epoch_started=time.time();lr=float(optimizer.param_groups[0]['lr'])
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                for batch,ids,inputs,targets,batch_identity in epoch_batches(dm,train,cfg,epoch,cli.max_batches):
                    order.update(ids.contiguous().numpy().tobytes())
                    x,y=inputs.to('cuda:0'),targets.to('cuda:0')
                    def closure(a,b):
                        logits=model(a)['logits']
                        return F.cross_entropy(logits,b,reduction='mean'),logits
                    actual_forward_count=[0]
                    def count_model_forward(*_):
                        actual_forward_count[0]+=1
                    forward_hook=model.register_forward_hook(count_model_forward)
                    try:
                        result=stepper.compute(x,y,closure,lr)
                        update=stepper.apply_update(lr,optimizer=optimizer)
                        committed=State(model)
                        try:
                            rng_set(result['context']['rng0'])
                            with torch.no_grad():after,_=closure(x,y)
                            loss_after=float(after)
                        finally:
                            committed.restore();diagnostic_checks=committed.check()
                    finally:
                        forward_hook.remove()
                    row=dict(result['record'],**update)
                    if row['ordinary_gradient_calls']!=READS[cfg['method']] or row['sgd_updates']!=1:
                        raise AssertionError('Gradient/actual SGD budget mismatch')
                    if actual_forward_count[0]!=READS[cfg['method']]+1:
                        raise AssertionError('Actual model forward hook count differs from gradient reads plus post-update observation')
                    row.update(step_index=steps,task=0,epoch=epoch,batch=batch,task_step_index=steps,
                        seed=2,train_scope=cfg['train_scope'],method=cfg['method'],identity=batch_identity,
                        class_slice=[0,200],seen_classes=200,fc_shape=[200,768],
                        train_ce=result['loss0'],post_update_ce=loss_after,
                        post_update_delta_ce=loss_after-result['loss0'],nominal_lr=lr,
                        batch_samples=int(y.numel()),diagnostic_loss_forward_count=1,
                        diagnostic_state_restoration=diagnostic_checks,
                        strict_deterministic=torch.are_deterministic_algorithms_enabled(),
                        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                        actual_model_forward_count=actual_forward_count[0],
                        actual_model_forward_count_measured=True,
                        probe_radius_coordinates='global L2 over '+('LoRA A/B plus FC' if cfg['train_scope']=='lora' else 'all ViT and FC tensors'))
                    if steps == 0:
                        input_match=all(batch_identity[key] == reference_first['identity'][key]
                            for key in ('input_sha256','target_sha256','indices_sha256','indices','data_seed'))
                        gradient_match=(row['g0_sha256'] == reference_first['g0_sha256']) if cfg['train_scope']=='lora' else None
                        if not input_match or gradient_match is False:
                            raise AssertionError('Actual first batch / LoRA clean gradient is not paired with reused results')
                        row['historical_first_input_exact']=input_match
                        row['historical_first_lora_g0_exact']=gradient_match
                        if cfg['train_scope']=='full_ft':
                            # Measure the actual rounded update, rather than the
                            # intended gradient or the all-trainable declaration.
                            group_squared={'backbone':0.,'fc':0.}
                            group_nonzero={'backbone':0,'fc':0}
                            for name,param,base_value in zip(trainable_names,params,result['context']['base']):
                                group='fc' if name.startswith('fc.') else 'backbone'
                                if group=='backbone' and not name.startswith('backbone.'):
                                    raise AssertionError('Unexpected full-FT trainable scope tensor')
                                delta=param.detach()-base_value
                                group_squared[group]+=float((delta.double()**2).sum())
                                group_nonzero[group]+=int(bool(torch.count_nonzero(delta)))
                            del delta
                            displacement={group:math.sqrt(value) for group,value in group_squared.items()}
                            if any(displacement[group]<=0. or group_nonzero[group]<=0 for group in ('backbone','fc')):
                                raise AssertionError('Actual full-FT first update must change both pretrained backbone and classifier')
                            row['actual_first_update_scope_displacement']=dict(
                                measurement='FP64 norm of actual rounded FP32 parameter deltas from entry base',
                                backbone_parameter_displacement_norm=displacement['backbone'],
                                fc_parameter_displacement_norm=displacement['fc'],
                                backbone_nonzero_updated_tensor_count=group_nonzero['backbone'],
                                fc_nonzero_updated_tensor_count=group_nonzero['fc'],
                                backbone_and_fc_both_updated=True)
                            summary['actual_first_update_scope_displacement']=row['actual_first_update_scope_displacement']
                    if audit.nonfinite_paths(row):raise FloatingPointError('Nonfinite step trace')
                    handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
                    steps+=1;samples+=y.numel();loss_sum+=result['loss0']*y.numel()
                    correct+=int(result['logits0'].argmax(-1).eq(y).sum())
                    counters['training_batches']+=1;counters['training_samples']+=y.numel()
                    counters['training_ce_gradient_passes']+=READS[cfg['method']]
                    counters['training_forward_count']+=READS[cfg['method']]
                    counters['training_sgd_updates']+=1;counters['diagnostic_loss_forward_count']+=1
                    counters['projection_calls']+=int(cfg['method']=='gam')
                    # Drop full-FT transient gradient/snapshot references before next batch.
                    del result,update,committed,row,x,y
                if batch+1!=expected_batches or (not cli.max_batches and samples!=5981):
                    raise AssertionError('Complete epoch sample budget mismatch')
                scheduler.step()
                torch.cuda.synchronize();seconds=time.time()-epoch_started
                test_metrics=evaluate(model,loader,torch.device('cuda:0'))
                epoch_row=dict(epoch=epoch,samples=samples,batches=expected_batches,train_ce=loss_sum/samples,
                    train_accuracy=100.*correct/samples,lr_used=[lr],
                    lr_after_scheduler=[float(optimizer.param_groups[0]['lr'])],
                    sampler_order_sha256=order.hexdigest(),seconds=seconds,
                    peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),**test_metrics)
                epochs.append(epoch_row)
                audit.atomic_json(out/'metrics.json',dict(epochs=epochs,counters=counters))
                audit.atomic_json(out/'epoch_test_curve.json',dict(epochs=epochs,
                    evaluation_scope='all1519 images; all200 classifier',
                    selection_rule='fixed endpoint; no best-epoch selection'))
                summary.update(completed_epochs=epoch,completed_steps=steps,counters=counters,
                    initial_identity=identity,endpoint_test=test_metrics)
                audit.atomic_json(out/'final_summary.json',summary)
                handle.flush();os.fsync(handle.fileno())
                print(json.dumps(dict(event='single200_epoch_complete',train_scope=cfg['train_scope'],
                    method=cfg['method'],completed_steps=steps,**epoch_row)),flush=True)
        expected_steps=(cli.max_epochs or 50)*expected_batches
        if steps!=expected_steps or counters['training_ce_gradient_passes']!=steps*READS[cfg['method']]:
            raise AssertionError('Incomplete formal/smoke budget')
        checkpoint=out/'checkpoints'/'final.pt';checkpoint.parent.mkdir(parents=True)
        torch.save(dict(model={n:p.detach().cpu() for n,p in model.state_dict().items()},
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epochs[-1]['epoch'],
            config=effective,rng=rng_get(),train_scope=cfg['train_scope']),checkpoint)
        integrity=source_integrity(source)
        summary.update(status='completed',completed_tasks=1,completed_epochs=len(epochs),completed_steps=steps,
            endpoint_test={k:epochs[-1][k] for k in test_metrics},counters=counters,
            initial_identity=identity,model_scope_receipt=model_receipt,dataset_partition_receipt=dataset,
            source_integrity=integrity,elapsed_seconds=time.time()-started,
            checkpoint_path=str(checkpoint),checkpoint_sha256=audit.digest(checkpoint),
            final_model_parameter_sha256=parameter_hash(model),step_trace_sha256=audit.digest(trace),
            effective_config_sha256=audit.digest(out/'effective_config.json'),
            epoch_curve_sha256=audit.digest(out/'epoch_test_curve.json'),
            checks=dict(finite=True,exact_snapshot_restore=True,one_sgd_per_batch=True,
                full_original_train_test_used=True,test_not_used_for_selection=True,strict_deterministic=True,
                AMP_disabled=True,TF32_disabled=True,no_history=True,no_fisher=True,no_weight_merges=True,
                source_immutable=True,initialization_reference_exact=True,complete_budget=True,
                assigned_gpu_match=True,no_final_normalization=True,no_HVP=True))
    except BaseException as exc:
        summary.update(status='failed',error=repr(exc),completed_steps=steps,completed_epochs=len(epochs));raise
    finally:
        audit.atomic_json(out/'final_summary.json',summary)


if __name__=='__main__':
    main()
