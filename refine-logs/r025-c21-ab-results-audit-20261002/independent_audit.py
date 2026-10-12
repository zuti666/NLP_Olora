import json, math, pathlib, statistics

DEST = pathlib.Path(__file__).parent
ROOT = pathlib.Path('/data/140-1/users/liying/CopyFlatCV/refine-logs')
OLD = ROOT / 's3-component-horizon-recovery-20261001T230000Z'
NEW = ROOT / 's3-r025-c21-ab-20261002T080000Z'
paths = sorted((OLD/'outputs').glob('screen_*')) + sorted((NEW/'outputs').glob('screen_*'))
assert len(paths) == 11
ref = OLD / 'outputs/screen_us_matched_r025_s2'
read = lambda p: json.loads(p.read_text())
reference_cfg = read(ref/'effective_config.json')
trace = lambda p: [json.loads(line) for line in (p/'step_trace.jsonl').read_text().splitlines()]
reference_trace = trace(ref)
reference_ids = read(ref/'task_identities.json')
reference_epochs = read(ref/'metrics.json')['epochs']
allowed_cfg = {'component_rule','component_scale','outputs_root','prefix','filepath','feature_flat_save_path'}
identity_keys = ['seed','class_slice','fc_shape','new_fc_rows_sha256','frozen_origin_sha256','data_seed_rule','native_global_rng_protocol_preserved']
rows = []
for p in paths:
    cfg = read(p/'effective_config.json')
    diff = {k:[reference_cfg.get(k),cfg.get(k)] for k in reference_cfg.keys()|cfg.keys() if reference_cfg.get(k)!=cfg.get(k)}
    assert not (diff.keys()-allowed_cfg), (p.name, diff)
    seq = trace(p)
    assert len(seq)==2550
    for i,(a,b) in enumerate(zip(reference_trace,seq)):
        for k in ['identity','task','epoch','batch','batch_samples','seen_classes','fc_shape','lr_nominal','training_ce_gradient_passes','training_forward_count']:
            assert a[k]==b[k],(p.name,i,k)
        for k in ['self_spec_passed','component_spec_exact','projector_spec_exact','base_sgd_exact','native_decomposition_exact']:
            assert b[k],(p.name,i,k)
        assert b['closure_count']==4
        if cfg['component_scale']=='natural':
            assert b['scale_over_nominal_lr']==1.0
    ids=read(p/'task_identities.json')
    assert len(ids)==10
    assert ids[0]['task_initial_parameter_sha256']==reference_ids[0]['task_initial_parameter_sha256']
    for a,b in zip(reference_ids,ids):
        for k in identity_keys:
            assert a[k]==b[k],(p.name,k)
        assert b['old_fc_rows_inherited_exact']
    epochs=read(p/'metrics.json')['epochs']
    assert len(epochs)==500
    for a,b in zip(reference_epochs,epochs):
        for k in ['task','epoch','samples','batches','lr_used','lr_after_scheduler','sampler_order_sha256']:
            assert a[k]==b[k],(p.name,k)
    tm=read(p/'task_matrix.json'); m=tm['matrix']
    assert [len(x) for x in m]==list(range(1,11))
    stages=[statistics.mean(x) for x in m]
    result={'AAA_task_macro':statistics.mean(stages),'FAA_task_macro':statistics.mean(m[-1]),'LA':statistics.mean(m[i][i] for i in range(10)),
            'BWT':statistics.mean(m[-1][i]-m[i][i] for i in range(9)),
            'forgetting':statistics.mean(max(m[t][i] for t in range(i,10))-m[-1][i] for i in range(9))}
    for k,v in result.items(): assert abs(v-tm[k])<1e-10,(p.name,k,v,tm[k])
    result.update({'id':p.name,'path':str(p),'rule':cfg['component_rule'],'scale':cfg['component_scale'],
                   'FAA_sample_weighted':tm['FAA_sample_weighted'],'AAA_sample_weighted':tm['AAA_sample_weighted'],
                   'stage_task_macro':stages,'final_task_accuracy':m[-1],
                   'learning_task_accuracy':[m[i][i] for i in range(10)],
                   'initial_parameters_match':True,'all_step_inputs_targets_indices_match':True,
                   'new_head_init_match':True,'epoch_lr_sampler_match':True,'non_target_cfg_match':True,
                   'config_difference_keys':sorted(diff),
                   'actual_step_ratio_mean':statistics.mean(x['actual_to_reference_norm_ratio'] for x in seq)})
    if cfg['component_rule'].startswith('c21_'):
        c=[x['c21_audit'] for x in seq]
        for x in c:
            assert x['projection_scope']=='all_trainable_LoRA_and_seen_FC'
            assert x['reduction_dtype']=='float64'
            assert x['selected_spec_exact'] and x['orthogonality_passed']
        result['c21_audit']={'projection_scope':'all_trainable_LoRA_and_seen_FC','steps':len(c),
            'a_mean':statistics.mean(x['coefficient_a'] for x in c),
            'a_min':min(x['coefficient_a'] for x in c),'a_max':max(x['coefficient_a'] for x in c),
            'perp_ratio_mean':statistics.mean(x['perp_over_c21'] for x in c),
            'orthogonality_relative_max':max(x['orthogonality_relative_error'] for x in c),
            'g1_zero_count':sum(x['g1_zero'] for x in c),'c21_zero_count':sum(x['c21_zero'] for x in c)}
    rows.append(result)

natural={x['rule']:x for x in rows if x['scale']=='natural' or x['rule']=='us'}
deltas={}
for a in ['c21_base','c21_perp']:
    for b in ['u0','us','uo','ufull','ug']:
        deltas[f'{a}_minus_{b}']={k:natural[a][k]-natural[b][k] for k in ['AAA_task_macro','FAA_task_macro','LA','BWT','forgetting']}
        deltas[f'{a}_minus_{b}']['final_task_win_count']=sum(x>y for x,y in zip(natural[a]['final_task_accuracy'],natural[b]['final_task_accuracy']))
audit={'status':'passed','full_runs':11,'steps_checked':28050,'steps_per_run':2550,'epochs_per_run':500,
       'reference':str(ref),'rows':rows,'deltas_natural':deltas,
       'notes':['Full config equality except rule/scale and run output labels/paths.',
                'g1 matched entry is exact nominal-SGD identity control; reuse in natural table.',
                'AAA/FAA here are task-equally-weighted; sample-weighted metrics are separate.',
                'Only a single seed at r=.25: no inferential significance, no Hessian causality demonstrated.']}
(DEST/'agent_results.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'status':'passed','full_runs':11,'steps_checked':28050,'rows':[{k:x[k] for k in ['rule','scale','AAA_task_macro','FAA_task_macro','LA','BWT','forgetting','actual_step_ratio_mean']} for x in rows],'deltas':deltas},indent=2))
