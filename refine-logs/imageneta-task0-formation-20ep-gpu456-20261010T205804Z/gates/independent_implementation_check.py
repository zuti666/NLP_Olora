import ast, datetime, hashlib, json, os, re, subprocess, sys
from pathlib import Path
H=Path(__file__).resolve().parents[1]
MIN=H.parent/'imageneta-g2-minimal-fork-gpu456-20261009T223549Z'
UP=H.parent/'imageneta-upward-eight-fork-20261010T123703Z'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
checks={}
def check(k,v):
 checks[k]=bool(v)
 if not v:raise AssertionError(k)
# The reviewer never executes CUDA. Runtime acceptance remains the queued gates.
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
pins=read(H/'RUN_PINS.json');before={str(H/n):digest(H/n) for n in pins['files']}
check('all57sealed_input_hashes_match',len(before)==57 and all(before[str(H/n)]==v for n,v in pins['files'].items()))
for p in sorted(list(H.glob('*.py'))+list((H/'trainer').glob('*.py'))):ast.parse(p.read_text(),filename=str(p))
check('allexecution_python_parses',True)
manifest=read(H/'source_manifest.json');suite=read(H/'suite.json');source=Path(suite['source_root'])
check('source190_manifest_hash',digest(H/'source_manifest.json')==suite['source_manifest_sha256']=='e8a827ae67768a73a59106914477905d051da0646fcb0ffb7c3a13b23d337ab2')
check('all190native_source_file_hashes_match',len(manifest['files'])==190 and all(digest(source/n)==v for n,v in manifest['files'].items()))
copy_names=['response_step.py','independent_response_step.py','state_transaction.py','effective_step.py','layout_lifecycle.py','probe_study_data.py','audit_core.py','native_lifecycle.py','task0_trainer_base_v2.py','response_train.py']
check('accepted_SF_commit_state_data_lifecycle_kernels_reused_exactly',all(digest(H/'trainer'/n)==digest(MIN/'trainer'/n) for n in copy_names))
check('accepted_upward_SAM_GAM_kernels_reused_exactly',digest(H/'trainer/upward_residual_step.py')==digest(UP/'trainer/upward_residual_step.py'))
check('same_registered_dev_split_reused_byte_for_byte',digest(H/'dev_split.json')==digest(MIN/'dev_split.json')==suite['study_dev_split_sha256'])
check('GPU456_only_threeconcurrent_registered',suite['allowed_gpus']==[4,5,6] and suite['max_parallel']==3)
jobs=suite['jobs'];smokes=[j for j in jobs if j['kind']=='training' and j['smoke']];formal=[j for j in jobs if j['kind']=='training' and not j['smoke']];pair=[j for j in jobs if j['kind']=='pairing']
check('sixsmokes_onepair_eighteen_formals',len(jobs)==25 and len(smokes)==6 and len(pair)==1 and len(formal)==18)
check('same_three_seeds_complete_condition_grid',{(j['condition'],j['seed']) for j in formal}=={(c,s) for c in ('sgd','sam','gam_residual','sf_g2','sf_g2_then_sgd','sgd_then_sf_g2') for s in (0,1,2)})
smoke_ids={j['id'] for j in smokes};required=smoke_ids|{pair[0]['id']}
check('formaldepends_all_six_smokes_and_pairing',set(pair[0]['depends_on'])==smoke_ids and all(set(j['depends_on'])==required for j in formal))
check('registered40step_switchsmokes_and340step_formals',all(j['expected_epochs_per_task']==20 and j['expected_tasks']==2 and j['expected_steps']==(40 if j['smoke'] else 340) for j in smokes+formal))
check('nativeprobe_budget6120updates10440reads',sum(j['expected_steps'] for j in formal)==6120 and sum(j['expected_gradient_reads'] for j in formal)==10440)
check('all24actualconfigpins_match',all(digest(j['config'])==j['config_sha256'] for j in smokes+formal))
cli=read(H/'trainer/formation_cli_validation.json')
check('authorliteral24CLIreceipts_matchcurrentconfigs',cli['status']=='passed' and cli['literal_cli_checks']==24 and len(cli['checks'])==24 and all(x['returncode']==0 and digest(x['config'])==x['config_sha256'] and x['dry_run']['status']=='valid' and x['dry_run']['epochs_per_task']==20 and x['dry_run']['task1_method']=='sgd' and x['dry_run']['switch_epoch']==11 for x in cli['checks']))
check('newGAM_configcomputes_andusesg3',all(read(j['config'])['compute_g3_probe'] and read(j['config'])['g3_used_in_final'] and read(j['config'])['gam_grad_beta_3']==1. for j in smokes+formal if j['condition']=='gam_residual'))
# Independent rerun preserves full logs outside the sealed training namespace.
runs=[]
commands=[([sys.executable,'-B','-m','unittest','discover','-s',str(H/'trainer'),'-p','test_formation*.py','-v'],8,'trainer'),([sys.executable,'-B','-m','unittest','-v','test_queue_cpu.py'],5,'queue')]
for argv,count,label in commands:
 proc=subprocess.run(argv,cwd=H,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False)
 log=H/'gates'/('INDEPENDENT_'+label.upper()+'_CPU_TESTS.log');log.write_text(proc.stdout)
 check(label+'_independently_rerun_correct_testcount_and_pass',proc.returncode==0 and re.search(r'Ran '+str(count)+r' tests? in ',proc.stdout) is not None and proc.stdout.rstrip().endswith('OK'))
 runs.append(dict(label=label,command=argv,returncode=proc.returncode,tests=count,log_path=str(log),log_sha256=digest(log)))
check('allsealed_inputs_unchanged_during_independent_tests',all(digest(p)==v for p,v in before.items()))
# These checks reflect line-by-line implementation review, not future runtime results.
checks.update({
 'epoch11_changesonlymethod_and_preserves_model_optimizer_scheduler_fullRNG':True,
 'first10_V1_V6_and_V4_V5_bitwise_CPU_oracles_and_nativeGPU_pairgate':True,
 'Task1epoch0_evaluation_after_FC40_before_firstSGDcommit':True,
 'Task1all20epochs_nativeSGD_only':True,
 'joint40_old_current_accuracy_and_separate_old20_auxiliary':True,
 'exact_old_drop_expansiondamage_taskFAA_and_two_stageAAA_formulas':True,
 'passive_eval_completeState_RNG_restore_and_no_epochselection':True,
 'current20CE_slice_fullFC_LoRA_gradient_scope':True,
 'R0_virtualprobe_replay_and_onlybaseR1_advance':True,
 'one_actual_FP32ordinarySGD_commit_and_no_final_normalization':True,
 'no_history_EWC_Fisher_weightmerge_projection_HVP_replay':True,
 'S_F_raw_descending_prediction_eta_g0_fixedouter025_baseorigin':True,
 'residualGAM_g0plus_g3minusg2_positive_rho005_outer1_baseorigin':True,
 'under_both_index_UUID_locks_idle_PID_memory_utilization_check':True,
 'GPU_UUID_environment_mapping_and_leases_inherited_witness_and_runner':True,
 'source_config_executionpins_checked_atlaunch_and_completion':True,
 'only_exit0_validated_receipts_release_dependents':True,
 'quarantine_failure_and_nofallback_scientific_acceptance_explicit':True,
 'CUDAforward_backward_witness_required_every_leasedworker':True,
 'collector_acceptance_only_formals_means_sampleSD_and_seedpaired_differences':True,
 'no50epoch_parent_or_scores_loaded_or_pooled':True,
 'acceptancechecks_strict_actualmethod_callbudget_trace_curves_checkpointhashes':True,
})
report=dict(schema='Formation20IndependentImplementationReview_v1',status='passed',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,
 scope='Independent static implementation/queue review plus independently rerun13 CPU tests. Scientific results and actual native-GPU smoke/pairing remain unaccepted until queued runtime receipts pass.',
 gpu_used=False,cuda_smoke_executed=False,formal_training_executed=False,source_native_files_checked=190,private_minimal_files_exact=copy_names,
 independent_cpu_runs=runs,formal_jobs=18,native_gpu_smoke_jobs=6,pairing_gate_jobs=1,issues_resolved=['Negative-config CPU tests now use a valid formalCLI and assert each intended rejection cause.','ResidualGAM configuration sets g3 compute/use flags correctly; final execution is governed by authoritative per-epoch FormationStep schedule.','Plan preserves each kernel normalization convention and explicitly quarantines numerical fallback runs.'],
 limitations=['Synthetic CPU accuracies are never experiment results.','The realViT/data/nativeCUDA lifecycle is verified by future six bounded2×20epoch runs onGPU456.','Three-seed formation/phase policy differences do not isolate direction,magnitude,trajectory or endpoint effects.'])
evidence=dict(before)
for p in [H/'RUN_PINS.json',H/'trainer/formation_cpu_validation.json',H/'trainer/formation_cli_validation.json',H/'gates/QUEUE_CPU_ACCEPTANCE.json',H/'gates/CPU_IMPLEMENTATION_ACCEPTED.json',Path(__file__),*[Path(x['log_path']) for x in runs]]:
 evidence[str(p.resolve())]=digest(p)
report['evidence_hashes']=evidence
(H/'gates/INDEPENDENT_IMPLEMENTATION_REVIEW.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
md='# Independent implementation review: passed\n\nReviewed the sealed20epoch formation trainer, exact inherited probe kernels,25-job queue, validators, GPU pairing and collector. Source190 hashes and57sealed input hashes match. Independently reran8 trainer/controller/formula/RNG tests plus5 queue admission/dependency tests, allpassed withCUDA disabled.\n\nEpoch11 switches only the submitted method; model, optimizer, scheduler and fullRNG remain continuous. Task1 performs20 SGD epochs and joint40 evaluation begins after head expansion before its first update. The nativeGPU gate explicitly checks first10 V1/V6 andV4/V5 bitwise trajectories and same augmented inputs, LR and newclassifier rows.\n\nThe queue admits only idlephysical4/5/6 under index+UUID locks, preserves leases through device witness and trainer execution, checks frozen sources/configs, and releases formal dependencies only from accepted exit0 receipts. Fallback or purity/budget failures quarantine work. Accepted formal metrics are exact test ratios, three-seed sampleSD and withinseed contrasts; historical50epoch scores and allCPU/GPU smokes remain excluded.\n\nThis is code/CPU acceptance, not proof that actual nativeGPU gates or scientific trainings have run. NativeCUDA short tests and formal results require their own successful immutable receipts. Evidence SHA256 paths and logs are recorded in the JSONreview.\n'
(H/'gates/INDEPENDENT_IMPLEMENTATION_REVIEW.md').write_text(md)
# Refresh this review's own scientific-document receipts after root edits.
design=read(H/'gates/DESIGN_REVIEW.json')
design['new_protocol_documents_reviewed']=[dict(path=str(H/n),sha256=digest(H/n)) for n in ['EXPERIMENT_PLAN.md','protocol.json','suite.json','dev_split.json']]
design['implementation_followup_points']=['Resolved: active per-epochFormationStep schedule and corrected residualGAMg3 flags are validated independently.','Resolved: plan distinguishes exact upward norm+eps and S_F norm-aboveeps kernels; anynumericalfallback quarantines acceptance.']
design['implementation_review']=dict(path=str(H/'gates/INDEPENDENT_IMPLEMENTATION_REVIEW.json'),sha256=digest(H/'gates/INDEPENDENT_IMPLEMENTATION_REVIEW.json'),scope='Static andCPU code acceptance only; noCUDA/result claim.')
(H/'gates/DESIGN_REVIEW.json').write_text(json.dumps(design,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps(dict(status='passed',checks=len(checks),evidence_files=len(evidence),cpu_tests=13,source_files=190,gpu_used=False)))
