#!/usr/bin/env python3
"""Fixed-checkpoint, read-only old/new conditional20 and joint40 evaluation.

No training, parameter selection, test-based checkpoint selection, or model
mutation is permitted. Fixed endpoints are e0/e1/e10/e25/e50 for both branches.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def group_metrics(logits, labels, group):
    """Partition joint errors and distinguish conditional mistakes/competition.

    CE sums intentionally use FP32 native PyTorch reduction for agreement with
    historical evaluation. Margins/probabilities are also saved per example.
    """
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    lo, hi = (0, 20) if group == 'old' else (20, 40)
    other_lo, other_hi = (20, 40) if group == 'old' else (0, 20)
    if logits.ndim != 2 or logits.shape[1] != 40:
        raise ValueError('Exactly 40 jointly competing class logits required')
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('Nonfinite checkpoint logits')
    if len(labels) != len(logits) or not bool(((labels >= lo) & (labels < hi)).all()):
        raise ValueError('Group labels outside fixed original twenty-class slice')
    own, other = logits[:, lo:hi], logits[:, other_lo:other_hi]
    joint_prediction = logits.argmax(-1)
    conditional_prediction = own.argmax(-1) + lo
    joint_correct = joint_prediction.eq(labels)
    conditional_correct = conditional_prediction.eq(labels)
    cross_group = (joint_prediction < lo) | (joint_prediction >= hi)
    within_group_error = ~joint_correct & ~cross_group
    joint_error = ~joint_correct
    if int(joint_error.sum()) != int(cross_group.sum() + within_group_error.sum()):
        raise AssertionError('Cross-group + within-group errors must equal joint40 errors')
    conditional_ce = F.cross_entropy(own, labels - lo, reduction='none')
    joint_ce = F.cross_entropy(logits, labels, reduction='none')
    competition_penalty = F.softplus(torch.logsumexp(other, -1) - torch.logsumexp(own, -1))
    identity_error = float((joint_ce - conditional_ce - competition_penalty).abs().max())
    if identity_error > 3e-6:
        raise AssertionError('Joint CE = conditional CE + competition penalty identity failed')
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    probabilities_joint = logits.softmax(-1).gather(1, labels[:, None]).squeeze(1)
    probabilities_conditional = own.softmax(-1).gather(1, (labels-lo)[:, None]).squeeze(1)
    count = len(labels)
    metrics = dict(samples=count, group=group,
        conditional20=dict(accuracy=100*int(conditional_correct.sum())/count,
                           ce=float(conditional_ce.sum())/count,
                           correct=int(conditional_correct.sum()), errors=int((~conditional_correct).sum())),
        joint40=dict(accuracy=100*int(joint_correct.sum())/count,
                     ce=float(joint_ce.sum())/count, correct=int(joint_correct.sum()), errors=int(joint_error.sum())),
        cross_group_error_count=int(cross_group.sum()),
        within_group_error_count=int(within_group_error.sum()),
        conditional_correct_lost_to_other_group_count=int((conditional_correct & cross_group).sum()),
        conditional_error_and_cross_group_count=int((~conditional_correct & cross_group).sum()),
        cross_group_error_rate=100*int(cross_group.sum())/count,
        within_group_error_rate=100*int(within_group_error.sum())/count,
        ce_competition_penalty=float(competition_penalty.mean()),
        ce_partition_max_abs_error=identity_error,
        true_class_probability_joint40_mean=float(probabilities_joint.mean()),
        true_class_probability_conditional20_mean=float(probabilities_conditional.mean()),
        own_best_minus_other_best_mean=float((own.max(-1).values-other.max(-1).values).mean()),
        true_class_minus_other_best_mean=float((true_logits-other.max(-1).values).mean()),
        partition_identity_passed=True)
    arrays = dict(labels=labels.cpu().numpy(), logits=logits.cpu().numpy(),
        prediction_joint40=joint_prediction.cpu().numpy(), prediction_conditional20=conditional_prediction.cpu().numpy(),
        cross_group_error=cross_group.cpu().numpy(), within_group_error=within_group_error.cpu().numpy(),
        conditional_ce=conditional_ce.cpu().numpy(), joint_ce=joint_ce.cpu().numpy(),
        ce_competition_penalty=competition_penalty.cpu().numpy(),
        true_class_probability_joint40=probabilities_joint.cpu().numpy(),
        true_class_probability_conditional20=probabilities_conditional.cpu().numpy(),
        own_best_minus_other_best=(own.max(-1).values-other.max(-1).values).cpu().numpy(),
        true_class_minus_other_best=(true_logits-other.max(-1).values).cpu().numpy())
    return metrics, arrays


def assert_historical_agreement(actual, reference, label):
    for key, tolerance in (('accuracy', 1e-8), ('ce', 1e-5)):
        if abs(actual[key]-reference[key]) > tolerance:
            raise AssertionError('%s %s differs: %.12g versus %.12g' %
                                 (label, key, actual[key], reference[key]))
    if actual.get('samples', reference['samples']) != reference['samples']:
        raise AssertionError(label+' sample count differs')
    return dict(accepted=True, accuracy_abs_error=abs(actual['accuracy']-reference['accuracy']),
                ce_abs_error=abs(actual['ce']-reference['ce']), accuracy_tolerance=1e-8, ce_tolerance=1e-5)


def configure_runtime():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Strict deterministic CUBLAS environment required')
    if not torch.cuda.is_available():
        raise RuntimeError('Checkpoint ViT jobs require one scheduled CUDA device')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '2'))))


def evaluate_dataset(model, dm, dataset, group, expected):
    if len(dataset) != expected:
        raise AssertionError('Expected %d examples, got %d' % (expected, len(dataset)))
    loader = dm.build_dataloader(dataset, batch_size=128, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(20261012), persistent_workers=False)
    logits_list, labels_list, index_list = [], [], []
    joint_ce_sum = conditional_ce_sum = 0.
    lo = 0 if group == 'old' else 20
    with torch.no_grad():
        for index, inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = model(inputs)['logits'][:, :40]
            # Preserve identical FP32 reduction/order to original batch128 evaluator.
            joint_ce_sum += float(F.cross_entropy(logits, targets, reduction='sum'))
            conditional_ce_sum += float(F.cross_entropy(logits[:, lo:lo+20], targets-lo, reduction='sum'))
            logits_list.append(logits.cpu()); labels_list.append(targets.cpu()); index_list.append(index.cpu())
    metrics, arrays = group_metrics(torch.cat(logits_list), torch.cat(labels_list), group)
    metrics['joint40']['ce'] = joint_ce_sum/expected
    metrics['conditional20']['ce'] = conditional_ce_sum/expected
    arrays['dataset_local_indices'] = torch.cat(index_list).numpy()
    arrays['raw_train_indices'] = np.asarray(getattr(dataset, 'study_full_train_indices', []), dtype=np.int64)
    return metrics, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--seed', type=int, choices=(0,1,2), required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pilot', action='store_true')
    cli = parser.parse_args()
    root, output = Path(cli.campaign).resolve(), Path(cli.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Fresh output directory required; historical results are immutable')
    output.mkdir(parents=True, exist_ok=True)
    configure_runtime()
    sys.path.insert(0, str(root/'trainer'))
    from response_chain_train import load_snapshot
    from probe_study_data import build_data_manager
    from state_transaction import State
    epochs = (1,) if cli.pilot else (0,1,10,25,50)
    aggregate = dict(protocol='three_gaps_fixed_checkpoint_eval_v1', seed=cli.seed, pilot=cli.pilot,
        status='running', submitted_training_updates=0, test_used_for_selection=False,
        fixed_epochs=list(epochs), independent_repeat_unit='training seed', results=[])
    write_json(output/'evaluation_results.json', aggregate)
    dm = None
    try:
        for branch in ('S','R'):
            summary = json.loads((root/'outputs'/('formal_%s_s%d'%(branch,cli.seed))/'final_summary.json').read_text())
            references = {r['epoch']:r for r in summary['retention_curve']}
            for epoch in epochs:
                checkpoint = root/'outputs'/('formal_%s_s%d'%(branch,cli.seed))/'checkpoints'/('epoch_%03d.pt'%epoch)
                payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
                source = payload['config']['study_source']
                if source not in sys.path: sys.path.insert(0, source)
                model, payload = load_snapshot(checkpoint, 'cuda:0', restore_rng=True)
                if int(payload['config']['seed']) != cli.seed: raise AssertionError('Checkpoint seed mismatch')
                if dm is None:
                    dm = build_data_manager(source, payload['config'], payload['config']['study_dev_split_path'])
                transaction = State(model)
                row = dict(seed=cli.seed, branch=branch, epoch=epoch, checkpoint=str(checkpoint),
                    checkpoint_sha256=digest(checkpoint), datasets={}, agreement={}, submitted_training_updates=0,
                    old_training_images='evaluation only, no replay/update', transform='native deterministic test transform',
                    test_used_for_selection=False,source=source,
                    full_model_parameter_sha256=payload['full_model_parameter_sha256'],
                    dev_split_sha256=dm.study_split_sha256)
                arrays_all = {}
                try:
                    model.eval()
                    specifications = (
                        ('train_old','old',dm.get_dataset(range(0,20),source='train',mode='test'),1141),
                        ('train_new','new',dm.get_dataset(range(20,40),source='train',mode='test'),906),
                        ('dev_old','old',dm.get_dev_dataset(0),118),
                        ('dev_new','new',dm.get_dev_dataset(1),91),
                        ('test_old','old',dm.get_dataset(range(0,20),source='test',mode='test'),307),
                        ('test_new','new',dm.get_dataset(range(20,40),source='test',mode='test'),254))
                    for name, group, dataset, expected in specifications:
                        metrics, arrays = evaluate_dataset(model, dm, dataset, group, expected)
                        row['datasets'][name] = metrics
                        arrays_all.update({name+'__'+key:value for key,value in arrays.items()})
                    reference = references[epoch]
                    for name, refkey in (('test_old','old_test_allseen40'),('test_new','current_test_allseen40')):
                        actual = dict(row['datasets'][name]['joint40'],samples=row['datasets'][name]['samples'])
                        row['agreement'][name+'_joint40'] = assert_historical_agreement(actual, reference[refkey],name)
                    if 'current_test_conditional20' in reference:
                        actual = dict(row['datasets']['test_new']['conditional20'],samples=254)
                        row['agreement']['test_new_conditional20'] = assert_historical_agreement(actual, reference['current_test_conditional20'],'new conditional')
                    clean_reference = reference.get('clean_training',summary['clean_training_start'] if epoch==0 else None)
                    if clean_reference is not None:
                        actual = dict(row['datasets']['train_new']['conditional20'],samples=906)
                        row['agreement']['train_new_clean20'] = assert_historical_agreement(actual,clean_reference,'new clean train')
                    row['FAA'] = (row['datasets']['test_old']['joint40']['accuracy']+row['datasets']['test_new']['joint40']['accuracy'])/2
                    row['old_drop_since_shared_start'] = references[0]['old_test_allseen40']['accuracy']-row['datasets']['test_old']['joint40']['accuracy']
                finally:
                    transaction.restore()
                    row['restoration'] = transaction.check()
                filename = 'branch%s_epoch%03d_logits.npz'%(branch,epoch)
                np.savez_compressed(output/filename, **arrays_all)
                row['logits_file'] = str(output/filename)
                row['logits_sha256'] = digest(output/filename)
                row['status'] = 'accepted'
                write_json(output/('branch%s_epoch%03d.json'%(branch,epoch)),row)
                aggregate['results'].append(row)
                write_json(output/'evaluation_results.json',aggregate)
                print(json.dumps(dict(event='evaluation_complete',seed=cli.seed,branch=branch,epoch=epoch,FAA=row['FAA'],restoration=row['restoration'])),flush=True)
                del model,payload,transaction,arrays_all
                torch.cuda.empty_cache()
        aggregate.update(status='completed',completed_checkpoints=len(aggregate['results']))
        write_json(output/'evaluation_results.json',aggregate)
        write_json(output/'completion_receipt.json',dict(status='accepted',submitted_training_updates=0,
            completed_checkpoints=len(aggregate['results']),historical_metrics_agree=True,state_rng_restored=True,
            results_sha256=digest(output/'evaluation_results.json'),script_sha256=digest(__file__)))
        return 0
    except Exception as error:
        aggregate.update(status='failed',exception_type=type(error).__name__,exception=str(error))
        write_json(output/'evaluation_results.json',aggregate)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
