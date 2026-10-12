#!/usr/bin/env python3
"""Verify ten completed fresh Task0 runs, tabulate them, then run five offline anchors."""
import csv,hashlib,json,os,subprocess,sys
from pathlib import Path
from common import atomic_json,sha256,now

ROOT=Path(__file__).resolve().parent

def main():
    suite=json.loads((ROOT/'suite.json').read_text());rows=[];summaries={};identities=[];orders=[];batchhashes=[]
    for job in suite['jobs']:
        path=Path(job['output']);summary=json.loads((path/'final_summary.json').read_text());metrics=json.loads((path/'metrics.json').read_text())['epochs']
        if summary['status']!='completed' or summary['smoke'] or summary['completed_epochs']!=50:raise RuntimeError('Incomplete '+job['id'])
        summaries[job['id']]=summary;identities.append(summary['initial_identity']['initial_model_sha256'])
        orders.append([e['sampler_order_sha256'] for e in metrics]);batchhashes.append([e['first_batch']['input_sha256'] for e in metrics])
        f=summary['final'];rows.append({'id':job['id'],'direction':summary['direction'],'r':summary['r'],'seed':2,'epoch':50,'a11_sample_accuracy':f['eval_accuracy'],'eval_ce':f['eval_ce'],'train_ce':f['train_ce'],'train_accuracy':f['train_accuracy'],'elapsed_seconds':summary['elapsed_seconds'],'output':str(path)})
    checks={'same_initial_model':len(set(identities))==1,'same_epoch_sample_orders':all(x==orders[0] for x in orders),'same_epoch_first_augmented_batch':all(x==batchhashes[0] for x in batchhashes)}
    atomic_json(ROOT/'TRAINING_COMPLETION_AUDIT.json',{'time':now(),'checks':checks,'rows':rows})
    if not all(checks.values()):raise RuntimeError('Cross-arm pairing audit failed: '+repr(checks))
    res=ROOT/'results';res.mkdir(exist_ok=True)
    with open(res/'task0_results.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(res/'task0_results.json',{'status':'completed','seeds':[2],'primary_metric':'epoch50 Task0 sample accuracy a11','rows':rows,'pairing_checks':checks,'selection_status':'pending_joint_accuracy_and_matched_diagnostic_review','next_training_stages':'not_queued'})
    text=['# 独立 Task0 结果（seed2）','','固定epoch50；相同初始化、每epoch数据顺序与首batch增强hash已核验。这里只报告单任务a11，不报告遗忘或AAA。','','| 配置 | r | a11 (%) | eval CE |','|---|---:|---:|---:|']
    text += ['| {direction} | {r:g} | {a11_sample_accuracy:.4f} | {eval_ce:.6f} |'.format(**r) for r in rows]
    text += ['','仅单seed开发结果；待匹配步幅/探测距离诊断完成后判断G*。两任务、三任务未加入自动队列。','']
    (res/'RESULTS.md').write_text('\n'.join(text))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(11,4))
        for job in suite['jobs']:
            label=job['label'];epochs=json.loads((Path(job['output'])/'metrics.json').read_text())['epochs']
            if label in ('ce','s','g','sum','mean','true_sam'):
                ax[0].plot([e['epoch'] for e in epochs],[e['eval_accuracy'] for e in epochs],label=label)
        points=sorted((r['r'],r['a11_sample_accuracy']) for r in rows if r['direction']=='g')
        ax[1].plot([p[0] for p in points],[p[1] for p in points],marker='o')
        ax[0].set(xlabel='Epoch',ylabel='Task 0 held-out accuracy (%)',title='Composition, seed 2');ax[0].legend()
        ax[1].set(xlabel='GAM outer radius r',ylabel='Final Task 0 accuracy (%)',title='g3 only, fixed inner rho=0.05')
        fig.tight_layout();fig.savefig(res/'task0_curves.pdf');fig.savefig(res/'task0_curves.png',dpi=180);plt.close(fig)
    except Exception as exc:atomic_json(res/'PLOT_ERROR.json',{'error':repr(exc)})
    # Four final checkpoints plus one common epoch-zero local diagnostic; no Task1/Task2 training is authorized here.
    diag=ROOT/'diagnostics';ds={k:suite[k] for k in ('python','execution_hostname','allowed_gpus','source_root','source_manifest','environment')}
    ds.update(campaign=str(ROOT),script=str(diag/'paired_gam_diagnostic.py'),runtime=str(diag/'runtime/formal'),max_parallel=2,summarize=True,file_pins={},jobs=[])
    for name in ('paired_gam_diagnostic.py','run_queue.py','summarize_diagnostics.py'):ds['file_pins'][str(diag/name)]=sha256(diag/name)
    for label,epoch in (('s',0),('s',50),('g',50),('sum',50),('mean',50)):
        job=next(j for j in suite['jobs'] if j['label']==label);cp=Path(job['output'])/'checkpoints'/('epoch_%03d'%epoch)
        ds['file_pins'][job['config']]=sha256(job['config'])
        for name in ('lora_w_a_0.pt','lora_w_b_0.pt','fc_state_0.pt','lora_meta_0.json'):ds['file_pins'][str(cp/name)]=sha256(cp/name)
        ds['jobs'].append({'id':label+'_epoch'+str(epoch),'task':0,'max_batches':4,'config':job['config'],'checkpoint_dir':str(cp),'output':str(diag/'results'/(label+'_epoch'+str(epoch)+'.json'))})
    sp=diag/'suite.json'
    if sp.exists():raise RuntimeError('Diagnostic suite already registered; no blind relaunch')
    atomic_json(sp,ds)
    subprocess.run([suite['python'],str(diag/'run_queue.py'),'--suite',str(sp),'--validate-only'],check=True)
    atomic_json(diag/'AUTO_LAUNCH.json',{'time':now(),'suite_sha256':sha256(sp),'trigger':'all ten task0 trainings verified','two_three_task_training_queued':False})
    subprocess.run([suite['python'],str(diag/'run_queue.py'),'--suite',str(sp)],check=True)

if __name__=='__main__':main()
