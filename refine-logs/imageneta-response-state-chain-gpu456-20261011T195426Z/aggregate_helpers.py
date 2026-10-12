"""Preregistered paired summaries and first-entry fit bins; no test-based selection."""
import math,statistics

def mean_sd(values):
 return dict(n=len(values),mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None)
def fit_match(rows,width=.05,maximum=5.):
 first={}
 for r in sorted(rows,key=lambda x:(x['seed'],x['branch'],x['epoch'])):
  ce=r['clean_training_ce']
  if not (0<=ce<maximum):continue
  b=math.floor(ce/width)
  first.setdefault((r['seed'],b,r['branch']),r)
 result=[]
 for seed,b in sorted({(s,b) for s,b,_ in first}):
  if (seed,b,'S') not in first or (seed,b,'R') not in first:continue
  s,r=(first[(seed,b,v)] for v in ['S','R'])
  result.append(dict(seed=seed,ce_bin_low=b*width,ce_bin_high=(b+1)*width,S_first_epoch=s['epoch'],R_first_epoch=r['epoch'],
    S_clean_training_ce=s['clean_training_ce'],R_clean_training_ce=r['clean_training_ce'],
    old_accuracy_R_minus_S=r['old_accuracy']-s['old_accuracy'],new_accuracy_R_minus_S=r['new_accuracy']-s['new_accuracy'],
    new_conditional20_ce_R_minus_S=r['new_conditional20_ce']-s['new_conditional20_ce'],selection='first epoch entering fixed clean-training-CE bin; no test-based choice'))
 return result
