#!/usr/bin/env python3
"""Read existing experiment records; write evidence only beside this script."""
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path('/data/140-1/users/liying/CopyFlatCV')
REFINE = ROOT / 'refine-logs'
TERMS = ['g1_plus_residual', 'sum_family_lambda', 'unit_interpolation_alpha',
         'g3_scale_only', 'residual_lambda', 'direction_interpolation']
PATTERN = re.compile('|'.join(TERMS), re.I)


def files(path):
    return [Path(x) for x in subprocess.check_output(
        ['rg', '--files', '--no-ignore', '--hidden', str(path)], text=True).splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


all_files = files(REFINE)
suites = [p for p in all_files if p.suffix == '.json' and 'suite' in p.name
          and any(part.startswith('s3-') for part in p.parts)]
suite_rows, job_rows = [], []
for p in suites:
    try:
        value = json.loads(p.read_text())
    except (ValueError, OSError):
        continue
    jobs = value.get('jobs', []) if isinstance(value, dict) else []
    jobs = [j for j in jobs if isinstance(j, dict)]
    suite_rows.append({'path': str(p), 'sha256': sha(p), 'job_records': len(jobs),
                       'script': value.get('script') if isinstance(value, dict) else None})
    job_rows.extend({'suite': str(p), **j} for j in jobs)
named_hits = [j for j in job_rows if re.search('residual|interpol|g1_plus|sum_family', j.get('id', ''), re.I)]
write('s3_suites_and_jobs.json', {'suites': suite_rows, 'job_records': job_rows,
                                 'residual_interpolation_named_jobs': named_hits})

# Include named config files AND arbitrary names inside configs directories.
metadata = [p for p in all_files if p.suffix == '.json' and
            ('configs' in p.parts or 'config' in p.name or 'suite' in p.name
             or p.name.startswith('JOBS') or p.name in ('completed.json', 'manifest.json'))]
meta_hits = []
for p in metadata:
    matches = sorted(set(m.group(0) for m in PATTERN.finditer(p.read_text(errors='replace'))))
    if matches:
        meta_hits.append({'path': str(p), 'terms': matches})
write('metadata_candidate_search.json', {'terms': TERMS, 'files_checked': len(metadata),
      'paths_checked': [str(p) for p in metadata], 'matches': meta_hits})

logs = files(ROOT / 'logs_try')
log_hits = []
yaml_generic = []
for p in logs:
    text = p.read_text(errors='replace')
    if PATTERN.search(text):
        log_hits.append(str(p))
    if p.suffix == '.yaml' and re.search('resid|interpol', text, re.I):
        yaml_generic.append({'path': str(p), 'matching_lines': [
            {'line': i, 'text': line} for i, line in enumerate(text.splitlines(), 1)
            if re.search('resid|interpol', line, re.I)]})
with (OUT / 'logs_try_files.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['path', 'size_bytes', 'sha256'])
    writer.writeheader()
    writer.writerows({'path': str(p), 'size_bytes': p.stat().st_size, 'sha256': sha(p)} for p in logs)
write('logs_try_search.json', {'file_count': len(logs), 'suffix_counts': dict(Counter(p.suffix for p in logs)),
      'specific_terms': TERMS, 'specific_term_matches': log_hits,
      'generic_resid_interpol_yaml_hits': yaml_generic,
      'excluded_generic_hit': 'tmp_s3_endpoint_wmerge_omega030.yaml describes endpoint weight interpolation, not g1/g3 direction interpolation.'})

diag = REFINE / 's3-g1g3-mechanism-run-20260930T210711Z/diagnostics/paired_gam_diagnostic.py'
source_lines = diag.read_text().splitlines()
ranges = {'projection': (611, 615), 'reference_step': (627, 631), 'candidate_definitions': (643, 650),
          'effective_polynomial': (202, 217), 'actual_match_and_restore': (387, 419),
          'CLI_protocol_defaults': (510, 518), 'evaluation_losses': (360, 384)}
write('code_evidence.json', {'source_path': str(diag), 'source_sha256': sha(diag),
      'line_references': {k: {'start': lo, 'end': hi, 'text': '\n'.join(
          '%d: %s' % (i, source_lines[i-1]) for i in range(lo, hi+1))} for k, (lo, hi) in ranges.items()}})

write('INVENTORY_SUMMARY.json', {'audited_at_utc': datetime.now(timezone.utc).isoformat(),
      'scope_root': str(ROOT), 'commands': ['rg --files --no-ignore --hidden PATH'],
      's3_suite_files': len(suites), 's3_job_records_including_repeated_smoke_and_recovery_suites': len(job_rows),
      'named_residual_interpolation_jobs': len(named_hits), 'metadata_files_checked': len(metadata),
      'metadata_candidate_hits': len(meta_hits), 'logs_try_files': len(logs),
      'logs_try_specific_hits': len(log_hits),
      'conclusion': 'The requested residual-lambda and normalized-direction-alpha families have offline counterfactual results. No independent full-training coefficient scan was found in the enumerated suites/configs/receipts/logs. Existing g1/g3/sum/mean training endpoints do not constitute that matched-step scan.',
      'limit': 'A bounded filesystem inventory, not a claim that no such run exists anywhere outside the enumerated scope. Suite job records include repeated copies and smoke/recovery entries, not unique scientific runs.'})
print(json.dumps(json.loads((OUT/'INVENTORY_SUMMARY.json').read_text()), indent=2))
