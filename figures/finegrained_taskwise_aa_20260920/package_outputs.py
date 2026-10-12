"""Package the figures, verified data, provenance, fonts and reproduction scripts."""
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'finegrained_taskwise_aa.zip'


def main():
    paths = sorted(p for p in HERE.rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts
                   and p.name not in {ARCHIVE.name, 'bundle_manifest.json'}
                   and p.suffix != '.pyc')
    manifest = {
        'description': 'Five dataset figures, original-precision frozen row means, provenance, and scripts.',
        'files': [{'path': str(p.relative_to(HERE)), 'bytes': p.stat().st_size,
                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
    }
    manifest_path = HERE / 'bundle_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    paths.append(manifest_path)
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, arcname=Path('finegrained_taskwise_aa') / p.relative_to(HERE))
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
    print(f'{ARCHIVE}: {len(paths)} files, {ARCHIVE.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
