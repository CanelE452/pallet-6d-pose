"""Verify only publication artifacts; no experiments, training, network or lifter work."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
import zlib

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_n3_static_closeout_v1'
CODE = ROOT / 'scripts/research/pallet_n3_static_closeout_v1'
REPORT = DOC / 'PUBLICATION_VALIDATION.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publication_files():
    return sorted([ROOT / 'readme.md'] + [p for folder in (DOC, CODE)
        for p in folder.rglob('*') if p.is_file() and not p.is_symlink()
        and '__pycache__' not in p.parts and p.suffix != '.pyc' and p != REPORT])


def main():
    files = publication_files()
    selected = set(files)
    archives = []
    for entry in json.loads((DOC / 'evidence/MANIFEST.json').read_text())['files']:
        path = DOC / 'evidence' / entry['archive']
        data = gzip.decompress(path.read_bytes())
        assert digest(path) == entry['archive_sha256']
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
        assert len(data) == entry['bytes']
        archives.append(entry['archive'])
    # All files delivered in the prior phase are immutable; excluded raw files
    # are also compared when present locally, but only archives are published.
    frozen = json.loads((DOC / 'LOCAL_DELIVERY.json').read_text())['files']
    unchanged, unavailable = [], []
    for entry in frozen:
        path = ROOT / entry['path']
        if not path.exists():
            unavailable.append(entry['path'])
            continue
        assert digest(path) == entry['sha256'], entry['path']
        unchanged.append(entry['path'])
    links = []
    for path in [DOC / 'README.md', DOC / 'evidence/README.md']:
        for dest in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if dest.startswith(('https:', 'http:', '#')):
                continue
            target = (path.parent / dest.split('#')[0]).resolve()
            assert target.exists(), (path, dest)
            if target.is_file():
                assert target in selected or target == REPORT, (path, dest)
            links.append({'document': str(path.relative_to(ROOT)), 'target': dest})
    pngs = []
    for p in sorted((DOC / 'figures').glob('*.png')):
        data = p.read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        pos = 8
        while pos < len(data):
            size = struct.unpack('>I', data[pos:pos+4])[0]
            chunk = data[pos+4:pos+8+size]
            assert zlib.crc32(chunk) & 0xffffffff == struct.unpack('>I', data[pos+8+size:pos+12+size])[0]
            pos += 12 + size
        assert pos == len(data)
        pngs.append({'path': str(p.relative_to(ROOT)), 'dimensions': list(struct.unpack('>II', data[16:24])), 'sha256': digest(p)})
    assert len(pngs) == 4
    for path in CODE.glob('*.py'):
        ast.parse(path.read_text())
    forbidden = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for path in files:
        assert path.stat().st_size < 50_000_000
        if path.suffix in ('.json', '.md', '.txt', '.py', '.csv', '.tex', '.html'):
            assert not forbidden.search(path.read_bytes()), 'Credential pattern in ' + str(path)
    assert (DOC/'generated_tables/tab_backbones.md').read_text().split('\n\n')[0] in (DOC/'README.md').read_text()
    report = dict(status='PASS', phase='publication_pre_push', baseline_main='a7fb68050e2207a5d8c379a1c1704f4e694b6675',
        source_results_changed=False, training_runs=0, optimizer_updates=0,
        archives_verified=archives, previous_delivery_files_unchanged=unchanged,
        previous_delivery_files_unavailable=unavailable, entrypoint_links_verified=links,
        pngs_verified=pngs, python_syntax='PASS', credential_pattern_scan='PASS',
        known_local_only=['review/*.html image dependencies (319 absolute symlinks omitted)',
            'full input images, original prediction inputs, annotations and checkpoints',
            'three inherited manuscript diagram images unavailable at source'],
        note='Historical no-push receipts retain their original meaning; this records the later explicitly authorized publication. Does not rerun experimental tests.',
        files=[dict(path=str(p.relative_to(ROOT)), sha256=digest(p), bytes=p.stat().st_size) for p in files])
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], files=len(files)+1, bytes=sum(p.stat().st_size for p in files)+REPORT.stat().st_size,
        frozen_unchanged=len(unchanged), unavailable=len(unavailable), archives=len(archives), images=len(pngs), links=len(links))))


if __name__ == '__main__':
    main()
