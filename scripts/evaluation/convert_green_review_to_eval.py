"""User-authorized split-only conversion of green REVIEW copies, with backups."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from scripts.evaluation.eval_workspace import atomic_write_text

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'outputs/green_paper_review_20260918'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def convert(review: Path):
    target = review / 'full_session_annotations'
    manifest = json.loads((review / 'full_sessions_manifest.json').read_text())
    allowed = {r['session'] + '_manual_gt' for r in manifest['records']}
    paths = sorted(target.glob('*/*.json'))
    planned = []
    for path in paths:
        if path.parent.name not in allowed or path.is_symlink():
            raise ValueError(f'Unexpected review annotation: {path}')
        before = path.read_bytes(); doc = json.loads(before)
        if len(doc['objects']) != 1:
            raise ValueError(f'Expected one pallet: {path}')
        obj = doc['objects'][0]
        if obj.get('object_type', doc.get('object_type')) != 'plastic_standard_110x110x15':
            raise ValueError(f'Wrong object type: {path}')
        if obj.get('split') not in ('train', 'eval'):
            raise ValueError(f'Unknown split: {path}')
        after = copy.deepcopy(doc); after['objects'][0]['split'] = 'eval'
        planned.append((path, before, doc, after))
    backup = review / 'backups' / ('train_to_eval_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    backup.mkdir(parents=True, exist_ok=False)
    for path, before, _, _ in planned:
        dest = backup / path.relative_to(target)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        if dest.read_bytes() != before:
            raise RuntimeError(f'Concurrent annotation edit during backup: {path}')
    records = []
    try:
        for path, before, doc, after in planned:
            if path.read_bytes() != before:
                raise RuntimeError(f'Concurrent edit; refusing overwrite: {path}')
            changed = doc['objects'][0]['split'] != 'eval'
            if changed:
                atomic_write_text(path, json.dumps(after, ensure_ascii=False, indent=2) + '\n')
            actual = json.loads(path.read_text())
            assert actual == after
            records.append(dict(path=str(path.relative_to(target)), changed=changed,
                original_split=doc['objects'][0]['split'], before_sha256=digest(before),
                after_sha256=digest(path.read_bytes())))
    finally:
        report = dict(scope='green review copies only; original training datasets unchanged',
            planned=len(planned), processed=len(records), changed=sum(r['changed'] for r in records),
            backup=str(backup), records=records,
            note='Bulk EVAL assignment is not individual quality review or independent-test admission.')
        atomic_write_text(backup / 'CONVERSION_AUDIT.json', json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args=parser.parse_args()
    if not args.apply:
        parser.error('--apply required; only the dedicated green review folder is writable')
    report=convert(REVIEW)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))
