"""Package the already-verified diagnostic as a standalone, offline HTML page."""
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/pallet_posefix_green_audit_v1'
DOC = ROOT / '_docs/experiments/pallet_posefix_green_audit_v1'


def read(path):
    return json.loads(path.read_text())


def encode(path, binding=None):
    raw = path.read_bytes()
    if binding:
        assert hashlib.sha256(raw).hexdigest() == binding['sha256'], path
    return 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')


def main():
    figures = read(OUT / 'FIGURES.json')
    contract = read(ROOT / '_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')
    for row in figures['figures']:
        row['src'] = encode(ROOT / row['image']['path'], row['image'])
        for model in row['models'].values():
            p, gt = model['predicted_xy'], row['focused_GT_xy']
            distance = ((p[0]-gt[0])**2 + (p[1]-gt[1])**2)**.5
            assert abs(distance-model['canonical_corner_error_px']) < 1e-6
    payload = dict(cases=figures['figures'], edges=contract['edges'],
                   subsets=read(DOC / 'NUMERIC_AUDIT.json')['subsets'],
                   heatmap=encode(OUT / 'heatmap_mechanisms.png'))
    template = (HERE / 'review_template.html').read_text()
    assert template.count('__AUDIT_PAYLOAD__') == 1
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    target = OUT / 'GREEN_REFINER_REVIEW.html'
    target.write_text(template.replace('__AUDIT_PAYLOAD__', data))
    print(target)
    print(f'{target.stat().st_size / 1024 / 1024:.2f} MiB; 4 original images + heatmap embedded; no server or network needed')


if __name__ == '__main__':
    main()
