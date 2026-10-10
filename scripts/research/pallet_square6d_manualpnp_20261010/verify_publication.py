"""Final public content checks; saved numeric files only, no F/model/RGB input."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOC = ROOT / '_docs/experiments/pallet_square6d_manualpnp_20261010'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc', type=Path, default=Path(os.environ.get('PALLET_SQUARE_OUTPUT', DEFAULT_DOC)))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    doc = args.doc
    output = args.output or doc / 'PUBLIC_CONTENT_CHECK.json'
    assert not output.exists(), 'Use a fresh verification output'
    metrics = json.loads((doc / 'METRICS.json').read_text())
    comparisons = 0
    with (doc / 'PAPER_TABLE.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    for row in rows:
        packet = metrics['backbones'][row['backbone']]['ALL118']['seed_mean'][row['method']]
        assert int(row['frames']) == 118 and int(row['available_n']) == packet['available']
        comparisons += 2
        assert float(row['success_rate']) == packet['success_rate']['rate']
        comparisons += 1
        for key in ('T_cm', 'R_deg', 'ADDsym_m', 'IoU3D'):
            for stat in ('mean', 'std', 'median', 'P90'):
                assert float(row[key + '_' + stat]) == packet['metrics'][key][stat]
                comparisons += 1
    links = []
    for path in sorted(doc.glob('*.md')):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            actual = (path.parent / target.split('#', 1)[0]).resolve()
            assert actual.is_file(), 'Missing Markdown target: ' + target
            links.append(dict(document=path.name, target=target))
    figures = json.loads((doc / 'FIGURE_INDEX.json').read_text())
    assert figures['numeric_only'] and not figures['raw_RGB_published']
    assert len(figures['figures']) == 4
    for record in figures['figures']:
        assert sha(doc / record['path']) == record['sha256']
        assert (doc / record['path']).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    entries = []
    for base in (doc, ROOT / 'scripts/research/pallet_square6d_manualpnp_20261010'):
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path == output:
                continue
            assert path.suffix.lower() not in ('.jpg', '.jpeg', '.bmp', '.tiff', '.pt', '.pth', '.npy', '.npz', '.bin', '.pyc')
            if path.suffix == '.png':
                assert path.parent == doc / 'figures'
            else:
                text = gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text()
                assert ('/' + 'home' + '/') not in text and ('/' + 'tmp' + '/') not in text
            entries.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size))
    result = dict(status='PASS', manuscript_csv_numeric_comparisons=comparisons,
                  manuscript_rows=12, markdown_targets_checked=len(links), links=links,
                  public_source_and_document_files_checked=len(entries), numeric_PNGs_checked=4,
                  personal_absolute_paths_found=0, raw_RGB_files_published=0,
                  additional_F_calls=0, additional_model_calls=0, files=entries)
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('SQUARE_PUBLIC_CONTENT_PASS', comparisons, len(links), len(entries))

if __name__ == '__main__':
    main()
