"""Verify the published square clean evidence using Python's standard library."""
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


DOC = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    receipt = json.loads((DOC / 'RECHECK.json').read_text())
    sources = []
    for source in receipt['sources']:
        archive = (DOC / source['archived_copy']).read_bytes()
        assert sha(archive) == source['archived_sha256']
        original = gzip.decompress(archive)
        assert sha(original) == source['sha256']
        sources.append(json.loads(original))
    expected = {}
    history_counts = []
    for population, snapshot, store in [('GREEN0918', sources[0], sources[1]),
                                         ('GREEN150', sources[2], sources[3])]:
        replay = {}
        for event in store['history']:
            if event['action'] == 'human_reset_to_unreviewed':
                replay.pop(event['frame_id'], None)
            else:
                assert event['action'] == 'explicit_class_selection'
                replay[event['frame_id']] = event['current']
        assert replay == store['records']
        history_counts.append(len(store['history']))
        for row in snapshot['records']:
            key = (population, row['id'])
            assert key not in expected
            expected[key] = (row, store['records'][population + '::' + row['id']])
    rows = list(csv.DictReader((DOC / 'ALL269_LABELS.csv').open()))
    clean = list(csv.DictReader((DOC / 'CLEAN106_IMAGES.csv').open()))
    assert len(rows) == len(expected) == 269
    assert len({(r['population'], r['id']) for r in rows}) == 269
    assert clean == [r for r in rows if r['severity'] == 'clean'] and len(clean) == 106
    for row in rows:
        original, tag = expected[(row['population'], row['id'])]
        assert row['original_image_path'] == original['image']['path']
        assert row['original_image_sha256'] == original['image']['sha256'] == tag['image_sha256']
        for key in ('severity', 'label_source', 'selected_at'):
            assert row[key] == tag[key]
        if row['preview']:
            image = DOC / row['preview']
            assert sha(image.read_bytes()) == row['preview_sha256']
            assert image.stat().st_size == int(row['preview_bytes'])
    counts = {population: dict(Counter(r['severity'] for r in rows if r['population'] == population))
              for population in ('GREEN0918', 'GREEN150')}
    assert counts == receipt['counts']
    assert counts['GREEN0918'] == dict(clean=3, moderate=85, severe=31)
    assert counts['GREEN150'] == dict(clean=103, moderate=44, severe=3)
    extra = json.loads((DOC / 'SOURCE_MEMBERS.json').read_text())
    assert len(extra['rows']) == 9 and extra['source_zip_image_count'] == 8361
    for row in extra['rows']:
        assert not row['official'] and not row['selected119_member']
        assert row['label_status'] == 'VISUAL_CANDIDATE'
        assert ('GREEN0918', row['id']) not in expected
        image = DOC / row['preview']
        assert sha(image.read_bytes()) == row['preview_sha256']
        assert image.stat().st_size == row['preview_bytes']
    files = []
    for line in (DOC / 'CHECKSUMS.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        path = DOC / relative
        assert path.resolve().is_relative_to(DOC)
        assert sha(path.read_bytes()) == digest, relative
        files.append(relative)
    actual = {str(p.relative_to(DOC)) for p in DOC.rglob('*') if p.is_file()
              and p.name != 'CHECKSUMS.sha256'}
    assert set(files) == actual and len(files) == len(actual)
    links = 0
    for path in DOC.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('https:', 'http:', '#')):
                continue
            assert (path.parent / target.split('#')[0]).exists(), target
            links += 1
    print('PASS: 269 labels; 106 clean previews; 9 candidate previews; '
          f'{len(files)} artifact hashes; {links} relative links; histories {history_counts}; NN/F=0')


if __name__ == '__main__':
    verify()
