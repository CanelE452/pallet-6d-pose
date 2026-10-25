"""Collect one public occlusion diagnostic chosen only by publisher annotation."""
from pathlib import Path, PurePosixPath
import csv
import hashlib
import io
import json
import math
import stat
import urllib.request

R = Path(__file__).absolute().parent
H = R / 'V259_ROOT_PUBLIC_OCCLUSION_PILOT'
CAPTURE = '0.55m_hall'


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for a in reversed([p, *p.parents]):
        try:
            assert not stat.S_ISLNK(a.lstat().st_mode), str(a)
        except FileNotFoundError:
            pass
    return p


def bind(p, expected=None):
    p = safe(p)
    b = p.read_bytes()
    digest = hashlib.sha256(b).hexdigest()
    assert expected is None or digest == expected
    return {'path': str(p), 'sha256': digest, 'size_bytes': len(b)}


def put(name, value):
    p = safe(H / name)
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return bind(p)


def main():
    api = bind(R / 'V255_PUBLIC_PALLET_METADATA_ROOT/OFFICIAL_API_RESPONSE.json',
               'df02a2d129392f76d65a1130f741e2de29d114fd3a608ef431229b313ddce3dd')
    held = bind(R / 'V258_PUBLIC_HELD_DIGEST_IDENTITY_B/HELD45_ORIGINAL_RGB_BYTE_DIGEST_ALLOWLIST.json',
                'eebb382e9512b1471ba443d17f42ae0c90d6d7adf5dfd65d18aeace39c10fabd')
    metadata = json.loads(safe(Path(api['path'])).read_bytes())
    registry = {('/'.join((row.get('directoryLabel', ''), row['label']))).strip('/'): row
                for row in metadata['data']['latestVersion']['files']}
    digests = json.loads(safe(Path(held['path'])).read_bytes())['exact_encoded_byte_SHA256_exclusion_set']
    assert len(set(digests)) == 45
    safe(H)
    assert not H.exists()
    H.mkdir()
    put('PROSPECTIVE_SCOPE.json', {
        'purpose': 'Public source geometry/occlusion diagnostic, not model evaluation or training.',
        'capture': CAPTURE,
        'selection_fixed_before_CSV_read': 'Maximum publisher occluded_percent among frames with all four components; ties use smallest numeric frame ID.',
        'publisher_category_is_not_native_point_gold': True,
        'metadata': api, 'held_encoded_identity_metadata': held,
        'code': bind(Path(__file__)), 'decode_or_model_forward_training': False,
        'protected_case_or_current223_access': False, 'automatic_retry': False,
    })
    downloaded = []

    def fetch(public_path, local_name):
        row = registry[public_path]
        data = row['dataFile']
        assert row['restricted'] is False
        assert data['checksum']['type'] == 'MD5'
        expected_size, expected_md5 = data['filesize'], data['checksum']['value']
        assert 0 < expected_size < 10000000
        url = 'https://rdr.kuleuven.be/api/access/datafile/' + str(data['id'])
        req = urllib.request.Request(url, headers={'Accept-Encoding': 'identity'})
        target = safe(H / local_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest, legacy, size = hashlib.sha256(), hashlib.md5(), 0
        with urllib.request.urlopen(req, timeout=30) as response:
            assert response.status == 200
            with target.open('xb') as f:
                while chunk := response.read(1048576):
                    size += len(chunk)
                    assert size <= expected_size
                    digest.update(chunk)
                    legacy.update(chunk)
                    f.write(chunk)
        assert size == expected_size and legacy.hexdigest() == expected_md5
        item = {'path': str(target), 'sha256': digest.hexdigest(), 'size_bytes': size,
                'publisher_MD5_verified': expected_md5, 'public_path': public_path, 'url': url}
        downloaded.append(item)
        return item

    try:
        csv_binding = fetch(CAPTURE + '/visibility_scores.csv', 'visibility_scores.csv')
        text = safe(Path(csv_binding['path'])).read_text()
        rows = list(csv.DictReader(io.StringIO(text)))
        assert rows and 'image' in rows[0] and 'occluded_percent' in rows[0]
        candidates = []
        for row in rows:
            stem = PurePosixPath(row['image'].strip()).stem
            assert stem.isdigit()
            frame = str(int(stem)).zfill(4)
            occlusion = float(row['occluded_percent'])
            assert math.isfinite(occlusion) and 0 <= occlusion <= 100
            paths = {
                'RGB': CAPTURE + '/rgb/' + frame + '.png',
                'DEPTH': CAPTURE + '/depth/' + frame + '.png',
                'POSE': CAPTURE + '/annotated_poses/' + frame + '.txt',
                'SAM': CAPTURE + '/masks/' + frame + '_sam1_mask.png',
            }
            if all(p in registry for p in paths.values()):
                candidates.append({'frame': frame, 'publisher_occluded_percent': occlusion,
                                   'publisher_row': row, 'paths': paths})
        assert candidates
        chosen = sorted(candidates, key=lambda x: (-x['publisher_occluded_percent'], int(x['frame'])))[0]
        selection = put('PUBLIC_PILOT_SELECTION.json', {
            'CSV_binding': csv_binding, 'complete_candidates': len(candidates),
            'selected': chosen, 'selection_uses_model_outputs_or_current_eval': False,
            'not_a_random_or_accuracy_evaluation_sample': True,
        })
        inputs = {key: fetch(path, 'raw/' + key + '_' + chosen['frame'] + Path(path).suffix)
                  for key, path in chosen['paths'].items()}
        inputs['K'] = bind(R / 'V258_ROOT_PUBLIC_POINT_TEACHER/public_0p55_hall_0000_v1/cam_K.txt',
                           '9408bd28ae0a51e23ebe81b63da6e8ff09997b241f35791e81e80e519dd5c7e0')
        inputs['OBJ'] = bind(R / 'V257_ROOT_PUBLIC_PALLET_GEOMETRY/pallox.obj',
                             '2c44a016ec657955af2d195876af1234fab143dc1651203d13bad1c44c836e0f')
        hits = int(inputs['RGB']['sha256'] in digests)
        identity = put('PUBLIC_FRAME_ENCODED_IDENTITY_QA.json', {
            'completed': True, 'PASS': hits == 0, 'public_RGB_binding': inputs['RGB'],
            'held_digest_metadata': held, 'exact_encoded_byte_hits': hits,
            'permits_this_public_frame_geometry_diagnostic': hits == 0,
            'permits_training_or_claims_full_capture_disjointness': False,
            'protected_held_case_reads': 0,
            'limitation': 'Encoded-byte exclusion only; reencoded/shared capture not excluded. Official capture declarations differ; geometry diagnostic only.',
        })
        assert hits == 0
        complete = {'completed': True, 'PASS': True, 'capture_frame': CAPTURE + '/' + chosen['frame'],
                    'inputs': inputs, 'identity_QA': identity, 'selection': selection,
                    'downloaded': downloaded, 'decoded_images_or_poses': 0, 'training': False}
        put('OPAQUE_DOWNLOAD_COMPLETE.json', complete)
        print(json.dumps({'completed': True, 'PASS': True, 'capture_frame': complete['capture_frame'],
                          'complete_candidates': len(candidates),
                          'publisher_occluded_percent': chosen['publisher_occluded_percent'],
                          'downloads': len(downloaded), 'decoded_images_or_poses': 0}), flush=True)
    except BaseException as exc:
        put('FAILURE.json', {'completed': False, 'PASS': False, 'error': str(exc),
                             'downloaded': downloaded, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
