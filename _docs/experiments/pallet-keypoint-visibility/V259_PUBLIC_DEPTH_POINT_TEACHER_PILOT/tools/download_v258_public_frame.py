"""Download one prospectively fixed public frame as opaque bytes, no decoding."""
from pathlib import Path
import hashlib
import json
import stat
import urllib.request

R = Path(__file__).absolute().parent
H = R / 'V258_ROOT_PUBLIC_POINT_TEACHER'
OUT = H / 'public_0p55_hall_0000_v1'
CATALOG = R / 'V257_PUBLIC_PALLET_TEACHER_READINESS_C/INERT_PUBLIC_FILE_REQUIREMENTS.json'
ROWS = (
    ('rgb_0000.png', 357953, 509841, '98d2aab8dd1a9e662b034175589de83b'),
    ('depth_0000.png', 357837, 70713, 'ee51034000cb590d0ec1e56251272c0f'),
    ('pose_0000.txt', 357303, 407, '7aa03bcce137de5321033cdeddb6fbfe'),
    ('mask_0000.png', 356802, 2429, 'bab7680b8455bd4dd211713f64ba9cea'),
    ('cam_K.txt', 357974, 198, '5ddf09d5a8fc883f5c7231440b538681'),
)


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


def put(name, v):
    with safe(OUT / name).open('x') as f:
        json.dump(v, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    policy = safe(H / 'PROSPECTIVE_DATA_ACCESS_SCOPE.json').read_bytes()
    assert hashlib.sha256(policy).hexdigest() == 'a5b8eff4ac71a66c6ad9803927712ab37f780ed5c277d9fa123974d854a3c933'
    cb = safe(CATALOG).read_bytes()
    assert hashlib.sha256(cb).hexdigest() == '4d29e87049d890f849aa3f1cf64bfb759fe31b026b4be6a6ed07065430a7e2bb'
    c = json.loads(cb)
    candidates = c['public_geometry_documentation_assets'] + c['single_frame_bundle_lookup_examples_only_not_data_selection']
    registry = {r['dataverse_file_id']: r for r in candidates}
    for name, file_id, n, md5 in ROWS:
        row = registry[file_id]
        assert row['public_relative_path'].startswith('0.55m_hall/')
        assert row['restricted'] is False and row['publisher_size_bytes'] == n
        assert row['publisher_checksum'] == {'type': 'MD5', 'value': md5}
    safe(OUT)
    assert not OUT.exists()
    OUT.mkdir()
    scope = {'approved_opaque_public_download': True,
             'selection': 'First lexicographic capture directory and frame0000 from complete metadata bundle, fixed before image inspection.',
             'dataset_DOI': '10.48804/DJLDFS', 'capture': '0.55m_hall', 'frame': '0000',
             'source_code_sha256': hashlib.sha256(safe(Path(__file__)).read_bytes()).hexdigest(),
             'files': [{'filename': n, 'file_id': i, 'size_bytes': s, 'MD5': m} for n, i, s, m in ROWS],
             'decoded_images_arrays_poses': 0,
             'requires_identity_and_separate_geometry_entry_before_decode': True,
             'protected_case_or_current223_reads': 0, 'training': False, 'automatic_retry': False}
    put('ROOT_OPAQUE_DOWNLOAD_SCOPE.json', scope)
    files = []
    try:
        for name, file_id, size, md5 in ROWS:
            url = 'https://rdr.kuleuven.be/api/access/datafile/' + str(file_id)
            target = safe(OUT / name)
            h, legacy, n = hashlib.sha256(), hashlib.md5(), 0
            req = urllib.request.Request(url, headers={'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(req, timeout=30) as response:
                assert response.status == 200
                with target.open('xb') as f:
                    while chunk := response.read(1048576):
                        n += len(chunk)
                        assert n <= size
                        h.update(chunk)
                        legacy.update(chunk)
                        f.write(chunk)
            assert n == size and legacy.hexdigest() == md5
            files.append({'path': str(target), 'sha256': h.hexdigest(), 'size_bytes': n,
                          'publisher_MD5_verified': md5, 'url': url,
                          'public_relative_path': registry[file_id]['public_relative_path']})
            print(json.dumps({'verified_public_download': name, 'bytes': n}), flush=True)
        put('OPAQUE_DOWNLOAD_COMPLETE.json', {'completed': True, 'PASS': True, 'files': files,
                                             'decoded_images_arrays_poses': 0, 'training': False})
        print(json.dumps({'completed': True, 'PASS': True, 'files': len(files),
                          'decoded_images_arrays_poses': 0}), flush=True)
    except BaseException as exc:
        put('FAILURE.json', {'completed': False, 'PASS': False, 'error': str(exc),
                             'verified_files': files, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
