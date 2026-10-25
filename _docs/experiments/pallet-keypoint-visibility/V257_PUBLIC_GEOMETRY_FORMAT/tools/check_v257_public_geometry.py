"""Acquire two public non-frame geometry assets and check their numeric format."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import math
import stat
import urllib.request

R = Path(__file__).absolute().parent
OUT = R / 'V257_ROOT_PUBLIC_PALLET_GEOMETRY'
CATALOG = R / 'V257_PUBLIC_PALLET_TEACHER_READINESS_C/INERT_PUBLIC_FILE_REQUIREMENTS.json'
EXPECTED_CATALOG = '4d29e87049d890f849aa3f1cf64bfb759fe31b026b4be6a6ed07065430a7e2bb'
ASSETS = (
    ('cam_K.txt', 356835, 198, '5ddf09d5a8fc883f5c7231440b538681'),
    ('pallox.obj', 357978, 8888206, 'c8aa430a11996540f84935fcea9f4c43'),
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


def write_json(name, obj):
    with safe(OUT / name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    catalog_bytes = safe(CATALOG).read_bytes()
    assert hashlib.sha256(catalog_bytes).hexdigest() == EXPECTED_CATALOG
    catalog = json.loads(catalog_bytes)
    listed = {x['dataverse_file_id']: x for x in catalog['public_geometry_documentation_assets']}
    for name, file_id, size, md5 in ASSETS:
        row = listed[file_id]
        assert not row['restricted'] and row['publisher_size_bytes'] == size
        assert row['publisher_checksum'] == {'type': 'MD5', 'value': md5}
    safe(OUT)
    assert not OUT.exists()
    OUT.mkdir()
    scope = {
        'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'PUBLIC_NON_FRAME_CALIBRATION_AND_MESH_FORMAT_ONLY',
        'approved_exact_public_file_ids': [x[1] for x in ASSETS],
        'catalog_sha256': EXPECTED_CATALOG,
        'root_code_sha256': hashlib.sha256(safe(Path(__file__)).read_bytes()).hexdigest(),
        'RGB_depth_pose_mask_visibilityCSV_or_protected_case_reads': 0,
        'training_evaluation_or_native_point_label_admission': False,
        'no_model_threshold_or_data_recipe_selection': True,
        'mesh_external_material_and_texture_references_inert': True,
        'automatic_retry': False,
    }
    write_json('ROOT_PROSPECTIVE_SCOPE.json', scope)
    downloaded = []
    try:
        for name, file_id, size, md5 in ASSETS:
            url = 'https://rdr.kuleuven.be/api/access/datafile/' + str(file_id)
            req = urllib.request.Request(url, headers={'Accept-Encoding': 'identity'})
            partial = safe(OUT / (name + '.partial'))
            h, legacy, n = hashlib.sha256(), hashlib.md5(), 0
            with urllib.request.urlopen(req, timeout=30) as response:
                assert response.status == 200
                content_type = response.headers.get('Content-Type')
                with partial.open('xb') as f:
                    while chunk := response.read(1048576):
                        n += len(chunk)
                        assert n <= size
                        h.update(chunk)
                        legacy.update(chunk)
                        f.write(chunk)
            assert n == size and legacy.hexdigest() == md5
            target = safe(OUT / name)
            assert not target.exists()
            partial.rename(target)
            downloaded.append({'filename': name, 'url': url, 'file_id': file_id,
                               'sha256': h.hexdigest(), 'size_bytes': n,
                               'publisher_MD5_verified': md5, 'HTTP_status': 200,
                               'content_type': content_type})
            print(json.dumps({'download_verified': name, 'bytes': n}), flush=True)
        values = [[float(x) for x in line.split()]
                  for line in safe(OUT / 'cam_K.txt').read_text().splitlines() if line.strip()]
        assert len(values) == 3 and all(len(row) == 3 for row in values)
        assert all(math.isfinite(x) for row in values for x in row)
        assert values[0][0] > 0 and values[1][1] > 0 and values[2] == [0, 0, 1]
        count, vertices = collections.Counter(), []
        declarations = []
        for line in safe(OUT / 'pallox.obj').read_text().splitlines():
            parts = line.split()
            if not parts or parts[0].startswith('#'):
                continue
            tag = parts[0]
            count[tag] += 1
            if tag == 'v':
                v = [float(x) for x in parts[1:4]]
                assert len(v) == 3 and all(math.isfinite(x) for x in v)
                vertices.append(v)
            elif tag == 'f':
                assert len(parts) >= 4
                for item in parts[1:]:
                    index = int(item.split('/')[0])
                    assert index != 0
                    assert index <= len(vertices) if index > 0 else -index <= len(vertices)
            elif tag in ('o', 'g', 'mtllib', 'usemtl'):
                declarations.append({'tag': tag, 'value': ' '.join(parts[1:])})
        assert vertices and count['f'] > 0
        mins = [min(v[i] for v in vertices) for i in range(3)]
        maxs = [max(v[i] for v in vertices) for i in range(3)]
        summary = {
            'completed': True, 'PASS': True, 'scope': scope, 'downloads': downloaded,
            'K_numeric': values, 'OBJ_tag_counts': dict(count),
            'OBJ_coordinate_min': mins, 'OBJ_coordinate_max': maxs,
            'OBJ_coordinate_extent': [maxs[i] - mins[i] for i in range(3)],
            'OBJ_declarations_inert': declarations,
            'format_checks_only_not_registration': True,
            'mesh_units_axis_origin_and_pose_convention': 'UNVERIFIED',
            'physical_native_keypoint_mapping': 'NOT_ESTABLISHED',
            'point_visibility_labels_created': 0,
            'frame_payload_downloads': 0,
            'camera_RGB_depth_alignment_or_point_teacher_accuracy_claim': False,
        }
        write_json('GEOMETRY_FORMAT_QA.json', summary)
        print(json.dumps({'completed': True, 'PASS': True, 'K_numeric': values,
                          'OBJ_tag_counts': dict(count), 'extent_unverified_units': summary['OBJ_coordinate_extent'],
                          'point_labels_created': 0}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        write_json('FAILURE.json', {'completed': False, 'PASS': False,
                                   'error': str(exc), 'verified_downloads': downloaded,
                                   'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
