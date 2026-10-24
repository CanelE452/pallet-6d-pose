"""Root-entry-only source metadata plan checker; no image/array/asset loader."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import stat
import finite_end_locality_contract as geometry

H = Path(__file__).absolute().parent
V = H.parent
PROTOCOL = H/'PROTOCOL_BEFORE_END_LOCALITY_SOURCE_BODIES.json'


def safe_path(value):
    path = Path(value)
    geometry.require(path.is_absolute() and '..' not in path.parts and
                     all('v112' not in p.casefold() for p in path.parts), 'Explicit safe path before filesystem access')
    return path


def read_bytes(value):
    path = safe_path(value)
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        final = os.open(path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    geometry.require(stat.S_ISREG(os.fstat(final).st_mode), 'Regular file only')
    with os.fdopen(final, 'rb') as stream:
        return stream.read()


def binding(value):
    return {'path': str(value), 'sha256': hashlib.sha256(read_bytes(value)).hexdigest()}


def checked(binding_value):
    data = read_bytes(binding_value['path'])
    geometry.require(hashlib.sha256(data).hexdigest() == binding_value['sha256'], 'Frozen metadata/code SHA drift')
    return data


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run(entry_path):
    e = json.loads(read_bytes(entry_path))
    geometry.require(e['approved_source_metadata_plan_audit'] is True, 'Root actual entry required')
    geometry.require(e['code'] == binding(Path(__file__).absolute()) and
                     e['pure_geometry'] == binding(H/'finite_end_locality_contract.py'), 'Exact frozen code')
    geometry.require(e['protocol']['path'] == str(PROTOCOL), 'Exact frozen protocol')
    protocol = json.loads(checked(e['protocol']))
    geometry.require(protocol['code'] == e['code'] and protocol['pure_geometry'] == e['pure_geometry'], 'Protocol/code closure')
    json.loads(checked(e['root_scope']))
    geometry.require(Path(e['frozen_plan_metadata']['path']).suffix == '.jsonl', 'JSONL metadata only; no array or image loader')
    records = [json.loads(row) for row in checked(e['frozen_plan_metadata']).splitlines()]
    geometry.require(len(records) == e['expected_rows'] and len({r['attempt_id'] for r in records}) == len(records), 'All frozen attempts and unique IDs')
    out = safe_path(e['output_directory'])
    geometry.require(out == V/'end_locality_metadata_plan_audit_v1', 'Fresh fixed metadata diagnostic output')
    out.mkdir(exist_ok=False)
    try:
        outputs, counts = [], {}
        for row in records:
            result = geometry.evaluate_plan(row['original_input'], row['endpoint_observation'], row['placement'])
            result['attempt_id'] = row['attempt_id']
            result['upstream_attempt_status'] = row.get('upstream_attempt_status')
            counts[result['status']] = counts.get(result['status'], 0) + 1
            outputs.append(result)
        target = out/'ALL_FROZEN_ATTEMPT_GEOMETRY_AND_UNKNOWN.jsonl'
        with target.open('x', encoding='utf-8') as stream:
            for row in outputs:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
        done = {'completed': True, 'PASS': True, 'entry': binding(entry_path), 'input': e['frozen_plan_metadata'],
                'rows': len(outputs), 'status_counts': counts, 'output': binding(target), 'all_attempts_retained': True,
                'no_actual_pixel_opacity_or_direct_end_labels_certified': True, 'training_admission': False,
                'asset_RGB_NPZ_header_depth_mask_mesh_model_or_evaluation_body_reads': 0}
        write_new(out/'COMPLETE.json', done)
        print(json.dumps(done), flush=True)
    except Exception as exc:
        write_new(out/'FAILURE.json', {'completed': False, 'PASS': False, 'error_type': type(exc).__name__, 'error': str(exc), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', type=Path, required=True)
    run(parser.parse_args().entry)
