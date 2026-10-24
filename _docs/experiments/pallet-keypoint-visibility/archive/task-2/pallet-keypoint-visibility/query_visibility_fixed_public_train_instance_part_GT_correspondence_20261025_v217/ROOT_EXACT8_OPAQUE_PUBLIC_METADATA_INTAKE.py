"""Exact eight predeclared public camera/info paths: bytes/SHA only, no value parse."""
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

V = Path(__file__).parent
H = V / 'fixed12_GT_payload_preparation_a'
OUT = V / 'ROOT_ACTUAL_EXACT8_OPAQUE_METADATA_INTAKE.json'
WL = H / 'EXACT_PUBLIC_GT_PAYLOAD_WHITELIST.json'
EXPECTED_WL = 'a65552c47f67861caf011091c9f33e010fd8896f6dac0ef502c46f50bf646ad0'
ENTRY = V / 'ROOT_ACTUAL_EXACT8_OPAQUE_PUBLIC_METADATA_ENTRY.json'


def raw(path, limit=67108864):
    path = str(path)
    assert 'v112' not in path.casefold() and os.path.isabs(path) and os.path.normpath(path) == path
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = path.strip('/').split('/')
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        f = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    assert stat.S_ISREG(os.fstat(f).st_mode)
    with os.fdopen(f, 'rb') as fh:
        data = fh.read(limit + 1)
    assert len(data) <= limit
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    eb = raw(ENTRY)
    entry = json.loads(eb)  # root metadata only
    assert entry['approved_actual_exact8_opaque_intake'] is True
    assert entry['code_sha256'] == sha(raw(__file__))
    assert entry['GT_values_may_be_decoded'] is False
    assert entry['download_or_fallback'] is False
    assert entry['output_path'] == str(OUT)
    wb = raw(WL)
    assert sha(wb) == EXPECTED_WL == entry['whitelist_sha256']
    wl = json.loads(wb)  # saved whitelist, not GT body
    records = [r for r in wl['payloads'] if r['kind'] in {'SCENE_GT_INFO_JSON', 'SCENE_CAMERA_JSON'}]
    assert len(records) == 8 and len({r['path'] for r in records}) == 8
    assert all(r['sha256'] is None and r['scene'] in {'000000', '000002', '000005', '000006'} for r in records)
    assert entry['exact8_paths'] == [r['path'] for r in records]
    findings = []
    for r in records:
        row = {'path': r['path'], 'scene': r['scene'], 'kind': r['kind'], 'prior_SHA': 'UNKNOWN', 'GT_value_decodes': 0}
        try:
            data = raw(r['path'])
        except FileNotFoundError:
            row.update(status='EXACT_DECLARED_LOCAL_PATH_MISSING', sha256=None, size_bytes=None)
        else:
            row.update(status='PRESENT_FIRST_OPAQUE_SHA_INTAKE', sha256=sha(data), size_bytes=len(data))
        findings.append(row)
    result = {'completed': True, 'PASS': True, 'entry': {'path': str(ENTRY), 'sha256': sha(eb)},
              'code': {'path': str(Path(__file__).absolute()), 'sha256': sha(raw(__file__))},
              'whitelist': {'path': str(WL), 'sha256': EXPECTED_WL}, 'findings8': findings,
              'present': sum(r['status'] == 'PRESENT_FIRST_OPAQUE_SHA_INTAKE' for r in findings),
              'missing': sum(r['status'] == 'EXACT_DECLARED_LOCAL_PATH_MISSING' for r in findings),
              'raw_metadata_values_JSON_decoded': 0, 'download': 0, 'discovery_or_alternate_path': 0,
              'model_image_array_PnP_training_or_protected_access': 0, 'training_admission': False,
              'UTC': datetime.now(timezone.utc).isoformat()}
    with OUT.open('x') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'completed': True, 'present': result['present'], 'missing': result['missing'],
                      'receipt': {'path': str(OUT), 'sha256': sha(raw(OUT))}}))


if __name__ == '__main__':
    main()
