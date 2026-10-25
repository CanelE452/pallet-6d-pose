"""명시 entry leaf만 읽는다. 원 source/가중치/RGB pointer는 추적하지 않는다."""
import hashlib
import json
import os
from pathlib import Path
import stat

def need(ok, message):
    if not ok:
        raise ValueError(message)

def lexical(value):
    s = str(value)
    need(not any(x in s.casefold() for x in ('v112', '__pycache__', '.pyc',
        'current223', 'publicheld', 'oldval', 'source_holdout', 'private_evaluation',
        '/v177', '/v178', '/v182')), '금지 경로: filesystem 전 거부')
    p = Path(s)
    need(p.is_absolute() and '..' not in p.parts, '절대 경로/부모 이동 금지')
    return p

def safe(value):
    p = lexical(value)
    for q in reversed(p.parents):
        need(stat.S_ISDIR(os.lstat(q).st_mode), '부모 symlink/non-directory 금지')
    if p.exists() or p.is_symlink():
        need(not stat.S_ISLNK(os.lstat(p).st_mode), 'leaf symlink 금지')
    return p

def identity(b):
    need(isinstance(b, dict) and {'path', 'sha256'} <= set(b) <=
         {'path', 'sha256', 'size_bytes'}, '정확 binding schema')
    return str(lexical(b['path'])), b['sha256']

def read_raw(path):
    with os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        return f.read()

def verify(b, white):
    need(identity(b) in {identity(x) for x in white}, '선언 whitelist 밖')
    p = safe(b['path']); h = hashlib.sha256(); n = 0
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block); n += len(block)
    need(h.hexdigest() == b['sha256'] and
         ('size_bytes' not in b or b['size_bytes'] == n), 'binding byte 변경')
    return p

def load_json(b, white):
    return json.loads(read_raw(verify(b, white)))

def binding(path):
    raw = read_raw(path)
    return dict(path=str(safe(path)), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))

def write_json(path, value):
    with os.fdopen(os.open(safe(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                          os.O_NOFOLLOW, 0o600), 'w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    return binding(path)

def open_entry(path, flag, code_path):
    e = json.loads(read_raw(path))
    need(e.get(flag) is True and e.get('source_only') is True and
         e.get('automatic_retry') is False, 'Root 별도 actual entry 필요')
    for k in ('code', 'protocol', 'io_code'):
        verify(e[k], e['code_whitelist'])
    need(identity(e['code']) == identity(binding(code_path)) and
         identity(e['io_code']) == identity(binding(__file__)), '실행 code pin')
    root = load_json(e['root_protocol'], e['metadata_whitelist'])
    need(e['root_protocol']['sha256'] ==
         '4cbbd1ede0420e1cc6dfe83446b2204141bd99ac29781836b9663c49c77fa021', '원 Root protocol pin')
    load_json(e['root_scope'], e['metadata_whitelist'])
    return e, root
