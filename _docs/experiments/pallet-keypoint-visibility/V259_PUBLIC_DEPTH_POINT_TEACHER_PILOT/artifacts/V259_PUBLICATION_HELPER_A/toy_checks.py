"""Invented temporary bytes/path checks only; no Source/Git operations."""
from pathlib import Path
import hashlib
import json
import tempfile
import types

H = Path(__file__).absolute().parent
code = (H / 'publish_public_point_teacher.py').read_bytes()
module = types.ModuleType('v259_helper_invented_toy')
module.__file__ = str(H / 'publish_public_point_teacher.py')
exec(compile(code, module.__file__, 'exec'), module.__dict__)


def rejects(fn):
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError('Expected fail-closed rejection')


checks = []
with tempfile.TemporaryDirectory(prefix='v259_publication_toy_') as tmp:
    root = Path(tmp)
    data = bytes([0, 1, 10, 255, 13, 0]) + b'raw bytes\n'
    source = root / 'source.bin'
    source.write_bytes(data)
    binding = {'path': str(source), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
    target = root / 'out' / 'copy.bin'
    module.copy_bound(binding, target)
    assert target.read_bytes() == data
    checks.append('binary copy SHA/size and byte preservation')
    rejects(lambda: module.copy_bound({**binding, 'sha256': '0'*64}, root / 'wrong.bin'))
    checks.append('incorrect SHA rejected, without source normalization')
    try:
        module.copy_bound(binding, target)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Existing destination was overwritten')
    checks.append('exclusive destination create')
    for value in ('/absolute', '../escape', 'a/../b', 'a//b', 'a/./b', 'v112/x', 'CaChe/x', 'x.PYC', 'a\\b'):
        rejects(lambda v=value: module.relative(v))
    checks.append('relative traversal/forbidden paths rejected')
    link = root / 'link'
    link.symlink_to(root / 'out', target_is_directory=True)
    rejects(lambda: module.safe(link / 'unopened.bin'))
    checks.append('ancestor symlink rejected before descendant open')
    rejects(lambda: module.validate_binding({'path': str(source), 'sha256': binding['sha256']}))
    checks.append('missing size rejected')
    false = root / 'false.json'
    false.write_text(json.dumps({'RootapprovedPublicPointTeacherPublication': False}) + '\n')
    rejects(lambda: module.validated_entry(false))
    checks.append('false entry stops before payload/Git/output access')
    raw = b'{"invented":"entry"}\n'
    embedded = module.json_bytes({'utf8': raw.decode('utf-8')})
    assert json.loads(embedded)['utf8'].encode('utf-8') == raw
    checks.append('original entry UTF8 roundtrip preserves original bytes')

result = {'completed': True, 'PASS': True, 'checks': checks,
          'helper_sha256': hashlib.sha256(code).hexdigest(),
          'invented_temporary_bytes_only': True, 'real_Source_payload_reads': 0,
          'Git_operations': 0, 'models_or_decoders': 0}
out = H / 'INVENTED_CHECKS.json'
with out.open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(result, ensure_ascii=False))
