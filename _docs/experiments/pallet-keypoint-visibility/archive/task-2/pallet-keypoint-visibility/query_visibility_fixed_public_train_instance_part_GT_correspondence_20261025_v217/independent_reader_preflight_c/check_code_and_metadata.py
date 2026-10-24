"""Bounded code/metadata preflight; never follows an inert GT payload pointer."""
import ast
import collections
import copy
import hashlib
import json
import os
import stat
from unittest.mock import patch

P = '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility'
D = P + '/query_visibility_fixed_public_train_instance_part_GT_correspondence_20261025_v217'
H = D + '/fixed12_GT_payload_preparation_a'
O = D + '/independent_reader_preflight_c'
reads = []

def raw(path):
    assert 'v112' not in path.casefold() and os.path.normpath(path) == path
    assert path.startswith(H + '/') or path in {
        D + '/ROOT_PROSPECTIVE_FIXED12_PUBLIC_GT_CORRESPONDENCE_AUDIT_SCOPE.json',
        D + '/ROOT_EXACT8_OPAQUE_PUBLIC_METADATA_INTAKE.py',
        D + '/ROOT_PROSPECTIVE_EXACT8_OPAQUE_PUBLIC_METADATA_INTAKE.json',
        __file__,
        P + '/query_visibility_public_train_part_query_support_contract_20261025_v216/joined_part_query_support_v2/ALL96_JOINED_CANDIDATES.json',
        P + '/query_visibility_public_train_part_query_support_contract_20261025_v216/component_split_preparation_b/INTERNAL_SOURCE_FIT_DEV_COMPONENT_SPLIT.json',
    }
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        parts = path.strip('/').split('/')
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = nxt
        f = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        try:
            assert stat.S_ISREG(os.fstat(f).st_mode)
            with os.fdopen(f, 'rb', closefd=False) as fh:
                b = fh.read(4000000)
        finally:
            os.close(f)
    finally:
        os.close(fd)
    reads.append({'path': path, 'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)})
    return b

def checked(binding):
    b = raw(binding['path'])
    assert hashlib.sha256(b).hexdigest() == binding['sha256']
    if 'size_bytes' in binding:
        assert len(b) == binding['size_bytes']
    return b

def save(name, doc):
    b = (json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    with open(O + '/' + name, 'xb') as fh:
        fh.write(b)
    return {'path': O + '/' + name, 'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)}

ready = json.loads(checked({'path': H + '/FINAL_READY.json', 'sha256': 'b97dc7de1d15e8e251bbb5dbd262611d69b7ab711a24c4a2b0f26f559af243e4'}))
scope = json.loads(checked(ready['scope']))
code = checked(ready['code'])
proto = json.loads(checked(ready['protocol']))
for k in ['checks', 'test_code', 'metadata_preparation_code', 'false_entry']:
    checked(ready[k])
docs = {n: json.loads(checked(b)) for n, b in ready['artifacts'].items() if n.endswith('.json')}
original = json.loads(checked(scope['preparation_inputs'][0]))['rows96']
split = json.loads(checked(scope['preparation_inputs'][1]))['rows96']
rows = docs['ALL96_ORIGINAL_INPUTS_AND_GT_POINTERS.json']['rows96']
frames = docs['FIXED12_TRAIN_FRAME_AND_INSTANCE_CANDIDATE_METADATA.json']['frames12']
wl = docs['EXACT_PUBLIC_GT_PAYLOAD_WHITELIST.json']['payloads']
assert len(rows) == len(original) == len(split) == 96 and len(frames) == 12
for r, j, s in zip(rows, original, split):
    i = j['input']
    for k in ['frame_key', 'original_frame_index', 'component_id', 'source_group_key', 'internal_source_role', 'row_id', 'original_global_index', 'native_slot', 'query_xy', 'L', 'original_query_available', 'original_flags']:
        assert r[k] == i[k] == s[k], (r['row_id'], k)
    assert r['frame_position'] == i['position'] == s['position']
    assert r['original_role'] == i['role'] == s['role'] == 'TRAIN'
    assert r['selected_gt_instance_index'] is None and r['training_admission'] is False
assert sum(not r['original_query_available'] for r in rows) == 11
assert collections.Counter(r['internal_source_role'] for r in rows) == {'SOURCE_FIT': 64, 'SOURCE_DEV': 32}
assert [f['frame_position'] for f in frames] == list(range(12))
assert collections.Counter(f['scene'] for f in frames) == {'000000': 3, '000002': 3, '000005': 3, '000006': 3}
assert len(wl) == len({r['path'] for r in wl}) == 55
declared = {r['path']: r for r in wl}
unknown = [r for r in wl if r['sha256'] is None]
assert len(unknown) == 8 and all(r['kind'] in {'SCENE_CAMERA_JSON', 'SCENE_GT_INFO_JSON'} for r in unknown)
assert all('v112' not in r['path'].casefold() for r in wl)
assert all(r['sha256'] is None or len(r['sha256']) == 64 for r in wl)
assert sum(len(f['obj1_candidate_gt_list_indices']) for f in frames) == 15
for f in frames:
    assert f['selected_gt_instance_index'] is None and f['training_admission'] is False
    assert sorted(map(int, f['candidate_masks'])) == f['obj1_candidate_gt_list_indices']
    for pair in f['candidate_masks'].values():
        for r in pair.values():
            assert r == declared[r['path']]
    for k in ['depth', 'nominal_CAD']:
        assert f[k] == declared[f[k]['path']]

ns = {'__name__': 'independent_in_memory_reader', '__file__': ready['code']['path']}
exec(compile(code, ready['code']['path'], 'exec'), ns)
checks = []
def bad(name, fn):
    try:
        fn()
    except (ValueError, AssertionError, OSError):
        checks.append(name); return
    raise AssertionError('Expected rejection: ' + name)

with patch.object(os, 'open', side_effect=AssertionError('No fixture filesystem access')):
    calls = []
    def dec(s):
        calls.append(s); return json.loads(s)
    result, stats = ns['select_top_level_values']('{"unselected":{"opaque":NOT_JSON,"quoted":"} , \\\""},"51":[{"obj_id":1}],"other":[1,2]}', {'51'}, dec)
    assert result == {'51': [{'obj_id': 1}]} and len(calls) == 1 and stats['skipped_values_JSON_decoded'] == 0
    checks.append('selected_values_only_decoder_spy_with_semantically_invalid_unselected_value')
    bad('missing_selected_key', lambda: ns['select_top_level_values']('{"52":[]}', {'51'}))
    bad('duplicate_unselected_key', lambda: ns['select_top_level_values']('{"x":0,"x":1,"51":[]}', {'51'}))
    bad('malformed_selected_value', lambda: ns['select_top_level_values']('{"51":NOT_JSON}', {'51'}))
    ns['validate_all8_inputs'](rows)
    x = copy.deepcopy(rows); x[1]['native_slot'] = x[0]['native_slot']
    bad('duplicate_native_membership', lambda: ns['validate_all8_inputs'](x))
    x = copy.deepcopy(rows); x[0]['original_role'] = 'HOLDOUT'
    bad('non_TRAIN_role', lambda: ns['validate_all8_inputs'](x))
    u = unknown[0]
    bad('missing_first_opaque_permission', lambda: ns['gate_payload'](u, declared, set()))
    ns['gate_payload'](u, declared, {u['path']})
    checks.append('explicit_first_opaque_permission_pure_acceptance')
    binary = next(r for r in wl if r['kind'] == 'DEPTH_PNG')
    bad('binary_kind_before_FS', lambda: ns['gate_payload'](binary, declared, set()))
    bad('changed_declared_identity_before_FS', lambda: ns['gate_payload']({**u, 'sha256': '0'*64}, declared, set()))
    bad('V112_before_any_FS', lambda: ns['opened']('/not/read/v112/data.json'))
    assert len(ns['requested_records'](wl, ['SCENE_GT_JSON'])) == 4
    assert len(ns['requested_records'](wl, sorted(ns['JSON_KINDS']))) == 12
    checks.append('known4_only_or_explicit12_JSON_branches')

hc = checked({'path': D + '/ROOT_EXACT8_OPAQUE_PUBLIC_METADATA_INTAKE.py', 'sha256': 'c41d15ababc16871eac64c7d2797c85aec91ebccb7be248a8e412c82987b37b4'})
hs = json.loads(checked({'path': D + '/ROOT_PROSPECTIVE_EXACT8_OPAQUE_PUBLIC_METADATA_INTAKE.json', 'sha256': '5b4752fc48305dbdd8c939550503e51ccb10801981a5361d8453c69a7bb96c31'}))
assert hs['approved_actual_exact8_opaque_intake'] is False
assert hs['exact8_paths'] == [r['path'] for r in wl if r['kind'] in {'SCENE_CAMERA_JSON', 'SCENE_GT_INFO_JSON'}]
tree = ast.parse(hc)
loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'json' and n.func.attr == 'loads']
assert [ast.unparse(n.args[0]) for n in loads] == ['eb', 'wb']
assert "with OUT.open('x')" in hc.decode() and 'except FileNotFoundError:' in hc.decode()
checks.append('opaque_helper_exact8_metadata_schema_no_payload_JSONloads_exclusive_output')
frozen_false = json.loads(checked(ready['false_entry']))
assert frozen_false['approved_actual_selected_GT_JSON_read'] is False and frozen_false['training_admission'] is False
assert frozen_false['authorize_first_opaque_SHA_intake_exact_paths'] == []
checks.append('false_actual_gates_preserved')
fixture = save('INDEPENDENT_CHECKS.json', {'completed': True, 'PASS': True, 'checks': checks, 'count': len(checks), 'original96_exact_input_fields_checked': True, 'source_payload_FS_calls': 0})
review = {
    'completed': True, 'PASS': True, 'prepared_only': True,
    'FINAL_READY': {'path': H + '/FINAL_READY.json', 'sha256': 'b97dc7de1d15e8e251bbb5dbd262611d69b7ab711a24c4a2b0f26f559af243e4'},
    'scope': ready['scope'], 'code': ready['code'], 'protocol': ready['protocol'],
    'root_opaque_intake_code': next(r for r in reads if r['path'].endswith('ROOT_EXACT8_OPAQUE_PUBLIC_METADATA_INTAKE.py')),
    'root_opaque_intake_false_scope': next(r for r in reads if r['path'].endswith('ROOT_PROSPECTIVE_EXACT8_OPAQUE_PUBLIC_METADATA_INTAKE.json')),
    'checker': {'path': __file__, 'sha256': hashlib.sha256(raw(__file__)).hexdigest()}, 'independent_checks': fixture,
    'counts': ready['counts'],
    'findings_KO': [
        '원 V216 두 메타데이터와 96행의 순서·ID·q/L/native·U11·원TRAIN·FIT64/DEV32·flags가 정확히 같다.',
        '선택 TRAIN key만 값 decoder로 보내며 비선택 값은 균형 구조 경계만 찾는다. 전체 JSON 의미 유효성 검증이라고 주장하지 않는다.',
        '4개 known GT만 읽는 분기가 있다. 미확인 camera/info 8경로는 별도 root actual의 정확한 first-opaque 승인 후 SHA를 먼저 기록한다. 누락은 자동 대체·다운로드하지 않는다.',
        '55개 경로는 inert whitelist다. 현재 reader는 scene JSON 3종만 허용하며 mask/depth/CAD는 별도 bounded reader/actual authority가 필요하다.',
        '15개 obj1 후보를 모두 보존하고 selected instance는 12개 전부 UNKNOWN/null이다. 첫 후보·GT-best·새 permutation fallback이 없다.',
        'nominal reference, mask support, depth, owner, occlusion, query/part/native 권한을 분리한다. mask 부재를 E로 복사하거나 완벽 mesh를 일괄 선결조건으로 삼지 않는다.'
    ],
    'limits_KO': [
        'code/저장 메타데이터/가상 fixture 검토이며 실제 GT 구조·파일 존재·SHA·선택 instance·native 가시성을 인증하지 않았다.',
        'runtime 원96 보존은 root가 이번 frozen selection/whitelist/코드/protocol SHA를 정확히 actual entry에 결합하는 조건이다.',
        'opaque helper PASS는 first-byte intake의 행정 완료이며 present/missing을 별도 보존한다. GT 값 해석 완료가 아니다.'
    ],
    'actual_GT_camera_info_mask_depth_CAD_RGB_NPZ_weight_FS_calls': 0,
    'model_projection_training_Git_network_or_protected_access': 0,
    'new_labels_or_training_admission': False,
    'actual_execution_requires_separate_root_entry': True,
    'metadata_read_bindings': reads,
    'own_folder_frozen_no_further_edits': True,
}
out = save('REVIEW.json', review)
with open(O + '/KO.md', 'x') as fh:
    fh.write('# V217 reader 독립 사전 검토\n\nPASS. 원 12 TRAIN 프레임·96행·U11·FIT64/DEV32와 q/L/native/flags를 저장 원문과 대조했다. 선택 key만 JSON 값으로 해석하며, obj1 후보 15개를 보존하고 instance는 모두 UNKNOWN이다.\n\nknown scene GT 4개만의 실행 또는 root가 정확히 승인한 camera/info 8개의 first-opaque SHA intake 후 해석이 가능하다. 미확인 SHA는 파일 부재나 영구 진행 금지를 뜻하지 않는다. 누락·불일치는 대체 없이 중지한다.\n\n실제 GT/이미지/배열/모델 파일은 열거나 stat/hash하지 않았다. mask/depth/CAD 진단과 nominal projection은 이번 reader 범위 밖이며 별도 실행 권한과 코드를 요구한다. 새 라벨·학습 채택은 없다.\n')
print(json.dumps({'completed': True, 'PASS': True, 'checks': len(checks), 'REVIEW': out, 'actual_payload_FS_calls': 0}, ensure_ascii=False))
