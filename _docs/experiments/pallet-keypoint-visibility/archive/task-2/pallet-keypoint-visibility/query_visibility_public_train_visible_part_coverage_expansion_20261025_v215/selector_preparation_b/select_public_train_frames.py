"""V215 metadata-only fixed SOURCE TRAIN frame selection. Never opens binary paths."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import traceback

H = Path(__file__).absolute().parent
V = H.parent
P = V.parent
SCOPE = {'path': str(V / 'ROOT_PROSPECTIVE_PUBLIC_SOURCE_TRAIN_COVERAGE_PILOT_SCOPE.json'), 'sha256': '417b08aa3bb0474aba448f2c25cd9099c1bf559a159e93538e3988e0e64a1383'}
RULE = {'path': str(H / 'PROSPECTIVE_SELECTION_RULE_BEFORE_METADATA.json'), 'sha256': '845a823e5d53cca8aacb15df05787d8766367ea590a75c8e62011d24a7e52daf'}
SEED = 'V215_public_SOURCE_visible_part_coverage_20261025'
PREFIXES = ('val/000000', 'val/000002', 'val/000005', 'val/000006')
QUOTA = 3
BINDINGS = {
    'rows': {'path': str(P / 'query_visibility_local_support_raw_head_20261024_v181/source_raw96_cache_v1/PUBLIC_TRAIN1480_ROWS.jsonl'), 'sha256': '08754bf88a98fcdc1a56cd0d84d6dd40804ff9de2f1ad7693a9e5bdd1fa31ff0'},
    'roles': {'path': str(P / 'query_visibility_source_group_split_direct_end_training_20261024_v194/source_component_split_v1/PUBLIC_TRAIN1480_SPLIT_ROWS.jsonl'), 'sha256': 'b04d5d7b1779b3df346a3fe6db26074b7376733317212471eef838439d72e0cc'},
    'frames': {'path': str(P / 'query_visibility_local_support_raw_head_20261024_v181/sam_input_preparation_c/PUBLIC185_FRAMES_METADATA_ONLY.jsonl'), 'sha256': '0cb2296e3dd786b294416a25d1c86e5af1331a43070270ae8f2828e245e74ca7'},
    'source_readiness': {'path': str(P / 'query_visibility_local_support_raw_head_20261024_v181/sam_input_preparation_c/PUBLIC_INPUT_READINESS.json'), 'sha256': '37270ea8a5c464a1d32ba310f46aae33817f067092d30ae0dc9c4fc3d02ebb6c'},
    'weak44_metadata': {'path': str(P / 'query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/WEAK44_INPUT_METADATA_AND_FUTURE_PNG_WHITELIST.json'), 'sha256': 'b857d89d8ff12f8d044f333e14b1055b7a36aacb35ea8ccd86c26097a58b8b0c'},
}


def need(value, message):
    if not value:
        raise AssertionError(message)


def lexical(path):
    value = str(path)
    need('v112' not in value.casefold(), 'V112 lexical rejection before filesystem access')
    p = Path(value)
    need(p.is_absolute() and '..' not in p.parts and P in p.parents, 'Exact absolute P-relative safe path')
    return p


def opened(path, write=False):
    p = lexical(path)
    fd = os.open(p.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL if write else os.O_RDONLY
        out = os.open(p.name, flags | os.O_NOFOLLOW, 0o644, dir_fd=fd)
    finally:
        os.close(fd)
    need(stat.S_ISREG(os.fstat(out).st_mode), 'Regular file required')
    return os.fdopen(out, 'wb' if write else 'rb')


def byte_read(path):
    with opened(path) as f:
        return f.read()


def bind(path):
    return {'path': str(lexical(path)), 'sha256': hashlib.sha256(byte_read(path)).hexdigest()}


def checked(binding):
    need(set(binding) == {'path', 'sha256'} and bind(binding['path']) == binding, 'Exact metadata SHA binding')
    return binding['path']


def read_binding(binding):
    return json.loads(byte_read(checked(binding)))


def lines_binding(binding):
    return [json.loads(line) for line in byte_read(checked(binding)).splitlines() if line.strip()]


def write(path, value):
    with opened(path, True) as f:
        f.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def write_text(path, value):
    with opened(path, True) as f:
        f.write(value.encode())


def same_value(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def stable_hash(prefix, source_id):
    return hashlib.sha256((SEED + '\n' + prefix + '\n' + source_id).encode('utf-8')).hexdigest()


def matching_possible(slots, candidates, used):
    """Bipartite prefix-slot to distinct source-component matching only."""
    match = {}
    def add(slot, seen):
        for comp in sorted({r['component_id'] for r in candidates if r['prefix'] == slot and r['component_id'] not in used}):
            if comp in seen:
                continue
            seen.add(comp)
            if comp not in match or add(match[comp], seen):
                match[comp] = slot
                return True
        return False
    return all(add(slot, set()) for slot in slots)


def select_twelve(candidates):
    slots = [p for p in PREFIXES for _ in range(QUOTA)]
    need(len({r['source_id'] for r in candidates}) == len(candidates), 'Unique original frame identity')
    need(matching_possible(slots, candidates, set()), 'Infeasible exact12 distinct components; no fallback')
    selected, used = [], set()
    for i, slot in enumerate(slots):
        ordered = sorted((r for r in candidates if r['prefix'] == slot and r['component_id'] not in used), key=lambda r: (r['stable_SHA256'], r['source_id']))
        eligible = [r for r in ordered if matching_possible(slots[i + 1:], candidates, used | {r['component_id']})]
        need(eligible, 'No exact continuation; no fallback')
        selected.append(eligible[0])
        used.add(eligible[0]['component_id'])
    need(len(selected) == len(used) == 12 and Counter(r['prefix'] for r in selected) == {p: 3 for p in PREFIXES}, '12 original frames / 12 components / exact4x3')
    return selected


def metadata():
    roles = lines_binding(BINDINGS['roles'])
    # Read only role + identity for non-TRAIN rows. Never inspect their q/L/targets.
    train_roles = {r['row_id']: r for r in roles if r['role'] == 'TRAIN'}
    need(len(roles) == 1480 and len(train_roles) == 1240, 'Original full1480 metadata / TRAIN1240 identities')
    rows = lines_binding(BINDINGS['rows'])
    groups = defaultdict(list)
    for r in rows:
        if r['row_id'] not in train_roles:
            continue
        z = train_roles[r['row_id']]
        for key in ('original_global_index', 'native_slot', 'original_query_available', 'L', 'query_xy'):
            need(same_value(r[key], z[key]), 'Original TRAIN META6 role identity and scalar parity: ' + key)
        o = r['original']
        need(o['split'] == 'SOURCE_DEVELOPMENT' and o['native_slot'] == r['native_slot'] and o['query_id'] == r['row_id'], 'SOURCE TRAIN input metadata only')
        groups[o['frame_index']].append(dict(row_id=r['row_id'], original_global_index=r['original_global_index'], native_slot=r['native_slot'], query_xy=r['query_xy'], L=r['L'], original_query_available=r['original_query_available'], raw_packet=r['raw_packet'], raw_row=r['raw_row'], RGB96_sha256=r['actual_raw_RGB96_sha256'], valid96_sha256=r['actual_raw_valid96_sha256'], source_id=o['source_id'], frame_id=o['frame_id'], original_frame_index=o['frame_index'], component_id=z['component_id'], role='TRAIN', original_flags={k: o.get(k) for k in ('feature_constructed', 'native_continuous_valid', 'nearest_valid', 'Q_inside', 'local_central4_available', 'context_available')}))
    ready = read_binding(BINDINGS['source_readiness'])
    need(ready['completed'] and ready['PASS'] and ready['frames_manifest'] == BINDINGS['frames'], 'SOURCE185 input-only readiness / exact manifest')
    frames = lines_binding(BINDINGS['frames'])
    need(len(frames) == 185 and len(groups) == 155, '185 metadata frames / 155 original TRAIN frames')
    # Non-TRAIN frame values beyond identity are not consumed.
    frame_map = {r['original_frame_index']: r for r in frames if r['original_frame_index'] in groups}
    weak = read_binding(BINDINGS['weak44_metadata'])
    weak_train = [r for r in weak['rows'] if r['component_role'] == 'TRAIN']
    need(len(weak_train) == 44 and len(weak['rows']) == 44, 'Weak44 source TRAIN authority only')
    weak_ids = {r['source_row_id'] for r in weak_train}
    weak_frames = list(weak['frames'].values())
    excluded_ids = {r['source_id'] for r in weak_frames}
    excluded_components = {r['component_id'] for r in weak_frames}
    need(len(weak_ids) == 8 and len(excluded_ids) == len(excluded_components) == 3, 'Exactly old weak8 cases / 3 frames and origin components')
    alltrain = []
    for fi, rr in sorted(groups.items()):
        rr = sorted(rr, key=lambda r: r['native_slot'])
        f = frame_map[fi]
        need(len(rr) == 8 and [r['native_slot'] for r in rr] == list(range(8)) and len({r['component_id'] for r in rr}) == 1 and len({r['source_id'] for r in rr}) == 1, 'Whole immutable native8 TRAIN frame')
        need(f['split'] == 'SOURCE_DEVELOPMENT' and f['source_id'] == rr[0]['source_id'] and f['frame_id'] == rr[0]['frame_id'], 'Full original TRAIN RGB metadata identity')
        for k, r in enumerate(rr):
            need(r['raw_row'] == k and r['raw_packet'] == f['saved_rgb96_packet'] and same_value(r['query_xy'], f['original_initial_points8'][k]) and same_value(r['L'], f['original_L8'][k]) and r['original_global_index'] == f['original_global_indices'][k] and r['row_id'] == f['original_query_ids'][k], 'Exact original q8/L8/native/cache binding')
            lexical(r['raw_packet']['path'])  # String-only safety, no binary FS operation.
        lexical(f['image']['binding']['path'])
        sid = rr[0]['source_id']
        prefix = sid.split(':')[0]
        need(prefix in PREFIXES, 'Existing SOURCE four-prefix namespace only')
        av = sum(r['original_query_available'] is True for r in rr)
        excluded = sid in excluded_ids or rr[0]['component_id'] in excluded_components
        alltrain.append(dict(source_id=sid, original_frame_id=sid, original_frame_index=fi, frame_id=f['frame_id'], prefix=prefix, component_id=rr[0]['component_id'], role='TRAIN', original_available_count=av, stable_SHA256=stable_hash(prefix, sid), excluded_old_weak_component=excluded, eligible=(not excluded and av >= 4), image=f['image'], original_encoded_RGB_provenance=f['original_RGB'], original_RGB_decoder=f['original_RGB_decoder'], native8=rr))
    need(sum(not r['original_query_available'] for f in alltrain for r in f['native8']) == 161 and len({f['component_id'] for f in alltrain}) == 103, 'Original TRAIN U161 and origin components103 preserved')
    candidates = [r for r in alltrain if r['eligible']]
    selected = select_twelve(candidates)
    need(not any(f['source_id'] in excluded_ids or f['component_id'] in excluded_components for f in selected), 'No prior weak frame/component selected')
    return selected, alltrain, dict(original_TRAIN_frames=155, original_TRAIN_native_rows=1240, original_TRAIN_inputU=161, original_TRAIN_components=103, original_TRAIN_prefix_counts=dict(Counter(r['prefix'] for r in alltrain)), old_weak_frame_ids=sorted(excluded_ids), excluded_origin_component_ids=sorted(excluded_components), frames_excluded_by_weak_component=sum(r['excluded_old_weak_component'] for r in alltrain), frames_below_four_available=sum(r['original_available_count'] < 4 for r in alltrain), eligible_frames=len(candidates), eligible_prefix_counts=dict(Counter(r['prefix'] for r in candidates)), eligible_components=len({r['component_id'] for r in candidates}), selected_frames=12, selected_native_rows=96, selected_original_inputU=sum(not r['original_query_available'] for f in selected for r in f['native8']), selected_distinct_components=12, selected_prefix_counts=dict(Counter(r['prefix'] for r in selected)), source_HOLDOUT_values_used=False, original_ROLE_unchanged=True)


def invented_tests():
    checks = []
    def check(name, condition):
        need(condition, name)
        checks.append(name)
    fake = [dict(source_id=p + ':' + str(i), prefix=p, component_id=p + ':' + str(i), stable_SHA256=stable_hash(p, p + ':' + str(i))) for p in PREFIXES for i in range(5)]
    a = select_twelve(fake)
    check('Exact4x3_unique12_and_input_order_independent', a == select_twelve(list(reversed(fake))) and len({r['component_id'] for r in a}) == 12)
    check('SHA256_exact_frozen_seed_delimiters_original_identity', stable_hash('p', 'i') == hashlib.sha256((SEED + '\np\ni').encode()).hexdigest())
    # The first prefix must skip a shared component when the last prefix needs it.
    cross = [dict(source_id=p + str(i), prefix=p, component_id=p + str(i), stable_SHA256=str(i)) for p in PREFIXES for i in range(3)]
    cross.append(dict(source_id='first-extra', prefix=PREFIXES[0], component_id='first-extra', stable_SHA256='9'))
    cross[0]['component_id'] = cross[-2]['component_id']
    check('Global_matching_preserves_last_prefix_feasibility', len({r['component_id'] for r in select_twelve(cross)}) == 12)
    for name, rows in [('Missing_prefix_no_fallback', [r for r in fake if r['prefix'] != PREFIXES[-1]]), ('Insufficient_components_no_duplicate_fallback', [dict(r, component_id='one') for r in fake])]:
        try:
            select_twelve(rows)
            ok = False
        except AssertionError:
            ok = True
        check(name, ok)
    check('Unavailable_null_query_not_imputed', same_value(None, None) and not same_value(None, [0., 0.]))
    try:
        lexical(P / 'forbidden_v112_no_stat')
        ok = False
    except AssertionError:
        ok = True
    check('Forbidden_name_rejected_before_any_FS', ok)
    return dict(completed=True, PASS=True, invented_only=True, count=len(checks), checks=checks, source_payload_access=0)


def prepare():
    scope, rule = read_binding(SCOPE), read_binding(RULE)
    need(scope['approved_code_metadata_preparation_only'] and not scope['approved_actual_RGB_array_or_annotation'], 'Root code/metadata-only scope')
    need(rule['seed'] == SEED and tuple(rule['prefix_order']) == PREFIXES and rule['quota_per_prefix'] == QUOTA and rule['no_fallback'], 'Prospective frozen rule')
    checks = invented_tests()
    write(H / 'INVENTED_SELECTION_CHECKS.json', checks)
    selected, alltrain, census = metadata()
    binary = [b for f in selected for b in (f['image']['binding'], f['native8'][0]['raw_packet'])]
    need(len(binary) == len({b['path'] for b in binary}) == 24, 'Only selected12 original RGB / 12 original frame8 packets')
    for b in binary:
        lexical(b['path'])  # No stat, hash, header or binary open.
        need(len(b['sha256']) == 64, 'Inherited payload SHA only')
    selection = dict(completed=True, PASS=True, phase='V215', selection_frozen_before_pixels=True, rule=RULE, code=bind(__file__), scope=SCOPE, input_bindings=BINDINGS, census=census, frames=selected, future_payload_whitelist=binary, payload_path_stat_hash_header_or_decode_count=0, new_training_admission=False, selected_model=None, no_fallback=True)
    write(H / 'SELECTED12_SOURCE_TRAIN_FRAMES_BEFORE_PIXELS.json', selection)
    write(H / 'SOURCE_TRAIN_SELECTION_CENSUS.json', dict(completed=True, PASS=True, census=census, frames=[{k: f[k] for k in ('source_id', 'original_frame_index', 'prefix', 'component_id', 'original_available_count', 'stable_SHA256', 'excluded_old_weak_component', 'eligible')} for f in alltrain]))
    fields = ['position', 'source_id', 'original_frame_index', 'prefix', 'component_id', 'native_slot', 'row_id', 'original_global_index', 'query_x', 'query_y', 'L', 'original_query_available', 'role', 'RGB96_sha256', 'valid96_sha256', 'raw_packet_path', 'raw_packet_sha256']
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for pos, f in enumerate(selected):
        for r in f['native8']:
            q = r['query_xy']
            writer.writerow(dict(position=pos, source_id=f['source_id'], original_frame_index=f['original_frame_index'], prefix=f['prefix'], component_id=f['component_id'], native_slot=r['native_slot'], row_id=r['row_id'], original_global_index=r['original_global_index'], query_x=None if q is None else q[0], query_y=None if q is None else q[1], L=r['L'], original_query_available=r['original_query_available'], role=r['role'], RGB96_sha256=r['RGB96_sha256'], valid96_sha256=r['valid96_sha256'], raw_packet_path=r['raw_packet']['path'], raw_packet_sha256=r['raw_packet']['sha256']))
    write_text(H / 'SELECTED96_ORIGINAL_NATIVE_INPUT_IDENTITY.csv', out.getvalue())
    write(H / 'FUTURE_PREVIEW_ENTRY_TEMPLATE_APPROVED_FALSE.json', dict(approved_selected_SOURCE_TRAIN_preview=False, scope=SCOPE, rule=RULE, selection=bind(H / 'SELECTED12_SOURCE_TRAIN_FRAMES_BEFORE_PIXELS.json'), exact_future_payload_whitelist=binary, source_binary_access_requires_new_root_entry=True, scores_GT_oldweak_models_allowed=False, original_native_qL_U_roles_unchanged=True, training_admission=False))
    doc = ['# V215 공개 SOURCE TRAIN 12프레임 고정 선택', '', '기존 TRAIN155에서 이전 weak3 및 같은 원본 component를 제외했다. 원 query availability가 4개 이상인 frame만 사전고정 hash순으로 4prefix×3개를 선택하며 12component를 유지한다. 조건이 부족하면 대체하지 않는다.', '', '선택은 새 RGB·crop·GT·모델·점수 접근 전에 완료했다. fullRGB canonical NPY와 원래 native8 INPUT_ONLY NPZ의 기존 SHA는 메타데이터에서만 전달했다. 파일 존재/내용을 다시 검증하지 않았다. 모든 원 q/L/native0..7/U/role은 그대로이며 새 라벨·학습 admission은 없다.', '', f"선택96slot 중 원 U={census['selected_original_inputU']}. graph component 구분은 물리적 독립성을 보증하지 않는다. 이후 fullframe+원crop의 의도부위 약관찰에서 D/E/S/UNKNOWN과 ON/NEAR/OFF를 분리하고 UNKNOWN·불일치를 보존해야 한다. ROI는 수정 keypoint나 physical gold가 아니다.", '', '|순서|원 frame|prefix|원 available/8|', '|---:|---|---|---:|']
    doc += [f"|{i}|{f['source_id']}|{f['prefix']}|{f['original_available_count']}|" for i, f in enumerate(selected)]
    doc += ['', '[선택 원문](SELECTED12_SOURCE_TRAIN_FRAMES_BEFORE_PIXELS.json) · [원96slot ID/qL](SELECTED96_ORIGINAL_NATIVE_INPUT_IDENTITY.csv) · [TRAIN 후보 census](SOURCE_TRAIN_SELECTION_CENSUS.json) · [사전 선택 규칙](PROSPECTIVE_SELECTION_RULE_BEFORE_METADATA.json)', '', '실제 preview는 별도 root 승인 entry 이후에만 실행한다. 선택·native·q/L·sampler를 score나 관찰결과로 바꾸지 않는다.', '']
    write_text(H / 'KO.md', '\n'.join(doc))
    ready = dict(completed=True, PASS=True, preparation_only=True, metadata_selection_completed=True, code=bind(__file__), scope=SCOPE, protocol=RULE, checks=bind(H / 'INVENTED_SELECTION_CHECKS.json'), selection=bind(H / 'SELECTED12_SOURCE_TRAIN_FRAMES_BEFORE_PIXELS.json'), census=bind(H / 'SOURCE_TRAIN_SELECTION_CENSUS.json'), identity_csv=bind(H / 'SELECTED96_ORIGINAL_NATIVE_INPUT_IDENTITY.csv'), false_entry=bind(H / 'FUTURE_PREVIEW_ENTRY_TEMPLATE_APPROVED_FALSE.json'), Korean_summary=bind(H / 'KO.md'), original_source_binary_stat_hash_header_decode_count=0, actual_preview_or_observations=False, training_admission=False, frozen_no_further_edits=True)
    write(H / 'FINAL_READY.json', ready)
    print(json.dumps(dict(completed=True, PASS=True, ready=bind(H / 'FINAL_READY.json'), selected=[f['source_id'] for f in selected], census=census), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true', required=True)
    parser.parse_args()
    try:
        prepare()
    except BaseException:
        write(H / 'FAILED_METADATA_PREPARATION_NO_AUTOMATIC_RETRY.json', dict(completed=False, PASS=False, traceback=traceback.format_exc(), source_binary_access=0))
        raise
