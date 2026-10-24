"""Independent selected-key SOURCE JSON QA. No producer import or binary access."""
import collections
import hashlib
import json
import math
import os
import stat
import traceback

P = '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility'
D = P + '/query_visibility_fixed_public_train_instance_part_GT_correspondence_20261025_v217'
H = D + '/fixed12_GT_payload_preparation_a'
BASE_O = D + '/independent_actual_selected_GT_review_c'
O = BASE_O + '/v2'
META_PREFIXES = (H + '/', D + '/fixed12_selected_GT_metadata_v1/', BASE_O + '/')
META_ROOT = {
    D + '/' + n for n in [
        'ROOT_ACTUAL_EXACT8_OPAQUE_METADATA_INTAKE.json',
        'ROOT_ACTUAL_FIXED12_SELECTED_TRAIN_GT_JSON_SCOPE.json',
        'ROOT_ACTUAL_FIXED12_SELECTED_TRAIN_GT_JSON_ENTRY.json',
        'ROOT_ACTUAL_SELECTED_GT_CLOSURE_AND_COUNT_QA.json',
    ]
}
RAW = {}
metadata_reads = []
raw_reads = []

def read(path, payload=False):
    assert 'v112' not in path.casefold() and os.path.isabs(path) and os.path.normpath(path) == path
    assert path in RAW if payload else (path in META_ROOT or path.startswith(META_PREFIXES))
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = path.strip('/').split('/')
        for x in parts[:-1]:
            nf = os.open(x, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = nf
        f = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        try:
            assert stat.S_ISREG(os.fstat(f).st_mode)
            with os.fdopen(f, 'rb', closefd=False) as fh:
                data = fh.read(67108865)
        finally:
            os.close(f)
    finally:
        os.close(fd)
    assert len(data) <= 67108864
    b = {'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
    (raw_reads if payload else metadata_reads).append(b)
    return data

def bound(b):
    data = read(b['path'])
    assert hashlib.sha256(data).hexdigest() == b['sha256'], b['path']
    if b.get('size_bytes') is not None:
        assert len(data) == b['size_bytes']
    return json.loads(data)

def write(name, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    with open(O + '/' + name, 'xb') as fh:
        fh.write(data)
    return {'path': O + '/' + name, 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}

def selective(text, allowed):
    """Locate balanced value slices independently; decode only allowed values."""
    n = len(text)
    def ws(i):
        while i < n and text[i] in ' \t\r\n': i += 1
        return i
    def quoted(i):
        assert text[i] == '"'
        i += 1
        while i < n:
            if text[i] == '\\': i += 2
            elif text[i] == '"': return i + 1
            else: i += 1
        raise ValueError('Unterminated string')
    def end(i):
        if text[i] == '"': return quoted(i)
        stack = []
        j = i
        while j < n:
            c = text[j]
            if c == '"': j = quoted(j); continue
            if c in '[{': stack.append(c)
            elif c in ']}':
                if not stack: return j
                if stack.pop() != {']': '[', '}': '{'}[c]: raise ValueError('Unbalanced structure')
                if not stack: return j + 1
            elif c == ',' and not stack: return j
            j += 1
        return j
    i = ws(0)
    assert i < n and text[i] == '{'
    i = ws(i + 1); result = {}; seen = set(); skipped = 0
    while i < n and text[i] != '}':
        k_end = quoted(i)
        key = json.loads(text[i:k_end])
        assert key not in seen
        seen.add(key); i = ws(k_end)
        assert text[i] == ':'
        start = ws(i + 1); stop = end(start)
        if key in allowed:
            result[key] = json.loads(text[start:stop])
        else:
            skipped += 1
        i = ws(stop)
        assert i < n and text[i] in ',}'
        if text[i] == '}': break
        i = ws(i + 1)
        assert i < n and text[i] != '}'
    assert i < n and text[i] == '}' and ws(i + 1) == n
    assert set(result) == set(allowed)
    return result, {'selected_value_count': len(result), 'skipped_value_count': skipped, 'skipped_values_JSON_decoded': 0}

def finite_flat(x):
    if isinstance(x, list):
        return [v for y in x for v in finite_flat(y)]
    assert isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    return [x]

def main():
    protocol_b = {'path': BASE_O + '/PROTOCOL_BEFORE_INDEPENDENT_SELECTED_GT_RECOMPUTE.json', 'sha256': 'b186b738de7a19e55295fc54ff97bb4366c680bb0cba4be310fa43c41b5440de'}
    protocol = bound(protocol_b)
    scope = bound(protocol['scope']); complete = bound(protocol['source_complete'])
    entry = bound(complete['entry'])
    assert complete['entry'] == protocol['original_entry']
    assert scope['approved_actual_selected_GT_JSON_read'] is entry['approved_actual_selected_GT_JSON_read'] is True
    assert entry['training_admission'] is complete['training_admission'] is False
    intake = bound(entry['first_opaque_intake'])
    assert intake['present'] == 8 and intake['missing'] == 0 and intake['raw_metadata_values_JSON_decoded'] == 0
    wl = bound(entry['payload_whitelist'])
    frames = bound(entry['fixed12_frames'])['frames12']
    rows = bound(entry['original96_inputs'])['rows96']
    saved = bound(complete['selected_GT_metadata'])
    opaque = bound(complete['opaque_bindings_before_values'])
    root_QA = bound({'path': D + '/ROOT_ACTUAL_SELECTED_GT_CLOSURE_AND_COUNT_QA.json', 'sha256': 'f8c026d857616b790ecf1ad9f49e054fed239e3ee207b29154f3a85cc1ae6732'})
    assert entry['original96_inputs']['sha256'] == '22803511e5abb9c9eb07d809bb5796c112fcdcd63ffe75c65bffb878549aa031'
    assert len(rows) == 96 and len(frames) == 12 and len({r['row_id'] for r in rows}) == 96
    assert collections.Counter(r['internal_source_role'] for r in rows) == {'SOURCE_FIT': 64, 'SOURCE_DEV': 32}
    assert sum(not r['original_query_available'] for r in rows) == 11
    assert all(r['original_role'] == 'TRAIN' and r['training_admission'] is False for r in rows)
    for f in frames:
        rr = [r for r in rows if r['frame_position'] == f['frame_position']]
        assert [r['native_slot'] for r in rr] == list(range(8))
        assert all(r['frame_key'] == f['frame_key'] and r['internal_source_role'] == f['internal_source_role'] for r in rr)
    assert opaque['GT_values_parsed_before_this_receipt'] == 0
    first = {r['path']: r for r in intake['findings8']}
    relevant = [r for r in wl['payloads'] if r['kind'] in {'SCENE_GT_JSON', 'SCENE_GT_INFO_JSON', 'SCENE_CAMERA_JSON'}]
    assert {r['path'] for r in relevant} == {r['path'] for r in protocol['exact12_raw_JSON_bindings']} == set(opaque['bindings'])
    for b in protocol['exact12_raw_JSON_bindings']:
        r = next(r for r in relevant if r['path'] == b['path'])
        assert b['sha256'] == (r['sha256'] if r['sha256'] is not None else first[r['path']]['sha256'])
        RAW[b['path']] = b
    # Pure fixtures happen before any permitted raw JSON reads.
    toy, counts = selective('{"held":{"never_decode":INVALID_TOKEN,"s":"} , \\\""},"51":[{"obj_id":1}],"later":[9]}', {'51'})
    assert toy == {'51': [{'obj_id': 1}]} and counts['skipped_value_count'] == 2
    for text, keys in [('{"52":[]}', {'51'}), ('{"51":[],"51":[]}', {'51'}), ('{"51":BAD}', {'51'})]:
        try: selective(text, keys)
        except (ValueError, AssertionError): pass
        else: raise AssertionError('Expected invented rejection')
    parsed = {}; statistics = {}
    for path, b in RAW.items():
        data = read(path, payload=True)
        assert hashlib.sha256(data).hexdigest() == b['sha256'] == opaque['bindings'][path]['sha256']
        assert len(data) == opaque['bindings'][path]['size_bytes']
        if b['size_bytes'] is not None: assert len(data) == b['size_bytes']
        values, stats = selective(data.decode('utf-8'), protocol['selected_scene_keys'][b['scene']])
        assert all(stats[k] == complete['selection_statistics'][path][k] for k in stats)
        parsed[(b['scene'], b['kind'])] = values; statistics[path] = stats
    assert sum(s['selected_value_count'] for s in statistics.values()) == 36
    assert sum(s['skipped_value_count'] for s in statistics.values()) == 519
    candidate_rows = []
    for f, s in zip(frames, saved['frames12']):
        for k in ['frame_position', 'frame_key', 'original_role', 'internal_source_role']:
            assert f[k] == s[k]
        assert s['selected_gt_instance_index'] is None and s['native_target_admission'] is False
        assert s['selected_instance_correspondence'] == 'UNKNOWN_PENDING_EXPLICIT_EVIDENCE'
        key = f['json_frame_key']; scene = f['scene']
        gt = parsed[(scene, 'SCENE_GT_JSON')][key]
        info = parsed[(scene, 'SCENE_GT_INFO_JSON')][key]
        camera = parsed[(scene, 'SCENE_CAMERA_JSON')][key]
        assert isinstance(gt, list) and isinstance(info, list) and len(gt) == len(info)
        indices = [j for j, g in enumerate(gt) if g['obj_id'] == 1]
        assert indices == f['obj1_candidate_gt_list_indices'] == [c['gt_list_index'] for c in s['obj1_candidates']]
        K = finite_flat(camera['cam_K'])
        assert len(K) == 9 and K[0] > 0 and K[4] > 0 and K[6:] == [0, 0, 1]
        assert isinstance(camera['depth_scale'], (int, float)) and math.isfinite(camera['depth_scale']) and camera['depth_scale'] > 0
        assert camera == s['camera']
        for c in s['obj1_candidates']:
            j = c['gt_list_index']; g = gt[j]; v = info[j]
            assert c['GT'] == g and c['GT_info'] == v and c['masks'] == f['candidate_masks'][str(j)]
            assert len(finite_flat(g['cam_R_m2c'])) == 9 and len(finite_flat(g['cam_t_m2c'])) == 3
            for m in ['full_mask', 'visible_mask']:
                base = c['masks'][m]['path'].rsplit('/', 1)[-1]
                assert base == f['frame_filename_stem'] + '_' + str(j).zfill(6) + '.png'
            for k in ['bbox_obj', 'bbox_visib']:
                assert len(finite_flat(v[k])) == 4
                if v[k] == [-1, -1, -1, -1]:
                    assert v['px_count_valid'] == v['px_count_visib'] == 0
                else:
                    assert v[k][2] >= 0 and v[k][3] >= 0
            for k in ['px_count_all', 'px_count_valid', 'px_count_visib']:
                assert isinstance(v[k], int) and v[k] >= 0
            fraction = v['visib_fract']; assert math.isfinite(fraction) and 0 <= fraction <= 1
            candidate_rows.append({'frame_position': f['frame_position'], 'frame_key': f['frame_key'], 'gt_list_index': j, 'bbox_obj': v['bbox_obj'], 'bbox_visib': v['bbox_visib'], 'px_count_all': v['px_count_all'], 'px_count_valid': v['px_count_valid'], 'px_count_visib': v['px_count_visib'], 'visib_fract': fraction, 'visible_count_exceeds_valid_depth_count': v['px_count_visib'] > v['px_count_valid'], 'bbox_obj_negative_origin': v['bbox_obj'] != [-1, -1, -1, -1] and min(v['bbox_obj'][:2]) < 0, 'bbox_metadata_state': 'NO_INFRAME_SUPPORT_UNKNOWN' if any(v[k] == [-1, -1, -1, -1] for k in ['bbox_obj', 'bbox_visib']) else 'BBOX_AVAILABLE_NO_NATIVE_INFERENCE', 'camera_depth_scale': camera['depth_scale']})
    assert len(candidate_rows) == 15 and saved['no_GT_best_candidate_selection'] is True and saved['nominal_projection_executed'] is False
    summary = {'files': len(RAW), 'selected_values': 36, 'opaque_skipped_values': 519, 'unselected_value_decodes': 0, 'frames': 12, 'candidate_instances': 15, 'instance_UNKNOWN_frames': 12, 'native_rows': 96, 'inputU': 11, 'SOURCE_FIT': 64, 'SOURCE_DEV': 32, 'camera_K_and_positive_depth_scale_valid_frames': 12, 'candidate_GT_info_and_mask_indices_exact': 15, 'visib_fract_range': [min(r['visib_fract'] for r in candidate_rows), max(r['visib_fract'] for r in candidate_rows)], 'visible_count_exceeds_valid_depth_count': sum(r['visible_count_exceeds_valid_depth_count'] for r in candidate_rows), 'bbox_obj_negative_origin_count': sum(r['bbox_obj_negative_origin'] for r in candidate_rows), 'sentinel_bbox_candidate_count': sum(r['bbox_metadata_state'] == 'NO_INFRAME_SUPPORT_UNKNOWN' for r in candidate_rows)}
    assert root_QA['selected_frame_values_parsed'] == summary['selected_values'] and root_QA['unselected_frame_values_opaque_skipped'] == summary['opaque_skipped_values']
    result = write('INDEPENDENT_RECOMPUTATION.json', {'completed': True, 'PASS': True, 'summary': summary, 'per_file': statistics, 'candidate_metadata_only': candidate_rows, 'raw_JSON_bindings_reverified': raw_reads, 'unselected_values_interpreted': 0})
    review = write('REVIEW.json', {'completed': True, 'PASS': True, 'actual_SOURCE_selected_metadata_QA': True, 'source_complete': protocol['source_complete'], 'entry': complete['entry'], 'scope': protocol['scope'], 'own_protocol': protocol_b, 'code': {'path': __file__, 'sha256': hashlib.sha256(read(__file__)).hexdigest()}, 'independent_recomputation': result, 'summary': summary, 'checks': ['Whole opaque SHA/size of exact12 files against first8/known4 and producer opaque receipt', 'Independent selective parser: 36 selected TRAIN values; 519 other values lexically skipped without semantic decoding', 'All15 obj1 candidate GT/info/mask-index and camera records exact against stored selection', 'Original frozen96 input metadata binding, all8 order/U11/SOURCE_FIT64/DEV32 preserved'], 'limits_KO': ['원 scene JSON 12개를 독립 재해시·선택 파싱했다. GT 값의 의미 접근은 고정 TRAIN 12 key에만 제한했다.', 'camera K/scale의 형식·유한값·양수 여부와 원값 복사 정합을 검사했다. 실제 카메라 보정 정확도나 depth 등록 정확도를 인증하지 않는다.', 'bbox 음수 원점은 프레임 경계를 넘는 범위의 일부 증거다. 실제 이미지 크기/마스크/depth를 읽지 않아 전체 in-frame 및 픽셀 가시성은 판정하지 않았다.', 'global visible fraction, visible와valid-depth pixel count의 차이는 native D/E의 근거로 자동 승격하지 않는다.', '15개 GT 후보 중 instance 선택은 여전히 UNKNOWN이다. nominal point projection, 새 native label, original q/L 변경 또는 학습 채택은 없다.'], 'image_mask_depth_CAD_array_model_or_protected_payload_reads': 0, 'projection_PnP_labels_fit_Git_network': 0, 'training_admission': False, 'metadata_read_bindings': metadata_reads, 'own_folder_frozen_no_further_edits': True})
    with open(O + '/KO.md', 'x') as fh:
        fh.write('# V217 실제 selected GT 독립 검산\n\nPASS. 승인된 12 scene JSON의 원 SHA/size와 선택 TRAIN 값 36개를 독립 검산했다. 다른 프레임 값 519개는 구조 경계만 스캔하고 해석하지 않았다. 원 96행/U11/FIT64·DEV32, 15개 obj1 후보 및 12개 instance UNKNOWN을 보존했다.\n\nGT/info와 mask 파일명 index 및 camera K/양의 depth_scale은 저장 결과와 일치한다. 실제 mask·depth·CAD·RGB는 열지 않았다. 전역 visible fraction이나 프레임 밖 bbox, valid-depth count 차이를 native E로 변환하지 않는다. instance/part와 픽셀 증거를 결합하는 다음 단계가 별도로 필요하며 새 라벨·학습 채택은 없다.\n')
    print(json.dumps({'completed': True, 'PASS': True, 'REVIEW': review, 'summary': summary}, ensure_ascii=False))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        write('FAILURE.json', {'completed': False, 'PASS': False, 'error_type': type(e).__name__, 'message': str(e), 'traceback': traceback.format_exc(), 'raw_files_already_read': raw_reads})
        raise
