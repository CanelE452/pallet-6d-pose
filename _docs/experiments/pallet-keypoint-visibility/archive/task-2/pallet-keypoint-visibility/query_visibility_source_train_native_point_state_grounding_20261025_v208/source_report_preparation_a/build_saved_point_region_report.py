#!/usr/bin/env python3
"""V208 sealed saved-output report only. Never follows original asset bindings."""
import argparse
import collections
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import stat

PHASE = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_source_train_native_point_state_grounding_20261025_v208')
PREP = PHASE / 'source_report_preparation_a'
CODE = PREP / 'build_saved_point_region_report.py'
PROTOCOL = PREP / 'PROTOCOL_BEFORE_SAVED_V2_REPORT.json'
COMPLETE = PHASE / 'source_train_native_point_grounding_v2/COMPLETE.json'
COMPLETE_SHA = '5b18efe0a89d31297d5bceff301e674d85ab2179199f506905733c16c650a6c6'
ALLOWED = {
    COMPLETE,
    PHASE / 'source_train_native_point_grounding_v2/ALL_ORIGINAL_BEFORE_NATIVE_POINT_STATE_AND_LOCALITY.jsonl',
    PHASE / 'source_train_native_point_grounding_v2/ALL_CONSTRUCTED_TRAIN_CONTROL_ROWS_TO_BEFORE_PARENT.jsonl',
    PHASE / 'source_train_native_point_grounding_v2/SUMMARY.json',
    PHASE / 'source_train_native_point_grounding_v1/COMPLETE.json',
    PHASE / 'source_train_native_point_grounding_v1/SUMMARY.json',
    PHASE / 'saved_schema_reader_v2_c/PROTOCOL_BEFORE_SAME_TRAIN_GT_RECORDS_CF_V2.json',
    CODE, PROTOCOL, PREP / 'ACTUAL_REPORT_ENTRY.json',
}
STATES = ['D', 'E', 'S', 'U', 'UNKNOWN']


def pathcheck(p):
    p = Path(p)
    if 'v112' in str(p).lower() or not p.is_absolute() or '..' in p.parts:
        raise ValueError('Forbidden or noncanonical path')
    return p


def opened(p):
    p = pathcheck(p)
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        parts = p.parts[1:]
        for i, part in enumerate(parts):
            nf = os.open(part, os.O_RDONLY | os.O_NOFOLLOW | (os.O_DIRECTORY if i + 1 < len(parts) else 0), dir_fd=fd)
            os.close(fd)
            fd = nf
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('Not a regular file')
        with os.fdopen(fd, 'rb') as f:
            fd = -1
            return f.read()
    finally:
        if fd >= 0:
            os.close(fd)


def read(p):
    p = pathcheck(p)
    if p not in ALLOWED:
        raise ValueError('Outside report read whitelist')
    return opened(p)


def binding(p):
    return {'path': str(p), 'sha256': hashlib.sha256(read(p)).hexdigest()}


def checked(b):
    raw = read(Path(b['path']))
    assert hashlib.sha256(raw).hexdigest() == b['sha256'], 'Saved input SHA mismatch'
    return raw


def jread(p):
    return json.loads(read(p))


def write(p, data):
    p = pathcheck(p)
    data = data.encode('utf8') if isinstance(data, str) else data
    # All parents are fixed existing phase/preparation directories, checked without following links.
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in p.parts[1:-1]:
            nf = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nf
        wf = os.open(p.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=fd)
        with os.fdopen(wf, 'wb') as f:
            f.write(data)
    finally:
        os.close(fd)
    return {'path': str(p), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def jd(v):
    return json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def counts(rows, key):
    c = collections.Counter(r[key] for r in rows)
    assert set(c) <= set(STATES)
    return {s: c[s] for s in STATES}


def table(headers, rows):
    def cell(v):
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(cell(v) for v in r) + ' |' for r in rows])


def html_table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in r) + '</tr>' for r in rows) + '</tbody></table></div>'


def local_links(text, kind):
    if kind == 'md':
        return re.findall(r'(?<!!)\[[^\]]*\]\(([^\s)]+)\)', text)
    return re.findall(r'(?:href|src)=["\']([^"\']+)', text)


def validate_links(text, kind):
    links = local_links(text, kind)
    for link in links:
        assert not link.startswith(('/', 'file:', 'http:', 'https:')), 'Portable links must be local relative'
        assert 'v112' not in link.lower()
        target = PHASE / link.split('#', 1)[0]
        assert '..' not in Path(link).parts
        opened(target)
    return len(links)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry', type=Path, required=True)
    args = ap.parse_args()
    assert args.entry == PREP / 'ACTUAL_REPORT_ENTRY.json'
    e = jread(args.entry)
    assert e['approved'] is True and e['saved_v2_report_only'] is True
    assert e['code'] == binding(CODE) and e['protocol'] == binding(PROTOCOL)
    assert e['complete'] == {'path': str(COMPLETE), 'sha256': COMPLETE_SHA}
    c = json.loads(checked(e['complete']))
    assert c['completed'] and c['PASS'] and c['point_state_rows'] == 328
    assert c['physical_TRAIN_control_rows_mapped'] == 1968 and c['GT_json_files_read'] == 41
    assert c['new_point_or_region_labels_created'] is False and c['training_admission'] is False
    b = c['outputs']
    rows = [json.loads(x) for x in checked(b['ALL_ORIGINAL_BEFORE_NATIVE_POINT_STATE_AND_LOCALITY.jsonl']).splitlines()]
    mapping = [json.loads(x) for x in checked(b['ALL_CONSTRUCTED_TRAIN_CONTROL_ROWS_TO_BEFORE_PARENT.jsonl']).splitlines()]
    saved_summary = json.loads(checked(b['SUMMARY.json']))
    assert len(rows) == 328 and len(mapping) == 1968
    assert len({(r['base_index'], r['native_slot']) for r in rows}) == 328
    assert len({r['base_index'] for r in rows}) == 41
    for base in {r['base_index'] for r in rows}:
        assert {r['native_slot'] for r in rows if r['base_index'] == base} == set(range(8))
    assert all(r['role'] == 'TRAIN' and r['source_variant'] == 'ORIGINAL_BEFORE_ONLY' for r in rows)
    assert all(r['training_admission'] is False and r['new_visibility_target_created'] is False for r in rows)
    assert all(r['role'] == 'TRAIN' and r['saved_native_states_apply_to'] == 'ORIGINAL_BEFORE_PARENT_ONLY' for r in mapping)
    assert all(r['changed_V189_COVER_SHAM_point_states'] == 'UNKNOWN_NOT_DERIVED_OR_ALIASED_FROM_FINITE_END' for r in mapping)
    assert len({r['attempt_id'] for r in mapping}) == 82
    assert collections.Counter(r['variant'] for r in mapping) == {'BEFORE': 656, 'COVER': 656, 'SHAM': 656}
    lookup = {(r['base_index'], r['native_slot']): r for r in rows}
    for m in mapping:
        r = lookup[tuple(m['parent_lookup_key'])]
        for key in ['native_slot', 'original_global_index', 'original_occurrence_id', 'query_xy', 'L', 'original_query_available', 'original_input_flags']:
            assert m[key] == r[key], 'Parent original identity drift'
    point = counts(rows, 'point0_combined_state')
    target = counts(rows, 'point0_target_alone_state')
    region = counts(rows, 'region_state')
    target_region = counts(rows, 'target_alone_region_state')
    assert point == target == {'D': 287, 'E': 0, 'S': 41, 'U': 0, 'UNKNOWN': 0}
    assert region == target_region == {'D': 164, 'E': 0, 'S': 41, 'U': 123, 'UNKNOWN': 0}
    assert {s: point[s] for s in saved_summary['point0_combined_states']} == saved_summary['point0_combined_states']
    assert {s: region[s] for s in saved_summary['region_states']} == saved_summary['region_states']
    cross = {p: {q: sum(r['point0_combined_state'] == p and r['region_state'] == q for r in rows) for q in STATES} for p in STATES}
    mismatch = cross['D']['U']
    assert mismatch == 123
    native = []
    for k in range(8):
        rr = [r for r in rows if r['native_slot'] == k]
        native.append({'native_slot': k, 'N': len(rr), 'point': counts(rr, 'point0_combined_state'), 'region': counts(rr, 'region_state'), 'input_U': sum(not r['original_query_available'] for r in rr), 'inside_sample_center_extent': sum(r['original_crop_reference_locality']['inside_sample_center_extent'] is True for r in rr), 'inside_canvas_extent': sum(r['original_crop_reference_locality']['inside_canvas_extent'] is True for r in rr), 'point_D_region_U': sum(r['point0_combined_state'] == 'D' and r['region_state'] == 'U' for r in rr)})
    locality = {k: sum(r['original_crop_reference_locality'][k] is True for r in rows) for k in ['inside_sample_center_extent', 'inside_canvas_extent']}
    locality['input_U'] = sum(not r['original_query_available'] for r in rows)
    locality['saved_reference_present'] = sum(r['saved_anchor_projected_reference_xy'] is not None for r in rows)
    assert locality == {'inside_sample_center_extent': 328, 'inside_canvas_extent': 328, 'input_U': 0, 'saved_reference_present': 328}
    v1_complete = PHASE / 'source_train_native_point_grounding_v1/COMPLETE.json'
    v1 = jread(v1_complete)
    v1_summary = json.loads(checked(v1['outputs']['SUMMARY.json']))
    assert v1_summary['point0_combined_states'] == {'UNKNOWN': 328}
    assert v1_summary['region_states'] == {'UNKNOWN': 328}
    schema_reasons = dict(collections.Counter(x['UNKNOWN_schema_reason'] for x in v1_summary['schemas']))
    assert schema_reasons == {'UNKNOWN_OR_AMBIGUOUS_SAVED_CF_RECORD_LIST': 41}
    csvbuf = io.StringIO(newline='')
    fields = ['base_index', 'native_slot', 'canonical_index', 'original_global_index', 'original_occurrence_id', 'role', 'original_query_available', 'query_x', 'query_y', 'L', 'point0_target_alone_state', 'point0_combined_state', 'region_state', 'target_alone_region_state', 'reference_x', 'reference_y', 'reference_u96', 'reference_v96', 'inside_sample_center_extent', 'inside_canvas_extent', 'source_variant', 'new_visibility_target_created', 'training_admission']
    cw = csv.DictWriter(csvbuf, fieldnames=fields)
    cw.writeheader()
    for r in sorted(rows, key=lambda z: (z['base_index'], z['native_slot'])):
        t = {k: r[k] for k in fields if k in r}
        t.update(query_x=r['query_xy'][0], query_y=r['query_xy'][1], reference_x=r['saved_anchor_projected_reference_xy'][0], reference_y=r['saved_anchor_projected_reference_xy'][1], reference_u96=r['original_crop_reference_locality']['raw96_reference_index_xy'][0], reference_v96=r['original_crop_reference_locality']['raw96_reference_index_xy'][1], inside_sample_center_extent=r['original_crop_reference_locality']['inside_sample_center_extent'], inside_canvas_extent=r['original_crop_reference_locality']['inside_canvas_extent'])
        cw.writerow(t)
    csv_binding = write(PHASE / 'NATIVE328_POINT_REGION_LOCALITY.csv', csvbuf.getvalue())
    state_headers = ['저장 판정', 'D', 'E', 'S', 'U', 'UNKNOWN', '전체']
    state_rows = [[label] + [ct[s] for s in STATES] + [sum(ct.values())] for label, ct in [('point0 target-alone', target), ('point0 combined', point), ('region', region), ('target-alone region', target_region)]]
    cross_headers = ['point0 combined → region'] + STATES + ['전체']
    cross_rows = [[p] + [cross[p][q] for q in STATES] + [sum(cross[p].values())] for p in STATES]
    native_headers = ['native', 'N', 'point D', 'point S', 'region D', 'region S', 'region U', 'point D·region U', 'input U', 'sample extent', 'canvas']
    native_rows = [[r['native_slot'], r['N'], r['point']['D'], r['point']['S'], r['region']['D'], r['region']['S'], r['region']['U'], r['point_D_region_U'], r['input_U'], r['inside_sample_center_extent'], r['inside_canvas_extent']] for r in native]
    denom_headers = ['단위', '개수', '해석']
    denom_rows = [['현재 TRAIN constructed physical rows', 1968, '82 attempts × 3 variants × 원 native8'], ['constructed TRAIN attempts', 82, 'admitted/기존 weak 상태로 추가 선별하지 않음'], ['중복 제거 original BEFORE 부모', 41, '현재 constructed TRAIN 행이 가리키는 고유 부모'], ['original BEFORE native 조회', 328, '41 parents × native0..7'], ['원 input U', 0, 'finite 여부로 보충하지 않고 원 availability 유지']]
    locality_rows = [['저장 reference 존재', 328, 328], ['sample-center extent [0,95]', locality['inside_sample_center_extent'], 328], ['canvas extent [-0.5,95.5)', locality['inside_canvas_extent'], 328]]
    limits = [
        'point0는 선언된 합성 source 내부 ray/기하 판정이며 실제 공개 영상의 물리 native gold가 아니다. region은 별도로 저장된 판정 단위다.',
        'point D·region U 123개는 서로 다른 판정의 차이다. 실제 annotation 오류 123개, 모델 실패 원인 또는 새 정답으로 해석하지 않는다.',
        '원 BEFORE에는 E가 없다. 새 V189 COVER/SHAM의 native point 상태는 UNKNOWN이며 BEFORE 상태·finite-end D/E/D·center owner를 복사하지 않는다.',
        'raw96 reference index는 47.5 + 96 × (saved reference − original q) / L의 좌표 대수다. q의 GT 오차나 실제 pixel 가시성 판정이 아니다.',
        '모든 reference가 canvas 및 sample-center 범위 안이라는 결과는 원영상·crop에서 부위가 보인다는 증거가 아니다. 실제 RGB/valid·source frame observability를 이번 보고서에서 재확인하지 않았다.',
        'native0..7을 모두 보존했다. native6/7의 이번 point 상태와 과거 finite-end direct target의 UNKNOWN은 서로 다른 필드이며 자동 학습 권한으로 연결하지 않는다.',
        'TRAIN source만 다뤘다. 보호 평가 점수·영상·모델·NPZ를 새로 읽지 않았고 projection/ray/학습/forward를 실행하지 않았다.',
        '이 단계는 저장된 source 정의를 확인한 진단이다. 실사 전이·모델 개선·채택을 입증하지 않는다. 최신 실제 실사 비교 V203은 그대로이며 전체 목표는 미해결이다.'
    ]
    results = {
        'completed': True, 'PASS': True, 'phase': 'V208', 'kind': 'SOURCE_TRAIN_SAVED_POINT_VERSUS_REGION_DIAGNOSTIC',
        'actual_v2_complete': e['complete'], 'builder': binding(CODE), 'report_protocol': binding(PROTOCOL), 'report_entry': binding(args.entry),
        'saved_output_bindings': b, 'denominators': saved_summary['selection_counts'],
        'point0_target_alone': target, 'point0_combined': point, 'region': region, 'target_alone_region': target_region,
        'point_region_cross': cross, 'native_rows': native, 'locality': locality,
        'point_D_region_U': {'count': mismatch, 'all_native_denominator': 328, 'fraction_all_native': mismatch / 328, 'point_D_denominator': point['D'], 'fraction_point_D': mismatch / point['D']},
        'constructed_mapping_variant_counts': dict(collections.Counter(r['variant'] for r in mapping)),
        'original_BEFORE_only': True, 'new_V189_COVER_SHAM_native_point_state': 'UNKNOWN_NOT_DERIVED_OR_ALIASED_FROM_FINITE_END',
        'history_v1': {'complete': binding(v1_complete), 'point_UNKNOWN': 328, 'region_UNKNOWN': 328, 'schema_reason_counts': schema_reasons, 'status': 'Process completed; reader could not identify explicit saved records_CF. Original UNKNOWN outputs remain preserved.'},
        'history_v2_protocol': binding(PHASE / 'saved_schema_reader_v2_c/PROTOCOL_BEFORE_SAME_TRAIN_GT_RECORDS_CF_V2.json'),
        'native328_csv': csv_binding, 'limits_ko': limits,
        'new_GT_binary_image_score_reads': 0, 'new_training_forward_projection_ray_calls': 0, 'training_admission': False,
        'latest_actual_real_comparison': 'V203_UNCHANGED', 'model_improved': False, 'overall_goal_solved': False,
        'independent_saved_GT_QA': 'Parallel independent review by B; not asserted complete by this builder.',
        'root_final_phase_freeze': False
    }
    result_binding = write(PHASE / 'ROOT_RESULTS_KO.json', jd(results))
    history_text = 'v1 실행 자체는 completed/PASS로 종료했지만, reader가 저장된 records_CF 형태를 찾지 못해 point·region 328개가 모두 UNKNOWN이었다. 41개 파일의 사유는 UNKNOWN_OR_AMBIGUOUS_SAVED_CF_RECORD_LIST였다. 원 v1 결과를 보존하고, v2에서 명시적인 records_CF schema adapter로 같은 선택·원 q/L/native/U를 다시 조회했다. 저장값을 고치거나 새 visibility를 계산하지 않았다.'
    md = '# V208 · 저장된 합성 source의 point0와 region 판정\n\n'
    md += '원 BEFORE 부모 41개·native 328개에서 point0 combined는 D 287 / S 41, region은 D 164 / U 123 / S 41이다. point D이면서 region U인 123개는 전체 328개의 **37.50%**, point D 287개의 **42.86%**다. 서로 다른 판정 단위의 저장값을 확인한 결과이며 annotation 오류 수나 모델 원인 증명은 아니다.\n\n'
    for title, headers, vals in [('범위와 분모', denom_headers, denom_rows), ('point0와 region 상태', state_headers, state_rows), ('교차표', cross_headers, cross_rows), ('원 native별 전체 보존', native_headers, native_rows), ('저장 reference의 원 crop 좌표 범위', ['진단', '해당', '분모'], locality_rows)]:
        md += '## ' + title + '\n\n' + table(headers, vals) + '\n\n'
    md += 'D는 저장된 point의 직접 가시, S는 저장된 자기 가림, E는 외부 가림, U는 저장된 region 판정의 미결정이다. UNKNOWN은 필드/schema를 읽지 못하는 상태와 구분해 표시했다. target-alone와 combined point 집계가 같고 region과 target-alone region 집계도 같다. 원 BEFORE만 조회했으므로 E 0을 외부 가림 학습 범위의 증거로 쓰지 않는다.\n\n'
    md += '## 보존된 reader 이력\n\n' + history_text + '\n\n'
    md += '## 해석의 경계\n\n' + '\n'.join('- ' + x for x in limits) + '\n\n'
    links = [
        ('수치 JSON', 'ROOT_RESULTS_KO.json'), ('원328행 CSV', 'NATIVE328_POINT_REGION_LOCALITY.csv'),
        ('v2 실제 완료', 'source_train_native_point_grounding_v2/COMPLETE.json'),
        ('원328행 저장 JSONL', 'source_train_native_point_grounding_v2/ALL_ORIGINAL_BEFORE_NATIVE_POINT_STATE_AND_LOCALITY.jsonl'),
        ('현재 TRAIN 1968행 부모 연결', 'source_train_native_point_grounding_v2/ALL_CONSTRUCTED_TRAIN_CONTROL_ROWS_TO_BEFORE_PARENT.jsonl'),
        ('v2 저장 요약', 'source_train_native_point_grounding_v2/SUMMARY.json'),
        ('보존 v1 UNKNOWN 요약', 'source_train_native_point_grounding_v1/SUMMARY.json'),
        ('v2 schema protocol', 'saved_schema_reader_v2_c/PROTOCOL_BEFORE_SAME_TRAIN_GT_RECORDS_CF_V2.json'),
        ('보고서 생성 코드', 'source_report_preparation_a/build_saved_point_region_report.py'),
        ('보고서 읽기 범위', 'source_report_preparation_a/PROTOCOL_BEFORE_SAVED_V2_REPORT.json')
    ]
    md += '## 원문과 재현\n\n' + '\n'.join('- [' + text + '](' + url + ')' for text, url in links) + '\n\n'
    md += '보고서는 저장된 v2 출력만 재집계했다. 독립 saved-GT QA는 별도 B 검토이며 이 생성기가 완료를 선행 주장하지 않는다. HTML은 오프라인 파일·링크 검사 대상이며 브라우저 렌더 검증을 주장하지 않는다.\n'
    h = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V208 source point·region 진단</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1180px;margin:36px auto;padding:0 20px;color:#142333;background:#f8fafc}h1{font-size:30px}h2{margin-top:34px}.lead{font-size:19px;background:#e8f0ff;padding:18px;border-radius:10px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px 11px;border:1px solid #cdd7e1;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#eaf0f7}a{color:#064fa3}li{margin:8px 0}.note{border-left:4px solid #c18426;padding:12px;background:#fff5df}</style><main>'
    h += '<h1>V208 · 저장된 합성 source의 point0와 region 판정</h1><p class="lead">원 BEFORE 41부모 × native8 = 328행. point D·region U는 <b>123/328 = 37.50%</b>, point D 안에서는 <b>123/287 = 42.86%</b>다. 판정 정의의 차이를 확인했으며 annotation 오류나 모델 원인으로 인증하지 않는다.</p>'
    for title, headers, vals in [('범위와 분모', denom_headers, denom_rows), ('point0와 region 상태', state_headers, state_rows), ('교차표', cross_headers, cross_rows), ('원 native별 전체 보존', native_headers, native_rows), ('저장 reference의 원 crop 좌표 범위', ['진단', '해당', '분모'], locality_rows)]:
        h += '<h2>' + html.escape(title) + '</h2>' + html_table(headers, vals)
    h += '<p>D=저장된 직접 가시, S=저장된 자기 가림, E=외부 가림, U=저장된 판정 미결정. UNKNOWN은 필드/schema 조회 미확정과 구분한다. 원 BEFORE E는 0이며 새 COVER/SHAM native 상태를 대신하지 않는다.</p>'
    h += '<h2>보존된 reader 이력</h2><p class="note">' + html.escape(history_text) + '</p><h2>해석의 경계</h2><ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in limits) + '</ul>'
    h += '<h2>원문과 재현</h2><ul>' + ''.join('<li><a href="' + html.escape(url) + '">' + html.escape(text) + '</a></li>' for text, url in links) + '</ul><p>독립 B saved-GT QA는 별도 진행·봉인한다. 새 학습 0, 원 GT·영상 재열기 0. 브라우저 렌더 검증을 주장하지 않는다.</p></main></html>\n'
    outputs = {'results': result_binding, 'native328_csv': csv_binding}
    outputs['root_md'] = write(PHASE / 'ROOT_REPORT_KO.md', md)
    outputs['portable_md'] = write(PHASE / 'PORTABLE_REPORT_KO.md', md)
    outputs['html'] = write(PHASE / 'ROOT_READOUT_KO.html', h)
    assert outputs['root_md']['sha256'] == outputs['portable_md']['sha256']
    nlinks = {'root_md': validate_links(md, 'md'), 'portable_md': validate_links(md, 'md'), 'html': validate_links(h, 'html')}
    for p in ALLOWED:
        # Rehash only the explicit saved-report input files, not any nested source bindings.
        read(p)
    done = {'completed': True, 'PASS': True, 'code': binding(CODE), 'protocol': binding(PROTOCOL), 'entry': binding(args.entry), 'actual_v2_complete': e['complete'], 'outputs': outputs, 'link_checks': nlinks, 'tables': {'state_rows': 4, 'cross_rows': 5, 'native_rows': 8, 'native_csv_rows': 328, 'mapping_rows_checked': 1968}, 'original_GT_NPZ_RGB_model_score_reads_or_calls': 0, 'browser_verified': False, 'scientific_phase_final_freeze': False}
    db = write(PREP / 'REPORT_BUILD_COMPLETE.json', jd(done))
    print(json.dumps({'PASS': True, 'completion': db, 'outputs': outputs, 'link_checks': nlinks}, ensure_ascii=False))


if __name__ == '__main__':
    main()
