"""Saved SOURCE TRAIN witness report. No training, evaluation or original arrays."""
from pathlib import Path
import hashlib
import html
import json
import os
import statistics

V = Path(__file__).absolute().parent
O = V / 'joint_witness_display_v1'


def safe(p):
    p = Path(p).absolute()
    assert V in p.parents and '..' not in p.parts and 'v112' not in str(p).lower()
    assert all(not x.is_symlink() for x in (p, *p.parents))
    return p


def bind(p):
    p = safe(p)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def read(p):
    return json.loads(safe(p).read_text())


def put(p, d):
    p = safe(p)
    assert not p.exists()
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def text(p, s):
    p = safe(p)
    assert not p.exists()
    p.write_text(s)


def main():
    complete = read(O / 'COMPLETE.json')
    joined = read(O / 'JOINED_WITNESS8.json')
    manifest = read(O / 'FIXED8_DISPLAY_MANIFEST.json')
    qa = read(V / 'ROOT_ACTUAL_FIXED8_WITNESS_VISUAL_QA.json')
    assert complete['PASS'] and joined['PASS'] and qa['PASS']
    rows = joined['rows']
    assert len(rows) == len(manifest['cards']) == len(qa['rows']) == 8
    values = [r['IoU'] for r in rows if r['IoU'] is not None]
    metrics = {'categories': complete['categories'], 'IoU_min': min(values),
               'IoU_median': statistics.median(values), 'IoU_max': max(values),
               'same_local_feature_complex_weak_support': sum(r['weak_corresponding_local_visible_feature_complex_supported'] for r in qa['rows']),
               'all_A_C_ROI_pixels_valid': all(all(v == 1.0 for v in r['ROI_valid_pixel_fractions'].values()) for r in qa['rows'])}
    lines = ['# V206 원 query를 유지한 공개 TRAIN 보임 부위 근거 8건', '',
             '원 crop에서 무엇을 보고 보임으로 판단했는지를 2D 부위 영역으로 남겼다. 두 assistant가 원 8개 crop을 독립적으로 확인해 각각 8개의 약한 영역을 표시했고, 8쌍 모두 일부 겹쳤다. Root가 고정 8카드와 원 q8 전체 맥락 3장을 실제로 확인했다.', '',
             '**이 결과는 가림 판별 성능 개선이나 새로운 정답 위치가 아니다.** 영역은 눈에 보이는 코너·끝면·지지블록 주변의 coarse 근거다. 물리 native vertex, 전체 부위 segmentation, 인간 gold, 새로운 외부 가림 E를 인증하지 않는다. 새 학습과 학습 채택은 없다.', '',
             '## 데이터와 관찰', '',
             'V205 SOURCE TRAIN에서 같은 D와 같은 ON/NEAR로 판단한 8건을 전부 고정했다. 원본 3프레임의 8개 native이며, 8프레임이나 독립 촬영 8회가 아니다. 7건의 기존 위치 가능성 표시 불일치도 유지했다. V205의 사전 엄격 결합 D1/E0 결과는 수정하지 않았다.', '',
             '원 q/L/native/U, raw96 RGB·validity, 입력 순서는 그대로다. ROI는 원96의 정수 inclusive [x0,y0,x1,y1], 0..95 좌표이며, query를 고치거나 ROI 중심으로 crop을 다시 만들지 않았다. q를 영역 안에 넣어야 한다는 조건, 최소 영역 크기, 완벽한 mesh/좌표 gold 조건은 두지 않았다. UNKNOWN/null도 허용한 상태에서 두 관찰 모두 이번 8건에 YES를 기록했다.', '',
             'A는 이번 단계에서 CLEAN96 8개와 원 q8 표시 3개, 고유 PNG 11개를 실제 확인했다. C는 허용한 CLEAN96·VALID96 각 8개와 3프레임의 fullRGB·q8·contact 각 3개, 고유 PNG 25개를 실제 확인했다. 두 관찰은 각자 봉인하기 전 상대의 새 ROI 값을 읽지 않았다. 이전 SOURCE 관찰 맥락은 알려져 있으므로 blinded 인간 연구로 해석하지 않는다.', '',
             '## 결과', '',
             f'- 두 YES 영역 겹침: {complete["categories"].get("BOTH_YES_OVERLAP", 0)}/8.',
             f'- IoU 최솟값/중앙값/최댓값: {metrics["IoU_min"]:.6f} / {metrics["IoU_median"]:.6f} / {metrics["IoU_max"]:.6f}. 영역 범위 차이의 기술 통계이며 채택 문턱이나 모델 정확도가 아니다.',
             '- A/C 원 영역은 모두 별도로 보존했다. 둘을 둘러싼 bbox도 기록했으나 정확한 union segmentation mask로 부르지 않는다.',
             '- 표시한 모든 A/C 영역의 픽셀은 원 validity 안에 있었다. 영상 하단이 잘린 사례는 유효한 상부 노출 조각만 근거로 남겼다.', '',
             '| 원 row ID | A 영역 | C 영역 | IoU | 실제 시각 검토 |', '|---|---|---|---:|---|']
    for r, q in zip(rows, qa['rows']):
        lines.append('| ' + ' | '.join([r['row_id'], str(r['ROI_A']), str(r['ROI_C']), f'{r["IoU"]:.6f}', q['semantic_evidence'].replace('|', '/')]) + ' |')
    lines += ['', '## 무엇이 해결됐고 무엇이 남았나', '',
              'query 중심이 나무 표면에 놓이지 않아도 원 crop 안의 의도 부위가 보이는 사례에서, 가시성 근거와 좌표 품질을 별도로 기록할 수 있게 됐다. A/C의 표시 범위는 다르지만 원 전체 맥락과 함께 보면 같은 국소 코너/끝/블록 복합체를 지지하는 약한 근거가 있었다. 정확히 같은 3D 점을 표시했다는 주장은 하지 않는다.', '',
              '이제 다음 개입에서 어떤 가시 근거를 실제로 덮고, 어떤 대조 조건에서 그대로 남겨야 하는지를 확인할 자료가 있다. 단순히 query 중심이나 별도의 finite-end를 덮고 native E로 복사하는 일을 피하기 위한 준비다. 외부 물체를 붙였다는 사실만으로 새 E를 만들지 않는다.', '',
              '아직 이 영역을 이용한 외부 가림·동일 물체 대조쌍은 만들지 않았고 모델도 학습하지 않았다. 이 3프레임만으로 다양한 목재/정사각형 팔레트, 자연 외부 가림의 범위를 충분히 확보한 것은 아니다. 현재223·publicheld·SOURCE HOLDOUT·COMMON·기존 VAL은 사용하지 않았고, PnP·새 initializer·원 query 보정도 없다. 최신 동결 실사 모델 비교는 V203 그대로다.', '',
              '## 직접 확인', '',
              '[고정 8카드와 두 원 영역](joint_witness_display_v1/PORTABLE_GALLERY_KO.md) · [원 영역 결합표](joint_witness_display_v1/JOINED_WITNESS8.csv) · [고정 8건 실제 시각 검토](ROOT_ACTUAL_FIXED8_WITNESS_VISUAL_QA.json)', '',
              '원 공개 이미지: [MR6D](https://huggingface.co/datasets/anas-gouda/mr6d), Anas Gouda 등, pinned revision `3c4ed8453312655499c3d3a3cf38b4929fae77ee`, 배포 CC-BY-4.0. 새 다운로드는 없다. 원 fullRGB·crop은 앞선 V205 게시 자료와 동일하다.', '']
    report = '\n'.join(lines)
    text(V / 'PORTABLE_REPORT_KO.md', report)
    text(V / 'ROOT_REPORT_KO.md', report)
    body = ['<!doctype html><meta charset="utf-8"><style>body{max-width:1500px;margin:24px auto;font:16px sans-serif}img{max-width:100%;height:auto}pre{white-space:pre-wrap}</style>',
            '<h1>V206 원 query를 유지한 보임 부위 근거 8건</h1>',
            '<p>두 독립 assistant의 원 영역을 보존했습니다. Root가 8카드와 3개 q8 맥락을 실제 확인했습니다. 영역은 약한 시각 근거이며 교정 keypoint·물리 gold·새 E 정답·학습 채택이 아닙니다.</p>',
            '<p><a href="PORTABLE_REPORT_KO.md">보고서·수치·한계</a> · <a href="joint_witness_display_v1/JOINED_WITNESS8.csv">결합표</a></p>']
    for c, r, q in zip(manifest['cards'], rows, qa['rows']):
        rel = Path(c['images']['card']['path']).relative_to(V).as_posix()
        body += ['<h2>' + html.escape(r['row_id']) + '</h2>', '<p>' + html.escape(q['semantic_evidence']) + '</p>', '<img src="' + html.escape(rel) + '">']
    body += ['<h2>원 전체 q8 맥락</h2>']
    for b in qa['actual_toolview_original_q8_contexts']:
        # Reference already published V205 images, never follow/copy their bindings here.
        rel = os.path.relpath(b['path'], V).replace(os.sep, '/')
        assert rel.startswith('../query_visibility_public_train_native_visibility_annotation_pilot_20261025_v205/')
        body += ['<img src="' + html.escape(rel) + '">']
    text(V / 'ROOT_READOUT_KO.html', '\n'.join(body))
    result = {'completed': True, 'PASS': True, 'phase': 'V206', 'fixed_source_cases': 8,
              'original_source_frames': 3, 'metrics': metrics, 'joined': bind(O / 'JOINED_WITNESS8.json'),
              'joint_complete': bind(O / 'COMPLETE.json'), 'actual_root_visual_QA': bind(V / 'ROOT_ACTUAL_FIXED8_WITNESS_VISUAL_QA.json'),
              'observations_A': joined['observer_A_original'], 'observations_C': joined['observer_C_original'],
              'new_external_E_model_fit_forward_admission_or_query_change': False,
              'physical_native_gold_or_model_accuracy_claim': False, 'goal_solved': False}
    put(V / 'ROOT_RESULTS_KO.json', result)
    put(V / 'ROOT_REPORT_BUILD_COMPLETE.json', {'completed': True, 'PASS': True, 'code': bind(__file__),
        'outputs': [bind(V / x) for x in ('PORTABLE_REPORT_KO.md', 'ROOT_REPORT_KO.md', 'ROOT_READOUT_KO.html', 'ROOT_RESULTS_KO.json')],
        'root_actual_fixed8_visual_review': bind(V / 'ROOT_ACTUAL_FIXED8_WITNESS_VISUAL_QA.json'),
        'actual_HTML_browser_render': False, 'new_fit_or_protected_data_access': False})
    print(json.dumps({'completed': True, 'PASS': True, 'metrics': metrics,
                      'report_complete': bind(V / 'ROOT_REPORT_BUILD_COMPLETE.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
