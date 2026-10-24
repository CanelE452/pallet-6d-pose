"""Report saved fixed source controls; never reads bound original source assets."""
from pathlib import Path
from collections import Counter
import csv, hashlib, html, json

V = Path(__file__).absolute().parent
H = V / 'raw96_weak_witness_control_pilot_v1'

def load(path):
    assert V in path.parents and 'v112' not in str(path).lower()
    return json.loads(path.read_text())

def bind(path):
    assert V in path.parents and 'v112' not in str(path).lower()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def main():
    complete = load(H / 'COMPLETE.json')
    assert complete['completed'] and complete['PASS'] and not complete['fit_admission']
    technical_file = V / 'independent_actual_raw96_control_review_c/REVIEW.json'
    technical = load(technical_file)
    assert technical['PASS'] is True
    visual_file = V / 'ROOT_ACTUAL_FIXED8_RAW96_CONTROL_VISUAL_QA.json'
    visual = load(visual_file)
    assert visual['PASS'] and visual['actual_unique_card_tool_views'] == 8
    rows = [json.loads(x) for x in (H / 'ALL80_ATTEMPTS.jsonl').read_text().splitlines()]
    assert len(rows) == 80 and len({r['attempt_id'] for r in rows}) == 80
    assert all(not r['new_native_E_label'] and not r['training_admission'] for r in rows)
    good = [r for r in rows if r['status'] == 'CONSTRUCTED_RAW96_CONTROL_NO_LABEL_ADMISSION']
    cases = []
    for i, view in enumerate(visual['records']):
        rr = [r for r in rows if r['case_position'] == i]
        assert len(rr) == 10
        cases.append({'position': i, 'row_id': view['row_id'], 'constructed': sum(r in good for r in rr),
                      'attempts': len(rr), 'statuses': dict(Counter(r['status'] for r in rr)),
                      'actual_visual_observation_KO': view['observation_KO']})
    results = {'completed': True, 'PASS': True, 'cases': cases, 'attempts': 80,
               'original_frames': 3, 'fixed_cases': 8, 'candidate_donors': 10,
               'constructed': len(good), 'statuses': dict(Counter(r['status'] for r in rows)),
               'cases_with_pair': sum(c['constructed'] > 0 for c in cases),
               'constructed_by_object_id': dict(Counter(r['object_id'] for r in good)),
               'authority': 'Post-sampler RAW96 2D weak local-feature controls only',
               'original_q_L_native_U_and_sampler_preserved': True,
               'new_native_E_labels_or_training_admission': False,
               'model_improvement_claim': False, 'goal_solved': False,
               'actual_complete': bind(H / 'COMPLETE.json'), 'independent_saved_pixel_QA': bind(technical_file),
               'actual_fixed8_visual_QA': bind(visual_file)}
    write(V / 'ROOT_RESULTS_KO.json', results)
    with (V / 'ROOT_CASE8_CONTROL_SUMMARY.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['position', 'row_id', 'constructed', 'attempts', 'actual_visual_observation_KO'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in cases)
    table = '\n'.join(f"| {c['position']:02d} | {c['row_id']} | {c['constructed']}/10 | {c['actual_visual_observation_KO']} |" for c in cases)
    objects = ', '.join(f"{k}: {n}쌍" for k, n in sorted(results['constructed_by_object_id'].items()))
    text = f'''# V207 공개 TRAIN 보임 부위를 덮는 가림과 남기는 대조 가림

고정한 원 crop 8개와 공개 YCB-V TRAIN 물체 10개로 80시도를 실제 수행했다. {len(good)}개는 두 조건을 모두 만들었고 {80-len(good)}개는 고정 배치 조건을 만족하지 못했다. 성공한 원 사례는 {results['cases_with_pair']}/8이다. 이 수치는 생성 가능 비율이며 가림 판별 정확도가 아니다.

원본은 3프레임이다. 80개 독립 촬영이나 80개 자연 외부 가림 정답이 아니다. 원 q/L/native/U와 기존 RAW96 샘플을 유지한 뒤 2D crop에만 개입했다. 전체 영상 물리 렌더링, 정확한 native E gold, 새 학습 채택은 없다.

## 어떻게 구성했나

V206에서 두 관찰자가 독립 표시한 노출 코너·끝면·블록 주변 영역의 픽셀 합집합을 고정했다. 교집합이나 둘을 둘러싼 bbox로 바꾸지 않았다. COVER는 이 합집합 전부를 불투명 alpha로 덮는다. SHAM은 같은 물체·같은 크기·같은 회전의 alpha와 RGB를 그대로 옮기되 합집합의 원 RGB를 남긴다. 두 조건은 같은 양의 alpha 픽셀 수를 가지며, alpha 밖 원 RGB도 각각 그대로다. 실제 RGB가 바뀐 픽셀 수와 alpha 면적은 구분한다.

물체는 기존 TRAIN 59개 중 object_id별 사전 hash로 10개를 골랐다. 각 시도의 7크기×4회전과 정수 배치는 사전 hash 순서에 따라 첫 기하적 가능 조건을 사용했다. 유효 영역 안에서만 붙이고 alpha 구멍은 채우지 않았다. 불가하면 q·ROI를 옮기거나 줄이지 않았으며 모든 80시도를 보존했다. 표시용 8카드는 사례별 가능한 시도 중 사전 assignment hash가 가장 작은 것을 사용했고, 불가 사례 2개도 빈 대조 패널로 남겼다.

## 실제 확인

| 고정 순서 | 원 native 사례 | 생성된 쌍 | Root 실제 시각 관찰 |
|---|---|---:|---|
{table}

Root가 고정 8카드 모두를 실제로 보았다. 6개에서는 표시한 국소 근거가 COVER에 가려지고 SHAM에서 보이는 것을 확인했다. 나머지 2개는 원 목재가 보이는 factual crop과 생성 불가 상태를 확인했다. 기술 독립 QA는 저장 RGB/alpha와 원 q/L/native/영역·원본 바이트 binding을 별도로 검사했다.

## 무엇이 해결됐고 무엇이 남았나

이전에는 외부 물체를 붙였다는 사실만으로 다른 부위의 가림 라벨을 query에 복사할 수 있었다. 이번에는 실제 보임 판단에 사용한 국소 부위를 덮는지, 같은 물체를 다른 곳에 놓았을 때 그 부위가 남는지 픽셀로 검사할 수 있게 됐다. 이는 개입과 약한 부위 근거의 대응을 만든 진전이며 모델의 실제 인과 의존성이나 성능 향상을 입증하지 않는다.

생성된 18쌍에 쓰인 물체 종류는 4개뿐이다({objects}). 62개 실패에는 crop 하단 padding, 넓은 약한 ROI, 동일 크기 물체를 피해서 놓을 공간 부족이 포함될 수 있으나 각 원인의 기여도는 분해하지 않았다. 이 조건에서 배치가 없다는 뜻이며 일반적인 가림 데이터 생성 불가능을 뜻하지 않는다. 원 부위를 줄여 성공으로 바꾸지 않았다.

조명·그림자·경계와 회전은 자연 영상 수준으로 맞추지 않았다. coarse ROI에는 주변 면·배경이 포함될 수 있고 물리 native 점의 정확한 위치를 인증하지 않는다. 일부 다른 목재가 남더라도 국소 코너가 가려진 경우이므로 팔레트 전체 부재와 동일시하지 않는다. 현재는 D/E 학습 라벨로 채택하지 않았다. 외부 가림 정답, 다양한 목재/정사각형 대상과 촬영 조건의 충분한 범위, 새로운 학습과 동결 비교가 남아 있다. 최신 실제 모델 비교는 V203 그대로다.

현재223·publicheld·SOURCE HOLDOUT·COMMON·기존 VAL은 사용하지 않았다. 새 initializer, 좌표 교정, GT recenter, PnP, 모델 추론·학습은 없다.

[원 crop·가림·대조 가림 8카드](raw96_weak_witness_control_pilot_v1/PORTABLE_GALLERY_KO.md) · [모든 80시도 CSV](raw96_weak_witness_control_pilot_v1/ALL80_STATUS.csv) · [상세 원 결과](raw96_weak_witness_control_pilot_v1/ALL80_ATTEMPTS.jsonl) · [사례 요약 CSV](ROOT_CASE8_CONTROL_SUMMARY.csv)
'''
    for name in ['ROOT_REPORT_KO.md', 'PORTABLE_REPORT_KO.md']:
        with (V / name).open('x', encoding='utf-8') as f:
            f.write(text)
    cards = ''.join(f"<section><h2>{c['position']:02d} {html.escape(c['row_id'])}</h2><img style='max-width:100%' src='raw96_weak_witness_control_pilot_v1/fixed8/CASE_{c['position']:02d}_CONTACT.png'></section>" for c in cases)
    with (V / 'ROOT_READOUT_KO.html').open('x', encoding='utf-8') as f:
        f.write('<!doctype html><meta charset="utf-8"><title>V207 공개 TRAIN 가림 대조</title><style>body{max-width:1100px;margin:auto;padding:24px;font-family:sans-serif;line-height:1.6}pre{white-space:pre-wrap}</style><a href="PORTABLE_REPORT_KO.md">보고서</a> · <a href="raw96_weak_witness_control_pilot_v1/ALL80_STATUS.csv">80시도 표</a>' + cards + '<pre>' + html.escape(text) + '</pre>')
    write(V / 'ROOT_REPORT_BUILD_COMPLETE.json', {'completed': True, 'PASS': True, 'code': bind(Path(__file__)),
          'outputs': {n: bind(V / n) for n in ['ROOT_RESULTS_KO.json', 'ROOT_CASE8_CONTROL_SUMMARY.csv', 'PORTABLE_REPORT_KO.md', 'ROOT_REPORT_KO.md', 'ROOT_READOUT_KO.html']},
          'new_source_asset_or_model_reads': 0, 'new_E_labels_or_training_admission': False})
    print(json.dumps({'completed': True, 'constructed': len(good), 'report': bind(V / 'PORTABLE_REPORT_KO.md')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
