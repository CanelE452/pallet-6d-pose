"""Render only completed, locked screen results; no evaluation or training changes."""
import screen as S


def fmt(value, digits=3):
    return '미정의' if value is None else f'{value:.{digits}f}'


def main():
    S.verify_lock()
    result = S.read(S.DOC / 'RESULTS.json')
    close = S.read(S.DOC / 'CLOSEOUT.json')
    audit = S.read(S.DOC / 'TRAINING_AUDIT.json')
    assert close['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    meanings = {
        'SIGNAL_FOR_CONFIRMATION': '동일한 짧은 조건에서 medium의 개선 신호를 확인했다. 독립 검증이나 전체 수렴 결과는 아니며, 본실험으로 넘어갈 근거다.',
        'NO_SIGNAL_WITHIN_1000_STEPS': '1,000step 예산에서 medium으로 넘어갈 사전 조건을 충족하지 못했다. 모델 용량 가설의 최종 기각이나 모든 머신러닝 방법의 실패를 뜻하지 않는다.',
        'INCONCLUSIVE_INSUFFICIENT_COVERAGE': '공통 검출 표본이 사전 최소치에 못 미쳐 판정 불가다. 짧은 초기 학습의 결과이며 용량 가설을 기각하지 않는다.',
    }
    lines = ['# YOLO26n/m 빠른 용량 비교 — 완료', '',
             f"판정: `{result['verdict']}`.", '', meanings[result['verdict']], '',
             '두 모델 모두 공식 COCO pose 사전학습에서 시작해 동일한 합성4,000장으로',
             '각각1,000회의 실제 optimizer update를 수행했다. 기존 팔레트 R0와 초기',
             'medium을 비교한 것이 아니다. seed는 각각1개다. 본학습 외에 폐기한',
             '메모리/배선 시험은 각5step(총10회)이며 추가 재학습은 없다.', '',
             '## 같은 실제 평가셋 결과', '',
             'DEV319 양성 +2,689 음성. 키포인트 통계는 각 시점 n/m의 공통 IoU50',
             '검출 프레임에서만 계산했다. gross20은 감독 대상 키포인트 중20px 초과',
             '비율이다. 시점마다 공통 집합이 다를 수 있어 시점 간 직접 차감은 피한다.',
             '포즈는 각 모델의 유효 MAIN 산출물 기준이며, 전체319장 대비 산출 수를 병기한다.', '',
             '| step | 모델 | AP50-95 | 검출/319 | 공통 kp median/P90 px | gross20 | 위치 cm | yaw deg | MAIN 산출/319 |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for step in S.MILESTONES:
        pair = result['common_geometry'][str(step)]
        for arm in ('n', 'm'):
            r = result['results'][str(step)][arm]
            p = r['pose']['ALL']
            g = pair[arm]
            lines.append(f"| {step} | {arm} | {fmt(r['metrics']['box_ap50_95'], 4)} | {round(r['Det'] * 319)} | "
                         f"{fmt(g['median_px'])}/{fmt(g['p90_px'])} | {fmt(g['gross20'])} | "
                         f"{fmt(p.get('translation_median_cm'))} | {fmt(p.get('yaw_median_deg'))} | {p['n']} |")
    lines += ['', '시점별 공통 검출 수: ' + ', '.join(
        f"{step}step={result['common_geometry'][str(step)]['frames']}장" for step in S.MILESTONES) + '.', '',
        '## 고정 gate', '', *[f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in result['checks'].items()], '',
        '250/500 결과를 보고 checkpoint를 골라내지 않았다. 최종1,000step 및 사전에',
        '정한500step 방향 확인 조건만 판정에 사용했다. R0 모델 교체는 하지 않았다.', '',
        '## 실행 검증과 GPU', '']
    for arm in ('n', 'm'):
        a = audit['arms'][arm]
        peak_t = max(float(t['gpu'].split(',')[1]) for t in a['telemetry'])
        lines.append(f"- {arm}: {a['init']['parameters']:,} parameters, {a['optimizer_updates']} updates, "
                     f"학습 {a['elapsed_seconds']:.1f}초, peak tensor allocation {a['peak_allocated_MiB']:.0f}MiB, "
                     f"샘플링된 최고 GPU 온도 {peak_t:.0f}°C.")
    lines += ['- 1,000개 전체 step의 실제 증강 이미지·라벨 텐서와 학습률 일치 PASS.',
              '- 두 모델의 초기 상태를 사전 시험 해시로 재검증. 6개 체크포인트의 업데이트 수·파라미터 수·유한성·해시 검증 PASS.',
              '- 선택한 원본4,000장의 이미지와 라벨 해시 불변 PASS.',
              '- 재부팅·드라이버·전원 변경·외부 작업 종료 없음. GitHub 업로드 없음.', '',
              '## 해석의 한계', '',
              '이 검사는 원래 R0의55,980장/60epoch 재현이 아니다. 균등 부분집합,',
              '배치4/FP32/짧은 스케줄, seed1 조건의 비교다. 양쪽은 같은 조건이나',
              '공식 COCO 사전학습 연산량까지 맞춘 순수 파라미터 수 실험은 아니다.',
              '전체 YOLO scale이 바뀌므로 backbone-only 효과라고 부르지 않는다.', '',
              '평가 데이터는 반복 검토한 역사적 DEV이며, 포즈 GT는 기하적으로 재구성된',
              '참조다. 짧은 학습에서 신호가 없다고 최종 수렴 성능이나 용량 가설을',
              '기각하지 않는다. 반대로 신호가 있어도 신규성·독립 일반화·논문 우위를',
              '입증한 것은 아니다. 양쪽 동일한 optional Albumentations 미적용과',
              'CuDNN fallback 경고는 PROTOCOL_KO.md에 기록했다.', '',
              '첫 평가 시 누락된 최상위 kpt_shape 메타데이터는 평가 로더에서만',
              '보완했다. 학습·저장 가중치·고정 gate는 변경하지 않았으며, 원래 오류',
              '로그와 정정 기록은 EVALUATION_CORRECTION_KO.md에 보존했다.', '',
              '## 근거 파일', '',
              '- [고정 조건](PROTOCOL_KO.md), [기계 판독 lock](PROTOCOL_LOCK.json)',
              '- [결과 및 gate](RESULTS.json), [학습 감사](TRAINING_AUDIT.json)',
              '- [실행 종료](CLOSEOUT.json)', '']
    payload = '\n'.join(lines)
    path = S.DOC / 'REPORT_KO.md'
    if path.exists():
        assert path.read_text() == payload
    else:
        with path.open('x') as f:
            f.write(payload)
    print(path)


if __name__ == '__main__':
    main()
