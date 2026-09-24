# 지시문 대조 재감사 — 완료 판정 정정

## 현재 상태: WAITING_FOR_HUMAN_QA (2점)

기존 보고서의 QA0은 **좌표 거리·메모 조건만** 검사한 값입니다.
지시문 10절의 기존 가시성 기록과 새 상태 사이의 충돌 검사가 빠졌습니다.
72개 검수점을 재대조하면 두 점이 다릅니다: 신규 DIRECT_VISIBLE ↔ 기존 occluded 1개,
신규 EXTERNAL_OCCLUDED ↔ 기존 visible 1개. 기존 source는 모두 unknown이므로
옛 기록을 정답으로 간주하거나 새 입력을 자동 변경하지 않습니다.

현재 66점 결과·그림은 이전 버전 그대로 보존하며, **이 두 점의 QA가 끝날 때까지 잠정 결과**입니다.
18장 재작업, 추가 사진, 강제 클릭, 학습은 하지 않습니다.
기존 정답·모델 예측을 숨긴 화면에서 현재 입력을 확인하고, 맞으면 K/Enter로 유지합니다.

## 지시문에서 남은 작업

|항목|현재 상태|
|---|---|
|선정/분할/근접중복/원본 hash|완료, 원래18장 유지|
|사람 좌표와 D/V 상태|사용자 요청대로 기존 annotation.py+PnP → 별도 상태 확인|
|모든18×8 상태|미완료, 미입력/비수동 점은 미분류·평가 제외. 완료라고 주장하지 않음|
|엄격한 PnP 없는 blind first pass|사용자 변경으로 충족하지 않음. 사후 재해석 불가|
|minimum visible coverage|66점으로 수량 기준 충족, 추가 사진 요구 안 함|
|좌표 차이/불확실 메모 QA|대상0개|
|기존 visibility metadata 충돌 QA|2점, 사람 확인 대기|
|R0/S1/T0/T1/T2 비교|기존 결과 보존, QA 후 재검증 필요|
|frozen teacher 보조 비교|동일 프레임 TYPE_REPLAY_PIPELINE cache 존재 확인, QA 후 실행|
|실제 클릭 이벤트 수|기존 annotation.py 이벤트 로그 없음. 저장75좌표를 클릭 횟수로 오인하지 않음|
|teacher 포함 최종 요약·지시문 출력|QA 완료 후 추가|
|새 학습/새 추론/기존 GT 변경|모두0|

이 재감사에서는 예전 가시성만 QA 조건으로 사용했고 모델 오차로 QA 점을 고르지 않았습니다.
정확한 frame/좌표/QA queue는 로컬 private 경로에만 보존합니다.

```bash
python -m scripts.research.pallet_verified_anchor_v1.review_metadata_qa
```

두 점을 확인하면 기존 first pass와 변경 이력을 유지한 채 최종 reference를 새 버전으로 잠그고,
원래 모델5종 재검증·teacher 보조 비교·보고서·push를 마무리합니다.
