# 다음 분기 — 계획만 작성

[추정] **Primary: ROUTE C — 데이터 다양성/인공–실제 가림 차이 분리. 보조 route는 지정하지 않는다.**

[확인] 한 recording의 인접 253개 이미지를 사용했다. 실제 가림 A11−A10은 음수이고, 합성 stress 복구 향상과 실제 가림 성능은 분리되어 있다. 데이터 다양성이 원인이라고 확정한 것은 아니다.

[추정] 다음 한 실험은 **다중 세션 clean 8/32/128 × CLEAN/OCC refiner-only 통제 비교**다. 총 update/실사 노출, 초기 checkpoint, source replay, optimizer, seed, 가림 recipe를 고정한다. 각 크기의 subset은 사전 고정하고 recording/viewpoint 다양성을 함께 기록한다. 실제 가림 direct contrast와 정상/source 보존을 함께 평가한다.

[확인] 기존 한 세션 결과와 새 결과 차이만으로 다양성의 인과 효과를 증명하지 못한다. 규모별 nested subset에서도 이미지 수와 다양성이 함께 변할 수 있다. 순수 다양성 분리가 필요하면 별도의 matched-size 세션 통제가 필요하며, 이 계획 자체를 검증 완료로 주장하지 않는다.

[추정] 새로운 clean 세션이 충분하지 않으면 준비 단계에서 멈춘다. 같은 영상의 인접 프레임을 복제/추가해 독립 데이터 규모가 늘었다고 해석하지 않는다. 평가 recording 제외 및 고정 pseudo gate는 유지한다.

[확인] ROUTE A는 frame headroom 2.945302pp<3이며 검증된 inference-time 선택 신호가 없다. ROUTE B는 외부/자기 가림 독립 subtype와 primary93 geometry oracle이 부족하다. ROUTE D 역시 geometry로 해결 불가능함이 검증되지 않아 곧바로 구조를 바꿀 근거가 부족하다.

[확인] 이 문서는 계획이다. 새 모델 학습, 추가 fine-tuning, E3~E6, 새 pseudo-label 생성, commit/push는 실행하지 않았다.
