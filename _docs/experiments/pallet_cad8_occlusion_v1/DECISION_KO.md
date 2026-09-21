# 최종 판정

인공 가림은 동일 포즈 전용 CLEAN 대조군보다 소폭 나았지만, 기존 R0 대비 일관된 개선은 확인되지 않았다. 원래 최종 모델은 유지한다. 추가 학습이나 sweep은 수행하지 않았다.

- 주 평가: 같은 occlusion96장,737개 평가 코너. CAD 세션 전체 및 Replay teacher 학습 세션 제외.
- R0: median11.39px / PCK10 41.79% / PCK20 62.82%.
- CLEAN: median11.69px / PCK10 40.43% / PCK20 62.96%.
- OCCLUDED: median11.52px / PCK10 40.98% / PCK20 63.64%.
- 검출 매칭은 세 모델 모두88/96. CLEAN/OCCLUDED의 검출 box/score가 R0와 수치적으로 정확히 일치했다.
- PCK20 기준 OCCLUDED는 CLEAN보다5개, R0보다6개 코너가 순증했다(464→469 /463→469). 작은 변화이며 단일 seed로 유의성을 주장하지 않는다. R0보다 PCK10은6개 코너가 순감했다(308→302).

실사8장 자체를 늘린 것이 아니라2560회 노출 중1930회 입력만 인공적으로 가렸다. pseudo 좌표와 학습 마스크는 유지했다. 두 실험은320회 업데이트,seed42,lr1e-4,합성 replay,입력 순서/기본 augmentation/타깃을 동일하게 유지했고 backbone/검출/모든 BN 버퍼를 동결했다. CLEAN과 OCCLUDED의320개 batch trace가 일치하는 것을 검사했다.

이 결과는 '좋은 깨끗한 수도레이블 + 이번 인공 가림 recipe'의 작은 탐색 결과이다. 인공 컬러 패치와 실제 물체 가림은 다르며,8장 장면의 제한과 수도레이블 오차가 남는다. 다른 학습 설계가 불가능하다는 결론이 아니다. 이 실험만으로 논문의 일반화 개선을 주장하거나 모델을 교체하지 않는다.

재현: PROTOCOL.json / FIT_CLEAN.json / FIT_OCCLUDED.json / AUDIT.json / RESULTS_KO.md.
시각자료: outputs/pallet_cad8_occlusion_v1/index.html (실제 학습 batch의 가림 전후8개 예시 포함).
