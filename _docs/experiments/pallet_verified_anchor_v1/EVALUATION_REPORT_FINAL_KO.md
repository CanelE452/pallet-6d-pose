# verified anchor — 2점 QA 이후 최종 평가

실제 사람 결정 2/2를 반영한 별도 V2 reference입니다. 기존 잠정 결과는 보존했습니다.
P4는 외부 가림→불확실로 바뀌어 계속 제외, P2는 직접 보임과 좌표를 유지했습니다.
선정 18장 / 저장 16장 / 최종 직접 보임 66점. 기존 대비 점수 변화 없음: **True**.

| 모델 | PCK5 | PCK10 | PCK20 | median px | P90 px | >20 |
|---|---:|---:|---:|---:|---:|---:|
|R0|21/66|44/66|60/66|7.1439|18.3897|6|
|OLD_S1|22/66|41/66|61/66|7.0331|17.4525|5|
|T0|27/66|44/66|61/66|7.0864|17.8804|5|
|T1|26/66|45/66|62/66|6.6149|19.3290|4|
|T2|25/66|43/66|61/66|6.3946|19.7548|5|
|TEACHER (보조)|30/66|54/66|63/66|5.2024|13.4087|3|

교사 coverage: {'available': 16, 'total': 16} frames, {'available': 66, 'total': 66} points.
교사는 winner 선정에서 제외했습니다. 고정 ID, symmetry-min 없음, 결측은 원영상 대각선 벌점입니다.
새 visible reference와 legacy 거리: median 2.8284px, P90 5.3742px, >20px 0점.

## 난도별 PCK10 / median px

| 난도 | R0 | OLD_S1 | T0 | T1 | T2 |
|---|---|---|---|---|---|
|CLEAN|22/30; 5.648|22/30; 5.645|23/30; 5.073|23/30; 5.369|21/30; 5.803|
|MODERATE_OCCLUSION|15/22; 7.673|11/22; 9.114|12/22; 9.062|12/22; 8.137|13/22; 8.012|
|SEVERE_OCCLUSION|7/14; 12.459|8/14; 7.533|9/14; 6.493|10/14; 5.509|9/14; 6.617|

## 해석과 한계

기존 다섯 모델 결론이 유지됩니다. T1의 PCK10 우위는 R0/T0 대비 1점이고 T2는 median이 가장 낮아, 전반적 우승 모델은 선언하지 않습니다.
수량 충족은 전체 144개 상태 완료를 의미하지 않습니다. 미분류/미입력은 계속 제외했고 추가 annotation은 요구하지 않습니다.
PnP 보조 first pass, 재사용 DEV, 관측 가능한 점의 선택 편향, P6 부재, 적은 recording 수의 한계를 유지합니다. 독립 6D GT가 아닙니다.

![QA 전후와 최종 평가](../pallet_011067_corner_contract_v1/figures/01_verified_anchor_qa_before_after.png)

## 근거

- [사람 QA 결과/보존 해시](METADATA_QA_FINAL.json)
- [난도·recording·corner별 결과](VERIFIED_RESULTS_FINAL.json)
- [frame win/loss/tie·leave-one-recording-out·잠정→최종 변화](MODEL_COMPARISON_FINAL.json)
- [교사 보조 비교](TEACHER_SUPPLEMENT_FINAL.json)
- [reference 비교](REFERENCE_DISAGREEMENT_FINAL.json)
- [최종 결정](FINAL_DECISION_V2.json)
- [다음 011067 contract 감사](../pallet_011067_corner_contract_v1/REPORT_KO.md)

`python -m scripts.research.pallet_verified_anchor_v1.finalize_metadata_qa` — 새 학습/추론 없음.
