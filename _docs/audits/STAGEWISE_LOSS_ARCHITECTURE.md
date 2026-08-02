# Loss architecture

architecture 변경 0.  belief stage 4~6 만 학습.  centroid channel 은 legacy MSE 만 유지.

```
L_total = L_legacy(6-stage belief MSE + affinity MSE)
        + lambda_mass     * L_mass
        + lambda_rank     * L_rank
        + lambda_distance * L_distance
        + lambda_progress * L_progress
```

- **mass**: P = softmax(H/0.1) full-map, GT 3x3 확률질량의 -log.
  stage weight 0.25/0.50/1.00.
- **rank**: score_gt = 0.1*logsumexp(GT window/0.1),
  score_wrong = GT 반경 4셀 밖 raw belief 최대.
  `softplus((score_wrong - score_gt + 0.10)/0.1)`.
- **distance**: 좌표 기대값이 아니라 **거리의 기대값** — bimodal map 의 평균 좌표가
  두 peak 사이로 가는 문제를 피한다.  bbox 대각선으로 정규화(무효 시 grid 대각선, 카운트 기록).
- **progress**: `relu(D5 - sg(D4)) + relu(D6 - sg(D5)) + relu(sg(M4)-M5) + relu(sg(M5)-M6)`.
  이전 stage 를 detach 해 "stage4 를 망쳐서 조건 충족" 을 막는다.

validity 는 raster all-zero 가 아니라 **변환된 GT center 가 grid 안**인지로 판단한다.
sigmoid·절대 threshold·local-window-only 확률 없음.

## Phase D synthetic gradient 검증 (학습 전 통과)

- confident-wrong map(wrong 0.9 @ 40,40 / GT @ 5,5, 거리 49셀):
  wrong peak gradient > 0(하강 시 감소), GT gradient < 0(하강 시 증가) [확인]
- 중간 영역(20,20)에도 gradient 존재 = 7x7 window 밖에도 전달 [확인]
- progress 는 인위적 회귀 예제에서 양수, stage4 gradient 0 [확인]
- border GT(0,0 / 49,49)에서 cropped window, 전부 finite [확인]
- centroid channel gradient 정확히 0 [확인]
