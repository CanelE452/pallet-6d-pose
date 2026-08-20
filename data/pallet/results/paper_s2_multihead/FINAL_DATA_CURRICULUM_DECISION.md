# FINAL DATA CURRICULUM DECISION

```
FINAL TRAINING DATA   Corner = BROAD 40k              (CORNER_LA 미채택)
                      Line   = BROAD 40k              (EDGE_HARD 미개봉)
FINAL POSE PATH       CORNER_PNP_PRIMARY
LONG CONFIRM          DO_NOT_RUN
```

근거는 `LOWANGLE_OBLIQUE_RESULT.md`. 요약하면 C1 이 사전등록 gate 를 통과하지 못했고
(rms 가 seed 간 부호 반전, t 가 두 seed 모두 미달, T3 안전 절 위반), paired bootstrap
24개 비교 중 CI 가 0 을 배제한 것이 손상 1건뿐이다.

⚠ 이 결정은 **"CORNER_LA 5K 가 쓸모없다"가 아니다.** target 표본이 seed 당 44~51 뿐이라
이 screen 은 20~30% 미만의 효과를 검출할 수 없다. 데이터셋 QA 는 전부 통과했고
(5,000장, 결손·중복 0, elevation/yaw 위반 0, BROAD 와 collision 0),
`LOWANGLE_OBLIQUE_DATA_SIGNAL = False` 는 **이 예산·이 평가셋에서 확립되지 않았다**는 뜻이다.

재시도한다면 바꿔야 할 것은 데이터가 아니라 설계다 — 표본이 큰 target 평가셋,
더 긴 예산, 또는 LA 노출 상향. 셋 다 **새 사전등록**이 필요하며 이번 결과를 보고
gate 를 완화하는 형태여서는 안 된다.

## 이번 screen 이 확실히 남긴 것

```
LINE_ISOLATION_EXACT = True
```

two-stream 커리큘럼에서 line branch 가 3,000 step 학습 후에도 param diff 0.000e+00 로
보존된다. 앞으로 branch 별 데이터 실험을 할 때 line 쪽을 percentage guard 로 감시할
필요가 없다 — 배선이 보장한다. 이건 재사용 가능한 자산이다.
