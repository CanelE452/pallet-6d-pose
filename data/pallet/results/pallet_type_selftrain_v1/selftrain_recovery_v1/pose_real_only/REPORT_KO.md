# 합성 replay 제거 통제 실험

기존 R0에서 독립 초기화, 검출·backbone·BN 동결, pose-only lr1e-5,5epochs/320steps. 기존217장과 코너 마스크·보정 좌표는 동일하다. 원래 실사512노출 multiset을 두 번 사용하여 실사1024, 합성0으로 바꿨다. 실사 노출 두 배 효과까지 포함한 비교이며 순수한 gradient 충돌만의 인과 증명은 아니다.

| 모델 | PCK20 %↑ | 대칭8점 median↓ | 대칭8점 P90↓ | IoU3D↑ |
|---|---:|---:|---:|---:|
| SYN_LR5 | 78.057 | 7.450 | 42.767 | 0.59421 |
| RAW_LR5 | 77.859 | 7.288 | 43.492 | 0.59411 |
| REF_LR5 | 78.718 | 6.722 | 43.643 | 0.59405 |
| RAW_REAL_ONLY | 76.999 | 7.567 | 43.828 | 0.58614 |
| REF_REAL_ONLY | 78.519 | 6.373 | 43.663 | 0.61208 |

사전 반복 진행 조건: {"PCK20_above_previous_REF": false, "PCK20_above_R0": true, "IoU3D_preserved": true, "matched8_P90_preserved": false}

반복 진행 조건 충족: False

세션-cluster bootstrap은 반복 사용 DEV에 대한 탐색적 진단이고 검색 횟수 보정은 하지 않았다. full194에는 과거 보정기 학습3장도 포함된다. 독립 확인 또는 확정된 일반화 성능 주장은 없다.

```json
{
  "R0": {
    "delta_pp": 0.6609385327164574,
    "CI95_pp": [
      -0.6590726016794732,
      1.7857998630762197
    ],
    "clusters": 9,
    "bootstrap_replicates": 10000,
    "exploratory": true,
    "multiple_search_adjustment": false
  },
  "SYN_LR5": {
    "delta_pp": 0.46265697290152014,
    "CI95_pp": [
      -1.1737089201877935,
      1.890493066255778
    ],
    "clusters": 9,
    "bootstrap_replicates": 10000,
    "exploratory": true,
    "multiple_search_adjustment": false
  },
  "REF_LR5": {
    "delta_pp": -0.1982815598149372,
    "CI95_pp": [
      -0.8647798742138365,
      0.20675574453163917
    ],
    "clusters": 9,
    "bootstrap_replicates": 10000,
    "exploratory": true,
    "multiple_search_adjustment": false
  },
  "RAW_REAL_ONLY": {
    "delta_pp": 1.520158625247852,
    "CI95_pp": [
      -0.3861003861003861,
      3.776782904349777
    ],
    "clusters": 9,
    "bootstrap_replicates": 10000,
    "exploratory": true,
    "multiple_search_adjustment": false
  }
}
```

전체194 검출 box/score exact parity와 동결 상태·320steps를 확인했다. 이 screen은 NEG2689 및 논문 fixed-index9점 지표를 새로 실행하지 않았다. 공식C2 평가, 기존 수도레이블, 최종 모델은 변경하지 않았다.
