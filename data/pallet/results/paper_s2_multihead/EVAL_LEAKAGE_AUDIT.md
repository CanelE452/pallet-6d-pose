# EVAL LEAKAGE AUDIT

```
결론   MAIN_TRAIN 과 평가 population 의 교집합 = 0
```

## 수집한 평가 population

```
population              n      MH_TRAIN 교집합
d2_mh_dev512           512            0
d3_mh_conf512          512            0
d4_theta_confirm512    512            0
d5_target_unseen       959            0
lcurve_population    1,089            0
────────────────────────────────────
고유 합계            2,885            0
```

전부 `MH_DEV`(6,242)에서 뽑았고 `MH_TRAIN`(33,758)과 겹치지 않는다.

## 구조적 이유

split 은 `mh_data.build_split` 이 **group 단위**로 나눈다(17 dev group).
평가 manifest 는 전부 `split == "MH_DEV"` 필터를 거쳐 생성됐다. 따라서 누수는
우연히 없는 게 아니라 구성상 불가능하다.

## train 프레임을 unseen 으로 표기한 흔적

없다. 반대로 그 함정을 명시적으로 피한 기록이 있다 —
`D5_TARGET_UNSEEN` 을 만들 때 "MH_TRAIN 에 같은 cell 이 706/950/931 장 있지만
E3@18k 가 그 pool 전체를 이미 학습했으므로 모델 입장에서 unseen 이 아니다" 라고
판단하고 train 을 제외했다(`d5_target_unseen_manifest.json` 의 `note`).

## 남은 제약 (누수는 아니지만 기록)

`MH_DEV` 는 학습에 쓰인 적이 없으나, **데이터 설계에는 관여했다** —
canonical risk map 과 CORNER_LA 의 겨냥 cell 이 D2+D3+D4 위에서 산출됐다.
따라서 이 population 들은 "학습 미사용" 이지 "설계 미접촉" 은 아니다.
paper final independent confirmation 으로 쓰려면 별도 holdout 이 필요하다.
