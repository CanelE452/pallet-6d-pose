# Annotation Reliability Plan

작성 2026-08-27. 목적은 **GT annotation noise floor 를 수치로 갖는 것**이다.
모델 성능을 재는 게 아니다.

현재 상태: `annotation_reliability_measured = false` (`DATASET_AUDIT.json`).

## 왜 필요한가

모델 A 와 B 의 차이가 **GT 자체의 흔들림보다 작으면** 그 차이는 방법의 차이가 아니다.
지금은 noise floor 를 모르므로 어떤 차이가 유의한지 판단할 근거가 없다.

이번 세션의 실측이 그 위험을 보여준다 — hard-negative 스크린에서 arm 간 corner median
차이가 12.7 ~ 14.8 px 였다. **annotation noise 가 이 크기라면 그 비교는 무의미하다.**

## 표본

```
30 ~ 50 장
stratified sampling 축:
  · DAY / NIGHT          (현재 비율 112:28 — NIGHT 을 과소표집하지 않는다)
  · capture session      (7개 set 전부에서 뽑는다)
```

세션별 최소 3장을 보장하고 나머지를 비율 배분하면 7세션 × 최소 3 = 21장 + 나머지,
30~50 범위에 들어온다.

⚠️ `eval_outside` 는 다른 세션에서 뽑아 모은 셋이므로(§CONTRACT §3) 독립 세션으로
세지 않는다. 층으로는 유지하되 "세션 다양성" 근거로는 쓰지 않는다.

## 절차

```
1순위   annotator 2명이 서로의 annotation 을 보지 않고 독립 annotation
2순위   (한 명만 가능하면) 시간 간격을 두고 blind re-annotation
```

두 경우 모두:

- 기존 GT 를 화면에 띄우지 않는다.
- 예측 결과를 보조로 쓰지 않는다(현 도구는 애초에 예측을 쓰지 않는다 — 실측 확인).
- 같은 도구·같은 convention(`camera_dynamic_0123_v4`, 9점)을 쓴다.

## 보고 지표 (최소)

```
2D
  keypoint NME median            (normalized mean error — 정규화 기준을 명시할 것)
  keypoint NME p90

Pose disagreement
  rotation geodesic median  (deg)
  translation median        (m)
  yaw median                (deg)
```

`yaw` 는 팔레트 자신의 up 축(local Y) 둘레 상대회전으로 잰다 —
`scripts/stage0/real_eval/re_metrics.py::yaw_error` 와 같은 정의를 쓴다.

### NME 정규화 기준

기준을 정해서 기록한다(TODO — 아직 확정 안 함). 후보:

```
(a) cuboid 투영 bbox 대각선     프레임마다 크기가 크게 달라(면적비 p10 0.028 / p90 0.655)
                                정규화가 필요하다는 근거가 있다
(b) object diameter 투영 길이
```

정하기 전에는 **raw pixel median/p90 도 함께 보고**해 비교 가능성을 남긴다.

## 이 수치의 용도

```
용도       GT annotation noise floor
용도 아님   모델 성능 · 모델 간 비교
```

**적용 규칙:** 모델 A/B 의 지표 차이가 이 noise floor 보다 작으면
**강한 superiority claim 을 하지 않는다.** 논문 본문에 이 규칙을 명시하고,
noise floor 값을 표 각주 또는 별도 절에 싣는다.

## 함께 처리해야 할 것

reliability 측정과 별개로, 아래 두 가지는 **측정 전에** 정리돼야 결과가 해석된다.

1. **per-keypoint 가시성 flag 부재** — 가려진 코너를 두 annotator 가 서로 다르게
   추정하면 그것이 noise 로 잡힌다. "보이는 점만" 을 구분할 수 없으므로,
   최소한 이번 재annotation 에서는 **가시성 flag 를 함께 기록**해 두는 것이 좋다.

2. **`dimensions_m` W/D 스왑** — 140장 중 (1.1,0.11,1.3) 81 / (1.3,0.11,1.1) 59.
   pose disagreement 를 재려면 두 annotation 이 같은 dims 규약을 써야 한다.
   재annotation 전에 규약을 하나로 고정한다.
   (관련: memory `evaluator-receives-gt-per-frame-axis-assignment`)

## 산출물 (제안 위치)

```
_docs/paper/current_real_dataset/annotation_reliability/
  SAMPLING.json        표본 membership · 층별 배분 · 선정 시드
  ANNOTATOR_A.json     독립 annotation 결과
  ANNOTATOR_B.json     (또는 REANNOTATION_T2.json)
  NOISE_FLOOR.json     위 지표 5종
  REPORT.md            해석 + "이 값보다 작은 차이는 주장하지 않는다" 규칙
```
