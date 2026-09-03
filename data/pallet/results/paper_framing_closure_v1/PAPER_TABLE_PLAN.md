# 표 계획

> 이미 생성된 표는 `_docs/paper/final/generated/` 에 있다
> (TABLE_FINAL_1~4, DIAGNOSTIC, POSE, POSE_BY_MATERIAL, POSE_BY_LIGHTING).
> 새 표를 만들지 않는다.  여기서는 **배치**와 **각 표에 반드시 붙어야 할 단서**만
> 정한다.

## 본문

실제 번호는 현재 manuscript 구조를 따르고 역할만 맞춘다 — 새 번호를 임의로 깨지 않는다.

```
T1  검출과 랭킹 (2D)                     TABLE_FINAL_1
      AP50-95 · AP50 · AUROC · FPR95 · pooled supervised keypoint median/p90
      필수 단서   AUROC/FPR95 의 frame-level 구간을 이번에 채웠다(부록 TA1).
                 session-clustered 구간은 negative 의 session 라벨 부재로 계산 불가

T2  미세 국소화 (2D)                     TABLE_FINAL_2
      R0 vs 여섯 adaptation arm, paired bootstrap 포함
      필수 단서   원본 이미지 픽셀 단위이며 6D pose 오차가 아니다
                 (METRIC_NAMING_LOCK 의 표현 그대로)

Pose Table  하류 6D 전체 평가          TABLE_FINAL_POSE      ★ 본문 확정 (2026-09-04)
      PoseCov · AxisAcc · R · yaw · t · IoU3D · ADDsym AUC, 7 arm
      2D/검출 표와 **합치지 않는다** — 다른 질문이고 모집단도 다르다
      필수 단서   개선 방향으로 session-cluster 구간이 0 을 배제한 metric block 0/24.
                 AxisAcc 를 반드시 옆에 둔다 — 축 정확도 변화가 pose 정확도로
                 비례해 옮겨가지 않는다.
                 R0_CONT 는 318 프레임(PoseCov 0.997), 나머지는 319 — 숨기지 않는다.
                 reference 는 geometry-reconstructed 6D reference pose 이며
                 센서 GT 가 아니다
```

## 부록

```
TA1  ranking 부트스트랩 구간              PAPER_STATIC_STAT_AUDIT.json G1 (이번 산출)
TA2  재료 · 조명 부분모집단               TABLE_FINAL_POSE_BY_{MATERIAL,LIGHTING}
       필수 단서   Nighttime N=50 은 plastic 전용.  Lighting_night N=106 과 절대
                  혼용하지 않는다.  N 을 항상 인쇄한다.  pose manifest 는 319 중
                  120 장만 조명 라벨을 갖는다
TA3  day/night 부록 arm (A8)             lock 이 아니라 EXPERIMENTS.md 등록 — 단서 표기
TA4  pseudo-label 필터 품질 · retention   구간 없음(BLOCKED_MISSING_ARTIFACT)
TA5  축 oracle 상한선                     ORACLE path.  배포 가능한 수치가 아니라고 명시
TA6  no-train pose formulation 스크린      V1 S1/S3/S4 + V1B C1/L2/L3/L4 한 표로 통합
       필수 단서   POST_STOP_EXPLORATORY_CORRECTION.  전부 음성.
                  S3 는 305 프레임에서 코너 선택을 **0 개** 바꿨다
```

## 표에 절대 넣지 않는 것

```
REALFT_A / B / LV1V2 를 adaptation arm 과 같은 열 블록에 두는 것
temporal 의 정식 수치 표 (적격 모집단 0)
depth Gate 1 수치 (실행되지 않음)
빈 pose 열 (있으면 열째로 빼고, 없으면 제거한다 — 대시로 남기지 않는다)
```
