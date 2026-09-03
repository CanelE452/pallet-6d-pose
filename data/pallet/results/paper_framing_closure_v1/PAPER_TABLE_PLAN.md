# 표 계획

> 이미 생성된 표는 `_docs/paper/final/generated/` 에 있다
> (TABLE_FINAL_1~4, DIAGNOSTIC, POSE, POSE_BY_MATERIAL, POSE_BY_LIGHTING).
> 새 표를 만들지 않는다.  여기서는 **배치**와 **각 표에 반드시 붙어야 할 단서**만
> 정한다.

## 본문

```
T1  검출과 랭킹 (2D)                     TABLE_FINAL_1
      AP50-95 · AP50 · AUROC · FPR95 · pooled supervised keypoint median/p90
      필수 단서   AUROC/FPR95 의 frame-level 구간을 이번에 채웠다(부록 TA1).
                 session-clustered 구간은 negative 의 session 라벨 부재로 계산 불가

T2  미세 국소화 (2D)                     TABLE_FINAL_2
      R0 vs 여섯 adaptation arm, paired bootstrap 포함
      필수 단서   원본 이미지 픽셀 단위이며 6D pose 오차가 아니다
                 (METRIC_NAMING_LOCK 의 표현 그대로)

T3  6D pose 주 비교                      TABLE_FINAL_POSE      ★ 배치 미정
      PoseCov · AxisAcc · R · yaw · t · IoU3D · ADDsym AUC, 7 arm
      필수 단서   24 개 session-cluster 구간이 전부 0 을 포함한다.
                 AxisAcc 를 반드시 옆에 둔다 — 축 정확도 변화가 pose 정확도로
                 비례해 옮겨가지 않는다
      ★ PAPER_REVIEWER_GAP_AUDIT §1 의 사용자 결정 전에는 본문/부록 확정 불가
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
