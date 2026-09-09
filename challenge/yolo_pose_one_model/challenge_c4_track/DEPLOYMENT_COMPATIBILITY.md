# DEPLOYMENT_COMPATIBILITY — `newauto` 가 지금 이 모델을 소비할 수 있는가

C4 metric 이 좋아졌다는 것만으로 배포를 정당화하지 않는다.  최종 소비처는
지게차 정렬 시스템이고, 그쪽 계약이 keypoint **번호의 의미**에 의존한다.

---

## 1. 실제 계약 (코드에서 읽음, 추정 아님)

저장소 `25y_automatic_lifter-master` (`doyoung0503/newauto` 계열 작업본).

```
depth_cam/calib/config.py
    POSE_FACE_KPTS   = (0, 1, 2, 3)      전면 4 점
    POSE_KPT_VIS_THR = 0.5
    PALLET_FACE_W    = 1.000             전면 가로 (x)
    PALLET_FACE_H    = 0.150             전면 높이 (y)
    PALLET_DEPTH     = 1.200             깊이 (z)
    MODEL_PATH       = '../pallet_yolo26n_pose_ft.pt'
```

```
depth_cam/calib/perception.py
    sel = k[list(POSE_FACE_KPTS)]                 # (4,3)
    네 점이 모두 visible 임계를 넘어야 kpts4 를 넘긴다
    detection 선택은 bbox 종횡비가 PALLET_FACE_W/H 에 가까운 것

depth_cam/calib/geometry.py:_box_object_points()
    "원점은 전면(포크 진입면) 중심 — tvec 이 그대로 전면 중심 3D 좌표"
    "행 순서는 키포인트 인덱스와 1:1 대응: 0~3 전면(좌상,우상,우하,좌하),
     4~7 후면(동일 순서)"
    1단계  전면 4 점 SOLVEPNP_IPPE      (항상)
    2단계  8 점 다 보이면 SOLVEPNP_ITERATIVE 로 정밀화
    yaw  = atan2(n[0], n[2]),  n = R·(0,0,1)      ← 전면 법선
    roll = atan2(-u[1], u[0]), u = R·(1,0,0)      ← 상변 방향
```

즉 **0~3 은 "특정 camera-facing face" 로 가정돼 있고, 그 면의 중심이 좌표 원점**이다.

## 2. 그래서 무엇이 위험한가

정사각 팔레트(1.10 × 1.10)에서 예측의 face phase 가 90도 바뀌면:

```
tvec   전면 중심이 인접 면 중심으로 이동
       두 면 중심 사이 거리 = 0.55·√2 ≈ 0.778 m
yaw    90도 점프
```

C4 loss 는 이 90도 차이를 **벌하지 않도록** 학습한다.  그것이 목적이다 — 실제
localization 실패와 임의의 번호 선택을 분리하는 것.  그러나 downstream 은 번호를
그대로 믿으므로, 학습이 phase 를 자유롭게 두면 프레임마다 0.78 m 씩 튀는 위치가
정렬 루프로 들어간다.

`_angles_from_R` 이 내는 yaw 도 90도 단위로 점프하므로, FSM 이 yaw 를 절대값으로
쓰는 한 그대로는 못 쓴다.

## 3. 그래서 이렇게 판정한다

C4-equivalent metric 이 좋아져도, 예측 face phase 가 프레임 간 불안정하면 판정은
`C4_LOCALIZATION_IMPROVED_BUT_NOT_DEPLOYABLE_WITH_CURRENT_INDEX_CONTRACT` 이고
기존 배포 weight 를 덮지 않는다.  그 경우 다음 작업은 **inference canonicalizer**
또는 **newauto 계약 변경**이지, 이번 loss 실험의 성공으로 봉합할 문제가 아니다.

측정은 `scripts/diagnose_downstream.py` 가 한다.  **GT 를 쓰지 않는다** — 배포
시점에는 GT 가 없기 때문이다.  연속 촬영에서 이웃한 두 프레임의 **예측끼리만**
비교해 "다음 프레임이 이전 대비 몇 도 돌아 보이는가" 를 세고, 전면 중심(0~3 평균)의
프레임 간 이동량을 잰다.  카메라와 물체가 연속적으로 움직이면 정답은 항상 0도다.

## 4. 부수적으로 발견한 계약 불일치 (이번 실험과 별개)

```
                       newauto config     실측 팔레트
전면 가로              1.000 m            1.10 m
전면 높이              0.150 m            0.15 m
깊이                   1.200 m            1.10 m
```

`PALLET_FACE_W` 와 `PALLET_DEPTH` 가 실측과 다르다.  이 값은 `solvePnP` 의
`objectPoints` 와 detection 선택 기준(`TARGET_AR = FACE_W/FACE_H`) 양쪽에 쓰인다.
이번 track 의 판정 대상은 아니지만, 배포 전에 맞춰야 거리·정렬이 정확해진다.

---

## 5. 결과 (학습·진단 후 채움)

### 측정 결과 — 연속 시퀀스에서는 phase 가 튀지 않는다

5 세션 · F0 638 pair / F1 627 pair, GT 미사용.

```
                              F0 안정률    F1 안정률
forklift_v4_20260904_142318    100.0%       100.0%
forklift_v4_20260904_103429     92.7%        98.9%
forklift_v4_20260904_105615    100.0%       100.0%
forklift_v4_174925             100.0%       100.0%
capture_20260902               100.0%        99.0%
────────────────────────────────────────────────────
합계                            99.06%       99.68%
```

전면 중심(0~3 평균)의 프레임 간 이동도 F1 이 같거나 작다.

### 그래서 판정

**`DEPLOYABLE_WITH_CAVEAT`** — 우려했던 "프레임마다 0도 → 90도 → 0도" 진동은
관측되지 않았다.  F1 은 한 접근 시퀀스 안에서 face phase 를 유지하므로, 정렬 루프
도중 `tvec` 이 0.78 m 튀는 사고는 일어나지 않는다.

다만 조건이 붙는다.

1. **세션(접근 시도) 사이에는 phase 가 다를 수 있다.**  val 155 장에서 F1 은 93 장을
   0도, 62 장을 270도로 골랐다.  정사각 팔레트는 네 면 어디로도 포크가 들어가므로
   "다른 면을 앞면으로 본다" 자체는 물리적으로 유효하다.  그러나 절대 yaw 를 기록·
   비교하거나 이전 세션의 자세와 대조하는 로직이 있다면 90도 배수 차이를 감안해야
   한다.
2. **소표본이다.**  F1 의 우위(collapse 2→0, 미검출 3→0, phase 불안정 6→2)는 전부
   한 자릿수 사건이고 seed 하나짜리다.  "C4 가 확실히 낫다" 고 배포 근거를 삼기에는
   얇다.
3. **동료가 흔들린다고 지목한 세션(`..._190700`)이 이 저장소에 없다.**  가장 직접적인
   반례가 될 수 있는 데이터에서 재보지 못했다.

### 배포 전에 먼저 고쳐야 할 것 — loss 와 무관한 쪽

위 §4 의 치수 불일치가 그대로 남아 있다.

```
                config      실측
PALLET_FACE_W   1.000 m     1.10 m
PALLET_DEPTH    1.200 m     1.10 m
```

정사각 물체를 직사각 모델로 `solvePnP` 하는 중이다.  keypoint 를 아무리 안정시켜도
이 오차는 남는다.  수정 비용은 config 두 줄이고, C4 재학습보다 훨씬 싸다.
**흔들림을 줄이려는 목적이라면 이것부터 고치고 재측정하는 편이 순서에 맞다.**
