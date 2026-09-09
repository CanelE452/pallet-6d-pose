# annotate.py — Manual Keypoint Annotation 가이드

`scripts/annotate/annotate.py`는 시퀀스 frame에 9 keypoint를 클릭으로 표시하고,
PnP를 자동으로 풀어 NDDS JSON GT를 저장한다. AprilTag가 없는 시퀀스에서 학습용
GT 를 직접 만들 때 쓴다.

---

## 1. 실행

```bash
conda activate pallet-pose

# 기본: capturepallet02 를 stride 30 으로 (≈10 frame 라벨링)
python scripts/annotate/annotate.py --seq data/outside/capturepallet02

# 다른 시퀀스, 더 촘촘하게
python scripts/annotate/annotate.py --seq data/outside/capturepallet07 --stride 20

# 중간부터 시작
python scripts/annotate/annotate.py --seq data/outside/capturepallet09 --stride 60 --start 300

# 저장 경로 지정
python scripts/annotate/annotate.py --seq data/outside/capturepallet07 \
  --out_dir challenge/data/01_real/manual_gt/capturepallet07_manual_gt
```

**기본 출력 경로**: `challenge/data/<시퀀스명>_manual_gt/`
**옵션 인자**:
- `--stride N` : N frame 마다 1 개 라벨링 (기본 30)
- `--start N` : 시작 frame index (이어서 작업할 때)

---

## 🎯 잘린 keypoint 처리 — MANIPULATE 모드

화면 가장자리에 가까워서 keypoint 가 잘리거나, 가려진 corner 가 안 보이면 클릭이
불가능하다. 이때는:

```
1. 보이는 4+ keypoint 클릭 (없는 건 None 으로 둠)
2. PnP 가 풀리면서 cuboid wireframe 이 그려짐 (잘린 부분은 image 밖)
3. m 키 → MANIPULATE 모드 진입 (image 테두리 파랑)
4. 6DoF 조정으로 cuboid 를 팔레트에 정확히 정렬:
     translate  a/d (X)  w/x (Y)  q/e (Z, 가까이/멀리)
     rotate     j/l yaw  i/k pitch  u/o roll
     step       1/2 trans /2 *2     3/4 rot /2 *2
5. 정렬 완료되면 S (대문자) 로 저장
     → 잘렸던 keypoint 도 cuboid 좌표 (image 밖이라도) 로 GT 에 들어감
6. m 다시 누르면 CLICK 모드로 복귀, Q (대문자) 로 종료
```

CLICK 모드의 소문자 `s` 와 MANIPULATE 모드의 대문자 **`S`** 차이:
- `s` (click 저장) : 사용자가 찍은 점만 manual_kps 로 저장
- `S` (manip 저장) : cuboid 의 8 corner 모두 manual_kps 로 저장 (잘린 점 포함)

## ⚠️ 양면 진입 가능 팔레트 — 라벨링 규칙 (Convention B: view-fixed + symmetric_loss)

실제 팔레트는 1.10 × 1.30 × 0.11m 직사각형. 두 가지 짧은 면(110cm)이 모두 포크
진입 가능. **학습 시 `--symmetric_loss` 로 110m 면 앞/뒤 swap 자동 허용**, 따라서:

```
✓ 110m 짧은 면이 정면으로 보이는 frame → 그 면을 0~3 으로 라벨링
  (앞쪽 110 면이든 뒤쪽 110 면이든 무방, symmetric_loss 가 처리)
✗ 130m 긴 면이 정면으로 보이는 frame → 라벨링 SKIP (학습 모순)
✗ 카메라가 측면을 비스듬히 봐서 110/130 둘 다 보이는 frame → 110 면을 0~3 으로
```

**110 vs 130 식별법**:
- image 에서 팔레트 가로 폭이 더 짧고 깊이가 더 길게 보이면 110 면 view
- 거꾸로면 130 면 view → skip
- 헷갈리면 tag 가 보이는 frame 위주로 라벨링 (tag = 110 면 부착)

## 2. 클릭 순서 (학습 데이터와 일치)

```
인덱스   이름                  화면(image plane) 기준 위치
─────────────────────────────────────────────────────────────
  0     FrontTopRight        앞면 + 화면 우측 + 화면 위
  1     FrontTopLeft         앞면 + 화면 좌측 + 화면 위
  2     FrontBottomLeft      앞면 + 화면 좌측 + 화면 아래
  3     FrontBottomRight     앞면 + 화면 우측 + 화면 아래
  4     RearTopRight         뒷면 + 화면 우측 + 화면 위
  5     RearTopLeft          뒷면 + 화면 좌측 + 화면 위
  6     RearBottomLeft       뒷면 + 화면 좌측 + 화면 아래
  7     RearBottomRight      뒷면 + 화면 우측 + 화면 아래
  8     Centroid             8 코너의 중심 (자동 계산 가능)
```

**중요**: 학습 데이터(`data/pallet/training_data/mixed_v8_train`)는 image plane
기준 좌/우다. `_docs/preprocessing/keypoint_definition.md` 의 ID 표는 3D 모델
(Y=UP) 의 X 축 기준이라 projection 시 좌/우가 image 와 반대로 보이지만, 학습은
**image plane 기준 좌/우**를 따른다. 위 표 그대로 따라 클릭하면 학습 데이터와
일치.

**앞면**(0~3) = 포크가 들어가는 면 — challenge 의 핵심. 먼저 앞면 4 개를 정확히
찍는 것이 우선.

**색상 가이드** (annotate.py 가 화면에 표시):
```
0 red,  1 orange,  2 yellow,  3 green        ← 앞면 4 개 (따뜻한 색)
4 cyan, 5 blue,    6 magenta,  7 purple      ← 뒷면 4 개 (차가운 색)
8 white                                       ← centroid
```

---

## 3. 기본 키 컨트롤

### 점 찍기
```
좌클릭          현재 활성 인덱스 위치에 점 찍기 → 자동으로 다음 인덱스로 이동
우클릭          현재 활성 인덱스 점 삭제 (실수 정정용)
0 ~ 8           해당 인덱스를 활성 (다시 찍어서 덮어쓰기)
```

### 진행
```
s              저장 (JSON + image) → 다음 frame 자동 이동
n              저장 없이 다음 frame  (미저장 변경 있으면 한 번 더 눌러야 진행)
p              이전 frame
q              종료
```

### SESSION 선택

화면의 `SESSION` 버튼에서는 현재 기존 평가 session 12개와 신규 촬영본의 zero-copy
`STAGING EDIT` 4개, 총 16개를 선택할 수 있다. 신규 DAY와 NIGHT는 각각 `PLASTIC`,
`WOOD` 두 행으로 표시된다. 두 행은 같은 raw capture를 복사하지 않고 참조하지만 서로
겹치지 않는 실제 frame subset만 보여주며, object별 geometry로 PnP를 푼다.

분류 정본은 각 raw frame을 정확히 한 번 기록한
`incoming/sessions/<capture>/manifests/frame_review.csv`다. 실제 픽셀을 프레임
단위로 검수해 `plastic`, `wood`, `exclude`로 나눴으며, `exclude`(파렛트 없음,
카메라 이동, 심한 motion blur)는 두 객체 view에서 모두 숨긴다. 큰 재질 경계는
DAY source ordinal 기준 `WOOD=69..5480`, `PLASTIC=5481..24193`,
`WOOD=24241..29028`이고, NIGHT는 `WOOD=1..4849`,
`PLASTIC=4850..13583`이다. 경계 안의 검수 제외 구간까지 적용한 최종 수는 DAY
`PLASTIC 17,917 / WOOD 9,362 / EXCLUDE 1,749`, NIGHT
`PLASTIC 7,913 / WOOD 4,546 / EXCLUDE 1,124`이다.
partition view 안의 `frame N/M`과 goto는 view-local 번호다. 패널에는 원본 기준
`source ordinal N/raw_total`도 함께 표시하며, 경계 추적은 source ordinal 또는
filename을 사용한다.

staging 저장 위치는 `incoming/annotations/<capture>__plastic/` 또는
`incoming/annotations/<capture>__wood/`다. PnP GT JSON, 호환 PNG,
`frame_tags.csv`, `_overlays/<stem>.png`는 이 객체별 출력 아래에만 생성되고
`incoming/sessions/<capture>/`의 raw 파일은 수정·이동하지 않는다.

신규 capture의 제공 intrinsics는 검증되지 않았으므로 GT에는
`intrinsics_quality=UNKNOWN`으로 기록하고, 원래 `PROVIDED_UNVERIFIED` 품질과
`camera_info.json` 출처는 `intrinsics_source`에 보존한다. staging에서 `s`로 저장하면
방금 검수한 frame만 대응 active evaluation session으로 독립 복사하고, condition tag를
포함해 평가 manifest와 통합 진행률 MD를 즉시 갱신한다. 별도 promotion 명령은 필요
없으며 다른 incoming frame을 함께 편입하지 않는다.

### 평가 조건 및 session 일괄 적용

`/`로 `CONDITIONS` 모드에 들어간다.

```
1 / 2          occlusion / truncation ON·OFF
3 / 4 / 5      elevation LOW / MID / HIGH
n / m / 6      distance NEAR / MID / FAR
u              distance를 UNKNOWN으로 되돌림
a, a           방금 바꾼 tag만 현재 session의 annotated frame 전체에 적용
s              현재 frame만 저장하고 다음으로 이동
/ 또는 Esc   CLICK 모드로 복귀
```

일괄 적용은 해당 session에서 annotation JSON이 있는 frame만 대상으로 하며,
미어노테이션 frame과 다른 기존 tag는 건드리지 않는다. 실수로 전체 session을
바꾸지 않도록 `a`를 두 번 눌러야 실제 저장된다.

현재 active evaluation condition은 camera와 pallet 사이의 물리적 거리인
`NEAR / MID / FAR`만 사용한다. 고정 meter threshold는 아직 정의하지 않았으므로
명확하지 않으면 `u`를 누른다. 현재 session의 annotated frame 전체를 distance
UNKNOWN으로 되돌릴 때는 `u`, `a`, `a` 순서로 누른다. 과거 CSV의 `size_bin` 열은
파일 호환을 위해 보존하지만 UI, 진행률, 논문 condition 표에서는 사용하지 않는다.

저장하면 `challenge/data/<시퀀스명>_manual_gt/<timestamp>.json` + `.png` 가 같이
저장된다.

이전에 저장한 frame 으로 돌아가면 (`p` 또는 다시 시작) 기존 annotation 을 로드해서
이어 작업 가능.

### 보조
```
c              centroid 자동 계산 (찍힌 corners 의 평균)
z              마지막 점 취소 + 활성 인덱스 한 칸 뒤로
r              모든 점 리셋 (전체 다시)
b              belief map 토글  ※ run_live.py 와 달리 여기선 사용 안 함
```

---

## 4. 확대/축소 (정밀 클릭용)

화면 작아서 정확히 찍기 어려울 때:

```
+  또는 =      줌인 (최대 ×4, 마우스 커서 위치 중심으로 확대)
-  또는 _      줌아웃 (×1 로 복귀 시 자동 중앙 정렬)
```

줌인 상태에서 화면을 다른 영역으로 옮기려면 (vim 스타일 pan):
```
h              왼쪽으로  20px
l              오른쪽으로 20px
k              위로     20px
j              아래로   20px
```

화면 우하단에 현재 줌 배율이 표시됨 (예: `zoom x2.3`).

---

## 5. 화면 표시 보는 법

### 상단 HUD
```
[3/25] 1778654...   (3/9 set)        ← 현재 frame index / 총 frames / 찍은 점 개수
Click #4: RearTopRight                ← 현재 활성 인덱스 + 이름
*UNSAVED*                             ← 미저장 변경 있음
PnP reproj=2.34px                     ← PnP 풀이 정확도
```

### PnP reprojection error 의미
```
< 5px        매우 좋음 — 모든 점이 cuboid 와 일관 → 그대로 저장
5 ~ 10px     보통 — 점 하나 흔들렸을 수 있음, 점 검토 권장
> 10px       나쁨 — 점 위치 부정확하거나 인덱스 잘못 (좌/우 뒤집힘 의심)
```

### Cuboid wireframe (실시간)
4 점 이상 찍히면 PnP 가 자동으로 풀려서 녹색 직육면체 wireframe 이 그려진다.
- **앞면 (0-1-2-3)** 굵은 line, **뒷면 (4-5-6-7)** 얇은 line, **수직 4 edge** 가
  앞뒤 연결
- 찍은 점과 wireframe 이 어긋나면 → 점 위치 수정 후 다시 풀림

---

## 6. 효율적 작업 흐름 (권장)

1. **앞면 4 개부터 정확히** (0 → 1 → 2 → 3) — 포크 진입에 가장 중요.
2. **뒷면 4 개** (4 → 5 → 6 → 7) — 가려진 corner는 추정으로 찍어도 PnP 가
   완화시켜줌.
3. **`c`** 로 centroid 자동.
4. 화면 우상단의 **reproj error** 확인 — `<5px` 이면 좋음.
5. **`s`** 로 저장하고 다음.

가려진 점은 어떻게? — `0~7` 모두 찍는 것을 권장. 가려져도 cuboid 기하학상의
위치를 추정해서 찍으면 학습에 도움된다. 도저히 위치 모르겠으면 그 점만 안 찍고
저장해도 4 점 이상이면 PnP 는 풀린다 (단 reproj 가 더 커질 수 있음).

---

## 7. 자주 묻는 것

**Q. 좌/우가 헷갈림** → 화면(image plane) 기준 좌/우. 사용자 입장에서 화면 우측에
보이는 것이 0 (FrontTopRight). 색상으로 외우면 편함 — 0 red 는 항상 우측.

**Q. 가려진 corner 는 어떻게?** → 다른 corner 의 기하학에서 추정. 학습 모델은
가려진 keypoint 도 belief map 에서 회귀하므로 GT 가 있어야 학습이 된다.
정확한 추정이 어려우면 그 점은 None 으로 두고 (key `z` 로 취소) 저장. 4 점 이상이면
PnP 는 풀린다.

**Q. 저장 후 다시 수정하고 싶음** → 같은 stride 로 도구를 다시 켜면 기존 JSON 을
자동 로드한다. `0~8` 으로 인덱스 골라 다시 찍기.

**Q. 한 stride 의 frame 만 만들면 충분?** → 처음에는 `--stride 30` (≈25 frame)
으로 시작해서 ft 한 번 돌려보고, 정확도 부족하면 `--stride 15` 로 더 추가.
또는 다른 시퀀스도 보강.

**Q. PnP reproj 가 항상 크게 나옴** → 좌/우 인덱스가 뒤집혔을 가능성. 0 이 image
우측에 가는지 확인. 또는 앞면/뒷면이 뒤집힌 경우 — 0,1,2,3 이 가까운 면 (포크
들어가는 면)이어야 한다.

**Q. 저장된 GT 가 어떤 모양?** → NDDS 호환 JSON.
```json
{
  "camera_data": {"width", "height", "intrinsics": {...}},
  "objects": [{
    "class": "pallet",
    "pose_transform": [4x4 R|t],
    "projected_cuboid": [[u,v]×8],
    "projected_cuboid_centroid": [u, v],
    "dimensions_m": {"width", "height", "depth"},
    "gt_source": "manual",
    "manual_kps": [[u,v] or null, ×9],
    "reproj_error_px": <float>
  }]
}
```
DOPE `CleanVisiiDopeLoader`가 그대로 읽을 수 있는 포맷이므로
`bash challenge/scripts/finetune.sh` 로 바로 ft 가능.

---

## 8. 출력 경로

```
challenge/data/<시퀀스명>_manual_gt/
├── 1778654...4.json    NDDS GT
├── 1778654...4.png     동일 stem 의 RGB (학습용)
└── ...
```

ft 시 `--train_dir challenge/data/<시퀀스명>_manual_gt` 로 지정하거나,
여러 시퀀스를 모은 디렉토리로 옮긴 후 사용.
