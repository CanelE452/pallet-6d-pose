# run_live.py — 실시간 시퀀스/카메라 추론 가이드

`challenge/scripts/run_live.py` 사용법. RealSense 라이브 또는 저장된 시퀀스
재생을 같은 코드로 처리한다. False positive를 막는 sanity gate가 적용되어 있고,
`task.yaml`(strict) / `task_loose.yaml`(baseline 재현) 두 가지 config 제공.

---

## 1. 실행 (가장 자주 쓰는 패턴)

먼저 conda env 활성:
```bash
conda activate pallet-pose
```

### (a) 저장된 시퀀스 재생 (카메라 없이)
```bash
# 기본 — STRICT (gate 적용)
python challenge/scripts/run_live.py --seq data/outside/capturepallet02

# fps 5로 천천히, 끝까지 가면 처음부터 다시
python challenge/scripts/run_live.py --seq data/outside/capturepallet07 --seq_fps 5 --seq_loop

# LOOSE (baseline 원래 동작 재현 — 매 프레임 false positive 확인용)
python challenge/scripts/run_live.py --seq data/outside/capturepallet09 \
  --config challenge/config/task_loose.yaml --label LOOSE
```

### (b) 두 창 동시 비교 (LOOSE vs STRICT)
다른 터미널 두 개에서 각각 실행:
```bash
# 터미널 A
python challenge/scripts/run_live.py --seq data/outside/capturepallet09 \
  --seq_fps 5 --seq_loop --config challenge/config/task_loose.yaml --label LOOSE

# 터미널 B
python challenge/scripts/run_live.py --seq data/outside/capturepallet09 \
  --seq_fps 5 --seq_loop --config challenge/config/task.yaml --label STRICT
```
`--label`이 창 이름 접미사로 들어가서 두 인스턴스가 충돌하지 않는다.

### (c) RealSense D435i 라이브
```bash
python challenge/scripts/run_live.py --realsense
python challenge/scripts/run_live.py --realsense --no_depth   # depth 안 쓰면
```

### (d) 일반 웹캠 (RealSense 없을 때, depth gate 자동 비활성)
```bash
python challenge/scripts/run_live.py --cam_id 0
```

### Windows 인코딩
PowerShell/cmd 에서 한글 콘솔이면 stdout 인코딩 에러가 날 수 있다. 환경변수로 회피:
```bash
PYTHONIOENCODING=utf-8 python challenge/scripts/run_live.py --seq ...
```

---

## 2. 키 컨트롤

### 공통 (라이브/시퀀스 모두)
```
q     종료
s     현재 frame 저장 (challenge/data/01_real/_live_captures/live_NNNN.png)
b     belief map 그리드 토글
r     belief 클릭 auto-tune 초기화
```

### 시퀀스 재생 전용
```
space      일시정지 / 재생
n          다음 frame    (+1, 자동으로 일시정지 진입)
p          이전 frame    (-1)
.          10 frame 앞으로 (+10, 자동 일시정지)
,          10 frame 뒤로  (-10)
]  또는 =  재생 속도 빠르게 (×1.5, 최대 60fps)
[  또는 -  재생 속도 느리게 (÷1.5, 최소 0.5fps)
```

화면 우상단에 현재 fps, 하단에 키 안내가 표시된다. 일시정지 상태에서
`n`/`p`로 한 프레임씩 정밀 분석 가능.

### Belief map 창
- 9 채널 (8 corner + 1 centroid)이 3x3 grid로 표시
- 각 셀에서 팔레트의 해당 keypoint 위치를 좌클릭하면 그 채널의 peak 값을
  학습해 자동으로 `threshold`/`thresh_map`/`thresh_pts` 슬라이더를 조정 (auto-tune)
- 여러 채널 클릭할수록 더 보수적 threshold (모두의 최소값)

---

## 3. 슬라이더 (Controls 창)

라이브로 임계값 튜닝. `task.yaml`/`task_loose.yaml`의 값이 기본값.

```
threshold (x1000)      belief peak가 valid keypoint로 간주되는 최소값
thresh_map (x1000)     find_object_poses 내부 belief map threshold
thresh_pts (x1000)     find_object_poses 내부 point threshold
thresh_ang (x100)      angle threshold (affinity field)
sigma                  belief gaussian smoothing
min_kp                 PnP 결과를 detection으로 받는 최소 keypoint 수 (1~9)
max_reproj_px          PnP reprojection error 상한 (px)
```

값을 좋게 찾으면 `challenge/config/task.yaml`의 `inference.belief.*` 와
`inference.gates.*` 에 반영하여 영구화.

---

## 4. 화면 표시 의미

### 좌상단 배너
```
NOT DETECTED (reason)   gate 통과 못 함 — reason 으로 원인 표시
                        예: kp=3<7         keypoint 부족
                            z=23.87m       PnP z가 범위 밖
                            reproj=17.2px  reprojection error 초과
                            depth_z_diff   depth/PnP z 불일치
                            no_result      PnP 실패
PENDING n/N             단일 frame은 통과, 연속 frame 미달 (temporal)
CONFIRMED               gate + temporal 모두 통과 — 신뢰 가능
```

### 시각화 마커
```
회색 점 + 수치           매 frame의 belief peak (gate 무관, 항상 표시)
녹색/노랑 원              검출된 corner keypoint (0~7)
빨강 원                   검출된 centroid (8)
녹색 사각형 marker        PnP가 푼 cuboid 8 corner를 image에 projection
녹색 wireframe (line)     검출된 corner를 line으로 연결한 직육면체
                         앞면(0-1-2-3) 굵게, 뒷면 얇게, 수직 4개 edge
노란 화살표 + Yaw N도     팔레트 정면 방향 (yaw)
```

---

## 5. 설정 파일

```
challenge/config/task.yaml         STRICT (운영용)
  inference.belief.threshold:    0.30
  inference.gates.min_kp:        7
  inference.gates.max_reproj_px: 8.0
  inference.gates.z_range:       [0.30, 5.00] m
  inference.temporal.confirm:    2 frames

challenge/config/task_loose.yaml   LOOSE (baseline 재현)
  threshold 0.10, gate 거의 무력화 (z_max=500m 등)
```

다른 weight 사용 시: `--weights challenge/weights/finetuned/...pth`

---

## 6. 출력 경로
```
challenge/data/01_real/_live_captures/live_NNNN.png   's' 키로 저장한 frame
```

---

## 7. 자주 묻는 것

**Q. fps 가 너무 빨라서 보기 힘듦** → `[` 또는 `-`로 천천히. 또는 명령에 `--seq_fps 2`.

**Q. 특정 frame 만 자세히 보고 싶음** → `space`로 멈춤 → `n`/`p`로 한 프레임씩. `,`/`.`로 10 프레임씩 점프. 줌이 필요하면 belief 창 클릭으로 채널 디버그.

**Q. STRICT 인데 거의 검출 안 됨** → baseline 모델이 outdoor에 약함. `min_kp` 슬라이더 낮춰서(예: 4) recall 올리거나, `task.yaml`의 gate 임계값 완화. 근본 해결은 finetune.

**Q. 좌상단에 `z=20m` 같은 게 자주 뜸** → PnP가 keypoint detection 결과로부터 비현실적 거리를 푼 것 = false positive. gate가 잘 막아내는 정상 동작.

**Q. 카메라 intrinsic** → 시퀀스 폴더에 `cam_K.txt`가 있으면 그걸 자동 로드. 없으면 `task.yaml`의 `camera.fx/fy/cx/cy` 사용.
