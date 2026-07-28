# 📦 Lifter Calibration (RGB-D Camera Task)

본 프로젝트는 **RGB-D 카메라(RealSense) 기반 파렛트 전면부 인식 및 리프터 정렬**을 위한 파이프라인 구현을 다룹니다.  
YOLO segmentation, 3D point cloud 구성, RANSAC 기반 평면 추정 등을 통해 주요 변수(**yaw, offset, distance, width**)를 계산하고 HUD에 표시합니다.

---

## 👥 프로젝트 수행 인원

- 김지훈  
- 오수진  
- 조연경  

---

## 🚀 실행 방법

홈 디렉터리에서 다음 명령어를 실행하세요:

```bash
python main.py
```
---

## 📂 파일 구조

```bash
depth_cam/
📁 calib/              보정/제어 핵심 모듈 모음
  📄 config.py         파라미터/임계값/색상 정의
  📄 control.py        저수준 동작 바인딩(전진/회전/후진/정지 등)
  📄 fsm.py            상태기계. 명령 결정·검수·유지 로직
  📄 geometry.py       인지 결과 산출·보정
  📄 perception.py     인지 결과 산출·보정
  📄 utils.py          공통 유틸
  📄 hud.py            HUD 렌더링
📁 data/
📁 docs/
📁 runs/
📄 main.py             파이프라인 실행 엔트리
📄 README.md
📄 requirements.txt
📄 save_rgbd.py        학습용 데이터 수집 코드
```

---

## 📊 주요 기능

- **RGB-D 입력 처리**
  - RealSense 카메라로부터 RGB + Depth 영상 입력
  - YOLO segmentation으로 파렛트 앞면(`front`) 탐지

- **3D 포인트 클라우드 구성**
  - segmentation mask 영역 내 depth 값 추출
  - 파렛트 전면부에 대한 point cloud 생성

- **RANSAC 기반 평면 적합**
  - 전면부 평면 모델: `aX + bY + c = Z`
  - 평면 수직 벡터를 이용해 파렛트 **yaw** 추정

- **주요 변수 계산**
  - **Yaw**: 파렛트 바라보는 방향
  - **Offset**: 파렛트 중심과 리프터 간 x축 차이
  - **Distance**: 카메라-파렛트 간 z축 거리
  - **Width**: 탐지된 파렛트의 가로 길이  
    → **1m 이상**일 경우, RGB-D 영상 내에서 **파렛트 전체가 정상적으로 탐지됨**을 의미

- **EMA 기반 smoothing**
  - `yaw_smooth`, `offset_smooth`, `dist_z_smooth`
  - 노이즈 완화 및 안정적 제어 가능

---
## ⚙ 설정 요약 (`calib/config.py`)

- (감지/허용치)
- **CONF_THR**: 0.30
- **YAW_TOL_DEG**: 2.0°
- **OFF_TOL_M**: 0.20 m
- **ALIGN_DIST_M**: 2.0 m
- **ALIGN_BAND_M**: 0.05 m
- **BACKOFF_MARGIN_M**: 1.00 m

- (검수/유지 파라미터)
- **CMD_STABLE_THR**: (config.py에서 정의; 값은 프로젝트 설정 사용) 프레임  ← 일반 명령(전진/회전/후진/정지)
- **OK_FRAMES_THR_ALIGN**: (config.py에서 정의; 값은 프로젝트 설정 사용) 프레임  ← 정렬 OK
- **HOLD_SEC**: (config.py에서 정의; 값은 프로젝트 설정 사용) s  ← 확정 명령 유지 시간
- **WIDTH_MIN_FULL**: (config.py에서 정의; 값은 프로젝트 설정 사용)  ← 폭 부족 판단( `width_smooth < WIDTH_MIN_FULL` )

> 실제 값은 `calib/config.py`를 기준으로 동작.

---
## 🔁 FSM 동작

### 공통
- **검수**: 같은 후보가 연속 N프레임 발생해야 확정
  - 정렬 OK: **10프레임**
  - 그 외 모든 명령: **5프레임**
- **유지**: 확정 명령은 **2초간** 동일 명령 유지(센서 변화 무시)
- **HUD**: 수치 미표기. 예) `[대기: 전진+우회전] [3/5]`, `[유지: 제자리 좌회전] [1.2/2.0s]`

### SEARCH
- 입력: `det_ok=False` **연속 5프레임**
- 출력: **정지(STOP)** 확정 → 2초 유지
- HUD: `[대기: 정지] [k/5]` → `[SEARCH] 정지`

### APPROACH (원거리: `dist_z > ALIGN_DIST_M + ALIGN_BAND_M`)
- 폭 부족: `width_smooth < WIDTH_MIN_FULL` → **제자리 회전**
  - `offset_x>0` → `ROT_RIGHT`
  - `offset_x<0` → `ROT_LEFT`
  - 검수 5프레임 → 확정 시 2초 유지
- 폭 충분: **전진/전진+회전**
  - `|yaw| ≤ YAW_TOL_DEG` → `FWD`
  - `yaw>+tol` → `FWD_RIGHT`
  - `yaw<-tol` → `FWD_LEFT`
  - `yaw=None` & `|offset_x|>OFF_TOL_M` → offset 부호로 좌/우, 아니면 `FWD`
  - 검수 5프레임 → 확정 시 2초 유지

### AT_2M_ALIGN_YAW (정렬대역: `ALIGN_DIST_M - ALIGN_BAND_M < dist_z ≤ ALIGN_DIST_M + ALIGN_BAND_M`)
- 폭 부족: `width_smooth < WIDTH_MIN_FULL` → offset 부호로 **제자리 회전**(5프레임 → 2초 유지)
- yaw 보정: `yaw=None` 또는 `|yaw| > YAW_TOL_DEG` → yaw 부호로 **제자리 회전**(없으면 `last_dir`)
- offset 정보 없음: HUD `[ALIGN @2m] offset_x N/A`
- offset OK: `|offset_x| ≤ OFF_TOL_M` → HUD `[STABILIZE] OK 유지`
- offset 과대: **BACKOFF** 후보(5프레임 → 2초 유지)
- 더 멀어짐: `dist_z > ALIGN_DIST_M + ALIGN_BAND_M` → 조용히 **APPROACH** 복귀

### 과근접 (`dist_z ≤ ALIGN_DIST_M`)
- 조건: `|yaw| ≤ YAW_TOL_DEG` **AND** `|offset_x| ≤ OFF_TOL_M`
- 결과: **OK 카운트 10프레임** → `[정렬 완료]`
- 불만족: **BACKOFF** 후보(5프레임 → 2초 유지)

### BACKOFF
- 항상: **BACKOFF** 후보를 5프레임 검수 → 확정 시 2초 유지
- 자동 복귀: 구현 여부는 프로젝트 사양 (필요 시 거리 조건으로 FSM 전이 추가 권장)

---

## 🧭 “측정값 → 명령” 요약표

- 미탐지 5프레임 → **STOP**(2초 유지)
- 원거리 & 폭 부족 → **ROT_(offset 부호)**(2초 유지)
- 원거리 & 폭 충분 → **FWD / FWD_RIGHT / FWD_LEFT**(yaw/offset 규칙, 2초 유지)
- 정렬대역 & 폭 부족 → **ROT_(offset 부호)**(2초 유지)
- 정렬대역 & yaw 보정 필요 → **ROT_(yaw 부호)**(2초 유지)
- 과근접 & (yaw/offset OK 10프레임) → **정렬 완료(DONE)**
- 과근접 & 조건 불만족 → **BACKOFF**(2초 유지)

---

## ▶ 실행
- `python main.py` : 전체 파이프라인 실행(프로젝트 런처에 맞게 수정 가능)
- `python save_rgbd.py` : RGB‑D 저장 유틸

---

## 커스터마이징 팁
- 임계값/정책은 `calib/config.py`에서 조정
- 장비 제어는 `calib/control.py`에 매핑
- BACKOFF 복귀 로직(거리 조건 등) 필요 시 `fsm.py`에 추가

---

## 📦 설치 방법

```bash
pip install -r requirements.txt
```

---

## 🎯 데이터셋

- 공개 데이터셋을 사용하여 학습  
- [Pallet Dataset (Roboflow)](https://universe.roboflow.com/www-i7p4n/pallet-5esjz/dataset/3)

---

## 📹 시연

- 실제 동작 사진

![Demo Image](docs/demo.png)
