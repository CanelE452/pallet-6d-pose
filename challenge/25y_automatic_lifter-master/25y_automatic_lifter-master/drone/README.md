# Drone-Based Lifter Tracking System

본 프로젝트는 DJI Tello 드론 카메라 영상을 기반으로 리프터 객체를 실시간 탐지하고,
YOLO 기반 탐지와 DeepSORT 또는 ByteTrack 기반 추적을 통해
드론이 대상 객체를 자동으로 따라가는 기능을 수행한다.

### 프로젝트 수행 인원
- 김희지

### 주요 기능
1. 객체 탐지 (YOLO) 
- 리프터(lifter) 및 선택적 클래스(person) 탐지
- 640x480 영상 입력 기반

2. 실시간 추적
- DeepSORT: Kalman Filter + Re-ID 기반 안정적 ID 유지
- ByteTrack: 단일 클래스 중심 환경에서 높은 FPS \
 빠른 매칭 알고리즘 기반 실시간 추적

3. 드론 자동 제어 (RC Control)
- bbox 중심(cx, cy)을 기준으로 드론 방향(yaw) 조정
- 객체 크기(height, width)를 이용한 전·후진(fb) 결정
- deadzone, EMA smoothing 적용으로 과제동 방지
- 탐지 상실 시 RC 값 0 유지(안전 정지)

### 하이퍼파라미터 요약

설정:
- CONF_THRESHOLD = 0.4

- CENTER_MARGIN = 90

- SMOOTH_ALPHA = 0.85

- RC 제한값: yaw ±40 / fb ±30

- ByteTrack 설정:

- track_buffer = 30

- match_thresh = 0.8

- frame_rate = 30