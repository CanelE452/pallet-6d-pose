Pallet Pickup FSM Birdview Simulator
====================================

FoundationPose 프로젝트용 시뮬레이터.
바닥에 놓인 팔레트를 포크리프트가 DOPE detection으로 인식하고 픽업하는 FSM을 검증한다.

실행 방법
1. 가장 간단한 방법: run_simulator.bat 실행
   - 기본 브라우저에서 index.html을 바로 엽니다.
   - JSON 파일 드롭은 동작합니다 (file:// 에서도 FileReader 사용).

2. 로컬 서버 방식: run_local_server.bat 실행
   - http://127.0.0.1:8765/ 로 실행됩니다.
   - PowerShell만 있으면 됩니다.
   - 종료하려면 서버 창에서 Ctrl+C를 누르세요.

장면 구성
- 바닥에 놓인 팔레트 1개 (트럭 없음)
- 팔레트 규격: 1.10 (W) x 1.30 (D) x 0.12 m  ← scan_cleanup/pallet_full.obj 일치
- 팔레트 pose 는 centroid + yaw 로 표현 (DOPE 6D pose 출력 그대로)
- 포크리프트 (소형 창고용): width 0.92m, body 1.38m, fork 1.05m, fork gap 0.30m
- 포크리프트 정면 0.45m 위치에 RealSense D435i 마운트 (FOV 87 deg, 0.30-3.5 m)

FSM 상태 (PDF "리프터_한글.pdf" 그림 11 + 25y_automatic_lifter align.py snapshot-based)
  START → SEARCH (det 대기)
        → YAW_CHECK (snapshot psi, d_lat)
             |psi| > YAW_TOL → YAW_CORRECT_RIGHT/LEFT (IMU rel_yaw)
                              → OFFSET_CHECK
             |psi|≤tol ∧ |d_lat|≤OFF_TOL → READY_TO_INSERT
             |psi|≤tol ∧ |d_lat|>tol → OFFSET_CHECK
        → OFFSET_CHECK → DIST_CHECK 또는 YAW_CHECK loop
        → DIST_CHECK
             |d_lat|>tol         → LATERAL chain
             d_fwd-ALIGN>BAND   → ALIGN_FWD_ADJUST (timer)
             ALIGN-d_fwd>BAND   → ALIGN_BWD_ADJUST (timer)
             else                → YAW_CHECK
        → LATERAL chain
             LATERAL_ROTATE_RIGHT/LEFT (90°, IMU)
              → FORWARD_AFTER_RIGHT/LEFT (timer t(|d_lat|))
              → LATERAL_ROTATE_LEFT_BACK/RIGHT_BACK (LATERAL_BACK_YAW)
              → YAW_CHECK
        → READY_TO_INSERT → INSERT (timer (d_fwd+POCKET)/slow) → DONE
  STOP_INTERLOCK: 모든 cmd 전환 사이 0.25 초 정지

원칙: "한 번 보고 눈 감고 동작"
- *_CHECK 진입 시 detection 새로 sample → snapshot (psi, d_lat, d_fwd) 캡쳐
- snapshot 으로 다음 cmd plan (회전각/전진시간) 사전 compute
- cmd 진행 중 (YAW_CORRECT, *_ADJUST, LATERAL chain, INSERT) perception 무시.
  IMU (fork.yaw 변화) 또는 timer 만 보고 종료 판정.
- cmd 완료 → STOP_INTERLOCK → 다음 *_CHECK 진입 → 새 snapshot.

좌표 매핑 (시뮬 birdview → PDF camera frame)
  psi_pallet > 0 (fork.yaw 가 target 보다 CCW) → ROT_RIGHT (fork.yaw 감소)
  psi_pallet < 0                               → ROT_LEFT  (fork.yaw 증가)
  dLateral > 0 (forklift 가 entry face +t 측)  → LATERAL_ROTATE_RIGHT 먼저
  dLateral < 0                                 → LATERAL_ROTATE_LEFT  먼저
  IMU rel_yaw = fork.yaw - ref_yaw (snapshot 시점)
  t(d) = d / moveSpeed (PDF 의 piecewise 모델 대용, 상수 미정)

Detection 모드
1. 합성 노이즈 (기본)
   - "Sample noisy detection" 버튼으로 yaw / x-offset / d-forward / reproj 노이즈 샘플
   - sigma 값과 reproj base는 Detection Model 패널에서 조절
   - reproj_error <= reprojAccept 이면 accept, 초과하면 reject

2. DOPE JSON 드롭
   - Detection Model 패널의 "Drop DOPE detection JSON" 영역에 .json 파일을 드래그
   - 또는 영역을 클릭해서 파일 선택
   - 스키마는 sample_detection.json 참고

좌표 정의
- pallet.x, pallet.y 는 cuboid 무게중심(centroid) 의 world 좌표.
  DOPE 6D pose translation 을 그대로 넣으면 됨.
- entry face 중심 C 는 시뮬레이터 내부에서 centroid + n*depth/2 로 유도.
- pallet.yaw 는 팔레트 tangent(=긴 옆면) 방향. yaw=0 이면 +x 방향.
- FSM 의 d_forward / d_lateral 은 entry face C 기준으로 계산.

JSON 스키마
{
  "pallet": {
    "x": 0.04,           // pallet centroid X, meters (world frame)
    "y": 2.27,           // pallet centroid Y, meters (world frame)
    "yaw": 2.5,          // degrees
    "reproj_error": 2.3  // mean projected_cuboid reproj error (px)
  }
}

호환 스키마 (DOPE 원본 형식 일부 지원)
{
  "objects": [{
    "class": "pallet",
    "x": 0.04, "y": 2.27, "yaw": 2.5,   // centroid X, Y (world frame)
    "projected_cuboid_error": 2.3
  }]
}

FOV check
- 카메라 마운트 기준 팔레트가 horizontal FOV 87 deg + range 0.30-3.5 m 안에 있으면 in FOV
- DETECTION 상태에서 out of FOV 또는 reproj > 임계값이면 자동 재시도

조절 가능한 파라미터
- Move m/s, Slow m/s: 일반/저속 이동 속도
- Rotate deg/s: 회전 속도
- Stand-off m: 팔레트 앞 정지 거리
- Reverse sec: 픽업 후 후진 시간 (후진 거리 = Move m/s * Reverse sec)
- Yaw / X-offset / D-forward sigma: 합성 detection 노이즈
- Reproj accept px: detection accept 임계값
- d_lat tol, psi tol: 정렬 종료 조건

DOPE-style psi 검증 (실차 시뮬)
- Detection Model 패널의 "Use DOPE-converted psi for FSM" 토글로 활성화.
- 이 모드에서는 depth_cam/calib/pose6d_adapter.py 의 변환식을 시뮬에 그대로
  reproduce 한 ψ/d_lat/d_fwd 가 FSM 입력 (snapshot) 으로 들어간다.
- 변환 단계:
    1) world 의 (forklift, camera, pallet) 으로부터 가상 3D pose 생성.
       camera frame: OpenCV (+X right, +Y down, +Z forward = 광축).
       pallet model: v4 (+X width, +Y height, +Z entry-face forward).
    2) R_cam_pallet = R_world_camera^T · R_world_pallet,
       t_cam_pallet = R_world_camera^T · (t_world_pallet − t_world_camera).
    3) ψ = wrap_to_180( atan2(R[0,2], R[2,2]) ° + 180 ).
    4) entry-face anchor: d_lat = t[0] + R[0,2]·depth/2,
                          d_fwd = t[2] + R[2,2]·depth/2.
- HUD 의 State Variables 에 "psi (geometric)" 과 "psi (DOPE-style)" 가 매 frame
  같이 표시됨. 부호 일치 / 반전 여부를 시각으로 비교 가능.
- 검증 결과 (현재 코드): 두 ψ 가 모든 자세에서 정확히 일치.
    pallet yaw=+5°  → 두 ψ 모두 −5°  → ψ>0 케이스 아님
    pallet yaw=−5°  → 두 ψ 모두 +5°  → ROT_RIGHT (yaw 감소) → ψ→0 ✓
    fork yaw=+95°   → 두 ψ 모두 +5°  → ROT_RIGHT → ψ→0 ✓
    fork yaw=+85°   → 두 ψ 모두 −5°  → ROT_LEFT  → ψ→0 ✓
  align.py 분기 (yaw>0 → ROT_RIGHT, yaw<0 → ROT_LEFT) 와 부호 일치.
- 결론: pose6d_adapter.py 의 `+180° shift + wrap` fix 는 옳음. 시뮬에서도
  같은 변환을 거쳐야 정렬 완료 = 0° convention 이 성립.

포함 파일
- index.html
- styles.css
- sim.js
- simulator_preview.png  (기존 트럭 시나리오 캡처, 참고용)
- sample_detection.json  (DOPE JSON 예시)
- run_simulator.bat
- run_local_server.bat
- serve.ps1
- _original/  (트럭 시나리오 원본 백업)
