# PURPOSE — direct 3DoF pallet pose track

## [소비처]
forklift alignment / approach logic (`25y_automatic_lifter-master/depth_cam/calib/fsm*`).
이 값이 들어가는 곳은 6DoF 포즈 소비자가 아니라 **정렬 제어기** 하나뿐이다.
따라서 `y / roll / pitch` 는 산출물이 아니다.

## [문장]
> RGB 한 장에서 Jetson Orin Nano 실시간으로 `x, z, yaw` 를 직접 내는 모델이,
> 기존 `keypoint → PnP → adapter` 경로만큼 정확하면서 프레임간 흔들림은 더 작다.

이 문장을 지지하지 못하면 트랙을 접는다.

## Baseline (건드리지 않는다)
```
RGB → YOLO26n pose → bbox + 9 kp → PnP(IPPE/ITERATIVE) → R,t → x, z, yaw
```
경로: `challenge/yolo_pose_one_model/` + lifter `depth_cam/calib/geometry.py`.
이 트랙은 baseline 을 **읽기만** 한다. 파일을 고치거나 체크포인트를 덮어쓰지 않는다.

## Candidate
```
RGB → YOLO26n shared backbone/neck ─┬─ detection head → bbox
                                    └─ direct 3DoF head → x_norm, z_norm, sin4ψ, cos4ψ
```

### 가설 A — `direct3dof`
불필요한 자유도(`y/roll/pitch`)를 없애고 task output 을 직접 학습하면,
프레임별 PnP 해가 흔들리며 생기는 jitter 가 줄어든다.

### 가설 B — `direct3dof_auxkp`
real 이 100여 장뿐이라 direct head 가 과적합하기 쉽다.
synthetic 으로 배운 9-keypoint geometry 감독을 auxiliary 로 남기면
regularizer 로 작동한다. 배포 시 kp branch 는 쓰지 않는다.

> 이번 단계에서 A/B 우열을 결론내지 않는다. 같은 조건으로 비교 가능하게만 만든다.

## 예상 결과
- synthetic pretrained backbone/neck 을 대부분 재사용 가능
- 출력이 `x,z,yaw` 에 직접 대응
- regression head 추가 비용은 작다
- jitter 감소 가능성 (미검증)

## 실패 조건 — 하나라도 해당하면 direct 방식이 목적을 지지하지 못한 것
1. independent GT 기준 x/z/yaw 정확도가 kp→PnP 보다 나쁘다
2. static sequence jitter 가 줄지 않는다
3. 거리/세션 일반화가 크게 악화된다
4. Jetson latency 가 실질적으로 악화된다
5. direct z 가 train distance 분포를 외운다 (held-out 거리 구간에서 붕괴)

## 이번 작업에서 하지 않는 것
- 장시간 학습 (smoke 는 ≤32장 / ≤50 step)
- depth 사용 (RGB only, RealSense 는 RGB 취득 장치로만)
- model prediction 을 real 의 supervised GT 로 사용

## 계약 근거
좌표계·대칭·단위는 `3DOF_CONTRACT.md` 가 유일한 출처다. 여기 복사해 두지 않는다.
