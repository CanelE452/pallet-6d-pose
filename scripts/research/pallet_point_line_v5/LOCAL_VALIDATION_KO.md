# 작성 환경에서 실제 확인한 범위

## 최종 검증

- **72 tests passed, failures0/errors0/skipped0.** 원본 JUnit: `evidence/pytest.xml`.
- **생성 RGB 검사에서 3개 arm ×8회 =24회의 실제 AdamW 업데이트.** backbone/ROI adapter/새 head의 가중치가 모두 변했다. `evidence/rgb_demo_final/DEMO.json`.
- 별도 cache CLI 통합 테스트에서 **3개 arm ×2회 =6회의 실제 업데이트**, 동일 shared 초기값/배치계획, checkpoint 저장·strict load·예측·오프라인 평가를 확인했다.
- 통합 테스트 중 **GT 파일을 실제로 삭제한 상태에서 세 arm을 모두 추론**했고, target loader 읽기 수0·observation 읽기 수3이 실제 카운터와 일치했다. 종료 뒤 테스트 fixture만 복구했다. 운영 syscall 전체를 감시한 실험으로 확대하지 않는다.
- `compileall`과 실제 train/score/evaluate/compare `--help`를 실행했다.

전체 기계 판정: `evidence/LOCAL_VALIDATION.json`.

## 의미 있는 테스트

1. 모든6자유도 투영 Jacobian을 finite difference와 비교. 회전 derivative가 RX가 아닌 RX+t를 잘못 쓰면 실패한다.
2. solver의 최종 depth가 관측점에 대해 가진 derivative를 finite difference로 재확인.
3. **line CE·point loss·ranking·seed auxiliary를 빼고 최종 translation loss만 사용해도 Hough 파라미터와 input feature의 gradient가 nonzero**인지 확인.
4. 정답에 가까운 시작점에서 point-only / line-only / point+line이 analytic pose를 복구하는 generated geometry 테스트. 이것은 global image-based recovery 성능이 아니다.
5. line_weight0의 point-only parity, 같은 선의 양끝점이 서로 다른 mode를 골라 가짜 0오차를 만드는 사례 방지, line 부호/mode순서 불변성.
6. C2/C4 전체 손실 불변성, 하나의 g를 점·선·pose에 공유, 중심고정, 직사각 C4/미검증 square C4 거부.
7. affine 왕복만이 아니라 **pixel center 값이 알려진 실제 ramp feature**로 비등방 stride/offset의 샘플링 위치를 확인. bilinear footprint의 invalid support 검사.
8. theta seam의 rho 반전, DHT constant-preserving normalization과 line-local gradient, 멀리 떨어진 mode를 평균하지 않는 decoder 검사.
9. GT extra field/SHA 변경/중복 source image split/금지 FINAL split 거부.
10. pose missing을1의 실패로 primary 분모에 포함, 수작업3–4–5 거리 지표 검산, session paired bootstrap 상수 fixture.

## 기본 설정의 새 head 파라미터 수

|군|학습 가능 파라미터|
|---|---:|
|point|24,927|
|direct line|26,879|
|DHT line|32,683|

backbone/ROI adapter는 제외한 수다. 작은 생성 demo의 설정은 기본보다 작으며 별도 값이 기록돼 있다. 세 군의 규모가 완전히 같다고 주장하지 않는다.

## 개발 중 발견해 고친 결함

초기 DHT constant-input 검사에서 48개 중1개가 실패했다. 거의 정수인 bin 위치의 부동소수점 잔여 때문에 매우 작은 가짜 mass가 생겼다. 허용오차를 키우는 대신, bin 위치가 정수에서1e-10 이내면 정확한 정수로 정리한 뒤 interpolation하도록 수정했다. 최종72개 검사가 통과했다. 초기 RGB demo 기록도 `evidence/development/`에 보존했다.

## 하지 않은 것

실제 팔레트 학습, 원래 YOLO checkpoint load/forward, 실제 source 이미지/feature alignment 인증, 원본 asset 대칭 측정, real DEV 평가, GPU/Jetson speed/export, 독립 FINAL, canonical metric parity는 하지 않았다.

생성 fixture의 ROI는 GT 투영에서 만든 진단용이다. 실제 팔레트 pipeline에서는 예측 ROI만 허용한다. 생성 데이터 손실은 수렴/일반화의 근거가 아니며 서로 다른 step의 다른 batch loss를 성능 비교로 쓰지 않는다.

환경: Python3.13.5, PyTorch2.10.0+cpu, NumPy2.3.5, OpenCV4.13.0. CUDA 미사용. 정확한 환경은 JSON에 남겼다.
