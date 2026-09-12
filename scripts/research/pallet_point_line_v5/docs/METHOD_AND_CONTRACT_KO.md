# 방법·데이터·해석 계약

## 1. 목적과 이번 변경의 위치

최상위 목표는 단안 RGB 팔레트 인식의 실제 위치/자세 정확도와 안정성이다. DHT 채택은 하위 선택이다. 치수·유효 대칭을 제공한 강한 점 모델이 더 좋으면 그 결과를 인정한다.

|목적|행동의 이유 두 가지|실행 조건|기대와 반대일 때|
|---|---|---|---|
|불필요한 번호 경쟁 제거|동등 상태에 모순된 감독을 주지 않기 / 점·선·자세의 배치 공유|실제 asset 대칭 확인, object frame 매핑, 동일 g의 모든 마스크|해당 사례가 C1/C2인데 90도 혼동이면 C4로 면제하지 않음|
|선의 추가 정보 확인|점이 약한 곳의 방향/수직 위치 보완 / 선 정보와 readout 용량 분리|P/D/H 모두 치수·초기상태·population·예산 일치|H 미개선이면 Hough를 성공으로 포장하지 않음|
|큰 오배치에서 접근 가능성 확보|점 PnP가 없을 때도 시작 / 단일 국소 basin에 갇힘을 진단|영상 기반 다중 시작점, GT-free 선택, 전체 시작점 기록|정답선도 못 쓰면 네트워크 추가 학습보다 디코더/정의 진단|
|정상 출력 보호|평균에 숨은 대형 손상 확인 / 결측 삭제로 허위 향상 방지|pose validity와 실패 분모, 새 catastrophic frame 수 보고|손상률만 작다고 안전하다고 하지 않음|

제외한 것: 새 렌더 대량 생성, 자유 Hungarian으로 순열 모두 허용, 모든 팔레트 C4 선언, PCGrad/계수 grid 재탐색, 여러 설계를 실사에서 고른 뒤 가장 좋은 것만 보고, 기존 실패 JSON 수정.

후보는 (A) 기존 모델에서 대칭/치수 감독만 정리, (B) 이 패키지의 물리 자세 코어, (C) 진입면/정렬 상태를 직접 출력하는 별도 과제다. 이번 산출물은 B의 소프트웨어와 A를 포함해야 할 비교 계획이다. C는 과제 출력 정의까지 바뀌므로 이번 실행에서 제외한다.

## 2. 새 좌표계

**이 키포인트 번호는 원 저장소의 camera_dynamic 번호와 같다고 가정하지 않는다.**

물체 중심=원점, +x=가로(width), +y=위(height), +z=깊이(depth). 치수 입력 순서는 `(W,D,H)` metres. 카메라는 OpenCV right/down/forward이고 `X_camera = R @ X_object + t`다. 왜곡 없는 이미지 픽셀 중심 좌표를 쓴다.

0..7은 `geometry.SIGNS`의 순서이며 8은 3D 중심이다. 각 좌표는 `(±W/2, ±H/2, ±D/2)`다. 카메라를 향한 앞면을 매번 새로 0..3으로 고르는 규칙이 아니다. 이 차이를 학습 GT 변환에서 해결하지 않으면 실험을 시작하지 않는다.

기존 물체 프레임과 새 프레임이 같은 중심이고 `X_old=A X_new`면 `R_new=R_old A, t_new=t_old`. A는 적절한 회전이어야 한다. 중심 원점도 다르면 회전만 곱하면 안 되며 translation도 별도로 바뀐다. 제공 `reframe_pose`는 **같은 중심의 회전만** 지원하므로 중심 변경을 이 함수로 몰래 처리하지 않는다.

훈련 레이블의 camera-facing→object frame 변환에 renderer metadata를 쓰는 것은 허용된다. 추론 중 예측 키포인트를 GT permutation으로 바꾸는 것은 금지다. 입력 W/D를 매 프레임 GT facing으로 교환해 주는 것도 금지다. 실제 배포에서 알 수 있는 물체 제원 축을 고정한다.

## 3. 대칭

- C1: identity. 대칭 미확인은 C1이며 “비대칭을 증명함”을 뜻하지 않는다.
- C2: 확인된 높이축 180도 동등성.
- C4: 확인된 높이축 90도 동등성과 정사각 기하를 모두 만족.
- 정사각 tolerance `1e-6m`는 저장된 값의 산술 검사이며 물리적 동등성을 판정하는 오차 한도가 아니다.
- 현재 구현은 upright yaw C1/C2/C4만 지원한다. 뒤집기·반사·위아래 교환·연속대칭은 지원하지 않는다.
- 직사각형의 `(W,D)` 교환과 프레임 90도 재정의는 물리 C4와 다르며 자동 적용하지 않는다.
- `SymmetrySpec.verified/basis`는 코드가 물리 대칭을 측정했다는 뜻이 아니다. 로컬 reviewer가 원본 CAD/렌더/작업 계약으로 확인해야 하는 외부 증거다.

손실은 물체마다 `min_g [point_g + line_g + pose_g + seed_g + ranking_g]`다. 점마다 g를 따로 고르지 않는다. 선 역할과 유효성 마스크도 같은 permutation에서 나온다. 8번 중심은 모든 g에서 동일하다.

추론에는 `Supervision`이나 g oracle이 들어가지 않는다. 모델이 예측한 point/line/pose를 evidence objective로 선택한다. 추론 출력이 C2/C4의 어느 대표인가와 로봇이 접근할 진입면 선택은 다른 문제다. 후자를 모델 index0..3에 고정해서는 안 된다.

## 4. 신경망

1. 검출된 instance의 native P3/P4에서 24×24 ROI를 샘플링한다. P4 업샘플을 P3로 부르지 않는다. 정확한 stride 및 feature index0의 input pixel center가 필요하다.
2. `raw_to_feature`와 content support로 샘플링한다. 함수는 full bilinear footprint가 유효하지 않은 query를 버린다. backbone 수용영역의 반사패딩 영향까지 없애지는 못한다.
3. 공통 영상 CNN과 치수/camera conditioning을 사용한다. `condition_dims=False`는 신경망 conditioning에서만 치수를 빼며, geometry와 metric-scale depth seed에는 여전히 치수를 쓴다. 치수 입력 효과만 분리하기 위한 설정이다.
4. 점 head는 heatmap 평균 + 무제한 residual로 9개 2D 관측을 만든다. residual 덕분에 가림/화면 밖 amodal 코너가 ROI 밖에 놓일 수 있다. 이 새 head는 원래 YOLO의 RLE head가 아니다.
5. `direct`는 global role-query descriptor로 line lattice를 점수화한다. `hough`는 실제 선에 따른 특징 voting 후 Hough convolution을 쓴다. `point`에는 line head가 없다.
6. line distribution을 바로 하나의 전역 soft-argmax로 평균하지 않는다. 3개의 분리된 mode를 찾고 각 mode 근방에서만 연속 평균한다. theta seam은 rho 부호를 반전한다. 보존한 mode의 정규화 질량은 전체 후보 중의 **조건부** 질량이지 정답 확률이 아니다.
7. 영상 특징+치수+K로 복수 초기 자세를 예측한다. 점검출/PnP 성공이 필요한 경로가 아니다. 그러나 학습된 초기값의 global coverage·mode collapse를 아직 검증하지 않았다.
8. 모든 시작점을 공동 point-line solver로 정제하고 관측 에너지와 수치 유효성만으로 최종 한 자세를 고른다. 회전들을 평균내어 비정상 회전을 만들지 않는다.

24×24 특징, theta36/rho65는 CPU에 적합한 **미검증 파일럿 기본값**이다. theta bin은 normalized ROI 좌표에서 5도이고 raw 이미지의 5도와 같지 않을 수 있다. 원영상 정밀도에 적합한지 학습 전 GT→격자→decode oracle ceiling을 따로 확인한다. 실패한 뒤 실사 결과를 보고 격자를 바꾸지 않는다.

## 5. 공동 기하

`Exp(dr)@R, t+dt`의 회전 Jacobian은 `-skew(RX)`이다. `-skew(RX+t)`가 아니다. 코드의 수치미분 테스트가 이 차이를 잡는다.

한 edge의 양 끝점은 같은 line mode의 mixture likelihood를 공유한다. 각 endpoint가 독립적으로 다른 선을 골라 오차0을 만들 수 없도록 한다. 선은 서로 다른 역할 사이에서 자유 매칭하지 않는다.

고정 관측 mask에서 Huber point residual과 line mixture를 합한다. IRLS/EM은 국소 quadratic surrogate이고, line search는 실제 robust mixture objective의 감소를 확인한다. 자세에 따라 사용하는 edge 수를 갑자기 바꾸지 않는다. 잘못된 깊이·비유한 연산은 후보 수락에서 배제한다.

`information_rank`는 수치적 관측 가능성 진단이다. 성공 확률·정확도 인증·보정된 posterior covariance가 아니다. `pose_valid=False`의 좌표는 선택0의 placeholder일 수 있으므로 로봇 제어에 보내면 안 된다.

center8은 point-head 감독에 포함되지만 solver의 독립적 3D 관측 개수에는 추가하지 않는다. 출력 center는 같은 R,t가 투영한 3D 중심이다.

## 6. 손실과 전역 한계

- Point SmoothL1: 원본 이미지 대각선으로 정규화한 2D 오차.
- Line soft CE: 같은 symmetry의 edge 양 끝점이 각 line grid에 주는 거리. GT support와 ROI 교차만 확인하며 물리적 가시성으로 해석하지 않는다.
- Pose: 각 후보 자세가 만든 8개의 metric 3D corner 오차를 object diagonal로 정규화. 후보 자세를 평균하지 않고 후보 **오차**를 evidence-based 확률로 가중한다.
- Seed: 학습 중 가장 가까운 초기 pose branch를 감독하는 best-of loss. inference에 GT branch를 주지 않는다.
- Ranking: 각 후보의 GT 오차로 학습용 목표 분포를 만들고 evidence energy가 이를 구분하도록 학습한다. 이것은 EPro-PnP의 연속 SE(3) 분포/KL 구현이 아니다.

원래 점/선 관측과 solver가 함께 잘못된 해를 좋아할 수 있다. GT loss가 그 위험을 줄일 수는 있지만 제거를 보장하지 않는다. implicit-independent uncertainty 가정의 정확성도 증명하지 않았다. mode/start 수·loss coefficient·fixed sigma는 모두 미검증 설정이며 실사로 튜닝하지 않는다.

## 7. 기존 구현과의 차이/반복 방지

기존 `dimension_guided_graph_pose.py`도 치수+점+선을 사용했다. 새로움은 그 제목이 아니라 (i) 영상 기반 시작점으로 point-PnP 실패에도 접근 가능하게 하는 경로, (ii) 학습 그래프 내 refinement와 최종 pose loss, (iii) 하나의 대칭을 모든 감독 항이 공유하도록 하는 실제 구현을 검증하는 데 있다. 이 항목들 역시 학술 novelty가 검증됐다는 선언이 아니다.

기존 structured-v2는 이미 endpoint/segment/전체배치 scorer였다. 그 재현을 “새로 선을 넣었다”고 하지 않는다. 기존 joint-v1은 이미 line feedback을 point head에 전달했다. 이번에는 feature feedback을 또 붙이는 대신 최종 physical state를 명시했다.

## 8. 최소 결과 해석

기하 테스트 성공 → 구현 조건 만족이지 line utility 증거 아님.
GT선으로 solver 개선 → 특정 starts/geometry의 기하 가능성이지 예측선 성공 아님.
H vs P 개선 → 추가 line branch/감독의 방법 수준 이득, Hough만의 효과 아님.
H vs D 개선 → 이 matched-data 전체 readout 비교에서 DHT 쪽이 좋음, params/연산 차이 단독 원인 미분리.
대칭 재평가 개선 → 적절한 동일성 정의 효과일 수 있음, 학습 정확도 증가와 구별.
고정 cache head 개선 → online 전체 모델 개선 아님.
재사용 DEV 개선 → 독립 unseen-session 일반화 아님.
