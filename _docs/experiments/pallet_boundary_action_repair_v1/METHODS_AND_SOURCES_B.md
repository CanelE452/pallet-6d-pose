# B 기하 수리: 읽은 원문·설치 동작과 실제 구현의 연결

이 문서는 알고리즘의 출처와 차이를 설명한다. 최종 자세 성능·채택 여부는 별도 전체 평가표를 따른다. 2026-10-18에 아래 공식 원문과 설치 소스를 실제 열람했다.

| 출처 | 실제 확인한 내용 | 이번 적용과 차이 |
|---|---|---|
| [SRT3D, IJCV 2022 §3.4 Eq.(30)](https://link.springer.com/article/10.1007/s11263-022-01579-8) | 회전·이동별 Tikhonov 행렬을 더한 Newton 갱신, 정보가 약한 방향의 안정화 | B_FINITE_LOCAL은 N0를 고정 기준으로 하는 누적 geodesic 회전/중심 이동 비용이다. 논문의 매 반복 증분 갱신식, tracker, temporal appearance update를 재현하지 않는다. |
| [SRT3D §5.4](https://link.springer.com/article/10.1007/s11263-022-01579-8) | 정확한 강체 모델·색 분리·초기화와 부분 가림에 따른 제약 | 단일 영상 virtual cuboid/profile의 실제 material owner는 미인증이다. 유한 선분 지지는 이 물리 대응 가정을 인증하지 않는다. |
| [SciPy 1.12 SLSQP 공식 문서](https://docs.scipy.org/doc/scipy-1.12.0/reference/optimize.minimize-slsqp.html)와 설치 `_slsqp_py._minimize_slsqp` | maxiter·ftol, analytic jac 전달, optimizer 종료 상태 | 기존 원 실험의 80 iteration/ftol1e-8을 유지한다. analytic inequality Jacobian을 주며 finite-difference eps를 유한 지지 ε로 사용하지 않는다. 종료 성공과 관측 지지는 따로 검사한다. |

**단위와 한 가지 고정 규칙.** 최적화 변수는 회전벡터 rad와 native t/등록 대각선이다. 국소 prior는 실제 중심 이동/동일 대각선과 Log(R0ᵀR)를 사용한다. 원 train768의 자동 관측에 대해 원 reliability·covariance·robust weight와 B의 전체 이산 profile Fisher 정보를 반영한 2JᵀJ의 block 평균 대각선을 얻고, frame 간 median을 취했다. complete/partial/B의 교체 관계를 유지하여 같은 관측을 이중 계수하지 않는다. λR=7750.212386319994, λt=104492.61843444691이며 0정보 하한은1이다. validation·실사·정답에 strength를 맞추지 않았다. λ는 여기의 **비용 계수**다. prior 비용 λ||δ||²의 Hessian은2λI이므로 논문 Eq.(30)의 행렬 숫자와 같다고 쓰지 않는다. 논문의5000/500000을 복사하지 않았다. 이 규칙은 별도 단일 영상 전이 설계이며 논문이 이 팔레트 규칙의 정확도를 보장하지 않는다.

**유한 지지와 analytic 제약.** 모든 원 supported profile에 대해 D=cross(n,b−a), N=cross(a−c,n)를 계산한다. offset은 원 RGB 분포에 그대로 평가한다. λ=N/D는 실제 두 투영 끝점의 범위에 들어야 한다. candidate별 관측 삭제·0비용·endpoint clip은 없다. seed의 고정 denominator 부호 아래 signed D, signed(N+ελD), signed((1+ελ)D−N)를 이용해 분모를 나누지 않는 제약과 도함수를 구성한다. depth·비퇴화·평행 상태는 별도로 검사한다. 원 고정 관측이 없는 frame은 원 전체 관측 비용을 유지한다.

**Restoration과 예산.** analytic 선형화 부등식에 damped correction과 merit backtracking을 적용한다. 최대20회 복원 step은 동일 start의 총80 iteration에서 차감한다. 추가 optimizer start를 만들지 않는다. full/edge 각8회, local2회의 상한과 실제 복원/호출/종료/feasibility를 모두 저장한다. 작은 개별 step을 누적 local 이동으로 오인하지 않도록 SE(3)·투영 누적 변화를 기록한다.

**검사의 의미.** 기대 조건을 먼저 적은10개 범주, analytic/one-sided/국소 prior 도함수, 해상도·Q·중복·원 관측 교체, 원 최악3장의 고정 pose 회귀 검사를 분리했다. 독립 checker는 원 XYZ/Q/K와 normal query를 직접 투영하고2×2 선형계로 교차를 푼다. 생산 finite/solver 함수를 import하지 않는다. 수치 경계 불일치는 별도 원장에 남긴다. 실제 material owner와 독립 실측6D 정답은 UNVERIFIED다.

원 B 목적함수에 이미 포함된 coarse-pose prior는 그대로 유지했다. 새 local Tikhonov 항은 추가 운영 정규화이며 새로운 독립 관측이 아니다. train 정보량 규칙에는 complete/partial/B 자동 관측만 사용하고 기존 prior를 다시 관측 정보로 계수하지 않았다. full/local의 차이를 유한 제약 한 항만의 순수 효과라고 해석하지 않는다.

**실제 전량 독립 검산과 남은 수치 차이.** 238장의 55,035 candidate–family 조합, 1,297,316 profile 계산에서 직접 physical R을 투영한 독립 2×2 교차와 생산식 사이 λ 산술 비교가 6,132회 허용 오차를 넘었다. 첫 v1 검사는 FAIL로 원 소스·실패 행 전체와 함께 보존했다. 모든 exact 지지 predicate/status, 선택 비용 순서, fallback, 반복 상한은 일치했고, 선택된 자세의 지지 불가 profile과 경계 predicate 불일치는 각각0이었다. v2는 원 2×2 식·predicate·비교 허용 오차를 바꾸지 않고, 생산 경로의 R→회전벡터→R과 native 이동 정규화만 별도 수치 표현으로 재생했다. 이 표현에서 기존6,132건이 모두 설명되어 상태는 `PASS_WITH_RAW_REPRESENTATION_DIFFERENCES`다. 직접 R 산술 검사의 FAIL을 전체 numeric PASS로 바꾸지 않는다. 모델·관측·자세·선택·지지 ε는 변경하지 않았다.

**실제 비용의 범위.** 본배치 4,284회와 별도 CAL 기술 실행18회를 합쳐 SLSQP 호출·반환은4,302회, 기술 예외는0회였다. 본배치 wall은1,504.967초이며 frame CPU 합계1,443.915초에는 결과 gzip 직렬화·부모 조율·최종 집계가 포함되지 않는다. train768 정보량 계산은19.881초이고 최적화 호출은0회였다. 전체 실행과 두 독립 검산의 측정 범위·원 v1 실패·출력 출처는 [B_COST_AND_EXECUTION_RECEIPT.json](B_COST_AND_EXECUTION_RECEIPT.json)에 구분했다. 이 수치는 정확도 성능이나 실제 material owner 인증을 뜻하지 않는다.
