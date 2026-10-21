> Public edition of an existing development audit. Identifiers and local paths are redacted. Publication adds no new experiment or improvement claim. Historical no-publication statements describe the original audit date; this edition is separately authorized.

**Read [v38 real-cost follow-up](v38_real_cost_followup_KO.md) after this report. The original text predates that completed matrix audit and the solver-stage I0 correction.**

# 저장 선택기 단일 사례 감사 v38 — 비공개 개발 진단


## 합성 VAL 단일 사례: frame_6bb3c82bb9c81749c6b0

같은 v31 bank의 실제 eligible action 94개와 저장 TRAIN normalizer로 frozen forward를 실행해 두 저장 선택을 정확히 재현했다. 새 성능 개선 실험이 아니다. 반복 노출 VAL의 설명 사례이다.

| 후보 | 저장 T/R | v32 logit / 순위 | v33 logit / 순위 | v33 box 평균 중화 logit / 순위 |
|---|---|---|---|---|
| G13/SWAPPED_WD/endpoint/173 | 4.890237cm / 0.311191° | -2.845579 / 1 | -2.584218 / 7 | -3.859872 / 7 |
| G13/ORIGINAL_WD/endpoint/143 | 9.098492cm / 89.772464° | -3.048943 / 11 | -2.567592 / 1 | -3.681288 / 1 |

초기 N0의 저장 오차는 18.203928cm / 88.727311°이다. 초기부터 방향이 틀렸다는 사실과, 이미 bank 안에 있는 좋은 후보를 v33이 놓쳤다는 사실은 별개다. 초기 오류를 visible/hidden 점, correspondence, upstream W/D 점수 또는 PnP 원인 중 하나에 귀속하는 실험은 이번에 하지 않았다.

v32 220D = 기존188 + ROI값16 + ROIvalidity16. v33 232D = 기존188 + ROI값16 + box값6 + ROIvalidity16 + boxvalidity6. 기존188 및 ROI값16은 정확히 같은 float32 배열이고 ROI validity도 위치를 정렬하면 정확히 같다. 188은 후보 local값/validity62, 같은 bank 평균62/최대62, log1p(action수), N0 flag이다. 31scalar 안의 bank gap도 저장 값을 그대로 썼다. 두 모델의 base normalizer는 같은 v31 TRAIN normalizer와 정확히 같고, 각 extra normalizer는 자기 저장 TRAIN 산출물과 정확히 일치한다. 모델 둘은 별도 재학습된 checkpoint이므로 box 입력 추가만 바뀐 동결 모델 대조가 아니다.

v33 중화는 모든94 action의 normalized box값 columns204:210만 0(서명log 공간 TRAIN평균)으로 바꿨다. validity6, ROI validity16, 기존 bank pooling/gap/count와 나머지226 columns 및 weights는 보존했다. 오답이 여전히1위다. 오답-좋은후보 logit차이는 +0.016626에서 +0.178584로 오히려 커졌다. 따라서 이 finite intervention은 실제 box값만이 역전의 충분한 원인이라는 설명을 지지하지 않는다. TRAIN평균 box와 다른 residual의 조합은 실현 불가능한 입력일 수 있어 인과증명·fallback 정책·개선으로 해석하지 않는다.

ROI 16개 raw residual RMSE/원영상대각선은 좋은후보0.004292, 오답0.105800이다. ROI 관측과의 부합에 큰 차이가 입력에 남아 있어도 v33 선택은 오답이다. 이것은 ROI가 독립 물리 정답이라는 주장이 아니며, 학습된 가중치/상호작용이 그 근거를 충분히 활용한다는 보장도 아니다.

## 관측 비용: 단일한 정답 지지가 아니다

양쪽 family별 5비용 전체는 원 packet·BOUNDARY_BUNDLE·고정 config와 두 저장 R/t에서 CPU로 재평가했다. 새 관측 추출/영상 forward/PnP는 없다. raw 재평가임을 파일명에 표시했다. 두 후보의 context family, 첫26개 signed-log 특징, 해당 validity는 기존 저장값과 오차0으로 일치한다. pair만 재평가했으므로 새 pair gap은 사용하지 않았고, 실제 H1 gap은 원94 bank 저장값을 사용했다.

동일 family·동일 mode 안에서만 비교하면 다음과 같다.

- A: 좋은후보3.375853 < 오답3.937736. 두 family 모두 A feasible이며 좋은후보 지지.
- AB: 좋은후보27.648595 > 오답25.974641. 두 family 모두 AB feasible이며 오답 지지.
- ORIGINAL_WD의 fixed/B/edge는 오답 지지(좋은후보는 이 family의 해당 제약에서 infeasible).
- SWAPPED_WD의 fixed/B/edge는 좋은후보 지지(오답은 이 family의 해당 제약에서 infeasible).

서로 다른 family의 context 비용을 빼서 공통 잔차처럼 해석하지 않았다. 관측 비용도 일부는 오답을 지지하므로 ‘모든 좋은 관측을 H1이 무시했다’는 설명은 기각한다. 좋은 정보가 전혀 없다는 설명도 ROI와 A 대조에 맞지 않는다. 구체적으로는 상충하는 관측/대응 근거와 재학습된 선택기의 결정이 함께 남아 있다. 단일 logit 차이로 재학습 전체의 원인까지 특정할 수 없다.

## 실사 보완 감사: frame_704b3381eea10171f7c8

PRIMARY 반복 개발 사례의 원 OCPR 기록만 열람했다. 새 frozen forward도 하지 않았다. N0의 R_physical/centroid는 정확히 보존됐고 eligible=true이다. 65 action 중 N0는7위(logit -1.566151), 선택 E0_REGENERATED는1위(-0.358492). 선택 alias에 G13/SWAPPED_WD/endpoint/103 등 여러 동치 pose 출처가 있다. family 문자열만으로 실제 회전오류를 판정하지 않는다. 기존 패널의 저장 오차 N0 2.26cm/0.94° → OCPR10.69cm/82.40°를 인용하며 새 채점은 하지 않았다.

저장 full/tight mask 모두 존재한다. 실제 mask contour를 원 사진에 그려 눈으로 확인했다. 두 mask는 명확한 무늬가 있는 상판 왼쪽의 넓은 어두운 영역까지 포함하고, 일부 B 부분선은 그 바깥 경계 근처에 있다. 이 사실은 그림자/바닥 경계 혼동을 검사할 구체적 단서이다. 그 검은 영역의 물체/그림자 경계에 대한 독립 주석이 없으므로 SAM 의미 오류를 확정하지 않는다. 관측이 원 상판 모서리보다 어느 물리 edge에 붙었는지도 native association 가설과 실제 owner를 구별해야 한다.

저장 상태는 I,B,B,B,B,B,S,B, initial_self_hidden은 native6만 true. S는 조회하지 않았다. native0 I는 backbone fallback 관측, native5 B는 B_TARGET_BACKGROUND_LINES 수정 관측, native1/3/4/7은 부분선, native2는 남은 full2D fallback이다. 부분선은 선 법선 제약이며 display용 점을 full2D 관측으로 취급하지 않았다. full/tight/reach 복구 단계는 fallback B1/2/3/4/7을 조회했지만 complete B/I를 바꾼 기록은 없다.

H1 전체 stored actions/logits와 65×188 입력을 저장 normalizer로 복원했다(모델 forward0). 현재 frame 캐시에는 N0/선택 pose 각각의 양쪽 family 전체5비용 행렬이 없다. 저장 context scalar와 endpoint의 자기 목적함수 기록은 있으나 다른 family의 비용을 대신하지 않는다. 이번 실사 범위는 캐시 열람만이므로 비용을 재생성하지 않았다. 따라서 실사에서 어느 family의 어떤 관측이 N0를 이겼는지의 통제 비교는 아직 미완이다. 합성 사례와 같은 원인으로 단정하지 않는다.

REAL_CASE/CACHED_MASK_OBSERVATION_POSE.png를 직접 열어 검사했다. 좌: full주황/tight청록 mask. 중: 실제 I청록/B주황, 실제 RGB부분선 및 native번호. 우: 저장 N0주황/선택청록 pose투영이며 관측 keypoint가 아니다. crop/좌표를 바꾸지 않은 원640×480의 세 패널이다. 이 그림과 원자료는 비공개이다.

## 문서 정정 및 읽기 제한

실제 paper_evaluate.calibrate는 fixed-index Gaussian targets CE를 사용하고 local_phase를 호출하지 않는다. retained CALIBRATION_AND_SELECTION.json objective도 명시적으로 같은 계약이다. local_phase는 TRAIN train_loss의 sym 분기에서 사용한다. 원 source docstring의 TRAIN/calibration 표현을 실제 호출 경로에 확장하면 부정확하다.

공개 ESSENTIAL_CONTRACTS.md의 해당 한 문장과 MANIFEST의 그 파일 크기/hash만 정정했다. 문장 정정 commit e3e73e3c11d193977a5dea7081aebc9a5688ab55, manifest 총 byte 합계 후속 보정 commit ae99e9f73015d0f98dffe64ae6627575fc86a851. 원격 main과 두 파일 bytes를 검증했다. 방법 코드·온도1/λ1/cap1%·역사 결과·gate는 바꾸지 않았다. 새 분석 packet은 공개하지 않았다.


## 실행·비용·다음 단계

Python pallet-pose, OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1. 아래 순서로 실행했다.

[Local execution commands omitted from public edition; preserved in private audit.]


다음 필요한 검증은 합성의 상충하는 A/AB 근거와 v33 재학습의 가중치 상호작용을 분리할 TRAIN 사례 대조, 실사의 동일 family 내 N0/선택 비용 대조이다. 이번 단일 VAL/PRIMARY 사례로 예외 규칙을 만들지 않는다. 실사 원인 확정에는 기존 캐시만으로 미산출인 동일family 비용 또는 독립적인 실제 물체/그림자 edge 판독이 필요하다. 새 실험·v37 학습은 시작하지 않았다.
