# PoseFix 제한 적응 및 출력 보존 — 미실행 계획

[확인] 이번 작업의 optimizer.step은 **0회**다. CPU에서 기존 checkpoint를 읽고 모듈을 열거했을 뿐 wrapper 삽입, gradient/forward parity 검사, LoRA 학습은 하지 않았다. 실험 rank/LR/손실 가중치는 아직 학습용으로 잠그지 않았다.

## 1. 무엇을 보존하려는가

[확인] base weight 보존, adapter off의 함수 복원, adapter on의 정상점 성능 보존은 서로 다르다. 저랭크 제약은 갱신 크기나 clean 출력 변화를 제한하는 수학적 상한이 아니다. 이번 손상을 catastrophic forgetting 하나로 단정할 근거도 없다.

[확인] LoRA 원 논문은 고정 weight에 저랭크 갱신을 학습하는 방법을 제안한다. 팔레트 CNN 성능 보장으로 읽지 않는다. [Microsoft / ICLR 2022](https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/).

[확인] “LoRA Learns Less and Forgets Less”는 언어모델의 수학·코딩 적응에서 fullFT보다 적응이 약하면서 원래 능력을 더 유지하는 관측을 보고한다. 여기서도 복구가 줄어든 것을 보존 성공으로 포장하면 안 된다. [TMLR 논문 arXiv v2](https://arxiv.org/abs/2405.09673v2). OpenReview는 이번 조회에서 브라우저 검증 페이지로 연결되어 본문을 읽지 못했으며 저자 논문 버전을 사용했다.

[추정] 아래 출력 보존은 LwF의 아이디어를 참고한 미검증 팔레트 응용이다. source GT replay를 포함하므로 원 논문의 공식 재현이 아니다. [ECCV 2016 원 논문 정보](https://experts.illinois.edu/en/publications/learning-without-forgetting/), [저자 구현](https://github.com/lizhitwo/LearningWithoutForgetting).

## 2. 실제 모델 구조 확인

[확인] base는 E2와 동일한 synthetic-only PoseFix PRIOR1 마지막 6000 step이다. 실제 path/SHA는 로컬 전용 `POSEFIX_MODULE_INVENTORY.json`에 보존하며 이번 공개 묶음에는 포함하지 않는다. R0/YOLO는 계속 완전 동결한다. N2는 배포 성능 비교군이지 LoRA base가 아니다.

[확인] RGB 3채널과 초기 9점 Gaussian map을 결합한 12채널 입력의 ResNet152 계열 CNN이다. `q_proj/v_proj` 모듈은 없다. 실제 로드한 base parameter 수는 68,661,641, 현재 requires_grad 파라미터는 0이다. Conv2d 1×1/group1 후보는 105개, BN은 158개다. 전체 shape/group/stride 목록은 inventory에 있다.

| [확인] P1 우선 검토 층 | 실제 weight shape | stride/group |
|---|---|---|
| blocks.3.0.conv1.conv | 512×1024×1×1 | 1 / 1 |
| blocks.3.0.conv3.conv | 2048×512×1×1 | 1 / 1 |
| blocks.3.1.conv1.conv | 512×2048×1×1 | 1 / 1 |
| blocks.3.1.conv3.conv | 2048×512×1×1 | 1 / 1 |
| blocks.3.2.conv1.conv | 512×2048×1×1 | 1 / 1 |
| blocks.3.2.conv3.conv | 2048×512×1×1 | 1 / 1 |

[추정] 후기 block의 이 6개 pointwise 층을 최초 검토 대상으로 제안한다. 3×3, stride 변화, shortcut, ConvTranspose2d upsampling은 첫 후보에서 제외해 검증 범위를 좁힌다. 이 선택이 최적이라는 증거는 없다.

[확인] r=4를 가정한 A/B 모양의 **산술적** 추가 파라미터 수는 Σr(Cin+Cout)=57,344이다. adapter를 실제 생성/학습해 측정한 수가 아니다. rank4는 미검증 제안이다. GPU 메모리·처리시간·추론 지연은 측정하지 않았으므로 수치 주장을 하지 않는다.

## 3. 세 가지 구조 후보

- [추정] **P1 Conv2d LoRA-only:** `y = Conv(W0,x) + (alpha/r) B(A(x))`. 위 1×1/group1/stride1 층에 한정한다. W0/bias, 기존 head, BN running statistics와 affine를 모두 고정한다.
- [추정] **P2 head-only:** 실제 `out=Conv2d(256,9,1)` weight/bias 2,313개만 학습하는 단순 비교군. head 파라미터 수는 실제 모듈에서 셌다. LoRA 고유 이득 주장에 필요한 비교이며, head 변경을 off하면 자동 원본 복원된다고 주장하지 않는다.
- [추정] **P3 고정 경로 + 잔차 경로:** `f0(x)+g(x)Δ(x)`. LoRA와 다른 구조다. g=0인 곳만 원래 출력을 보장하며, g의 판단 오류는 별도 문제다. 기존 selector 실패를 고려해 자동으로 추가하지 않는다.

## 4. 구현 전 필수 검사 — 모두 계획, 현재 미실행

[확인] 공식 Conv 경로를 확인한 commit은 `c4593f060e6a368d7bb5af5273b8e42810cdef90`이다. 이 구현은 내부 convolution을 새로 만들고 reset하며, A random/B zero 초기화와 train/eval 시 weight merge를 수행한다. 이미 학습된 weight가 자동 보존된다고 가정하면 안 된다. [고정 버전 공식 layers.py](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py#L229). 라이브러리는 설치하거나 실행하지 않았다.

1. [추정] pretrained module을 참조하는 unmerged wrapper 또는 명시적 원본 weight/bias 복사 후 bit-exact 비교를 사용한다. train/eval 호출만으로 원본 weight가 변하지 않게 한다. 임의로 ConvTranspose2d를 Conv2d로 바꾸지 않는다.
2. [추정] A는 비영 초기화, B=0으로 delta=0을 만든다. 둘 다 0으로 두지 않는다. 최초 backward에서 B의 gradient가 살아 있는지 검사한다. A gradient가 첫 순간 0일 수 있는 것은 정상이며, 이후 별도 시험으로 경로를 확인한다.
3. [추정] adapter off 및 on-at-init의 logits, expectation 좌표, 원영상 inverse crop, bbox/score/center/candidate parity를 검사한다. 임의의 adapter 상태에서도 off로 돌아오면 base 출력이 재현되는지 확인한다. 평가 데이터로 튜닝하지 않는다.
4. [추정] BN running mean/var/count 및 affine까지 동결하고 optimizer에 A/B만 들어가는지 parameter identity로 감사한다. bias/head/BN까지 학습했다면 off=완전 base 복원 주장을 금지한다.
5. [확인] 기존 `losses()` L2는 `Conv2d`와 `ConvTranspose2d` weight만 순회한다. wrapper 안 frozen Conv만 남으면 A/B에는 이 L2 gradient가 전달되지 않는다. [추정] A/B 직접 penalty와 ΔW penalty는 서로 다르므로 명시적으로 설계/잠금해야 한다. frozen weight L2는 상수이며 보존 loss가 아니다. 이 미결정을 해결하기 전 학습하지 않는다.
6. [추정] r/alpha/LR/optimizer/L2 항목과 lambda들을 실험 전에 잠근다. 이번에는 sweep도 잠정 학습도 하지 않는다. shape 및 zero-delta parity를 통과한 뒤 실제 trainable count·peak memory·step/inference latency를 측정한다.

## 5. source replay와 출력 보존 손실 분리

[확인] 기존 replay는 synthetic GT에 normal/stress 입력을 다시 맞추는 학습이다. 정상 입력의 기존 함수 출력을 유지하는 손실과 같지 않다.

```text
L = L_adapt + lambda_source * L_source_GT + lambda_preserve * L_preserve
```

[추정] L_preserve는 **synthetic TRAIN**의 normal 입력 중 base 예측이 GT 기준 실제로 맞는 유효 코너에만 적용하는 후보를 우선 검토한다. 예컨대 기존 보조 지표의 <5px를 계획 기준으로 삼되 실제 mask 규칙은 실행 전 잠근다. heldout/eval GT로 mask를 만들지 않는다. 실사의 높은 R0 confidence를 정상 정답 증거로 쓰지 않는다. 실사 보존 좌표를 추가 수작업으로 만든다면 레이블 비용 조건 변경이므로 별도 승인이 필요하다.

[추정] 교사/학생에 같은 RGB/crop/초기점/유효 mask를 넣고 교사는 stop-gradient한다. 원래 base가 맞는 synthetic TRAIN 코너를 whole-object 승인 대칭으로 고정 매핑해 mask를 만들고, native heatmap 채널의 soft distribution 또는 기대 좌표를 보존한다. 학생마다 유리한 점별 GT 분기를 재선택하지 않는다. 중심점 8은 기존 고정 계약을 유지한다. 다중 모드 heatmap을 무조건 argmax one-hot으로 바꾸지 않는다.

[추정] 가림 입력에서 이미 틀린 base 좌표를 무조건 distill하지 않는다. clean normal 보존과 가림 복원 타깃은 분리한다. 기존 source batch의 normal 슬롯을 활용하는 등 데이터/노출을 맞추고, teacher forward 비용은 별도 기록한다. 효과를 보고 lambda를 같은 DEV에서 구제 탐색하지 않는다.

## 6. 최소 인과 비교 — 미실행

| [추정] 군 | base | 학습 범위 | 고정 source GT replay | 출력 보존 |
|---|---|---|---|---|
| F0 | PRIOR1 | 없음 | 학습 없음 | 원본 |
| F1 | PRIOR1 | full-refiner convolution/head | 있음 | 없음 |
| F2 | PRIOR1 | LoRA A/B만 | 동일 | 없음 |
| F3 | PRIOR1 | F1과 동일 | 동일 | 있음 |
| F4 | PRIOR1 | F2와 동일 | 동일 | 있음 |
| head-only 추가 대조 | PRIOR1 | out weight/bias만 | 동일 | 사전 지정 |

[추정] 공통 BN 통계/affine 동결 정책을 F1~F4에 적용할 경우 과거 E2의 BN-affine-trainable 조건과 다르므로 기존 A11을 F1로 재사용하지 않는다. 과거 값을 참고로만 둔다. A10을 새 base로 택하려면 모든 군과 fullFT 대조군도 A10에서 시작하는 별도 설계가 필요하다. PRIOR1 대조군과 혼합 비교하지 않는다.

[추정] 같은 입력/타깃/mask/가림 plan/실사·합성 노출/update/seed/last-checkpoint 규칙을 고정한다. fullFT와 LoRA가 같은 LR에서 최적이라는 가정은 하지 않는다. 한 개 사전설정 파일럿은 제한된 비교이며, 일반적 우월성에는 별도 validation-only 설정 예산과 복구–손상 곡선이 필요하다. 이번에 설정 탐색은 하지 않는다.

## 7. 성공 해석과 미확정 조건

[추정] 보존만 좋아지고 복구가 F0 수준으로 사라지면 성공이 아니다. R0 20–40px 복구 획득과 5–20px에서 base가 맞히던 코너 손실을 함께 본다. base ≤10→new >10 전체 전이와 R0 <5→new >10도 구분한다. clean/GREEN manual/source normal/source stress/실제 occlusion을 별도로 평가한다. source 보존이 GREEN 보존을 보장하지 않는다.

[추정] 출력 보존 loss만으로 해결되면 LoRA 필요성은 지지되지 않는다. LoRA+보존에서만 동시 개선되더라도 각 대조군과 독립 확인이 필요하다. 손상 감소와 함께 복구도 줄면 단순 적응량 축소일 수 있다. 어떤 후보도 숨은 위치를 찾지 못하면 관측 표현/기하/타깃 문제는 별도로 남는다.

[확인] 아직 rank/alpha/lambda/regularization/공정 LR 예산/독립 검증 데이터/가시성·좌표 신뢰도 확정이 남아 있다. 계획서 작성 완료는 학습 승인이나 효과 입증이 아니다. 원래 N2 유지 여부도 이번 oracle만으로 자동 변경하지 않는다.
