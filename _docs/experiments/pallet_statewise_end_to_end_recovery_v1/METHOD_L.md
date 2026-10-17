# L: 유한 후보 자세 감독과 동일 입력의 선형 대조군

L은 원 RGB/관측에서 이미 실제 계산한 자세 후보 중 하나를 선택하거나 보류하는 작은 모듈이다. 연속 자세 분포·Monte Carlo KL·EPro-PnP 네트워크를 구현하지 않았으므로 **EPro-PnP 재현이라고 부르지 않는다**. 최종 자세를 구별하는 감독 원리만 적용한 신규 이산 대조군이다. YOLO·N3·SubPix·PoseFix·상태 backbone은 재학습하지 않는다.

## 같은 자료·후보·입력

합성자료의 원 renderer pose, source별 고정 대칭, 원 K·크기·축 계약을 따르는 감독만 사용한다. real CAL/PRIMARY192나 설명6장의 GT·수동 경계는 새 학습에 사용하지 않는다. 학습/validation source asset은 완전히 분리하며 split lock은 새 후보 라벨 전에 고정했다. 확장 train768장은 원 C1 266장+C2 502장이고 validation235장은 전부 원 C1이다. PRIMARY192는 기존 C2다. 이 감독 범위 차이와 upstream이 과거 같은 assets를 보았던 사실을 그대로 보고한다.

root의 공통 `solver.features_for_bank`가 합성·실사에 같은 계산을 수행한다. 입력은31개의 signed-log1p scalar와31개의 validity bit다. 항은 고정/A/B/AB/동일지지edge 비용, point/partial/outline 비용과 관측 수, owner/none 모호성, B 분포 지지·entropy, fallback 비율, covariance·잔차, N0와의 자세 차이, 최소 제약 slack, 양의 깊이 비율, 같은 bank 내 점수 차이 등이다. GT 오차·성공 여부·oracle rank·ID·asset/material/session·원 실패 분류는 특징에 없다.

root의 등록된 후보 eligibility를 두 모델이 그대로 공유한다. 수치적으로 같은 물리 pose는 root의 기존 엄격한 허용차로 중복제거한다. C1 source에서는180° phase가 같은 pose가 아니므로 임의로 지우지 않는다. invalid/ineligible 후보는 장부에 남지만 softmax에서 마스킹한다. N0/E0는 원 baseline 대안으로 보존하되 joint 제약 해라고 인증하지 않는다. 세부 source-alias 처리도 root의 공통 장부를 따른다.

## 손실과 결측

프레임의 eligible 후보 중 같은 pose에서 중심 위치≤5cm AND 회전≤5°인 집합을 P라 한다. P가 있으면 각 원소의 목표 확률은1/|P|, 나머지와 보류는0이다. P가 비면 `no_reliable_candidate`가 유일한 목표다. 실제 후보와 보류 점수의 softmax에 표준 cross entropy를 사용하고, 프레임별 loss를 동일 가중 평균한다. 좋은 후보가 없는 프레임을 버리지 않는다.

표준화는 **train의 eligible 후보에서 유효한 scalar 값만** 사용한다. validity bits는 유지한다. 결측·train에서 한 번도 관측되지 않은 scalar는 정규화 좌표0으로 채우고 실제 유효성 bit를 함께 전달한다. validation/PRIMARY의 평균·분산으로 normalization을 다시 fit하지 않는다.

각 후보의62차원 특징 외에 해당 프레임의 eligible 후보 특징 mean/max, log1p(후보 수)를 같은 모델에 제공한다. 보류는 로컬 특징0과 동일한 global pool, 별도 none bit를 받는다. 총 입력188차원이다. mean/max는 후보 순서와 무관하며 padding은 softmax에서 완전히 마스킹한다.

- `L_LINEAR`:188→1 affine.
- `L_HEAD`:188→64→64→1, 두 hidden layer ReLU.

자료·목표·normalize·pool·후보 bank·eligibility·보류 정책은 같다. 헤드의 표현력 외 차이를 새 관측 효과로 섞지 않는다. softmax 수치는 **교정된 자세 안전 확률이 아니다**.

## 고정 훈련·모델 선택

seed는20261018/20261019/20261020, 첫 seed20261018을 사전 대표로 고정한다. 선형과 head를 각각 세 번, 총 최대6회 실행한다. PRIMARY 최고 seed를 선택하지 않는다.

후보 생성 전에 봉인된 `CONFIG_FROZEN.json`의 운영 gate는 eligible 공동성공 후보를 가진 영상이 train32장 이상·validation16장 이상인 것이다. 이는 일반화 충분성 인증이 아니며 gate에 못 미치면 학습0회로 남긴다. 새 라벨 결과에 맞춰 이 수를 바꾸지 않는다.

AdamW lr1e-3, weight decay1e-4, batch32프레임, 최대50epoch, validation loss patience5를 고정한다. strict loss 개선일 때 best checkpoint를 저장하고 last checkpoint와 모든 epoch history도 보존한다. 작은 head는CPU1thread에서 실행하여 MobileSAM과 GPU를 경쟁하지 않는다. 가장 낮은 validation loss의 epoch는 해당 run의 모델 선택이며 validation loss가 head를 지지하지 않아도 새 head 구조를 반복 설계하지 않는다.

실제 실행 수·epoch·optimizer step·frame exposure·checkpoint SHA와 비용은 `TRAINING_MANIFEST.json` 및 개별 `HISTORY.json`이 권위 기록이다. 선언한 최대6회를 실행했다고 미리 적지 않는다.

## 추론과 보류

별도 `predict` 프로세스는 고정 checkpoint·train normalizer와 봉인된 실제 특징만 읽는다. `SYNTH_GT_LABELS`, synthetic head target, fixed real reference, annotation, scored ledger의 읽기를 거부한다. 모든6seed 출력은 사전에 등록한 `L_LINEAR_s20261018` 등 이름으로 보존한다.

보류가 선택되면 모든 영상에서 원 N0로 복귀하고 `LEARNED_ABSTAIN_FALLBACK_N0`로 표시한다. 이는 좋은 자세·물리 안전을 보장하지 않는다. N0로 유지된 성공을 새 보정의 회복으로 주장하지 않는다. 후보를 고르면 실제 저장 pose 하나를 출력한다. 각 frame의 위치·회전 성공은 반드시 그 같은 pose에서 판정한다.

## 검사와 명령

후보 순서 변경·ID/session 변경에 대한 점수 불변, ineligible 후보의 pool 무영향, 실제0과 결측의 구분, padding softmax 무영향, reference 읽기 차단을 실제 단위검사한다. 새 head의 accuracy는 별도 후봉인 평가기에서 검산한다.

```bash
python -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.learned_head prepare-labels --features <sealed HEAD_FEATURES.jsonl.gz> --feature-seal <feature seal> --targets <SYNTH_GT_LABELS.jsonl.gz> --target-seal <PROVISION_SEAL.json> --split-lock <DATA_SPLIT_LOCK.json> --output <new label dir>
python -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.learned_head train --features <same features> --feature-seal <same seal> --labels-dir <sealed label dir> --output <new six-run training dir>
python -B -m scripts.research.pallet_statewise_end_to_end_recovery_v1.learned_head predict --features <actual prediction-only feature bank> --feature-seal <feature seal> --training-dir <frozen training dir> --output <new prediction dir>
```

새 실제 feature/label 분포에서 긍정 후보가 희소하거나 head가 전이되지 않으면 그 실패를 보존한다. 이 데이터에서 성공률이 높아져도 독립 세션 검증이나 실제 physical owner가 인증된 것은 아니다.
