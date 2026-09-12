# 실행 지시문 — 치수·대칭 일관 점·선 공동 자세 v5

## 0. 목표·범위·성공 의미

`CanelE452/pallet-6d-pose`에서 첨부 패키지의 이미 구현된 코어를 실제 데이터/모델 경로에 연결하라.

최상위 목표: 합성 데이터만으로 학습한 모델이 실제 팔레트의 위치·자세를 안정적으로 예측하는 것. DHT 자체를 반드시 채택하는 것이 아니다. 소비처는 사용자의 연구 설계·아키텍처 채택 판단이다.

이번 지시문을 전달받으면 다음 제한된 작업을 수행한다:
1. 기존 기록/소스/GT/치수/프레임 계약 확인.
2. 제공 코어를 그대로 검증하고 실제 로컬 연결부를 작성.
3. 합성 기반 기하·초기값·정답선 진단과 작은 실데이터 plumbing.
4. 아래 선결조건이 충족되고 이를 파일로 고정한 경우에만, 합성의 고정-feature matched pilot P/D/H×3seed, 각2,000step까지 수행.
5. 결과를 보고하고 종료. **실사 새 평가 및 전체 YOLO online 학습은 이 지시문만으로 자동 확대하지 않는다.** 그 단계의 계획과 구체적인 연결부까지 준비할 수 있지만 별도 승인 없이는 실행하지 않는다.

원래 v4 결과는 그대로 보존한다. v5가 v4의 합성 미달 판정을 무효화하지 않는다. 양호한 테스트/출력 파일/손실 감소/정답선 oracle은 성능 달성의 증거가 아니다.

금지: 새 렌더 대량 생성, real 레이블 학습, self-training, FINAL 읽기, 기존 annotation/checkpoint 덮어쓰기, 사용자 파일 삭제, 전역 Ultralytics 수정, commit/push, 외부 알림. 이전 다른 작업에서의 push/알림 승인을 이번 작업까지 확장하지 않는다.

## 1. 중요한 정정과 기존 작업 재사용

- 기존 structured-v2/v4는 이미 endpoint/finite-segment/배치 scorer다. “처음 점과 선을 연결했다”고 하지 않는다.
- 기존 joint-v1은 실제 DHT 특징을 point head에 전달하고 함께 학습했다.
- 기존 DGP도 치수+point+line을 하나의 pose optimizer에 사용했다. point-PnP 초기화 의존과 국소도달/가시성 불연속의 한계를 먼저 읽는다.
- v5는 **새 object-frame point head + direct 또는 DHT line head + image-derived multiple starts + unrolled common pose**다. 이전 YOLO point head를 보존한 작은 개선이라고 부르면 안 된다.
- 모델 규모, 초기 점 출력, loss가 바뀌므로 v5 H 대 예전 R0 차이만으로 DHT 효과를 말하지 않는다. v5 P/D/H의 동일 조건 비교가 먼저다.

이미 제공된 `plpose_v5/`를 새로 작성하지 마라. 결함이 있으면 원본을 보존하고 최소 수정 및 회귀 테스트/실제 사용 SHA를 남긴다. 어려운 solver를 삭제하거나, 최종 pose loss를 끊은 dummy stub로 바꿔 성공했다고 하지 않는다.

## 2. 먼저 로컬 상태 보존

```bash
REPO="$(git rev-parse --show-toplevel)"
KIT="$REPO/scripts/research/pallet_point_line_v5"
RUN="$REPO/data/pallet/results/pallet_point_line_v5"
export PYTHONPATH="$KIT${PYTHONPATH:+:$PYTHONPATH}"
```

경로는 실제 root 확인 후 위처럼 사용한다. KIT/RUN이 이미 있으면 덮어쓰지 말고 현재 파일/완료 상태를 확인해 충돌로 보고한다. git reset/clean/checkout은 하지 않는다. root 경로를 메모리의 `/home/...` 문자열로 고정하지 않는다.

압축은 새 KIT에 풀고 실행 전 `git status --short`, 관련 diff, 현재 HEAD, Python/torch/ultralytics/CUDA 버전을 RUN의 provenance에 저장한다. `data/pallet/results/pallet_point_line_v4`와 기존 모델/GT는 수정하지 않는다.

설치 재구성 대신 현재 검증된 환경을 사용한다. CPU 코어 검증이 성공했다는 이유로 원래 CUDA torch를 재설치하지 않는다.

```bash
cd "$KIT"
python verify_bundle.py
python -m pytest tests -q
python -m plpose_v5.runner --help
python -m plpose_v5.assessment --help
```

여기서 받은 `evidence/`는 작성 환경의 검증이다. 로컬 검증은 새 경로에 별도로 기록하고 기존 evidence를 덮어쓰지 않는다.

## 3. 먼저 읽을 실제 소스

아래 경로들은 앞선 분석에서 확인했다. 각 파일의 실제 로컬 버전/내부 key는 직접 읽고 해시한다.

- `scripts/research/pallet_dht_joint_v1/{integration,hough_block,line_targets,train,evaluate}.py`
- `scripts/research/pallet_dht_structured_v2/SEMANTICS.md`, `scripts/research/pallet_dht_structured_v2/{model,proposals,cache,infer}.py`
- `scripts/research/pallet_line_pose_v1/source_data.py`
- `Deep_Object_Pose/common/dimension_guided_graph_pose.py`
- `_docs/audits/ARCHITECTURE_GATE_DECISION_palletgraph_line.md`
- `_docs/experiments/pallet_translation_loss_v1/{GEOMETRY_METADATA_AUDIT,SYNTHETIC_DIMENSION_AUDIT,LOSS_DESIGN}.md`
- `_docs/experiments/pallet_translation_loss_v1/LOSS_SYMMETRY_CONTRACT.json`
- v4 정정본·외부 감사. 로컬에 없으면 해당 원격 branch/ref에서 작은 문서만 읽어라.

최근 검토 기준 `6a68452ea9a17a69860981fea1c5a3dc234e3dfa`의 `integration.py`가 선언하는 R0 SHA는 `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7`이다. 실제 파일이 일치하는지 확인한 뒤 사용한다. `0960...` joint seed1을 같은 R0라고 바꿔 쓰지 않는다. 다른 init을 쓰려면 별도 계약/비교군으로 명시한다.

`SOURCE_REGISTRY.json`에 실제 checkpoint 경로/SHA, source 파일/SHA, 라이브러리 버전, 실제 dataset 원본 locator, source frame·치수 축·K 변환 정보를 넣는다. 못 본 내부 key 이름을 추측해 배선하지 않는다.

## 4. 상류 게이트 — 통과하지 않으면 학습하지 않는다

### 4.1 새 object frame 연결

제공 코어: +x=width, +y=up, +z=depth, geometric center origin, dims=(W,D,H) metres. 기존 camera-facing 번호와 다르다.

renderer의 원래 cuboid/pose/치수와 camera-facing permutation의 계보를 읽어 새 object frame의 GT로 변환한다. 같은 중심이면 `reframe_pose`; 중심이 다르면 translation도 실제 식으로 변환하라. 알 수 없으면 `BLOCKED_OBJECT_FRAME_MAPPING`이다.

합성 전체 export에서 새 `K,R,t,dims`가 새 순서 2D GT를 재투영하는지 확인한다. raw-image pixel 비교 max0.1px를 예비 산술 기준으로 사용하되 이는 [미검증]이고 annotation 정확도 문턱이 아니다. 실패를 좌표 덮어쓰기나 큰 tolerance로 봉합하지 않는다. 의미 계약의 정당성을 별도로 검토한다.

학습 GT 정렬은 원본 metadata를 쓸 수 있다. 추론 pred points를 GT permutation으로 고치거나 source frame의 “정답 앞면 폭”을 input width로 넣으면 안 된다.

### 4.2 치수와 대칭

외부 제원으로 추론 시 얻을 수 있는 dims를 쓴다. pose GT로 W/D 순서를 선택하지 않는다. 오차 없이 정확한 per-frame 치수만 쓰는 실험이면 `KNOWN_EXACT_DIMENSION_ASSUMPTION`으로 한정한다. 치수 노이즈·실측오차 일반화는 아직 평가되지 않았다.

원래 asset별 확인 계약을 읽고 C1/C2/C4를 지정한다. shape만으로 대칭을 승격하지 않는다. square tolerance는 수치 비교일 뿐이다. 실질 C4가0이면 `C4_NOT_EVALUATED`로 보고하며 “정사각/직사각 혼합 성공”을 주장하지 않는다.

현재 계약으로 90도 혼동을 면제할 수 없는 물체는 잘못된 대응으로 유지한다. `(W,D)` 교환까지 허용하는 별도 좌표계 재표현은 이 코어에 자동 추가하지 않는다.

### 4.3 실제 특징 정합

같은 predictor의 실제 1회 forward에서 검출 output과 native P3/P4를 모두 보존한다. hook 위치를 한 곳으로 고정한다. Hough를 추가하기 전 stock R0 feature를 기본으로 하되 실제 graph를 읽고 확인한다.

제공 `capture_module_inputs`는 features만 캡처하므로 predictor 반환값 보존·비교는 연결부의 책임이다. P4만 bit-equal한 것을 “points/box/line까지 같은 forward”의 증거로 쓰지 않는다.

feature stride와 index0 중심 offset은 실제 Conv/resize 경로에서 확인한다. impulse/ramp 이미지의 독립 예상 위치와 실제 raw→feature 샘플을 비교한다. 단순 A·A^-1 왕복만으로 끝내지 않는다. raw의 모서리/이미지 경계/reflect 영역/letterbox에 대한 실제 시각 및 수치 표본을 남긴다.

`MultiScaleROI`에 주는 box는 예측 box다. GT box/crop으로 실사 성능을 높이지 않는다. ROI가 line 및 point가 보는 조건을 결정하므로 세 arm 모두 같은 box와 샘플을 쓴다.

### 4.4 입력/실패 모집단

prepared-feature pilot은 검출된 per-instance 데이터에 조건부일 수 있다. 그 경우 “all319 전체성능”으로 표시하지 않는다. 원본 전체 이미지 manifest, false negative, unmatched instance, negative detection을 별도 ledger로 보존한다. 누락을 숨기지 않으며 pseudo ROI를 GT로 만들어 복구하지 않는다.

실제 deployable pipeline에서 missing detection은 pose 실패로 남아야 한다. `assessment.py`는 manifest 안에서 pose 실패를 분모에 포함하지만 **manifest 밖으로 삭제된 검출 실패를 복원하지 못한다.** 전체 모수 검사는 로컬 evaluator가 담당한다.

## 5. 초기 기하/정보 진단 — 소수 합성에서만

제공 `diagnostics.compare_line_sources`를 사용해 동일한 실제 시작점·치수·K·점 관측에서 다음을 비교하는 로컬 wrapper를 작성한다.

A. point-only solver
B. predicted point+line solver
C. **GT geometry line oracle solver**

C는 네트워크에 들어가는 GT 입력이 아니라 기하·초기값의 **진단 상한**이다. 이 결과를 실제 성능표에 섞지 않는다. GT line은 raw annotation의 두 2D점을 다시 잇는 대신, 검증된 K,R,t,dims의3D edge를 투영해 만든다. 그래야 새 object frame과 일치한다. 단, raw annotation과 validated reprojection 자체의 차이도 별도 보고한다.

초기 pose도 동일해야 한다. current predicted starts / GT-near starts 두 population을 분리한다. GT pose를 초기값으로 쓰는 것은 두 번째 oracle 진단뿐이며 실제 inference 성공으로 보고하지 않는다.

해석:
- C가 current starts에서 P보다 의미 있게 좋아지지 않으면 `GEOMETRY_OR_INITIALIZATION_BLOCKED`; line head 전체학습으로 바로 가지 않는다.
- C는 GT-near에서만 좋아지면 solver basin/global-start 문제가 먼저다.
- C가 current starts에서 도움이 되고 B가 못하면 predicted line evidence/reliability가 병목 후보다.
- B/C 모두 P보다 좋아지면 line 사용 가능성은 있으나 trainable Hough 우위는 아직 아니다.

solver 자체를 결과에 맞게 튜닝하지 않는다. 이 진단 전에 iteration/trust/sigma를 동결한다.

## 6. 실제 export와 누출 게이트

기존 데이터의 원본 renderer JSON은 학습용 GT에만 사용한다. 관측 텐서와 target 파일은 분리한다. manifest는 두 파일의 SHA를 따로 묶는다. score 단계가 target을 열면 실패한다.

`Observation`은 예측/입력만:
- multiscale features+affines+content masks
- predicted box/instance score와 raw shape
- K와 **배포 시 알고 있다고 가정한** dims/symmetry id
- ROI용 predicted bbox
- source/frame id와 provenance

`Supervision`만:
- 새 object-frame 2D/3D corner target/masks
- R,t
- line support/mask

GT pose/corner/visibility-derived selector/permutation을 `Observation.extra`에 넣지 않는다. `extra`는 model forward에서 forbidden이다.

`original_image`가 없으면 바이트 비중복을 직접 검증했다고 말하지 않는다. provided audit가 `PARTIAL_INPUT_VERIFICATION`을 반환하는 것은 정직한 결과다. source 이미지가 raw인지 padded 파일인지 SHA 기준을 통일한다.

실제 source key→새 key, object-axis 변환, crop/feature affine, checkpoint 매핑을 `ADAPTER_MAPPING.md`에 작성한다. 코드에서 조회하지 않은 기존 key가 이미 있다고 가정하지 않는다.

데이터는 기존 source train1792/cal256/synth_val512를 기본 계보로 삼되, 실제 ID/SHA/split을 확인한다. 이것은 새 head 기준 분리이며 backbone의 완전 미사용 holdout이 아니다. 바꾼 크기의 데이터 사용은 원 계획과 별도 기록한다.

## 7. 실제 cache 학습 — 계약 고정 뒤만

`point`, `direct`, `hough` 세 군 모두:
- 같은 stock backbone와 export features/predicted ROIs/K/dims/GT
- 같은 native 데이터 (v4처럼 임의 C4 교란이 main 데이터가 아님)
- 치수 conditioning=true, 같은 유효 symmetry, same seed generator/solver
- 같은 shared initial tensors와 minibatch trace, 같은 actual optimizer steps

line branch 크기는 다르다. 실제 trainable/active counts를 보고한다. `direct`와 `hough`는 선 감독도 같으므로 H−D는 line readout 비교이고, H−P는 line branch+supervision을 더한 방법 비교다. 모델 규모 단독 효과는 아직 분리되지 않았다고 적는다.

**현재 core point head는 새 head다. 원래 YOLO keypoint loss를 무조건 같이 더하지 않는다.** 기존 camera-facing pose/RLE loss가 남으면 새로운 object-frame 대칭 감독과 충돌할 수 있다. 고정-cache stage에서는 legacy point loss를 계산하지 않는다. online stage의 box/class와 pose 항 분리는 설치된 실제 loss 코드를 읽어 별도로 검증해야 한다.

모든 config/sigma/loss weight/seed 수/시작점 수/solver budget를 저장해 동결한다. main checkpoint는 최종 step만 쓴다. validation으로 epoch·방법·margin 선택 금지.

```bash
EXPORT="$RUN/export"  # 로컬 연결부가 실제 생성한 manifest 위치
python -m plpose_v5.runner audit \
  --manifests "$EXPORT/train.json" "$EXPORT/calibration.json" "$EXPORT/synth_val.json" \
  --output "$RUN/EXPORT_AUDIT.json"

python -m plpose_v5.runner train \
  --manifest "$EXPORT/train.json" \
  --config "$KIT/configs/model_pilot.json" \
  --output "$RUN/cache_heads" \
  --steps 2000 --batch 4 --seeds 1 2 3 --lr 0.0001 --device cuda:0
```

runner는 3군×3seed=9개 실제 학습을 수행한다. 캐시는 기존 backbone을 학습하지 않는다. 자원 때문에 batch/grid/steps가 바뀌면 main 전에 동결값을 갱신하고 이유를 남긴다. main 도중 실패한 경우 자동으로 batch를 바꿔 완결하지 않는다.

숫자 제약은 [미검증 엔지니어링 예산]이다. 초기32frame/소규모GPU plumbing은 각 arm 최대16step, cache main각2000step, 총18000 optimizer steps가 상한이다. 소스 계약 실패이면 본학습0, NaN/좌표계/출처 오류이면 해당 단계중단, 동일 무결성 결함 수정시도 최대2회 후 차단사유를 보고한다. 성능 실패 후 다른 설정으로 반복하지 않는다.

오래된 `pgrep -f` 조건으로 GPU를 기다리지 않는다. GPU 점유는 numeric PID/메모리로 확인하고 타인 프로세스를 종료하지 않는다. runner는 동기 실행이며 표준출력에 진행을 남긴다. 동일 결과root 동시실행 금지. 중단 경로는 실패JSON을 보존하고 침묵한 재시작/overwrite를 하지 않는다.

## 8. 평가 순서와 판정

이 버전은 calibration margin selector가 없다. configuration 자체가 고정된 추론 규칙이다. 모든 모델을 최종 checkpoint로 완료한 뒤, 동일한 고정 config로 calibration과 validation을 각각 추론한다. 이후 sigma/start/계수/score temperature를 다시 고르지 않는다.

```bash
for ARM in point direct hough; do
  for SEED in 1 2 3; do
    for SPLIT in calibration synth_val; do
      python -m plpose_v5.runner score \
        --manifest "$EXPORT/$SPLIT.json" \
        --checkpoint "$RUN/cache_heads/${ARM}_seed${SEED}/checkpoint_final.pt" \
        --output "$RUN/predictions/${ARM}_seed${SEED}_${SPLIT}.json" --device cuda:0
      python -m plpose_v5.assessment evaluate \
        --manifest "$EXPORT/$SPLIT.json" \
        --predictions "$RUN/predictions/${ARM}_seed${SEED}_${SPLIT}.json" \
        --output "$RUN/evaluations/${ARM}_seed${SEED}_${SPLIT}.json"
    done
  done
done

python -m plpose_v5.assessment compare \
  --reference "$RUN/evaluations/point_seed1_synth_val.json" "$RUN/evaluations/point_seed2_synth_val.json" "$RUN/evaluations/point_seed3_synth_val.json" \
  --method "$RUN/evaluations/hough_seed1_synth_val.json" "$RUN/evaluations/hough_seed2_synth_val.json" "$RUN/evaluations/hough_seed3_synth_val.json" \
  --output "$RUN/H_VS_P.json"
```

H−D도 같은 명령의 reference를 direct로 바꿔 작성한다. 제공 compare는 수치만 계산하며 scientific_success를 자동으로 true로 쓰지 않는다.

주 지표: 같은 합성 population에서 유효대칭 하나를 프레임 전체에 적용한 8코너 평균오차/raw diagonal; point당1로cap, missing pose는1. 조건부 raw pixel median/P90와 실패 coverage를 별도로 보고한다. 이것은 transparent pilot metric이며 기존 MAIN6D/BOP와 동일하다고 부르지 않는다. 좌표계 검증 후 local canonical evaluator에 연결해 실제 동치/차이를 보고한다.

세 seed 각각 H가 matched P 대비 primary1% 이상 개선하고 median/P90 및 coverage 비악화, 원래 frame 평균≤10px가 >50px로 변한 새 catastrophic frame0일 때만 다음 실사 검토 후보로 본다. **1%,10/50px,0건은 미검증 예비 투자 기준**이며 안전 인증이 아니다. 이미 나온 결과를 이 기준으로 재판정하지 않는다. 강한 원래 모델과의 comparable한 운영 성능 참조도 빠지면 채택 근거로 승격하지 않는다.

H가 P는 이기고 D와 같다면 line branch 효과 후보이지 DHT 고유 효과는 미확립이다. 검증집합에서 C4표본0이면 C4주장미평가로 남긴다. 평균 개선이1~2프레임에 몰리는지 perframe 손익, 손상크기, seed 반복성을 보고한다. 후보/GT보정된 oracle 수치는 주 성능표와 분리한다.

Primary 비교 H−P 하나를 고정한다. H−D와 보조 비교의 통계적 주장은 별도 다중비교 계획을 먼저 고정해야 한다. 제공 session bootstrap은 seed를 먼저 평균한 paired frame 차이를 session 단위로 resample한다. synthetic renderer의 session/group가 실제 독립 단위인지 검증되지 않았으면 CI를 통계적 일반화 증거로 쓰지 않는다. 3seed를 독립24frame 같은 방식으로 부풀리지 않는다.

## 9. online 단계는 준비만 하고 실행하지 않음

합성 pilot에 유의미한 근거가 있어도 본 지시문 범위는 여기서 종료한다. 이후 전체 모델을 준비할 때:

- 공통 R0 backbone/neck/detector + 실제 native features + MultiScaleROI + 새 head/solver를 하나의 nn.Module로 연결.
- hook/detach 때문에 gradient가 끊기지 않는지 loss→solver→point/line→ROI→backbone의 실제 norm 및 parameter 변화 확인.
- GT를 읽는 loss와 observation forward를 분리. soft GT-crop teacher 강제입력 금지.
- stock box/class loss 유지와 기존 indexed keypoint/RLE loss 제거/일관변환을 실제 설치 소스 기준으로 검증. loss vector의 임의 index를 추측해 제외하지 않음.
- point/direct/hough 모두 같은 실제 data/augmentation/update/detector budget. 치수/K/좌표를 깨뜨리는 augmentation은 변환식을 검증하거나 모든 arm에서 공통 제외.
- 바뀐 모델마다 positive와 negative 전체 새 forward. 이전 negative output을 복사해서 새 모델 AP라고 하지 않음.
- module 가중치가 바뀌었는데 old cache로 성능을 계산하지 않음.
- 실제 진입면 정렬은 pose equivalence와 다른 작업. pose_valid=false를 제어에 전송하지 않음.
- dense reference DHT/다중 LM은 Jetson용 효율이 입증되지 않음. 전체 latency·peak memory·raw output parity·export 지원을 따로 보고.

학습 가능한 gradient 존재와 실제 경사로 최종 정확도가 좋아지는 것은 별개의 주장이다.

## 10. 완료 산출물

성공/실패 모두 아래를 남긴다:

1. `REPORT_KO.md`: 문제→변경 이유→소스/좌표계→실제 실행→수치→진행판정→한계.
2. `SOURCE_REGISTRY.json`, `ADAPTER_MAPPING.md`, `OBJECT_FRAME_CONTRACT.json`, `DIMENSION_SYMMETRY_AUDIT.json`.
3. 실제 train/cal/val manifest/SHA, detector 실패 포함 원모집단 ledger, 이미지 바이트 확인 범위.
4. raw feature/전체 predictor parity, independent coordinate alignment, generated/local/GPU 검증과 실패 기록.
5. 사용 config·source freeze, 초기 state SHA, 실제 batch trace, optimizer step, checkpoint SHA, 손실/gradient.
6. 각 군/seed의 예측좌표·R,t·모든 후보energy·validity·선택, perframe 평가, 원점수와 선택의 구분.
7. 제공 metric과 canonical metric의 대조 또는 `NOT_RUN` 사유. source/ROI/GT누출 및 입력파일 불변성 검토.
8. 성공 문구와 별개인 실행상태, 실제 수행 범위, 모델/데이터 한계.
9. 대용량 특징/가중치 없이 재계산 가능한 작은 수치·코드·문서를 `pallet_point_line_v5_review.zip`으로 묶음. 비밀키/토큰/알림url 제외. 원파일은 삭제하지 않음.

차단되었을 때는 차단사유와 확인한 자료까지 제시한다. 누락을 메우려고 무관한 새 모델/렌더/GT 수정으로 우회하지 않는다.
