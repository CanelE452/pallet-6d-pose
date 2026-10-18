# 보관 결과 검산과 소스 재사용

이 안내는 실제 캠페인의 등록 입력·관측·후보·가중치·split·seal을 보존한 상태에서 사용하는 명령 목록이다. 공개 파일만으로 private RGB·후보 bank·학습 checkpoint·정답을 복원할 수 없다. 원 데이터와 이전 실행 디렉터리는 읽기 전용으로 두고, 각 생성 단계는 비어 있는 새 출력 디렉터리를 사용한다. 이미 봉인한 결과를 덮어쓰거나 과거의 중단·복구 receipt를 새 실행에 만들어 넣지 않는다.

[보고서](REPORT_KO.md), [기계 결과](EVALUATION.json), [입력 요약](INPUTS_PUBLIC_SUMMARY.json), [실행 환경](ENVIRONMENT.json), [모델·감독 정의](MODEL_AND_SUPERVISION.md), [방법·근거](METHOD_AND_SOURCES.md), [전체 비용](COST_AND_EXECUTION.json), [F1 추가 학습 비용](R_F1_COST_AND_EXECUTION.json), [오프라인 공개 검사](PUBLIC_PACK_AUDIT.json)를 함께 읽는다. [소스 디렉터리](../../../scripts/research/pallet_risk_information_v1/)의 CLI가 실제 인자와 판정의 기준이다.

## 1. 실행 환경과 입력 복원

`PY`는 등록 Python 환경의 실행 파일, `P_REPO`는 체크아웃, `P_RUN`은 이 캠페인의 private 보관 루트, `P_OLD`는 원 D1 bank, `P_REF`는 봉인한 실사 참조 디렉터리, `P_REAL_FEATURES`는 등록한 원 실사 feature 파일을 가리킨다. 각 경로는 로컬 보관 위치로 지정한다. 공개 요약의 치환 경로는 실행할 수 있는 실제 입력 경로가 아니다.

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "${P_REPO}"
"${PY}" -m scripts.research.pallet_risk_information_v1.publish_pack --help
```

실제 등록 환경은 Python 3.10.20, NumPy 1.26.4, Torch 2.1.1+cu118이다. 세부 버전·원 파일 SHA·plan SHA는 환경과 입력 manifest에 보존한다. CPU 학습의 Torch intra/inter-op thread도 각각1이다. 공개된 선택 자세는 [SELECTED_POSES.jsonl.gz](SELECTED_POSES.jsonl.gz)에 있고, 원 bank·전체 logits·가중치는 공개하지 않는다.

새 namespace의 작은 구성·계약 시험은 다음과 같다. package-relative import를 위해 `-t .`를 포함한다. 이를 생략한 최초 discovery의 CLI 오류와 원 exit1 로그는 실행 기록에 남기며, 수정 명령의 성공으로 지우지 않는다.

```bash
"${PY}" -m unittest discover \
  -s scripts/research/pallet_risk_information_v1 -t . -p 'test_*.py' -v
```

## 2. 기존 봉인 결과의 독립 숫자 검산

아래 명령은 보관한 `EXPERIMENT_PLAN.json`, `PREDICTION_MANIFEST.json`, 원 `PREDICTIONS.jsonl.gz`, 원 실사 feature, reference seal, `evaluation_v1`을 연결한다. 출력은 새로운 경로다. 모델 forward·PnP·후보 생성·새 학습은 수행하지 않는다. 공개 선택 ledger는 원 prediction archive의 대체 파일이 아니다.

```bash
"${PY}" -m scripts.research.pallet_risk_information_v1.independent_evaluation \
  --run-dir "${P_RUN}" --reference-dir "${P_REF}" \
  --features "${P_REAL_FEATURES}" \
  --evaluation-dir "${P_RUN}/evaluation_v1" \
  --output "${P_RECHECK}/independent_evaluation" --scoring-GO
```

원 생성기의 실제 feature 파일명은 `INPUT_MANIFEST.json`의 원 SHA와 대조해 지정하고 존재하지 않는 경로를 추정해 쓰지 않는다. 원 평가를 다시 만드는 경우에도 `evaluate.py --help`에 따라 별도 출력 경로를 사용한다. 참조를 읽기 전에 모든 실제 등록 군과238장의 예측이 봉인됐는지 검사한다.

## 3. 특징·감독·학습 생성기의 재사용 범위

이 캠페인은 원 D1의1,003 합성장과768/235 split을 재사용한다. 원 F0/action/eligibility는 고정이다. 아래는 단계별 primitive이며, 공개 파일만으로 전체 실행을 시작하는 명령이 아니다. 필요한 원 cache·config·seal과 사전 고정 plan을 먼저 복원해야 한다.

```bash
"${PY}" -m scripts.research.pallet_risk_information_v1.feature_support produce --help
"${PY}" -m scripts.research.pallet_risk_information_v1.feature_support normalizer --help
"${PY}" -m scripts.research.pallet_risk_information_v1.train --help
"${PY}" -m scripts.research.pallet_risk_information_v1.train_visual --help
"${PY}" -m scripts.research.pallet_risk_information_v1.predict --help
```

F1 `produce`는 `--features --feature-seal --cache --config --output --plan-go --spec`으로 원 고정 자세만 평가한다. `--deny`와 `--denied-root`에는 실제 참조·평가·B 진단 경로를 명시한다. normalizer는 `--features --feature-seal --train-ids --output`으로 train768의 유효 scalar만 fit한다. 추가144값+144validity와 원 F0를 연결하고, ZERO는 동일 입력의 추가288 전체만0으로 만든다.

R/F 학습은 `train`의 `--features --feature-seal --labels-dir --targets --target-seal --split-lock --output --arm --seed --training-GO`를 사용한다. F1/ZERO에는 동일한 `--extra-features --extra-seal --extra-normalizer`를 추가한다. 대표 seed는20261018이며 세 seed20261018/20261019/20261020을 모두 보존한다. 기본12 fits와 사전 V gate를 통과한 조건부6 fits만 허용하며 전역 optimizer attempt 상한은18이다. 실패 attempt를 지우고 재시도를 새 실험처럼 세지 않는다. V 생성기는 별도 `train_visual` launcher의 gate·RGB cache·추가 특징 인자를 필요로 한다. 군별 synthetic validation objective의 첫 strict minimum으로 checkpoint를 고르며 서로 다른 loss 숫자를 직접 비교하지 않는다.

B 정답 대입은 `reference_support --help`의 별도 CLI로만 실행한다. `--references --reference-seal`은 이 진단 프로세스에만 전달하고 출력은 `b_reference_*` private 디렉터리로 제한한다. 이 결과를 F1·모델·정상화·후보 bank에 넣지 않는다. 원 C1/C2의 허용 phase와 원 phase0을 분리하며, 원 GT를 정규직교화하거나90° 대칭을 추가하지 않는다.

## 4. 공개 패키지와 오프라인 검증

이 단계는 root가 실제 예측·평가·그림·갤러리·모델 시각 검토를 마친 뒤에만 실행한다. `RAW_PRIMARY_RGB_INVENTORY.json`은 private의 실제192 원본 경로·SHA·canvas 목록이고, privacy plan은 등록한 원 display-only plan이다. 공개 권한은 PRIMARY192의 파생 RGB 패널만 포함한다. 합성 RGB와 원 RGB는 포함하지 않는다.

```bash
"${PY}" -m scripts.research.pallet_risk_information_v1.publish_pack prepare \
  --run-dir "${P_RUN}" --repo "${P_REPO}" \
  --public-root "${P_REPO}/_docs/experiments/pallet_risk_information_v1" \
  --evaluation "${P_RUN}/evaluation_v1" \
  --rgb-inventory "${P_RUN}/RAW_PRIMARY_RGB_INVENTORY.json" \
  --privacy-plan "${P_DISPLAY_PLAN}" \
  --rights-receipt "${P_REPO}/_docs/experiments/pallet_risk_information_v1/RIGHTS_AND_PRIVACY.json" \
  --publication-GO
```

동일 인자로 `prepare`를 `verify`로 바꿔 검사한다. 추가 소형 요약은 `--extra-map`에 상대 경로와 `summary/csv` 변환을 명시한다. 생성된 파생 파일을 다시 만들 때만 `--replace-derived`를 사용하며 이전 bytes를 private에 보존한다. 이 옵션도 보고서·그림·알 수 없는 기존 파일의 덮어쓰기를 허용하지 않는다. 검사기는 원 숫자의 단순 변환을 재현하고 전량 denominator·원 선택 자세·relative link·파일 SHA·PNG/JPEG decode·EXIF·경로/token·원 RGB와 동일 bytes/pixels를 확인한다. 이미지 decode는 실제 픽셀 검토나 물리 정답 인증을 대신하지 않는다.

`PUBLIC_PACK_MANIFEST.json`, 자체 audit와 명시한 후속 remote receipt는 순환 SHA를 피하도록 payload hash 집합에서 제외한다. 실제 결과 commit A를 push한 뒤 별도 네트워크·blob 검증 receipt가 불변 A를 연결하고, 그 사실을 기록한 metadata commit B를 만든다. 패키저는 Git/network 변경과 remote 완료 판정을 수행하지 않는다. 현재 원격 상태는 실제 publication receipt만 근거로 한다.

## 5. 실패·권리·보기

[최초 원본 보존 검사 FAIL](ORIGINAL_PRESERVATION_INITIAL_FAIL.json)은 status enumeration 옵션 차이로 생겼으며 삭제하지 않는다. 동일한 원 START 명령으로 한 [normal-mode 재검사](ORIGINAL_PRESERVATION_VERIFICATION.json)와 각각의 원 SHA를 함께 보존한다. 대형 원 status 파일명 목록은 private에 남긴다. B의 원 strict roundtrip FAIL과 raw/backend 표현 차이도 별도 검산 기록에 남긴다. 기타 실패·미실행·부분 비용·미계측 시간은 원 receipt의 상태 또는 null로 유지한다.

[갤러리](gallery/index.html)는 HTML과 하위 자산을 함께 내려받아 로컬 브라우저에서 연다. 저장소 웹의 HTML 미리보기는 실행 페이지가 아닐 수 있다. 공개 redaction은 표시 패널에만 적용하고 추론·계측 좌표를 바꾸지 않는다. 모델 픽셀 검토는 `PIXEL_REVIEW.json`의 실제 파일·SHA에 한정되며 human review와 독립 실측 정답은 주장하지 않는다. 새 독립 실사 test는0장이고, 운영 기본값·논문·과거 결과는 자동 교체하지 않는다.
