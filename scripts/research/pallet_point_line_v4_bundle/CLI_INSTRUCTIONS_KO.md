# 실행 지시문 — 팔레트 점·선 증거 분리 실험 v4

## 0. 작업 범위와 최상위 목표

`CanelE452/pallet-6d-pose`에서 이 패키지를 사용해 **합성 데이터만으로 학습하는 점·선 아키텍처**를 검증하라. 목표는 **원래 정확한 점을 보존하면서, 실사에서 잘못된 위치 또는 코너 번호 배치를 교정하는 것**이다. 소비처는 사용자의 아키텍처 채택 판단 및 향후 논문의 방법·실험 절이다. 프로그램 실행 완료, 떨어지는 학습 손실, 한 장의 성공, oracle 수치, HTML 생성은 목표 달성이 아니다.

이 지시문을 사용자가 CLI에 전달한 것을 아래의 제한된 실행 범위 승인으로 취급한다. 단계마다 같은 승인을 반복해서 묻지 않는다. 데이터/정답 계약이 해결되지 않으면 추측하지 말고 차단 사유를 산출한다.

**실행 범위:** 소스·데이터 계약 확인 → 후보 가용성 확인 → 네 군 × 세 seed의 작은 캐시 모델 학습 → 합성 검증 → 기준 통과 시 기존 실사 DEV 평가 → 실사 기준까지 통과 시 네 군 × 세 seed의 제한된 전체 네트워크 공동학습과 평가. 앞 단계가 실패하면 다음 단계는 실행하지 않는다. 독립 FINAL은 열지 않는다.

**수정 금지:** 기존 실험 폴더, 기존 GT·키포인트 번호·카메라 규약, 기존 checkpoint, 기존 성능 판정, 전역 Ultralytics 설치, unrelated working-tree changes. 새 렌더, 실사 레이블 학습, pseudo-label, self-training, 정답 기반 추론 후보 생성/선택, 결과를 본 뒤 지표 변경은 이번 범위에 없다. Git commit/push, 외부 서비스 알림, 사용자 파일 삭제는 실행하지 않는다.

[확인] 이 패키지를 작성할 때 확인한 저장소 기준은 `6a68452ea9a17a69860981fea1c5a3dc234e3dfa`다. 최신 로컬 변경이 이와 다르면 아래 소스 바인딩부터 대조한다. [확인] 여기서 제공하는 코드에는 실제 팔레트 이미지·checkpoint·GPU가 없었다. 제공 코드의 검증은 CPU 소프트웨어 테스트다. 정확한 결과는 `evidence/LOCAL_VALIDATION.json`을 읽는다.

## 1. 먼저 바로잡을 기존 해석

[확인] 기존 `pallet_dht_structured_v2/model.py`는 이미 **끝점 패치, 순서를 보존한 유한 선분 내부 특징, 전체 배치 Transformer, 역할별 Hough 단서**를 사용한다. `data_ops.py`는 이미 후보별 정답 오차를 사용한 listwise ranking, 비용 회귀, 코너 오차 회귀를 수행한다. 따라서 다음을 새 기여처럼 제안하거나 재구현하지 않는다.

- “이번에 처음 끝점과 선분을 분리했다.”
- “이번에 처음 후보의 정답 오차로 선택기를 감독한다.”
- “지금까지 점과 선은 연결되지 않았다.”

[확인] 기존 v2는 같은 ID의 baseline 위치·변위·confidence와, ID 순서가 있는 baseline context도 입력한다. [추정] 이것이 잘못된 번호를 유지하게 하는 지름길일 가능성은 있으나 원인으로 확정된 것은 아니다.

**이번의 추가 검증 질문은 다음으로 제한한다.**

1. 동일한 새 읽기 모듈에서, 끝점 외의 실제 유한 선분 내부 관측이 추가로 유용한가?
2. 동일 후보·영상 특징 조건에서 역할별 DHT 단서가 추가로 유용한가?
3. 동일 ID의 원래 위치·confidence에 의존하지 않는 표현이 큰 번호 오류를 줄이는가?

P3+P4 다중 해상도 읽기는 네 군 모두 공유한다. 이번 네 군 비교는 **해상도 단독 효과**를 식별하지 않는다. 과거 P4-only v2와 수치가 달라져도 그 차이를 P3의 인과적 기여라고 쓰지 않는다.

## 2. 목적 트리와 가지치기

| 목적 경로 | 필요한 행동과 이유 두 가지 | 제대로 달성하기 위한 조건 |
|---|---|---|
| 최상위 → 상보적인 관측인지 식별 | 동일 후보와 backbone에서 증거만 바꾼다. 후보 생성 이득과 읽기 이득을 분리하고, 정보가 없어서 실패한 경우와 선택을 못한 경우를 나눈다. | 네 군의 후보 좌표·유효성·GT·초기값·batch trace 동일. P/S도 공통 DHT backbone/candidate를 공유한다고 명시. |
| 최상위 → 정상 점 보호 | identity를 선택지와 채택 기준 양쪽에 둔다. 나쁜 후보들의 최선을 고르는 오류를 막고, 평균 개선에 가려진 정상 점 손상을 측정한다. | 합성 calibration만으로 margin 선택. 원래 ≤10px 점의 >10px 전환율 별도 보고. |
| 최상위 → 번호 오류와 위치 오류 분리 | 원래 점의 집합 기반 reference와 같은-ID reference를 대조한다. 번호 재대응을 물리적인 큰 이동으로 처벌하는 효과와 실제 위치 보정 효과를 나눈다. | 원래 GT ID 유지. 전체 배치 후보 사용. GT permutation은 oracle 설명에만 사용. |
| 최상위 → 단일 학습 모델로 검증 | 작은 읽기 실험 후 통과한 경우만 전체 네트워크에서 다시 비교한다. 고정 표현의 한계를 드러내고, 실제 최종 출력·6D·검출의 손익을 함께 판단한다. | 캐시 head의 성공을 전체 모델 성공으로 보고하지 않음. 전체 모델은 새 positive/negative 추론 필수. |

**가지치기:** PCGrad 재탐색, 선 손실 계수 grid, 전역 DLT/beam 확대, 새 렌더, GT 수정, 임의 전체 C4-equivalent 채점은 제거한다. 기존 근거를 넘어 목적을 넓히지 않는다. 양호점 보호를 이유로 작은 이동 상한을 걸어 수백 px 번호 재배정을 구조적으로 불가능하게 만들지 않는다.

| 단계 | 예상 결과 | 최상위 목적 지지 | 첫 독자 질문에 대한 답 |
|---|---|---|---|
| 후보 진단 | 자연 발생 오차에서 기존 후보가 기준보다 나은 경우가 실제로 존재 | 부분. 선택기를 배울 여지는 있지만 정확도는 아님 | “정답에 가까운 후보가 애초에 있어?” |
| 작은 네 군 비교 | H가 P와 기준 출력보다 좋아지고 정상 점 손상이 제한됨 | 부분. 고정 backbone 조건의 증거 | “선 증거를 추가해서 좋아진 거야, 모델이 커져서야?” |
| 전체 네트워크 | 동일 예산 비교에서도 최종 2D/6D·검출 손익이 유지 | 도달 가능. 여전히 재사용 DEV 한정 | “실제로 학습·추론하는 하나의 모델에서도 좋아져?” |
| 실패 기록 | 후보/읽기/역할 대응/전이 중 실패 위치를 구분 | 다음 목적 선택에는 도움, 성능 기여는 아님 | “무엇을 실패했다고 말할 수 있어?” |

## 3. 제공 코드와 CLI 담당 범위

**이미 작성된 코드를 다시 만들어서 대체하지 말라.** 먼저 그대로 테스트한 뒤, 필요한 저장소 연결 부분만 작성한다.

| 제공 파일 | 기능 |
|---|---|
| `pointline_v4/geometry.py` | 명시적 affine, content-mask 기반 특징 샘플링, 유한 선분 내부 읽기, 선·두 끝점의 동일 mode 관계 |
| `model.py` | P/S/H/HA 네 군의 다중 해상도 전체 배치 scorer |
| `objective.py` | 후보별 정답 오차, listwise 및 직접 회귀. GT는 모델 입력 밖에 있고 detach됨 |
| `policy.py` | identity를 포함한 이산 배치 선택. signed gate/보정 잔차 없음 |
| `adapter_helpers.py` | P3/P4 단일-forward hook, 명시적 feature affine·content mask, 확인한 v2 입력→새 Observation 매핑 |
| `cache_io.py` | 새 export 계약, 파일 SHA 검증, 관측/GT 별도 로딩 |
| `train_cache.py` | 실제 AdamW 캐시 head 학습, 초기값·plan·optimizer step·checkpoint 기록 |
| `score_cache.py` | GT 파일을 열지 않는 실제 모델 추론 |
| `evaluate_scores.py` | 합성 margin 보정 및 고정 규칙 평가. 8코너/9점 분리 |
| `oracle_audit.py` | 기존 전체 배치 후보에 대한 합성 전용 oracle 상한 진단 |
| `compare_evaluations.py` | 사전 지정 H 대 P/기준 비교, 세션 bootstrap 및 seed/입력 바인딩 확인 |
| `tests/` | 소프트웨어·좌표·마스크·경사·정보경로·CLI 통합 테스트 |

**CLI가 새로 해야 할 일:** 실제 로컬 원본 경로·checkpoint 확인, 제공 `adapter_helpers`를 검증된 predictor의 정확한 객체/forward에 연결, 새 export 생성, 실제 dataset/GT matching 매핑, 전체 단계 runner, 기존 canonical evaluator 연결, 통과 시 online 전체 네트워크 학습 adapter, 결과 보고서. 제공 코드는 이 부분을 이미 수행했다고 주장하지 않는다.

외부 계정 연결, 새 Hough 구현, 새 데이터 생성, 새 GT 도구는 필요 없다.

## 4. 로컬 보존과 소스 등록

1. 저장소 루트를 `git rev-parse --show-toplevel`로 확인하고 `git status --short` 및 관련 diff를 저장한다. `git reset`, `clean`, 강제 checkout은 금지한다.
2. 패키지는 새 경로 `scripts/research/pallet_point_line_v4_bundle/`에 둔다. 새 결과 루트는 `data/pallet/results/pallet_point_line_v4/`로 하되 기존 내용이 있으면 덮어쓰지 말고 충돌로 중단한다.
3. 원본 경로를 추측하지 말고 다음 **실제로 확인한 파일**에서 읽는다.
   - `scripts/research/pallet_dht_structured_v2/{model,data_ops,proposals,cache,infer,train,evaluation}.py`
   - `scripts/research/pallet_dht_decoder_probe_v1/{cache,geometry}.py`
   - `scripts/research/pallet_dht_joint_v1/{hough_block,integration,line_targets,train,evaluate}.py`
   - `scripts/research/pallet_line_pose_v1/source_data.py`
   - `_docs/notes/{pallet_dht_structured,pallet_dht_local,pallet_dht_decoder_probe,pallet_dht_global_layout,pallet_dht_joint,pallet_dht_coupling}.md`
   위 파일들의 **내부 객체·키는 각 파일을 읽고 확인한 뒤** 쓴다. 이 패키지의 export 필드와 원래 cache 필드를 혼동하지 않는다.
4. `SOURCE_REGISTRY.json`에 소스 commit, 실제 파일 SHA-256, checkpoint 절대 경로/SHA, 환경 버전, 원본 split/annotation locator, 기존 predictor 계약을 등록한다.
5. 관찰과 판정을 나눠 적는다. 문서상 주장만 확인했으면 “보고서 기록 확인”으로, 실제 배열/실행으로 확인했으면 그 범위를 적는다. 사실은 [확인], 원인 해석은 [추정], 새 숫자/설계 선택은 [미검증 제안]으로 구분한다.
6. 공개 GitHub에는 큰 cache·checkpoint가 없을 수 있다. 로컬에서 없으면 `BLOCKED_MISSING_ARTIFACT`로 정확한 누락 경로/필요 SHA를 보고한다. 작은 공개 파일을 실제 특징 cache로 취급하지 않는다.

다음 코드는 패키지 설치 후 실제로 존재하는 명령이다.

```bash
REPO="$(git rev-parse --show-toplevel)"
KIT="$REPO/scripts/research/pallet_point_line_v4_bundle"
export PYTHONPATH="$KIT${PYTHONPATH:+:$PYTHONPATH}"
cd "$KIT"
python -m pytest tests -q
python -m pointline_v4.train_cache --help
python -m pointline_v4.score_cache --help
python -m pointline_v4.evaluate_scores --help
```

Python 환경은 현재 저장소에서 사용하던 PyTorch 환경을 이용한다. 패키지가 있다는 이유로 전역 라이브러리를 업그레이드하거나 CUDA용 Torch를 CPU용으로 교체하지 않는다. 불필요한 모델/논문 repository 전체 다운로드도 하지 않는다.

## 5. 네 실험군 — 변경 축을 고정하라

| 군 | 끝점/코너 영상 | 실제 선분 내부 영상 | 명시적 역할 DHT | baseline reference |
|---|---|---|---|---|
| P | P3+P4 | 읽지 않음. 끝점 특징 보간으로 같은 크기 입력 구성 | 제외 | 점 집합 기반 |
| S | P3+P4 | 읽음 | 제외 | 점 집합 기반 |
| H — 주 방법 | P3+P4 | 읽음 | 포함 | 점 집합 기반 |
| HA — 역할 anchor 대조 | P3+P4 | 읽음 | 포함 | 같은 ID의 기존 좌표/confidence |

[확인: 제공 구현] 기본 설정 `channels=(64,128), visual=16, width=64, layers=2`에서 네 군의 등록·학습 가능 파라미터는 각각 **160,226개**다. [중요] P/S에서 명시적 Hough 입력 열은 비활성이다. 따라서 **등록 파라미터/초기 tensor shape 일치**이지 모든 군의 실제 활성 용량이 엄밀히 같다는 주장은 금지한다. gradient 원소 수와 모듈별 norm도 따로 보고한다.

공통점: 같은 backbone snapshot, 원래 예측, 후보 bank, P3/P4, content mask, GT와 loss, 초기 state, 업데이트 수, batch 순서, margin grid. candidate index, 후보 생성 family ID, GT-derived visibility, GT-goodness 플래그를 scorer 입력에 추가하지 않는다.

고정 주 비교: **H 대 P 및 변경하지 않은 공통 backbone 출력**. 부 비교: S−P(유한 선분 추가), H−S(명시 DHT 추가), H−HA(same-ID reference 변화). 부 비교의 통계적 주장은 세 비교 Bonferroni 95% family 구간을 사용한다. 가장 잘 나온 군을 사후 주 방법으로 바꾸지 않는다.

**완전한 no-Hough 비교가 아님:** P/S 역시 Hough로 학습된 공통 backbone과 Hough 후보를 공유한다. 여기서 측정하는 것은 그 조건에서 읽기 증거의 증분 효과다. H가 P보다 좋아도 “DHT 전체의 효과를 순수하게 증명했다”고 쓰지 않는다.

HA는 과거 v2의 완전한 재현이 아니라 **새 scorer에서 같은-ID reference만 복원한 대조군**이다. 과거 v2의 다른 context 표현까지 복원했다고 주장하지 않는다.

**기하적 역할 번호는 유지한다:** 8개 코너와 12개 구조 edge는 기존 camera-facing 계약을 따른다. 원래 8번 중심은 후보마다 정확하게 유지한다. “C4 후보”는 기존 번호 재대응 후보이지, 모든 팔레트가 물리적으로 C4 대칭이라는 선언이 아니다. 실사 GT 번호 변경이나 임의 Hungarian matching을 주 평가에 넣지 않는다.

## 6. 데이터 계약과 cache export

### 6.1 공통 backbone과 비교 모집단

[확인: 소스] 기존 structured-v2 raw inference는 joint seed1 snapshot SHA `0960fb32fd99fd0837a588792e07727298fc6f88f1e4ec3464efd69bd5574d37`를 명시한다. 로컬 실제 checkpoint와 해당 프로토콜을 검증한 뒤 사용한다. 다른 모델을 같은 baseline 이름으로 대체하지 않는다.

작은 실험은 기존 decoder/structured 계열의 source pool을 재사용한다. **source train 1,792 / calibration 256 / synth_val 512**를 원본 manifest에서 확인하고 고정한다. 기존 2,048 source train 묶음 중 1,792/256 분할의 계보를 보존한다. 어떤 파일이 어떤 split인지 폴더 이름만으로 추정하지 않는다.

이 데이터는 새 head 기준 분리이지, 기존 backbone이 한 번도 보지 않은 holdout을 의미하지 않는다. 실사 **positive319, 기존13세션**은 이미 여러 차례 본 DEV다. FINAL은 별도이며 이번 실행에서 읽지 않는다. 프레임 ID와 이미지 SHA 양쪽의 중복을 확인하고 burst/session 경계를 실제 메타데이터로 확인한다. 실제 중복이면 조용히 삭제하지 말고 계약 충돌로 보고한다.

주 학습에는 **native 예측만** 사용한다. 이전 v2의 인위적인 번호교란·국소변형 복구율로 실사 성능의 진입 기준을 대신하지 않는다. 인위적 교란은 별도의 기능 진단이며 main 학습/모델 선택/성공 판정에 쓰지 않는다. GT 근처 후보를 새로 주입하지 않는다.

### 6.2 실제 특징 읽기

공통 predictor 한 번의 실제 forward에서 원래 box/score/9점, 역할별 Hough 예측, **동일 forward의 native P3/P4**를 캡처한다. 기존 `HoughFeatureFusion`의 feature 입력/출력 위치를 확인하고 hook 위치를 한 곳으로 고정한다. 권장 계약은 해당 블록 **입력**의 P3/P4다. 채널·stride·shape는 hook 실제 출력으로 확인한다. 저장소 기본 구성은 64/128이지만 실행 결과를 검증하지 않고 단정하지 않는다.

제공 `capture_native_features`는 forward를 실행하지 않으며 검증된 블록 입력을 한 번만 캡처한다. 원래 predictor 호출은 CLI가 연결한다. `explicit_feature_affine`의 stride/offset은 실제 계약을 읽어 명시하고, `from_verified_v2_batch`는 확인한 v2 model-input dict만 허용한다. 전달 snapshot의 동일성까지 자동 증명하지는 않는다.

P4를 단순 upsample한 것을 native P3라고 부르지 않는다. 학습용 캡처는 backbone eval/frozen이며 BN 갱신을 차단한다. GPU 연산이 비결정적이면 캐시의 차이를 숨기지 않는다. 새 cache로 일관되게 정의한 baseline과 기존 baseline 차이를 별도로 보고하며, 엄격 parity 실패를 PASS로 덮지 않는다. 원래 수치 허용차를 결과에 맞춰 늘리지 않는다.

각 특징 plane에 **raw 좌표 → 해당 feature의 픽셀 중심 좌표** affine을 명시한다. 여기서 feature 중심 index는 0,1,...,W−1이다. 원래 코드의 input 좌표/stride 관계와 half-pixel 기준을 impulse/ramp fixture 및 실제 좌표 왕복으로 검증한다. 제공 sampler는 이 affine을 신뢰할 뿐 YOLO offset을 추측하지 않는다.

`content_valid`는 raw image가 존재하는 특징 cell을 표시하는 예측/전처리 기반 마스크다. GT 객체 마스크가 아니다. reflect100과 LetterBox, 사각 입력 bottom-padding을 구분한다. 제공 코드는 bilinear support가 content 밖에 걸리면 그 query를 버린다. **이미 backbone 수용영역에 들어온 패딩 영향까지 제거한 것은 아니다.** 숨은 구조선이 실제 보이는 경계라는 가시성 감독도 아니다.

### 6.3 후보 bank

기존 structured-v2의 **GT-free 전체 배치 후보 생성**을 재사용한다. 실제 `build_proposals`와 호출부를 읽고 입력·출력을 확인한다. 64슬롯, identity, 번호 재배열, 교점 후보의 실제 의미를 `CANDIDATE_CONTRACT.json`에 기록한다. 기존 코드를 변경한다면 그 순간 별도 실험으로 취급하며 자동 진행하지 않는다.

네 군의 candidate tensor는 byte-identical이어야 한다. 사전 유효성 검사에서 invalid인 슬롯은 값으로 baseline을 복사하고 mask=false로 내보낸다. identity index0은 항상 원래 예측이다. 모델 score는 후보 index를 보지 않는다. 중복된 후보 수와 별도 identity 중복 여부를 감사하되 결과를 본 뒤 중복 제거 정책을 바꾸지 않는다.

원래 예측점 중 누락이 있으면 해당 프레임은 **identity-only**로 유지한다. 추론에서 GT를 보며 “복구 가능한 프레임”만 골라 처리하지 않는다. 숫자 전송용 결측 placeholder와 원래 결측 mask를 구분하고 공식 출력은 원래 결측 의미를 유지한다.

### 6.4 새 export schema

아래는 **이 패키지가 정의한 새 인터페이스**다. 기존 repository cache가 이 필드를 이미 갖는다는 뜻이 아니다. CLI가 확인한 실제 source key → 아래 key의 매핑을 `ADAPTER_MAPPING.md`에 작성하고 fixture/실제 표본으로 검사한다.

관측 파일은 `Observation`의 필드만 가진 tensor dictionary이며 record당 B=1이다.

- `features`: P3/P4 FP32 tensor tuple. batch 처리를 위한 padding은 mask와 affine을 함께 보존한다.
- `raw_to_feature`: plane별 FP32 `[1,2,3]` affine tuple. 고정밀 FP64 원본은 provenance 메타데이터에 별도 보존한다.
- `content_valid`: plane별 bool `[1,1,H,W]` tuple.
- `baseline`, `point_valid`, `point_conf`: 각각 FP32 `[1,9,2]`, bool `[1,9]`, FP32 `[1,9]`. 유효성은 **예측 기반**이다.
- `layouts`, `candidate_valid`: FP32 `[1,K,9,2]`, bool `[1,K]`.
- `line_h`, `line_logits`, `line_valid`: `[1,12,4,3]`, `[1,12,4]`, `[1,12,4]`. FP32 raw 공간의 homogeneous line, FP32 원래 logit, bool 예측 mask다.
- `diagonal`: FP32 raw 영상 대각선 `[1]`.

별도 supervision 파일은 `points:[1,9,2]`, `supervised:bool[1,9]`, `matched:bool[1]`만 가진다. `supervised`는 물리적 가시성을 뜻하지 않는다. `matched`의 실제 IoU/검출 선택 계약은 기존 canonical scorer와 맞춘다. GT 기반 mask는 이 파일 밖, 특히 Observation으로 전달하지 않는다.

각 split manifest는 `schema="pointline_v4_export_1"`, `role`, `channels`, `records`를 가진다. record에는 `frame_id`, `session_id`, `origin`, `state="native"`, `observation`, `observation_sha256`, `supervision`, `supervision_sha256`가 있다. training/calibration origin은 `source_synthetic`, real은 `real_dev`로 기록한다. 실제 session_id를 파일마다 새로 만들어 bootstrap 표본 수를 부풀리지 않는다.

파일 경로는 manifest 폴더 아래의 상대경로이며 symlink로 밖을 가리키지 않는다. `torch.load(weights_only=True)`로 열 수 있는 Tensor/기본 자료형만 저장한다. pickled model/object를 데이터로 넣지 않는다. 원본 파일은 복사·변환 출처를 기록하고 그대로 보존한다.

## 7. 실행 전 무결성 gate

제공 단위 테스트 통과만으로 실제 연결을 PASS 처리하지 않는다. 다음을 실제 데이터 표본으로 확인한다.

**G1 — 좌표/정보 흐름:** raw↔input↔P3/P4 affine, 회전/수직/수평 선, theta seam, line equation 단위, reflect/LetterBox 구분, 숨은/화면 밖 코너 처리. 잘못된 기존 Hough 구현을 새로 만들지 말고 검증된 predictor의 mode와 raw line 변환을 재사용한다. 같은 두 끝점은 같은 mode와 대조한다.

**G2 — 데이터:** train/cal/val ID·image hash 분리, source-only 감독, GT 변경0, 후보 생성 GT 의존0, inference가 GT 파일 없이도 실행됨. 실제 scoring 단계에서 supervision 파일 읽기 hook으로 접근 0을 검사한다.

**G3 — 비교:** 네 군 같은 candidate/feature/prediction/mask, 같은 seed의 초기 state hash와 실제 batch trace, loss mask와 source snapshot 동일. P/S의 explicit Hough cue 0, P의 disjoint interior query 0, H의 Hough 입력 경사/민감도, H의 baseline-ID 재정렬 불변성 및 HA의 대조 민감도를 확인한다. 불변성 검사는 baseline0이 아니라 고정된 비identity 후보의 점수로 한다.

**G4 — 목적 도달 가능성:** train native 예측의 전체 배치 oracle 상대 이득≥1%, 원래 프레임 평균보다 ≥1raw px 나은 후보가 있는 자연 발생 train 프레임≥32. 이는 [미검증 제안]인 예산 보호 기준이지 논문 표준이 아니다. identity를 포함하므로 oracle 비악화는 자명하며 그것만으로 PASS하지 않는다. 후보별 독립 코너 oracle과 실제 하나의 전체 배치 oracle을 섞지 않는다.

G4가 실패하면 `NATIVE_PROPOSAL_SIGNAL_INSUFFICIENT`로 종료하고 어떤 후보가 부족한지 보고한다. 이를 모든 점·선 아키텍처의 불가능성으로 해석하지 않는다. 인위적 교란이나 GT 후보 주입으로 G4를 통과시키지 않는다.

**G5 — 실행 자원:** GPU 가용성, FP32 실제 batch16 smoke, CPU/disk 여유, 기존 프로세스 소유관계. 충분하지 않으면 기다리는 무한 loop 대신 `BLOCKED_RESOURCE`로 종료한다. batch 축소가 필요하면 본학습 전에 microbatch·accumulation·effective batch와 모든 군의 동일성을 잠그고 audit한다. 진행 중 한 군만 학습 조건을 바꾸지 않는다.

문제가 없는 경우 `PREFLIGHT.json`과 `ADAPTER_MAPPING.md`를 저장한다. `PROTOCOL_TEMPLATE.json`의 runtime-bound 필드를 실제 값으로 채우고 `locked=true`인 `PROTOCOL_LOCK.json`을 별도 생성한다. 원래 템플릿은 수정하지 않는다. manifest SHA와 모든 core `.py` SHA를 채우지 않으면 제공 trainer가 본학습을 거부한다.

## 8. 단계 A — 작은 캐시 모델 실제 학습

[미검증 제안: 고정 예산] 네 군 P/S/H/HA × seeds1,2,3, 각 **2,000 실제 optimizer updates**, batch16, AdamW lr0.001, weight_decay0.0001, FP32, gradient clip5. cosine LR의 최저비율0.1. 데이터는 native source train 1,792장. 마지막 checkpoint만 평가한다. 중간 best checkpoint·best seed 선택은 없다.

기본 모델은 visual16/width64/2 layers다. loss는 기존 v2 계열의 후보 quality 학습을 유지하는 새 코드이며 temperature0.25, SmoothL1 beta0.1, regression weight1, corner weight0.25로 고정한다. 이것들은 [미검증 제안]인 이번 설정이다. v2에서 그대로 복사한 하이퍼파라미터라고 쓰지 않는다.

정답 비용은 **중심 제외 8코너**의 유효점 평균 오차에 `log1p(error/(0.01*raw_diagonal))`를 적용한다. 원래 점 또는 GT 좌표를 이 비용으로 직접 최적화하지 않고 target을 detach한다. 후보 scorer에는 첫 step부터 직접 감독이 간다. 이산 선택에 대한 미분 가능성을 주장하지 않는다.

```bash
# RUN, EXPORT는 CLI가 실제로 만든 새 경로다. 아래 이름은 새 export 계약이다.
python -m pointline_v4.oracle_audit \
  --manifest "$EXPORT/train.json" --output "$RUN/NATIVE_ORACLE_AUDIT.json"

# 실제 실행에서 12개 군을 순서대로 실행하고, 각 완료 후 바인딩을 검사한다.
python -m pointline_v4.train_cache \
  --manifest "$EXPORT/train.json" --protocol "$RUN/PROTOCOL_LOCK.json" \
  --output "$RUN/heads/H_seed1" --arm H --seed 1 --device cuda:0
```

제공 trainer는 전체 matrix scheduler가 아니다. CLI가 얇은 foreground driver를 작성하되, 제공 trainer 내부를 재작성하지 않는다. 실패 시 다음 군을 무조건 계속 돌리지 않는다. 같은 예산을 다 채우지 않은 군을 완료 모델로 취급하지 않는다. 생성 fixture 전용 `--allow-generated-fixture`와 `--smoke`를 본학습에 사용하지 않는다.

모든 checkpoint의 실제 update 수, optimizer state, 초기 state, final state, batch plan을 검증한다. seeds만 다르고 실제 state/trace가 같은 경우 3회 반복으로 인정하지 않는다. 캐시 학습 3seed는 **고정 backbone 위 새 head의 3seed**이며 backbone 사전학습 전체의 3회 독립 반복이 아니다.

## 9. 단계 B — 합성 규칙 선택과 고정 평가

### 9.1 합성 calibration

각 head의 calibration256 **모든 후보 score를 먼저 저장**한다. 그다음 GT를 읽어 고정 grid `identity-only, 1, 0.5, 0.25, 0.1, 0.05, 0.025, 0`에서 margin을 선택한다. 숫자는 학습된 log-error cost의 차이이지 확률·픽셀 오차 허용치가 아니다.

선택 목적은 유효 프레임별 8코너 평균 오차/raw diagonal의 평균이다. median/P90 비악화 및 원래 ≤10px 점의 >10px 전환율≤1%를 제약으로 둔다. 같은 목적값이면 더 보수적인 먼저 열거한 규칙을 유지한다. 유효한 향상이 없으면 identity-only가 선택되어야 한다. “나쁜 후보 중 최선”을 채택하지 않는다.

```bash
python -m pointline_v4.score_cache --manifest "$EXPORT/calibration.json" \
  --checkpoint "$RUN/heads/H_seed1/checkpoint_final.pt" --output "$RUN/scores/H_seed1_cal.json" --device cuda:0
python -m pointline_v4.evaluate_scores calibrate --manifest "$EXPORT/calibration.json" \
  --scores "$RUN/scores/H_seed1_cal.json" --output "$RUN/policies/H_seed1.json"
```

12개 정책을 모두 저장하고 SHA를 고정한 뒤 synth_val512를 연다. calibration에서 지표가 좋다고 본 실험 성공으로 기록하지 않는다.

### 9.2 합성 검증

```bash
python -m pointline_v4.score_cache --manifest "$EXPORT/synth_val.json" \
  --checkpoint "$RUN/heads/H_seed1/checkpoint_final.pt" --output "$RUN/scores/H_seed1_synth.json" --device cuda:0
python -m pointline_v4.evaluate_scores evaluate --manifest "$EXPORT/synth_val.json" \
  --scores "$RUN/scores/H_seed1_synth.json" --policy "$RUN/policies/H_seed1.json" \
  --output "$RUN/evaluations/H_seed1_synth.json"
```

네 군의 seed1/2/3 파일을 각각 제공 `compare_evaluations`의 `--P`, `--S`, `--H`, `--HA`에 순서대로 넘긴다. 각 인자는 3개 경로를 받는다. CLI가 경로를 실제 생성한 파일에 맞춰 구성한다.

실사 진입 조건은 H가 **세 seed 각각** 다음을 만족하는 것이다.

- 기준 출력 및 P보다 주 지표가 각각 1% 이상 낮음.
- 원본 픽셀 pooled 8코너 median/P90이 기준과 P보다 나빠지지 않음.
- 양호점 손상≤1%; 동일 mask·coverage 유지; identity-only로 모든 결과가 같은 경우 통과 불가.

[미검증 제안] 1%와 양호점 손상1%는 이번 제한된 예비실험의 실용 기준이다. 기존 판정을 새 기준으로 뒤집지 않는다. 실패하면 학습 확대·실사 새 추론 없이 `NO_SYNTHETIC_ADVANCEMENT_SIGNAL`로 마무리한다. S가 좋아도 H를 성공으로 대체하지 않는다. S의 결과는 후속 목적 재선택 근거로만 보고한다.

### 9.3 실사 DEV

합성 기준 통과 시에만 기존 positive319의 GT-free 관측을 export하고 같은 12개 head를 평가한다. margin/모델/seed/규칙을 바꾸지 않는다. GT는 scoring 완료 후 평가 단계에서 연다. 기존 실패 사례 `eval_pallet07:1778652166837872128`는 설명용이며 규칙 선택용이 아니다.

주 보고는 8코너, 기존 보고서와의 교차검증은 중심 포함9점이다. 합성/실사, 8코너/9점, backbone identity/공동학습 후 identity, pooled 통계/프레임 평균을 표 제목에 모두 구분한다. 319개 프레임 수와 매칭/관측점 분모는 별도 보고한다.

고정 backbone이라도 6D 출력은 점 변경으로 달라질 수 있다. 기존 canonical evaluator로 원래9점 출력→동일 PnP/pose 평가를 실행한다. 점 오차가 좋아졌다는 이유만으로 6D를 생략하거나 향상으로 단정하지 않는다.

실사 progression 조건: 위 수치·안전 조건을 유지하고, 공통 프레임에서 seed를 먼저 평균한 **H−P 및 H−기준** 주 지표 차이의 13세션 paired bootstrap 95% 구간 상한이 모두0 미만. 20,000 draws, seed20260909. 세션을 재표집할 때 그 안의 모든 paired frame을 유지한다. 학습 seed 불확실성은 별도 SD/개별표로 보고한다. 독립 세션 개수가 적고 DEV를 반복 사용했음을 명시한다.

H−S가 개선되지 않으면 명시 DHT 추가 효과는 확인되지 않은 것이다. S−P가 개선되지 않으면 유한 내부 읽기 효과도 확인되지 않은 것이다. H−HA가 개선되지 않으면 번호 anchor 원인 가설을 확정하지 않는다. wrong-line 교란으로 성능이 떨어졌다는 사실만으로 상보성을 주장하지 않는다.

`compare_evaluations`의 숫자 PASS 외에 canonical metric parity·데이터 무결성·원본 불변성을 별도 검사한 뒤 전체 학습 진입 여부를 결정한다. JSON의 `official_metric_parity_checked=false`를 근거 없이 true로 편집하지 않는다. 별도 실제 parity 검증 receipt를 만든다.

## 10. 단계 C — 통과한 경우 제한된 전체 네트워크 실험

이 단계는 작은 cache 실험에서 **합성 및 실사 progression을 모두 통과하고, 실제 adapter/canonical 검증까지 유효할 때만** 수행한다. 그렇지 않으면 미실행으로 명시한다. 단순 코드 테스트 PASS가 실행 조건이 아니다.

[미검증 제안] 전체 네트워크에서 네 군 `J0, J_P, J_S, J_H` × seeds1,2,3을 비교한다.

- J0: **같은 joint backbone snapshot**의 원래 점·box 출력 경로를 동일 예산으로 더 학습한다. 순수 stock YOLO라는 이름을 붙이지 않는다.
- J_P/S/H: 동일 snapshot과 각 증거 읽기 구조를 한 모델에 연결하여 학습한다.
- 작은 head를 재사용한다면 각 seed의 대응 head로 P/S/H 모두 동일하게 재사용한다. 주 방법 H만 예열 checkpoint를 쓰지 않는다.
- 원래 joint의 stock loss·line loss·기존 detach 경계/EMA/증강 scheduler를 먼저 읽어 그대로 공통 유지한다. 새 verifier loss의 weight는 **0.1 고정**으로 본학습 전에 잠근다. 이는 이번 [미검증 제안]이며 성공 보장이 아니다. 첫 smoke에서 기울기 norm을 기록하되 실사 결과로 계수를 탐색하지 않는다.
- live feature tensor가 제공 scorer까지 이어지고 backbone/P3/P4로 실제 gradient가 전달되어야 한다. NumPy/CPU cache 변환을 online forward 안에 넣어 끊지 않는다. 후보 제안/정답 비용은 detach해 proposal이 target 비용을 조작하지 못하게 한다. 별도 원래 점 loss는 좌표 경로를 계속 감독한다.
- 최종 hard selection 자체는 미분 불가능하다. “하나의 네트워크가 공동학습된다”와 “모든 추론 연산이 매끄럽게 미분 가능하다”를 구분한다.

공통 source train55,980/val4,020의 실제 manifest를 확인한 뒤, **각 2epochs, FP32 effective batch16, 실제 optimizer updates6,998**를 상한 예산으로 사용한다. 실제 loader의 마지막 batch/drop_last/accumulation 때문에 수가 다르면 6,998을 보고서에 강제로 넣지 않는다. 본학습 전에 계약을 해소하고 모든 군의 sample/update 수를 같게 만든다. 주 optimizer 설정은 기존 joint protocol을 그대로 복사·검증하며 이번 네 군에서만 새 탐색하지 않는다.

학습 규칙은 마지막 epoch EMA 고정이다. 전체 네트워크가 달라졌으므로 작은 head의 margin을 그대로 재사용해도 최적이라는 보장은 없다. **동일한 source calibration256과 같은 고정 grid로 12개 새 policy를 다시 정하고, 실사 전에 모두 고정**한다. J0의 정책은 identity-only다. Calibration/validation이 backbone 학습에서 엄밀히 독립이 아님을 유지한다.

모든 새 전체 모델은 positive319+negative2689를 실제로 추론한다. 이전 negative cache 복사로 대체하지 않는다. 검출 confidence/box/9점 모두 바뀔 수 있다. 원래 evaluator의 matching/AP/FP/PCK/MAIN6D 계약을 그대로 적용한다.

전체 모델은 매칭 집합이 달라질 수 있으므로 cache용 `compare_evaluations`를 억지로 재사용하지 않는다. 모든 비교 모델·seed에서 관측 가능한 **사전 정의 공통 프레임**의 paired 차이와, 전체 모집단의 검출/관측 coverage·미매칭 수·전체 GT 분모 PCK를 함께 보고한다. 유리한 프레임만 남겨 조건부 정확도를 높이지 않는다. coverage가 기준보다 감소한 seed를 숨기거나 drop하지 않는다.

주 성능 기준은 J_H 대 J_P 및 J0의 normalized frame-mean 8코너 오차 1% 이상 감소와 세션95%CI 상한<0, pooled median/P90 비악화, 양호점 손상≤1%, 각 seed에서 검출/관측 coverage 보존이다. 6D rotation/translation/IoU3D/ADDsym AUC와 AP/FP도 모두 보고한다. **2D만 좋아지면 결과명을 2D 개선으로 제한**하며 6D/검출이 나빠졌는데 종합 향상이라고 쓰지 않는다. 초기 목적이 종합 아키텍처 채택이므로 그 경우 배포 승격은 하지 않는다.

실제 input→전처리→backbone/DHT→후보→verifier→원좌표의 시간을 같은 hardware/FP32/batch1에서 측정한다. 이미지 decoding과 PnP 포함 여부를 따로 명시한다. strict output parity 실패는 원래 tolerance 그대로 보고하고 accuracy의 다중-pixel 오류 원인으로 단정하지 않는다.

**최대 학습 예산:** 작은 head 12×2,000=24,000 updates. 전체 단계에 진입한 경우에만 12×6,998=83,976 updates 추가. 실패 후 epoch 추가, seed 추가, 세 번째 loss, 후보폭 grid, 데이터 증량은 승인 범위 밖이다. 본학습이 아니라 버그 수정용 smoke는 군당4updates로 별도 폴더에 저장한다.

## 11. Runner — 실행하지 않는 대기 루프를 만들지 말라

전체 runner는 foreground에서 현재 작업을 수행한다. 넓은 `pgrep -f`로 다른 과제/자기 명령까지 매칭해 대기하는 방식은 금지한다. 새 run별 lock을 사용하고 소유 PID/시작시간/command를 기록한다. 다른 사용자의 프로세스를 종료하지 않는다.

학습/eval 각각 자식 process의 exit code와 실제 artifact를 검사한다. 완료 marker는 기대 step 수·checkpoint SHA·원본 불변성 검증 후에만 기록한다. 완료된 군을 재사용하려면 protocol/source/input/initial/trace binding을 전부 검증한다. checkpoint 파일이 존재한다는 이유만으로 완료라고 쓰지 않는다.

부분 실행이 남아 있으면 보존하고 `INCOMPLETE_CELL_REQUIRES_RECONCILIATION`로 중단한다. 예고 없이 덮어쓰기나 자동 처음부터 재학습을 하지 않는다. 진행 실패·GPU 메모리 부족·계약 위반 시 무한 재시도하지 않는다. 동일 오류 자동 재시도 상한은0회다. 명확한 구현 버그를 고친 경우 별도 patch log와 영향 분석 후 새 smoke를 수행하고, 결과를 본 상태에서 학습 의미를 바꾸면 새 실험으로 분리한다.

120초 동안 step/heartbeat가 없거나 자식이 종료되면 상태를 점검하고, 원인이 확인되지 않으면 소유 자식만 정리해 실패를 기록한다. 이 숫자는 미래 완료 시간의 예측이 아니라 stall 감지용 [미검증 제안]이다. 무거운 초기 cache 작업은 진행 건수 heartbeat를 따로 내어 정상 작업을 학습 정지로 오인하지 않게 한다.

## 12. 실제로 남겨야 할 결과

최종 결과에 `execution_completed`, `integrity_valid`, `synthetic_signal`, `real_cached_head_signal`, `full_network_executed`, `accuracy_claim_scope`를 분리한다. 예를 들어 학습이 모두 끝나도 향상 기준을 못 넘으면 completed=true, accuracy_improved=false다. 차단 상태라면 어떤 단계와 어떤 증거가 부족했는지 적는다.

필수 산출물은 `PURPOSE.md`, `SOURCE_REGISTRY.json`, `ADAPTER_MAPPING.md`, `DATA_CONTRACT.json`, `PREFLIGHT.json`, `PROTOCOL_LOCK.json`, native oracle 진단, 실제 training trace/checkpoint/초기값 검증, 정책·평가·원좌표, paired 비교, `VERDICT.json`, `REPORT_KO.md`다. 생성 여부가 아니라 각 파일의 실제 내용·출처를 감사한다.

보고서의 독자 질문 순서는 다음을 지킨다: 문제 → 기존 실패가 남긴 구체적인 가설 → 네 비교군에서 바뀐 것 → 데이터/정답/예산 근거 → 결과가 어떤 주장까지 지지하는지 → 한계/다음. 특히 다음 구멍은 결과보다 앞에 보고한다.

- 이전 v2와 실제로 무엇이 새로 다른가?
- P/S는 정말 Hough 없는 모델인가? 아니라면 조건부 효과임을 밝혔는가?
- 후보가 원래 예측보다 나을 여지가 있는가?
- source native에서 좋아지는가, 인위적 번호교란에서만 좋아지는가?
- 좋은 후보가 있는데 점수가 싫어한 것인가, 후보가 없었던 것인가?
- 양호한 점을 망가뜨리지 않고 큰 번호 오류를 줄였는가?
- 8코너·9점·매칭 분모·seed별 평균·공통 paired 비교를 구분했는가?
- 전체 네트워크는 실제로 실행했는가? 실행하지 않았다면 왜인가?

시각화는 정량 판정 이후 설명용이다. 세션별 사전 SHA 표본1장과 기존 사용자 실패 사례를 사용하고 성공 사례만 고르지 않는다. 원본/GT/원래 ID/선택 후보/최종 ID/선분 내부 query를 동일 좌표로 보여준다. GT가 정밀 물리 edge라고 인증되지 않은 경우 그 한계를 캡션에 남긴다. 완성 HTML·브라우저·알림 제작을 정확도 진단보다 먼저 하지 않는다. 외부 전송/Discord 호출은 이번 지시 범위에 없다.

최종 답변에는 실제 수행한 단계, 핵심 수치와 분모, 미수행 단계와 이유, 실패 원인 중 [확인]과 [추정], 다음으로 허용되는 행동 한 가지를 적는다. 성공 보장이나 “진행 중이니 나중에 전달”이라는 말로 끝내지 않는다.
